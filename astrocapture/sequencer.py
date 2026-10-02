"""Capture sequencer: runs a plan against a mount + camera.

State machine::

    IDLE -> SLEWING -> EXPOSING -> (DITHERING -> EXPOSING)* -> DONE
       |        |          |
       v        v          v
     PAUSED <-> (resume)  ABORTED -> ERROR?

``pause()`` / ``resume()`` / ``abort()`` are thread-safe and intended to
be called from another thread (the CLI, a GUI, a signal handler) while
``run()`` blocks in the main thread.
"""

from __future__ import annotations

import enum
import threading
import time

from astrocapture import config as plan_config
from astrocapture.drivers.base import Camera, Mount
from astrocapture.session import (
    Ditherer,
    Session,
    check_meridian_flip,
    plate_solve,
)


class SeqState(enum.Enum):
    IDLE = "idle"
    SLEWING = "slewing"
    EXPOSING = "exposing"
    DITHERING = "dithering"
    PAUSED = "paused"
    ABORTED = "aborted"
    DONE = "done"
    ERROR = "error"


class Sequencer:
    def __init__(
        self,
        plan: plan_config.Plan,
        mount: Mount,
        camera: Camera,
        session: Session | None = None,
        dither_max_arcsec: float = 15.0,
        lst_hours: float | None = None,
    ) -> None:
        self.plan = plan
        self.mount = mount
        self.camera = camera
        self.session = session or Session(plan)
        self.log = self.session.log
        self.ditherer = Ditherer(dither_max_arcsec)
        self.lst_hours = lst_hours  # LST for meridian-flip check; None = skip
        self.state = SeqState.IDLE
        self.frames_taken = 0
        self._pause = threading.Event()
        self._pause.set()  # set == not paused
        self._abort = threading.Event()
        self.error: str | None = None

    # -- external control -------------------------------------------------
    def pause(self) -> None:
        if self.state in (SeqState.SLEWING, SeqState.EXPOSING, SeqState.DITHERING):
            self.log.info("pause requested")
            self._pause.clear()

    def resume(self) -> None:
        self.log.info("resume requested")
        self._pause.set()

    def abort(self) -> None:
        self.log.warning("abort requested")
        self._abort.set()
        self._pause.set()  # unblock a paused wait so abort lands

    # -- main run ----------------------------------------------------------
    def run(self) -> SeqState:
        log = self.log
        try:
            self._connect_all()
            self._run_sequence()
        except Exception as exc:  # noqa: BLE001 - report, don't crash silently
            self.state = SeqState.ERROR
            self.error = str(exc)
            log.exception("sequencer failed: %s", exc)
        finally:
            self._shutdown()
        return self.state

    # -- internals ----------------------------------------------------------
    def _wait(self, seconds: float, poll: float = 0.2) -> bool:
        """Sleep in slices; returns False if aborted during the wait."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._abort.is_set():
                return False
            self._pause.wait()  # blocks here while paused
            if self._abort.is_set():
                return False
            time.sleep(min(poll, max(0.0, deadline - time.monotonic())))
        return True

    def _checkpoint(self, what: str) -> bool:
        """Handle pause/abort between operations. False -> stop the run."""
        if self._abort.is_set():
            self.state = SeqState.ABORTED
            self.log.warning("aborted %s", what)
            return False
        if not self._pause.is_set():
            self.state = SeqState.PAUSED
            self.log.info("paused %s; waiting for resume", what)
            self._pause.wait()
            self.state = SeqState.IDLE  # will be set properly by next phase
            if self._abort.is_set():
                self.state = SeqState.ABORTED
                return False
        return True

    def _connect_all(self) -> None:
        self.log.info("connecting mount (%s) and camera (%s)",
                      self.plan.mount.driver, self.plan.camera.driver)
        self.mount.connect()
        self.camera.connect()
        if self.plan.cooler_temp_c is not None and self.camera.has_cooler:
            self.camera.set_cooler(self.plan.cooler_temp_c)
            self.log.info("cooler setpoint %.1f C", self.plan.cooler_temp_c)
        self.mount.unpark()
        self.log.info("mount unparked")

    def _slew_to_target(self) -> bool:
        tgt = self.plan.target
        self.state = SeqState.SLEWING
        self.log.info("slewing to %s (RA %.4fh Dec %+.4f°)",
                      tgt.name, tgt.ra_hours, tgt.dec_deg)
        self.mount.goto(tgt.ra_hours, tgt.dec_deg)
        while not self.mount.slew_complete():
            if not self._checkpoint("during slew"):
                return False
            time.sleep(0.2)
        self.mount.start_tracking()
        ra, dec = self.mount.position
        self.log.info("on target: RA %.4fh Dec %+.4f°", ra, dec)
        return True

    def _run_sequence(self) -> None:
        if not self._slew_to_target():
            return
        tgt = self.plan.target
        for step in self.plan.steps:
            if not self._checkpoint(f"before {step.type} block"):
                return
            if step.type == "light" and self.plan.meridian_flip and self.lst_hours is not None:
                if check_meridian_flip(tgt.ra_hours, self.lst_hours, self.log):
                    self.state = SeqState.PAUSED
                    self.log.warning("paused for manual meridian flip")
                    self._pause.clear()
                    self._pause.wait()
                    self._pause.set()
            self._run_step(step)
            if self.state == SeqState.ABORTED:
                self.log.warning("sequence aborted; stopping")
                return
        self.state = SeqState.DONE
        self.log.info("sequence complete: %d frames", self.frames_taken)

    def _run_step(self, step: plan_config.Step) -> None:
        tgt = self.plan.target
        for i in range(step.count):
            if not self._checkpoint(f"before frame {i+1}/{step.count}"):
                return
            # Dither between light frames.
            if step.type == "light" and step.dither_every and i > 0 \
                    and i % step.dither_every == 0:
                self.state = SeqState.DITHERING
                dra, ddec = self.ditherer.next_offset()
                self.log.info("dithering by %+.1f\" / %+.1f\"", dra, ddec)
                self.mount.offset(dra, ddec)
                if not self._wait(2.0):  # settle
                    self.state = SeqState.ABORTED
                    return
                if hasattr(self.camera, "apply_dither"):
                    # SimCamera keeps its synthetic sky aligned with the
                    # dithered pointing. 1px ~= 2" in the sim.
                    self.camera.apply_dither(dra / 2.0, ddec / 2.0)

            self.state = SeqState.EXPOSING
            self.log.info("exposing %s %d/%d: %.1fs gain=%g",
                          step.type, i + 1, step.count, step.exposure, step.gain)
            self.camera.set_exposure_settings(step.exposure, step.gain, step.binning)
            self.camera.start_exposure()
            while not self.camera.exposure_complete():
                if not self._wait(0.2):
                    try:
                        self.camera.abort_exposure()
                    except Exception:  # noqa: BLE001 - best effort
                        pass
                    self.state = SeqState.ABORTED
                    return
            image = self.camera.download_image()
            ra, dec = self.mount.position
            temp = self.camera.get_temperature()
            path = self.session.save_frame(image, step.type, step,
                                           ra, dec, temp)
            self.frames_taken += 1
            if self.plan.plate_solve and step.type == "light" and i == 0:
                plate_solve(path, self.log)

    def _shutdown(self) -> None:
        if self.state == SeqState.DONE:
            self.log.info("parking mount")
            try:
                self.mount.stop_tracking()
                self.mount.park()
            except Exception as exc:  # noqa: BLE001 - park is best effort
                self.log.warning("park failed: %s", exc)
        for dev, name in ((self.camera, "camera"), (self.mount, "mount")):
            try:
                dev.disconnect()
            except Exception as exc:  # noqa: BLE001
                self.log.warning("%s disconnect failed: %s", name, exc)
        if self.state not in (SeqState.DONE, SeqState.ABORTED, SeqState.ERROR):
            self.state = SeqState.ABORTED
