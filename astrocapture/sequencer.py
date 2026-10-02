"""Capture sequencer: runs a plan against a mount + camera.

State machine::

    IDLE -> SLEWING -> EXPOSING -> (DITHERING -> EXPOSING)* -> DONE
       |        |          |
       v        v          v
     PAUSED <-> (resume)  ABORTED -> ERROR?

``pause()`` / ``resume()`` / ``abort()`` are thread-safe and intended to
be called from another thread (the CLI, a GUI, a signal handler) while
``run()`` blocks in the main thread.

Autonomous imaging (all optional plan blocks, all off by default):

* per-target platesolve: after each slew, ``platesolve.recenter()``
  closed-loop centers the target; consecutive failures trip the
  watchdog, which parks the mount and stops the run.
* autofocus: V-curve ``focus.autofocus()`` before the first light frame,
  on a timer, and when light-frame HFR degrades; falls back to the
  manual Bahtinov-mask ``assist_mode()`` when no motorized focuser is
  reachable (the honest path for a stock NexStar 6SE).
* PHD2 guiding: dithers go through ``PHD2Client.dither()`` instead of
  the blind mount nudge; a lost guide star goes through the watchdog's
  retry budget before parking.
* safety watchdog: wraps the run loop — any exception parks the mount
  (swallow-after-parking), and the wall-clock session limit stops the
  run gracefully.
"""

from __future__ import annotations

import enum
import math
import shutil
import threading
import time

import numpy as np

from astrocapture import config as plan_config
from astrocapture.drivers.base import Camera, Mount
from astrocapture.focus import (
    INDIFocuser,
    SimFocuser,
    assist_mode,
    autofocus,
    measure_hfr,
)
from astrocapture.phd2 import PHD2Client, PHD2Error
from astrocapture.platesolve import FakeSolver, PlateSolver, recenter
from astrocapture.safety import AlertLog, Watchdog, WebhookAlert
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
        solver: PlateSolver | FakeSolver | None = None,
        focuser=None,
        phd2_client: PHD2Client | None = None,
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
        # -- autonomous-imaging wiring (all optional) -------------------
        self._solver = solver  # FakeSolver in tests; None -> real PlateSolver
        self._focuser = focuser  # injected Focuser (tests); else auto-setup
        self._focuser_client = None  # INDIClient owned by the focuser path
        self._phd2 = phd2_client  # injected PHD2 client (tests)
        self._guiding_started = False
        self.watchdog: Watchdog | None = None
        self._last_focus_monotonic: float | None = None
        self._focus_baseline_hfr = float("nan")
        self._hfr_recent: list[float] = []
        self.last_focus_position: int | None = None

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
        self._build_watchdog()
        try:
            self._connect_all()
            self._run_sequence()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - watchdog parks, then stop
            # Swallow-after-parking: the watchdog parks the mount, releases
            # the camera and alerts; the sequence ends ABORTED, not ERROR.
            assert self.watchdog is not None
            self.watchdog.handle_exception(exc)
            self.state = SeqState.ABORTED
            self.error = str(exc)
            log.error("sequencer aborted after exception: %s", exc)
        finally:
            self._shutdown()
        return self.state

    def _build_watchdog(self) -> None:
        safety = self.plan.safety
        url = self.plan.alerts.webhook_url
        max_session_seconds = (
            safety.max_session_hours * 3600.0
            if safety.max_session_hours else None
        )
        base_alert = WebhookAlert(url).alert if url else AlertLog().alert

        def alert(msg: str) -> None:
            # Every watchdog alert is also written to session.log so the
            # unattended run's record shows why it parked/stopped.
            self.log.warning("ALERT: %s", msg)
            base_alert(msg)

        self.watchdog = Watchdog(
            park_mount=self.mount.park,
            stop_sequence=self.abort,
            alert=alert,
            release_camera=self.camera.disconnect,
            max_solve_failures=safety.max_solve_failures,
            guide_retry_budget=safety.guide_retry_budget,
            max_session_seconds=max_session_seconds,
        )

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
        self._setup_guiding()
        self._setup_focuser()

    def _is_sim_rig(self) -> bool:
        return (self.plan.mount.driver == "sim"
                and self.plan.camera.driver == "sim")

    # -- platesolve ---------------------------------------------------------
    def _platesolve_recenter(self, entry: plan_config.TargetEntry) -> bool:
        """Closed-loop centering after the slew.

        Solve-failure accounting policy (a deliberate choice): a missing
        ``solve-field`` binary is detected up front, logged as "skipped",
        and never counted — that solve is *inconclusive*, not failed.
        Only a solver that runs but fails to converge within
        ``max_iterations`` (``recenter`` returning False) counts toward
        the watchdog's consecutive-failure budget, which parks the mount
        and stops the run at the threshold.
        """
        cfg = entry.platesolve
        tgt = entry.target
        solver = self._solver
        if solver is None:
            if shutil.which("solve-field") is None:
                self.log.info("platesolve: solve-field not installed — "
                              "skipping recenter on %s", tgt.name)
                return True
            solver = PlateSolver()
        self.log.info("platesolve: recentering on %s (tol %.1f', up to %d "
                      "iterations)", tgt.name, cfg.tolerance_arcmin,
                      cfg.max_iterations)
        ok = recenter(self.mount, self.camera, self.session,
                      tgt.ra_hours * 15.0, tgt.dec_deg,
                      tolerance_arcmin=cfg.tolerance_arcmin,
                      max_iterations=cfg.max_iterations,
                      exposure_s=cfg.exposure_s,
                      solver=solver)
        assert self.watchdog is not None
        if ok:
            self.watchdog.note_solve_success()
            return True
        self.log.warning("platesolve: failed to converge on %s within %d "
                         "iterations", tgt.name, cfg.max_iterations)
        self.watchdog.note_solve_failure()
        if self._abort.is_set():
            # The watchdog tripped: mount parked, sequence stopped.
            self.state = SeqState.ABORTED
            return False
        return True

    # -- autofocus ----------------------------------------------------------
    def _setup_focuser(self):
        """Build the focuser, or fall back to manual assist mode."""
        if self._focuser is not None:  # injected (tests)
            return
        cfg = self.plan.autofocus
        if not cfg.enabled:
            return
        if self._is_sim_rig():
            self._focuser = SimFocuser()
            self.log.info("autofocus: using SimFocuser (sim rig)")
            return
        if cfg.focuser_device and self.plan.mount.driver == "indi":
            focuser = self._connect_indi_focuser(cfg)
            if focuser is not None:
                self._focuser = focuser
                return
        # No motorized focuser: the honest path for a stock 6SE.
        assist_mode(self.plan.target.name)
        self.log.info("autofocus: no motorized focuser configured — "
                      "manual-assist mode (see printed Bahtinov-mask guide)")

    def _connect_indi_focuser(self, cfg: plan_config.AutofocusConfig):
        """Connect a motorized focuser through the mount's indiserver."""
        from astrocapture.drivers.indi import INDIClient

        opts = self.plan.mount.options
        client = INDIClient(host=str(opts.get("host", "localhost")),
                            port=int(opts.get("port", 7624)))
        try:
            client.connect()
            client.wait_for_device(cfg.focuser_device, timeout=10.0)
            focuser = INDIFocuser(client, cfg.focuser_device)
            pos = focuser.get_position()  # reachability probe
        except Exception as exc:  # noqa: BLE001 - fall back to assist mode
            self.log.warning("autofocus: focuser device %r not reachable "
                             "(%s) — falling back to manual assist",
                             cfg.focuser_device, exc)
            try:
                client.disconnect()
            except Exception:  # noqa: BLE001 - best effort
                pass
            return None
        self._focuser_client = client
        self.log.info("autofocus: INDI focuser %r at position %d",
                      cfg.focuser_device, pos)
        return focuser

    def _focus_measure(self) -> float:
        """One HFR measurement for the autofocus loop.

        Focusers that provide their own ``measure()`` (SimFocuser) are
        used directly — fast and deterministic.  Otherwise a short
        exposure is taken and measured with ``measure_hfr``.
        """
        measure = getattr(self._focuser, "measure", None)
        if callable(measure):
            return float(measure())
        cfg = self.plan.autofocus
        self.camera.set_exposure_settings(cfg.focus_exposure_s)
        self.camera.start_exposure()
        while not self.camera.exposure_complete():
            time.sleep(0.1)
        return measure_hfr(self.camera.download_image())

    def _run_autofocus(self, reason: str) -> None:
        cfg = self.plan.autofocus
        focuser = self._focuser
        if focuser is None:
            return
        self.log.info("autofocus: starting (%s)", reason)
        try:
            final = autofocus(focuser, self._focus_measure,
                              n_positions=cfg.n_positions, step=cfg.step,
                              settle_s=1.0, samples=3)
        except Exception as exc:  # noqa: BLE001 - focus failure isn't fatal
            self.log.warning("autofocus: failed (%s) — continuing unfocused",
                             exc)
            return
        self.last_focus_position = final
        self._last_focus_monotonic = time.monotonic()
        self._hfr_recent.clear()
        self._focus_baseline_hfr = self._focus_measure()
        self.log.info("autofocus: done at position %d "
                      "(baseline HFR %.2f px)", final,
                      self._focus_baseline_hfr)

    def _maybe_scheduled_focus(self) -> None:
        cfg = self.plan.autofocus
        if not (cfg.enabled and self._focuser is not None):
            return
        if self._last_focus_monotonic is None:
            return  # initial focus runs separately, before the first light
        elapsed = time.monotonic() - self._last_focus_monotonic
        if elapsed >= cfg.every_minutes * 60.0:
            self._run_autofocus(
                f"scheduled refocus ({elapsed / 60.0:.1f} min since last)")

    def _track_frame_hfr(self, image) -> None:
        """Refocus when light-frame HFR degrades vs the post-focus baseline."""
        cfg = self.plan.autofocus
        if not (cfg.enabled and self._focuser is not None):
            return
        if not math.isfinite(self._focus_baseline_hfr):
            return
        try:
            hfr = measure_hfr(image)
        except Exception as exc:  # noqa: BLE001 - never fail a frame on HFR
            self.log.debug("HFR measurement failed: %s", exc)
            return
        if not math.isfinite(hfr):
            return  # no usable stars: no measurement, never "in focus"
        self._hfr_recent.append(hfr)
        self._hfr_recent = self._hfr_recent[-3:]
        if len(self._hfr_recent) >= 2:
            med = float(np.median(self._hfr_recent))
            if med >= self._focus_baseline_hfr * cfg.hfr_degradation_trigger:
                self.log.info("HFR degraded: median %.2f px vs baseline "
                              "%.2f px (trigger x%.2f) — refocusing",
                              med, self._focus_baseline_hfr,
                              cfg.hfr_degradation_trigger)
                self._run_autofocus("HFR degradation trigger")

    # -- guiding ------------------------------------------------------------
    def _setup_guiding(self) -> None:
        cfg = self.plan.guiding
        if not cfg.enabled:
            return
        if self._phd2 is not None:  # injected (tests)
            self.log.info("guiding: using injected PHD2 client")
            return
        client = PHD2Client()
        try:
            client.connect(cfg.phd2_host, cfg.phd2_port)
        except PHD2Error as exc:
            # Failure to reach PHD2 is a configuration problem, not a
            # crash: fall back to unguided imaging with blind dithers.
            self.log.warning("guiding: %s — continuing unguided", exc)
            return
        self._phd2 = client
        self.log.info("guiding: connected to PHD2 at %s:%d",
                      cfg.phd2_host, cfg.phd2_port)

    def _start_guiding_once(self) -> None:
        if self._guiding_started or self._phd2 is None:
            return
        self._guiding_started = True
        cfg = self.plan.guiding
        if self._phd2.start_guiding(cfg.settle_pixels, cfg.settle_time_s):
            self.log.info("guiding: PHD2 guiding started")
        else:
            self.log.warning("guiding: PHD2 failed to start guiding — "
                             "continuing unguided")
            try:
                self._phd2.close()
            except Exception:  # noqa: BLE001 - best effort
                pass
            self._phd2 = None

    def _poll_guiding(self) -> bool:
        """Poll the guide-star state. False -> watchdog parked: stop."""
        client = self._phd2
        if client is None:
            return True
        if not client.star_lost:
            return True
        assert self.watchdog is not None
        decision = self.watchdog.note_guide_lost()
        if decision == Watchdog.PARK:
            self.log.error("guiding: guide star lost repeatedly — "
                           "watchdog parked the mount")
            self.state = SeqState.ABORTED
            return False
        self.log.warning("guiding: guide star lost — attempting resume")
        cfg = self.plan.guiding
        if client.start_guiding(cfg.settle_pixels, cfg.settle_time_s):
            client.clear_star_lost()
            self.watchdog.note_guide_recovered()
            self.log.info("guiding: resumed after star-lost")
        return True

    def _dither(self) -> bool:
        """Dither between light frames. False -> abort the run."""
        if self._phd2 is not None:
            cfg = self.plan.guiding
            self.log.info("dithering via PHD2 (%.1f px)", cfg.dither_pixels)
            if not self._phd2.dither(cfg.dither_pixels, cfg.settle_pixels,
                                     cfg.settle_time_s):
                self.log.warning("PHD2 dither failed to settle — continuing")
            return True
        dra, ddec = self.ditherer.next_offset()
        self.log.info("dithering by %+.1f\" / %+.1f\"", dra, ddec)
        self.mount.offset(dra, ddec)
        if not self._wait(2.0):  # settle
            self.state = SeqState.ABORTED
            return False
        if hasattr(self.camera, "apply_dither"):
            # SimCamera keeps its synthetic sky aligned with the
            # dithered pointing. 1px ~= 2" in the sim.
            self.camera.apply_dither(dra / 2.0, ddec / 2.0)
        return True

    # -- sequence -----------------------------------------------------------
    def _slew_to_target(self, tgt: plan_config.Target) -> bool:
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
        first_target = True
        for entry in self.plan.targets:
            tgt = entry.target
            if not self._slew_to_target(tgt):
                return
            if first_target:
                first_target = False
                self._start_guiding_once()
                if self.plan.autofocus.enabled and self._focuser is not None:
                    self._run_autofocus("initial focus before first light")
            if entry.platesolve.enabled and not self._platesolve_recenter(entry):
                return
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
                self._run_step(step, tgt)
                if self.state == SeqState.ABORTED:
                    self.log.warning("sequence aborted; stopping")
                    return
        self.state = SeqState.DONE
        self.log.info("sequence complete: %d frames", self.frames_taken)

    def _run_step(self, step: plan_config.Step, tgt: plan_config.Target) -> None:
        for i in range(step.count):
            if not self._checkpoint(f"before frame {i+1}/{step.count}"):
                return
            if self.watchdog is not None and self.watchdog.check_session_limits():
                self.state = SeqState.ABORTED
                self.log.warning("session time limit reached; stopping")
                return
            # Dither between light frames.
            if step.type == "light" and step.dither_every and i > 0 \
                    and i % step.dither_every == 0:
                self.state = SeqState.DITHERING
                if not self._dither():
                    return
            if step.type == "light":
                self._maybe_scheduled_focus()
                if not self._poll_guiding():
                    return

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
                                           ra, dec, temp, target=tgt)
            self.frames_taken += 1
            if step.type == "light":
                self._track_frame_hfr(image)
            if self.plan.plate_solve and step.type == "light" and i == 0:
                plate_solve(path, self.log)

    def _shutdown(self) -> None:
        if self._phd2 is not None:
            try:
                self._phd2.stop_guiding()
            except Exception as exc:  # noqa: BLE001 - best effort
                self.log.warning("PHD2 stop_guiding failed: %s", exc)
            try:
                self._phd2.close()
            except Exception as exc:  # noqa: BLE001 - best effort
                self.log.warning("PHD2 close failed: %s", exc)
            self._phd2 = None
        if self._focuser_client is not None:
            try:
                self._focuser_client.disconnect()
            except Exception as exc:  # noqa: BLE001 - best effort
                self.log.warning("focuser INDI disconnect failed: %s", exc)
            self._focuser_client = None
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
