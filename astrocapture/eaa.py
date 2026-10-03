"""Electronically-Assisted Astronomy: live stacking while you watch.

:class:`LiveStacker` accumulates frames into a progressively improving
mean stack — the EAA workflow (short subs, watch the target emerge in
near-real time).  Each new frame is optionally calibrated with master
frames supplied at construction, then translation-registered to the
**original reference** (the first frame), never to the running stack:
registering to a moving target would let small per-frame errors
accumulate into drift.

The ``eaa`` CLI command drives this with the simplest robust path: its
own small capture loop (``make_camera`` + direct expose/download, no
mount, no Session files) over the plan's light steps.  It deliberately
does *not* reuse :class:`~astrocapture.sequencer.Sequencer` — the
sequencer writes files, dithers, plate-solves and manages mount state,
none of which a live EAA view needs, and skipping it keeps the loop
tight and failure-proof.

Registration is translation-only (see
:func:`astrocapture.imaging.estimate_shift`); field rotation between
subs is NOT corrected.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from astrocapture import config
from astrocapture.drivers import make_camera
from astrocapture.imaging import (
    detect_star_centroids,
    estimate_shift,
    shift_image,
)

log = logging.getLogger(__name__)


class LiveStacker:
    """Progressive mean stack with per-frame registration to frame one.

    Parameters
    ----------
    master_bias, master_dark, master_flat:
        Optional calibration masters (numpy arrays); applied to every
        frame before registration, exactly like
        :func:`astrocapture.process.calibrate_light`.
    max_shift_px:
        Largest frame-to-frame translation trusted for registration.
    """

    def __init__(
        self,
        master_bias: np.ndarray | None = None,
        master_dark: np.ndarray | None = None,
        master_flat: np.ndarray | None = None,
        max_shift_px: float = 50.0,
    ) -> None:
        self.master_bias = master_bias
        self.master_dark = master_dark
        self.master_flat = master_flat
        self.max_shift_px = max_shift_px
        self._stack: np.ndarray | None = None
        self._n = 0
        self._ref_centroids: list[tuple[float, float]] | None = None
        self._last_shift: tuple[float, float] = (0.0, 0.0)

    # -- capture --------------------------------------------------------
    def add_frame(self, data: np.ndarray) -> None:
        """Calibrate, register and fold one frame into the running stack.

        The first frame becomes the registration reference (its centroids
        are kept); every later frame is shifted to match *it*.  Frames
        that fail registration are still stacked, unshifted — a live
        view must never silently drop frames, and one unregistered sub
        only softens the stack slightly.  (The offline
        :func:`~astrocapture.process.stack_lights` is stricter: it drops
        failed frames and counts them.)
        """
        from astrocapture.process import calibrate_light

        frame = calibrate_light(
            np.asarray(data, dtype=np.float64),
            self.master_bias,
            self.master_dark,
            self.master_flat,
        )
        if self._n == 0:
            self._ref_centroids = detect_star_centroids(frame)
            self._stack = frame.astype(np.float64)
            self._last_shift = (0.0, 0.0)
        else:
            dx, dy = estimate_shift(
                self._ref_centroids or [],
                detect_star_centroids(frame),
                max_shift_px=self.max_shift_px,
            )
            if (dx, dy) != (0.0, 0.0):
                frame = shift_image(frame, dx, dy)
            self._last_shift = (dx, dy)
            # Progressive mean: numerically stable, no frame history kept.
            self._stack = (self._stack * self._n + frame) / (self._n + 1)
        self._n += 1

    # -- readouts ---------------------------------------------------------
    @property
    def stack(self) -> np.ndarray | None:
        """Current float32 stack, or None before the first frame."""
        if self._stack is None:
            return None
        return self._stack.astype(np.float32)

    @property
    def n_frames(self) -> int:
        return self._n

    @property
    def last_shift(self) -> tuple[float, float]:
        """Registration shift applied to the most recent frame."""
        return self._last_shift

    def snr_db(self) -> float:
        """Rough signal-to-noise estimate in dB (documented, not metrology).

        Signal = mean of the brightest 1% of pixels minus the median
        (a proxy for "how far the stars stick up"); noise = robust
        background sigma (1.4826 × MAD, so bright stars don't inflate
        it).  Good enough to watch the stack improve live — roughly
        +1.5 dB every time the frame count doubles — but don't quote
        it in a paper.
        """
        s = self._stack
        if s is None or self._n == 0:
            return 0.0
        med = float(np.median(s))
        signal = float(np.mean(s[s > np.percentile(s, 99.0)])) - med
        noise = float(1.4826 * np.median(np.abs(s - med)))
        if not np.isfinite(signal) or not np.isfinite(noise) or noise <= 0:
            return 0.0
        ratio = max(signal / noise, 1e-12)
        return float(20.0 * np.log10(ratio))


# ---------------------------------------------------------------------------
# CLI: astrocapture eaa --config plan.yaml [--frames N]
# ---------------------------------------------------------------------------


def run_eaa(config_path: str | Path, max_frames: int | None = None) -> int:
    """Live-stack the plan's light frames, printing running SNR.

    Own small loop: camera only (no mount, no Session files).  Works
    with the sim driver for testing and any real camera driver for
    actual EAA.
    """
    plan = config.load_plan(config_path)
    print(config.plan_summary(plan))
    print()

    light_steps = [s for s in plan.steps if s.type == "light"]
    if not light_steps:
        print("Plan has no light steps: nothing to live-stack.")
        return 1

    camera = make_camera(plan.camera.driver, **plan.camera.options)
    camera.connect()
    stacker = LiveStacker()
    try:
        n = 0
        for step in light_steps:
            for i in range(step.count):
                if max_frames is not None and n >= max_frames:
                    break
                camera.set_exposure_settings(
                    step.exposure, gain=step.gain, binning=step.binning
                )
                camera.start_exposure()
                while not camera.exposure_complete():
                    time.sleep(0.05)
                frame = camera.download_image()
                stacker.add_frame(frame)
                n += 1
                dx, dy = stacker.last_shift
                print(
                    f"frame {n:3d}  SNR {stacker.snr_db():6.1f} dB  "
                    f"shift ({dx:+.1f}, {dy:+.1f}) px",
                    flush=True,
                )
            if max_frames is not None and n >= max_frames:
                break
    except KeyboardInterrupt:
        print("\nCtrl-C: stopping live stack…")
    finally:
        try:
            camera.disconnect()
        except Exception:  # noqa: BLE001 - best effort
            pass
    print(f"\nLive stack: {stacker.n_frames} frames, "
          f"final SNR {stacker.snr_db():.1f} dB")
    return 0
