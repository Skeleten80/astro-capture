"""Focus measurement and autofocus.

HONEST HARDWARE NOTE — read this before wiring autofocus into a plan:
Mathias's Celestron NexStar 6SE has **no stock motorized focuser**.
The ``autofocus()`` routine below drives an INDI focuser *when one is
present* (e.g. a Celestron SCT focus motor on the rear cell, or any
third-party focuser with an INDI driver exposing ``ABS_FOCUS_POSITION``).
With the stock 6SE there is nothing to drive — for that setup the
sequencer should call ``assist_mode()`` instead, which prints
Bahtinov-mask instructions for manual focusing.

``measure_hfr()`` works with any camera backend: it takes a numpy image
and returns the median half-flux radius (HFR) of the detected stars —
the standard "lower is sharper" focus metric.  Only numpy is used (no
scipy), so this module imports anywhere the rest of AstroCapture does.
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from astrocapture.drivers.indi import INDIClient, INDIError

log = logging.getLogger(__name__)

# HFR of a 2-D Gaussian PSF: the radius enclosing half the total flux.
# (Integral of exp(-r^2/2s^2) to R = half  <=>  R = s*sqrt(2*ln2).)
GAUSSIAN_HFR_FACTOR = math.sqrt(2.0 * math.log(2.0))  # ≈ 1.1774


# ---------------------------------------------------------------------------
# HFR measurement
# ---------------------------------------------------------------------------


def measure_hfr(
    image: np.ndarray,
    threshold_sigma: float = 3.0,
    max_stars: int = 50,
    box_radius: int = 8,
    window: int = 5,
) -> float:
    """Median half-flux radius of stars in ``image`` (pixels).

    Pipeline: threshold = median + ``threshold_sigma`` * std; find local
    maxima above threshold (maximum filter via numpy stride tricks);
    centroid each of the brightest ``max_stars`` in a small box; compute
    the radius enclosing half the background-subtracted flux per star;
    return the median.  Single-pixel-dominated detections (hot pixels,
    cosmic rays) are rejected.

    Returns NaN when no usable stars are found (empty/clouded frame) —
    callers should treat NaN as "no measurement", never as "in focus".
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError(f"measure_hfr needs a 2-D image, got shape {img.shape}")

    thresh = float(np.median(img) + threshold_sigma * np.std(img))

    # Local maxima: a pixel is a peak if it equals the max of its window.
    k = window if window % 2 == 1 else window + 1
    pad = k // 2
    padded = np.pad(img, pad, mode="reflect")
    local_max = sliding_window_view(padded, (k, k)).max(axis=(-2, -1))
    peaks = (img == local_max) & (img > thresh)
    ys, xs = np.nonzero(peaks)
    if len(xs) == 0:
        return float("nan")

    # Brightest first, with a cheap non-maximum suppression so a flat
    # saturated core doesn't count as a dozen stars.
    order = np.argsort(img[ys, xs])[::-1]
    kept: list[tuple[int, int]] = []
    for idx in order:
        x0, y0 = int(xs[idx]), int(ys[idx])
        if all(abs(x0 - kx) > pad or abs(y0 - ky) > pad for kx, ky in kept):
            kept.append((x0, y0))
        if len(kept) >= max_stars:
            break

    h, w = img.shape
    hfrs: list[float] = []
    for x0, y0 in kept:
        x1, x2 = max(0, x0 - box_radius), min(w, x0 + box_radius + 1)
        y1, y2 = max(0, y0 - box_radius), min(h, y0 + box_radius + 1)
        box = img[y1:y2, x1:x2]
        sig = box - np.median(box)  # local background subtraction
        sig[sig < 0.0] = 0.0
        total = float(sig.sum())
        if total <= 0.0:
            continue
        # Hot-pixel / cosmic-ray rejection: a real star spreads its flux
        # over many pixels; a single hot pixel keeps ~all of it in one.
        if float(sig.max()) / total > 0.75:
            continue
        yy, xx = np.mgrid[y1:y2, x1:x2]
        cx = float((xx * sig).sum() / total)
        cy = float((yy * sig).sum() / total)
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).ravel()
        s = sig.ravel()
        rank = np.argsort(r)
        cum = np.cumsum(s[rank])
        j = int(np.searchsorted(cum, 0.5 * total))
        j = min(j, len(rank) - 1)
        hfrs.append(float(r[rank[j]]))

    if not hfrs:
        return float("nan")
    return float(np.median(hfrs))


# ---------------------------------------------------------------------------
# Focuser drivers
# ---------------------------------------------------------------------------


class Focuser(ABC):
    """Motorized focuser. Positions are integer steps (driver units)."""

    @abstractmethod
    def get_position(self) -> int: ...

    @abstractmethod
    def move_to(self, pos: int) -> None:
        """Absolute move. Returns once the focuser reports idle."""

    @abstractmethod
    def move_by(self, delta: int) -> None:
        """Relative move by ``delta`` steps."""


class INDIFocuser(Focuser):
    """Focuser via an INDI focuser driver, through an existing INDIClient.

    Drives the ``ABS_FOCUS_POSITION`` number property (item name is read
    from the def vector — usually ``FOCUS_ABSOLUTE_POSITION``).  INDI
    focuser property names vary between drivers (Celestron SCT focus
    motor, MoonLite, ZWO EAF, ...): verify with ``indi_getprop`` on first
    light and pass ``prop=``/``item=`` overrides if yours differ.
    """

    def __init__(
        self,
        client: INDIClient,
        device: str,
        prop: str = "ABS_FOCUS_POSITION",
        item: str | None = None,
        timeout_s: float = 20.0,
    ) -> None:
        if not device:
            raise ValueError("INDIFocuser needs a device name")
        self._client = client
        self._device = device
        self._prop = prop
        self._item = item
        self._timeout_s = timeout_s

    def _item_name(self) -> str:
        items = self._client.items_of(self._device, self._prop)
        if not items:
            raise INDIError(
                f"INDI {self._device}: no items on {self._prop} "
                "(wrong property name? check indi_getprop)"
            )
        if self._item is not None:
            if self._item not in items:
                raise INDIError(
                    f"INDI {self._device}: {self._prop} has no item "
                    f"{self._item!r} (has: {sorted(items)})"
                )
            return self._item
        return next(iter(items))

    def get_position(self) -> int:
        return int(round(float(self._client.items_of(
            self._device, self._prop)[self._item_name()])))

    def move_to(self, pos: int) -> None:
        name = self._item_name()
        self._client.send_number(self._device, self._prop, {name: float(pos)})
        # wait_for_state raises INDIError on Alert and TimeoutError on
        # timeout — both are honest "the focuser didn't get there" signals
        # the caller (autofocus / sequencer) should see.
        self._client.wait_for_state(
            self._device, self._prop, ("Ok", "Idle"), timeout=self._timeout_s
        )

    def move_by(self, delta: int) -> None:
        self.move_to(self.get_position() + delta)


class SimFocuser(Focuser):
    """Simulated focuser for testing: HFR follows a parabola.

    ``true_hfr(pos) = hfr_min + curvature * (pos - best_position)^2``;
    ``measure()`` adds Gaussian noise, like a real HFR measurement.
    Positions are clamped to ``[min_pos, max_pos]`` (travel limits).
    """

    def __init__(
        self,
        best_position: int = 25000,
        hfr_min: float = 1.8,
        curvature: float = 2e-6,
        noise: float = 0.05,
        position: int | None = None,
        min_pos: int = 0,
        max_pos: int = 50000,
        seed: int = 42,
    ) -> None:
        self.best_position = best_position
        self.hfr_min = hfr_min
        self.curvature = curvature
        self.noise = noise
        self.min_pos = min_pos
        self.max_pos = max_pos
        self._rng = np.random.default_rng(seed)
        self._pos = self._clamp(best_position if position is None else position)

    def _clamp(self, pos: int) -> int:
        return max(self.min_pos, min(self.max_pos, int(round(pos))))

    def get_position(self) -> int:
        return self._pos

    def move_to(self, pos: int) -> None:
        self._pos = self._clamp(pos)

    def move_by(self, delta: int) -> None:
        self.move_to(self._pos + delta)

    def true_hfr(self, pos: int | None = None) -> float:
        """Noiseless HFR at ``pos`` (current position if omitted)."""
        p = self._pos if pos is None else pos
        return self.hfr_min + self.curvature * (p - self.best_position) ** 2

    def measure(self) -> float:
        """Noisy HFR measurement at the current position."""
        return self.true_hfr() + self._rng.normal(0.0, self.noise)


# ---------------------------------------------------------------------------
# Autofocus (V-curve)
# ---------------------------------------------------------------------------


def autofocus(
    focuser: Focuser,
    measure_fn,
    n_positions: int = 7,
    step: int = 50,
    settle_s: float = 1.0,
    samples: int = 3,
) -> int:
    """V-curve autofocus: scan, fit a parabola, move to best focus.

    Steps ``n_positions`` points centered on the current position,
    measures median HFR (``samples`` readings each), fits
    ``HFR = a·p² + b·p + c`` with ``np.polyfit``, and moves to the
    vertex ``-b/2a``.  If the fit is degenerate (``a <= 0``) or the
    vertex extrapolates far outside the scanned range, falls back to
    the scanned position with the lowest measured HFR.

    Returns the final focuser position (int).  Requires a motorized
    focuser — on the stock 6SE (no focus motor) use ``assist_mode()``.
    """
    if n_positions < 3:
        raise ValueError("autofocus needs at least 3 positions")
    start = focuser.get_position()
    span = (n_positions - 1) / 2.0
    positions = [start + int(round((i - span) * step))
                 for i in range(n_positions)]

    hfrs: list[float] = []
    for pos in positions:
        focuser.move_to(pos)
        time.sleep(settle_s)
        vals = [float(measure_fn()) for _ in range(samples)]
        finite = [v for v in vals if math.isfinite(v)]
        hfr = float(np.median(finite)) if finite else float("nan")
        hfrs.append(hfr)
        log.info("autofocus: pos=%d HFR=%.2f", pos, hfr)

    finite_pts = [(p, h) for p, h in zip(positions, hfrs)
                  if math.isfinite(h)]
    if len(finite_pts) >= 3:
        xs = np.array([p for p, _ in finite_pts], dtype=float)
        ys = np.array([h for _, h in finite_pts], dtype=float)
        a, b, _ = np.polyfit(xs, ys, 2)
        vertex = -b / (2.0 * a) if a > 0 else None
        lo, hi = min(positions) - step, max(positions) + step
        if vertex is not None and lo <= vertex <= hi:
            best = int(round(vertex))
            log.info("autofocus: parabola vertex at %d (a=%.3g)", best, a)
            focuser.move_to(best)
            return best
        log.warning("autofocus: degenerate fit (a=%.3g) — using measured minimum", a)
    else:
        log.warning("autofocus: too few valid measurements — using measured minimum")

    best = finite_pts[int(np.argmin([h for _, h in finite_pts]))][0] \
        if finite_pts else start
    focuser.move_to(best)
    return best


# ---------------------------------------------------------------------------
# Manual focusing aid (the honest path for a stock 6SE)
# ---------------------------------------------------------------------------

_BAHTINOV_GUIDE = """\
Manual focusing with a Bahtinov mask (no motorized focuser needed):

  1. Point at a bright star (mag 2-4) and center it. Use a short
     exposure or live view at high ISO so the star is clearly visible.
  2. Slip the Bahtinov mask over the front of the scope (6SE: over the
     corrector plate / dew shield). The mask's slots produce a
     distinctive diffraction-spike pattern: an X with a third spike
     through the middle.
  3. Watch the central spike: it sits LEFT of center when focus is too
     far IN, RIGHT of center when too far OUT.
  4. Turn the 6SE's rear focus knob in SMALL increments (quarter turns),
     pausing a few seconds each time — the 6SE's moving-mirror focus
     shifts the image, so let it settle before judging.
  5. When the central spike is exactly centered in the X, you are at
     best focus. Remove the mask WITHOUT touching the focus knob.
  6. Re-check every 30-60 minutes and whenever temperature drops
     noticeably — the SCT's focus drifts as the tube cools.

Tip: take a 5 s test exposure at each tweak and zoom in on the star;
the screen lies less than your eyes do.
"""


def assist_mode(target_name: str = "your target") -> None:
    """Print manual focusing instructions (Bahtinov mask steps).

    This is the focusing path for the stock NexStar 6SE, which has no
    motorized focuser for ``autofocus()`` to drive.  The sequencer
    should offer this (pause for the user) instead of attempting an
    automated focus run on hardware that cannot move.
    """
    print(f"Focus assist — {target_name}")
    print(_BAHTINOV_GUIDE)
