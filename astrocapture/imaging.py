"""Shared image helpers: star detection, registration, shifting, stretching.

numpy/astropy/Pillow only — no scipy, so these import anywhere.

DUPLICATION NOTE: the star-detection pipeline below intentionally mirrors
the one in :func:`astrocapture.focus.measure_hfr` (threshold → local
maxima → brightest-first non-maximum suppression → boxed centroid with
hot-pixel rejection) instead of refactoring it.  ``measure_hfr``'s exact
numerical behaviour is pinned by its tests, and the safer route was to
leave ``focus.py`` untouched rather than risk a subtle change in its HFR
numbers.  If both ever need to change together, factor the shared core
then — but keep focus's tests green while doing it.
"""

from __future__ import annotations

import io

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:  # Pillow is optional at import time; stretch_png raises without it.
    from PIL import Image
except ImportError:  # pragma: no cover - pillow is a real dependency
    Image = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Star detection
# ---------------------------------------------------------------------------


def detect_star_centroids(
    image: np.ndarray,
    threshold_sigma: float = 3.0,
    max_stars: int = 50,
    window: int = 5,
    box_radius: int = 8,
) -> list[tuple[float, float]]:
    """Flux-weighted (x, y) centroids of stars in ``image``.

    Pipeline: threshold = median + ``threshold_sigma`` * std; local maxima
    above threshold (maximum filter via ``sliding_window_view``);
    brightest-first with non-maximum suppression so a flat saturated core
    doesn't count as a dozen stars; flux-weighted centroid in a
    ``box_radius`` box around each peak after local background
    subtraction.  Single-pixel-dominated detections (hot pixels, cosmic
    rays: one pixel holding > 75% of the box flux) are rejected.

    Returns a (possibly empty) list of ``(x, y)`` float centroids in
    pixel coordinates.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError(
            f"detect_star_centroids needs a 2-D image, got shape {img.shape}"
        )

    thresh = float(np.median(img) + threshold_sigma * np.std(img))

    # Local maxima: a pixel is a peak if it equals the max of its window.
    k = window if window % 2 == 1 else window + 1
    pad = k // 2
    padded = np.pad(img, pad, mode="reflect")
    local_max = sliding_window_view(padded, (k, k)).max(axis=(-2, -1))
    peaks = (img == local_max) & (img > thresh)
    ys, xs = np.nonzero(peaks)
    if len(xs) == 0:
        return []

    # Brightest first, with a cheap non-maximum suppression.
    order = np.argsort(img[ys, xs])[::-1]
    kept: list[tuple[int, int]] = []
    for idx in order:
        x0, y0 = int(xs[idx]), int(ys[idx])
        if all(abs(x0 - kx) > pad or abs(y0 - ky) > pad for kx, ky in kept):
            kept.append((x0, y0))
        if len(kept) >= max_stars:
            break

    h, w = img.shape
    out: list[tuple[float, float]] = []
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
        out.append((cx, cy))
    return out


# ---------------------------------------------------------------------------
# Translation-only registration
# ---------------------------------------------------------------------------


def _nearest(img: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each row of ``pts``, index and distance of the nearest row of ``img``."""
    d2 = ((img[None, :, :] - pts[:, None, :]) ** 2).sum(axis=2)
    idx = d2.argmin(axis=1)
    return idx, np.sqrt(d2[np.arange(len(pts)), idx])


def _vote_shift(
    ref: np.ndarray,
    img: np.ndarray,
    max_shift_px: float,
    match_tol_px: float,
) -> tuple[np.ndarray | None, int]:
    """Best shift hypothesis and its inlier count (None when hopeless)."""
    best_shift: np.ndarray | None = None
    best_inliers = 0
    # Try the brightest-first few reference stars as hypothesis anchors.
    for r in ref[:8]:
        dist = np.hypot(img[:, 0] - r[0], img[:, 1] - r[1])
        j = int(dist.argmin())
        if dist[j] > max_shift_px:
            continue
        cand = r - img[j]  # shift to *apply* to img under this hypothesis
        # Predicted image positions of the ref stars: ref - cand.
        _, d = _nearest(img, ref - cand)
        n_in = int((d < match_tol_px).sum())
        if n_in > best_inliers:
            best_inliers = n_in
            best_shift = cand
    return best_shift, best_inliers


def estimate_shift(
    ref_centroids: list[tuple[float, float]] | np.ndarray,
    img_centroids: list[tuple[float, float]] | np.ndarray,
    max_shift_px: float = 50.0,
    match_tol_px: float = 2.0,
    min_inliers: int = 3,
) -> tuple[float, float]:
    """Translation ``(dx, dy)`` to apply to ``img`` to align it with ``ref``.

    Robust voting over shift hypotheses: for each of the first few
    reference stars, hypothesize the shift that maps it onto its nearest
    image star (within ``max_shift_px``); count how many reference stars
    land within ``match_tol_px`` of an image star under that hypothesis;
    keep the hypothesis with the most inliers and refine it with the
    median inlier residual.

    Returns ``(0.0, 0.0)`` when no hypothesis reaches ``min_inliers``
    inliers — callers should treat that as a registration failure, not as
    "no shift".  (The ``(0, 0)`` return is genuinely ambiguous with a
    true zero shift; internal callers that need to tell them apart use
    :func:`_vote_shift` directly.)

    HONEST NOTE: translation only.  Field rotation between subs is NOT
    corrected — fine for short alt-az subs (the 6SE's field rotates
    slowly); full rotation alignment is a follow-up.
    """
    ref = np.asarray(ref_centroids, dtype=float).reshape(-1, 2)
    img = np.asarray(img_centroids, dtype=float).reshape(-1, 2)
    if len(ref) == 0 or len(img) == 0:
        return (0.0, 0.0)

    best_shift, best_inliers = _vote_shift(ref, img, max_shift_px, match_tol_px)
    if best_shift is None or best_inliers < min_inliers:
        return (0.0, 0.0)

    # Refine: median residual of the inlier matches.
    pred = ref - best_shift
    idx, d = _nearest(img, pred)
    inlier = d < match_tol_px
    residuals = ref[inlier] - img[idx[inlier]]
    med = np.median(residuals, axis=0)
    return (float(med[0]), float(med[1]))


# ---------------------------------------------------------------------------
# Subpixel shifting
# ---------------------------------------------------------------------------


def shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift ``image`` by ``(dx, dy)`` pixels (bilinear, numpy only).

    Content moves +dx in x and +dy in y:
    ``out[y, x] = image`` sampled at ``(y - dy, x - dx)``.  Same shape as
    the input; out-of-bounds samples replicate the edge pixels.  Returns
    float64; callers cast back as needed.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"shift_image needs a 2-D image, got shape {img.shape}")
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    sx = xx - dx
    sy = yy - dy
    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    fx = sx - x0
    fy = sy - y0
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    return (
        img[y0c, x0c] * (1.0 - fx) * (1.0 - fy)
        + img[y0c, x1c] * fx * (1.0 - fy)
        + img[y1c, x0c] * (1.0 - fx) * fy
        + img[y1c, x1c] * fx * fy
    )


# ---------------------------------------------------------------------------
# Auto-stretch to PNG
# ---------------------------------------------------------------------------


def stretch_png(data: np.ndarray, max_px: int = 640) -> bytes:
    """Auto-stretched PNG bytes of a 2-D array (percentile stretch).

    The exact stretch formerly living in
    :meth:`astrocapture.dash.AstroDash.thumbnail_png`: 1st/99.5th
    percentile black/white points (computed on a stride-sampled copy for
    sensors over 1 Mpix), linear clip to uint8, LANCZOS thumbnail to
    ``max_px``.  Raises ``RuntimeError`` when Pillow is not installed.
    """
    if Image is None:
        raise RuntimeError("Pillow is not installed; cannot render PNG")
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"stretch_png needs a 2-D array, got shape {arr.shape}")
    # Percentile stretch on a stride for very large sensors.
    sample = arr
    if sample.size > 1_000_000:
        stride = int((sample.size / 1_000_000) ** 0.5) + 1
        sample = arr[::stride, ::stride]
    lo, hi = (float(v) for v in np.percentile(sample, (1.0, 99.5)))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            hi = lo + 1.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    img = Image.fromarray((norm * 255.0).astype(np.uint8))
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
