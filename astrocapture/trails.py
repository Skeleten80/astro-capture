"""Satellite / airplane trail detection (numpy-only).

Method, documented honestly: stars are masked out (boxes around the
centroids from :func:`astrocapture.imaging.detect_star_centroids`), the
remaining bright pixels are thresholded, and lines are found by
RANSAC-style voting — repeatedly sample two bright pixels, count how
many lie within 2.5 px of the line, keep the best hypothesis, and accept
it as a trail only if the inlier run spans at least ``min_length_px``.
Inliers are removed and the search repeats (up to ``max_trails``).

This is deliberately simple: it finds the long, bright, straight streaks
that ruin a sub.  Faint/short segments below the length threshold are
ignored (sigma-clipped stacking already handles those), and a frame
with a huge bright-pixel count (clouds) is subsampled so the vote stays
fast.  Cosmic-ray hits are single pixels — they never form a long run.
"""

from __future__ import annotations

import numpy as np

from astrocapture.imaging import detect_star_centroids


def _eccentricity(img: np.ndarray, cx: float, cy: float,
                  box_radius: int = 8) -> float:
    """Eccentricity of the source at (cx, cy): 0 = round, ->1 = streak."""
    h, w = img.shape
    x0, y0 = int(round(cx)), int(round(cy))
    x1, x2 = max(0, x0 - box_radius), min(w, x0 + box_radius + 1)
    y1, y2 = max(0, y0 - box_radius), min(h, y0 + box_radius + 1)
    box = img[y1:y2, x1:x2]
    sig = box - np.median(box)
    sig[sig < 0.0] = 0.0
    total = float(sig.sum())
    if total <= 0.0:
        return 0.0
    yy, xx = np.mgrid[y1:y2, x1:x2]
    dx, dy = xx - cx, yy - cy
    ixx = float((sig * dx * dx).sum() / total)
    iyy = float((sig * dy * dy).sum() / total)
    ixy = float((sig * dx * dy).sum() / total)
    tr, det = ixx + iyy, ixx * iyy - ixy * ixy
    disc = max(0.0, tr * tr / 4.0 - det)
    lam_max = tr / 2.0 + np.sqrt(disc)
    lam_min = max(1e-12, tr / 2.0 - np.sqrt(disc))
    if lam_max <= 0:
        return 0.0
    return float(np.sqrt(max(0.0, 1.0 - lam_min / lam_max)))


def _mask_stars(
    image: np.ndarray,
    star_mask_radius: int = 12,
    max_stars: int = 100,
) -> np.ndarray:
    """Copy of ``image`` with star boxes filled by the global median.

    Only *round* detections are masked: a bright satellite trail is a
    chain of flat-topped local maxima that the centroid detector happily
    reports as "stars", so each candidate is eccentricity-gated first —
    masking the trail itself would erase the very thing we hunt.
    """
    img = np.asarray(image, dtype=float)
    med = float(np.median(img))
    out = img.copy()
    h, w = img.shape
    for cx, cy in detect_star_centroids(img, max_stars=max_stars):
        if _eccentricity(img, cx, cy) > 0.6:
            continue  # elongated: trail pixel or trailed star, not a star
        x0, y0 = int(round(cx)), int(round(cy))
        x1, x2 = max(0, x0 - star_mask_radius), min(w, x0 + star_mask_radius + 1)
        y1, y2 = max(0, y0 - star_mask_radius), min(h, y0 + star_mask_radius + 1)
        out[y1:y2, x1:x2] = med
    return out


def detect_trails(
    image: np.ndarray,
    *,
    star_mask_radius: int = 12,
    bright_sigma: float = 6.0,
    min_length_px: float = 60.0,
    max_trails: int = 3,
    ransac_iters: int = 200,
    inlier_tol_px: float = 2.5,
    seed: int | None = None,
) -> list[dict]:
    """Find satellite/airplane trails in ``image``.

    Returns a list of trail dicts ``{"angle_deg", "length_px",
    "n_inliers", "points"}`` (``points`` = (N, 2) ``(x, y)`` inlier
    array); empty list when no trail is found.  Deterministic for a
    given ``seed``.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError(f"detect_trails needs a 2-D image, got {img.shape}")
    rng = np.random.default_rng(seed)

    masked = _mask_stars(img, star_mask_radius=star_mask_radius)
    med = float(np.median(masked))
    std = float(np.std(masked))
    if not np.isfinite(std) or std <= 0:
        return []
    ys, xs = np.nonzero(masked > med + bright_sigma * std)
    pts = np.stack([xs.astype(float), ys.astype(float)], axis=1)
    if len(pts) < 12:
        return []
    # Cloudy frames produce oceans of bright pixels: subsample so the
    # RANSAC vote stays fast (a real trail survives subsampling).
    if len(pts) > 30000:
        keep = rng.choice(len(pts), 30000, replace=False)
        pts = pts[keep]

    trails: list[dict] = []
    remaining = pts
    for _ in range(max_trails):
        if len(remaining) < 12:
            break
        best_inliers: np.ndarray | None = None
        best_n = 0
        for _ in range(ransac_iters):
            i, j = rng.choice(len(remaining), 2, replace=False)
            p1, p2 = remaining[i], remaining[j]
            d = p2 - p1
            seg_len = float(np.hypot(d[0], d[1]))
            if seg_len < min_length_px:
                continue  # hypothesis can't span a trail: skip early
            # Distance of every point to the infinite line through p1,p2.
            n = np.array([-d[1], d[0]]) / seg_len
            dist = np.abs((remaining - p1) @ n)
            inl = np.nonzero(dist < inlier_tol_px)[0]
            if len(inl) > best_n:
                best_n = len(inl)
                best_inliers = inl
        if best_inliers is None or best_n < 12:
            break
        inlier_pts = remaining[best_inliers]
        # Span of the inlier run along the line direction.
        p1, p2 = inlier_pts[0], inlier_pts[-1]
        d = p2 - p1
        seg_len = float(np.hypot(d[0], d[1]))
        if seg_len < 1e-9:
            break
        direction = d / seg_len
        proj = (inlier_pts - p1) @ direction
        # A trail must be a *continuous* run: longest gap-bounded span
        # along the line.  Star cores that merely align leave big gaps
        # and are rejected here, not stacked against.
        run_span = _longest_continuous_run(np.sort(proj))
        if run_span < min_length_px:
            break  # a clump, not a streak
        angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
        trails.append({
            "angle_deg": angle,
            "length_px": run_span,
            "n_inliers": int(best_n),
            "points": inlier_pts,
        })
        # Remove this trail's inliers and look for another.
        keep = np.ones(len(remaining), dtype=bool)
        keep[best_inliers] = False
        remaining = remaining[keep]
    return trails


def _longest_continuous_run(
    proj_sorted: np.ndarray,
    max_gap_px: float = 20.0,
) -> float:
    """Longest span of ``proj_sorted`` with no gap bigger than ``max_gap_px``.

    A real satellite trail is a *continuous* streak — bright at (nearly)
    every pixel along its length.  Random star cores that happen to
    roughly align leave hundred-pixel gaps; this is what separates the
    two.  Returns the longest continuous span in pixels.
    """
    if len(proj_sorted) < 2:
        return 0.0
    best = 0.0
    run_start = float(proj_sorted[0])
    prev = float(proj_sorted[0])
    for p in proj_sorted[1:]:
        p = float(p)
        if p - prev > max_gap_px:
            best = max(best, prev - run_start)
            run_start = p
        prev = p
    return max(best, prev - run_start)


def trail_mask(
    shape: tuple[int, int],
    trails: list[dict],
    width_px: float = 3.0,
) -> np.ndarray:
    """Boolean mask covering ``trails`` (thickened to ``width_px``)."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if not trails:
        return mask
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    for t in trails:
        pts = np.asarray(t["points"], dtype=float)
        if len(pts) < 2:
            continue
        p1, p2 = pts[0], pts[-1]
        d = p2 - p1
        seg_len = float(np.hypot(d[0], d[1]))
        if seg_len < 1e-9:
            continue
        n = np.array([-d[1], d[0]]) / seg_len
        dist = np.abs((np.stack([xx.ravel(), yy.ravel()], axis=1) - p1) @ n)
        # Also require the pixel to project within the segment span.
        direction = d / seg_len
        proj = (np.stack([xx.ravel(), yy.ravel()], axis=1) - p1) @ direction
        inside = (dist.reshape(h, w) < width_px / 2.0) & (
            proj.reshape(h, w) > -width_px) & (
            proj.reshape(h, w) < seg_len + width_px)
        mask |= inside
    return mask


def has_trail(image: np.ndarray, **kwargs) -> bool:
    """Convenience: True when at least one trail is detected."""
    return bool(detect_trails(image, **kwargs))
