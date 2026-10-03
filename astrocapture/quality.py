"""Per-frame quality scoring (heuristic v1) + training-data export.

HONEST NOTE: this is the *heuristic* quality model — star-trailing from
second moments, clouds from background/noise drift, focus from HFR.
There is deliberately no trained neural network here yet: a CNN needs a
labelled dataset this program is only starting to collect (see
:func:`export_training_data`).  The :class:`QualityModel` interface is
the seam a future learned model plugs into — implement
``score(frame) -> float`` and the pipeline (``process.py`` rejection,
training export) works unchanged.

Scores are 0..1 (1 = pristine).  The overall score is the geometric mean
of the components so one catastrophic axis (a cloud bank, heavy
trailing) tanks the frame even if the others look fine.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from astropy.io import fits

from astrocapture.focus import measure_hfr
from astrocapture.imaging import detect_star_centroids, stretch_png


class QualityModel(ABC):
    """Anything that scores a 2-D frame 0..1 (1 = pristine)."""

    @abstractmethod
    def score(self, frame: np.ndarray) -> float:
        """Overall quality of ``frame`` in [0, 1]."""
        ...

    def details(self, frame: np.ndarray) -> dict:
        """Per-component breakdown; default is just the overall score."""
        return {"overall": self.score(frame)}

    def fit(self, frames: list[np.ndarray]) -> None:
        """Learn session baselines from ``frames`` (optional)."""


# ---------------------------------------------------------------------------
# Star shape (trailing) via second moments
# ---------------------------------------------------------------------------


def star_eccentricities(
    image: np.ndarray,
    box_radius: int = 8,
    max_stars: int = 40,
) -> list[float]:
    """Eccentricity of each detected star: 0 = round, ->1 = streak.

    Second central moments in a background-subtracted box around each
    centroid from :func:`detect_star_centroids`; eccentricity from the
    eigenvalue ratio of the 2x2 moment matrix.  Hot pixels are already
    rejected by the centroid detector.
    """
    img = np.asarray(image, dtype=float)
    h, w = img.shape
    out: list[float] = []
    for cx, cy in detect_star_centroids(img, max_stars=max_stars):
        x0, y0 = int(round(cx)), int(round(cy))
        x1, x2 = max(0, x0 - box_radius), min(w, x0 + box_radius + 1)
        y1, y2 = max(0, y0 - box_radius), min(h, y0 + box_radius + 1)
        box = img[y1:y2, x1:x2]
        sig = box - np.median(box)
        sig[sig < 0.0] = 0.0
        total = float(sig.sum())
        if total <= 0.0:
            continue
        yy, xx = np.mgrid[y1:y2, x1:x2]
        dx = xx - cx
        dy = yy - cy
        ixx = float((sig * dx * dx).sum() / total)
        iyy = float((sig * dy * dy).sum() / total)
        ixy = float((sig * dx * dy).sum() / total)
        # Eigenvalues of [[ixx, ixy], [ixy, iyy]].
        tr = ixx + iyy
        det = ixx * iyy - ixy * ixy
        disc = max(0.0, tr * tr / 4.0 - det)
        lam_max = tr / 2.0 + np.sqrt(disc)
        lam_min = max(1e-12, tr / 2.0 - np.sqrt(disc))
        if lam_max <= 0:
            continue
        out.append(float(np.sqrt(max(0.0, 1.0 - lam_min / lam_max))))
    return out


# ---------------------------------------------------------------------------
# Heuristic model (v1 — no ML)
# ---------------------------------------------------------------------------


class HeuristicQualityModel(QualityModel):
    """Numpy-only quality model: trailing + cloud + focus heuristics.

    Baselines (background level/noise, best HFR) are learned by
    :meth:`fit` from the session's own frames — a frame is "cloudy"
    relative to *its own night*, which is what matters for stacking.
    Without :meth:`fit`, the cloud/focus components return neutral 1.0
    and only trailing is measured.
    """

    #: Eccentricity at which the trailing component hits 0.
    trail_ecc_full_bad: float = 0.6

    def __init__(self) -> None:
        self.baseline_bg: float | None = None
        self.baseline_noise: float | None = None
        self.best_hfr: float | None = None

    # -- baselines ------------------------------------------------------
    def fit(self, frames: list[np.ndarray]) -> None:
        bgs, noises, hfrs = [], [], []
        for f in frames:
            img = np.asarray(f, dtype=float)
            bgs.append(float(np.median(img)))
            mad = float(np.median(np.abs(img - np.median(img))))
            noises.append(1.4826 * mad)
            h = measure_hfr(img)
            if np.isfinite(h):
                hfrs.append(h)
        if bgs:
            self.baseline_bg = float(np.median(bgs))
            self.baseline_noise = float(np.median(noises))
        if hfrs:
            self.best_hfr = float(min(hfrs))

    # -- components -----------------------------------------------------
    def trailing_score(self, frame: np.ndarray) -> float:
        eccs = star_eccentricities(frame)
        if not eccs:
            return 0.5  # no stars to judge: neutral, don't dominate
        med = float(np.median(eccs))
        return float(max(0.0, 1.0 - med / self.trail_ecc_full_bad))

    def cloud_score(self, frame: np.ndarray) -> float:
        if self.baseline_bg is None or self.baseline_noise is None:
            return 1.0  # no baseline: neutral
        img = np.asarray(frame, dtype=float)
        bg = float(np.median(img))
        mad = float(np.median(np.abs(img - bg)))
        noise = 1.4826 * mad
        bg_ratio = bg / max(1e-9, self.baseline_bg)
        noise_ratio = noise / max(1e-9, self.baseline_noise)
        # Clouds raise the background and the noise; penalize drift.
        penalty = max(0.0, bg_ratio - 1.0) + max(0.0, noise_ratio - 1.0)
        return float(np.exp(-2.0 * penalty))

    def focus_score(self, frame: np.ndarray) -> float:
        if self.best_hfr is None:
            return 1.0  # no baseline: neutral
        h = measure_hfr(np.asarray(frame, dtype=float))
        if not np.isfinite(h) or h <= 0:
            return 0.5  # unmeasurable: mild penalty, don't dominate
        return float(min(1.0, self.best_hfr / h))

    # -- QualityModel API ------------------------------------------------
    def details(self, frame: np.ndarray) -> dict:
        t = self.trailing_score(frame)
        c = self.cloud_score(frame)
        f = self.focus_score(frame)
        return {
            "trailing": t,
            "cloud": c,
            "focus": f,
            "overall": float((t * c * f) ** (1.0 / 3.0)),
        }

    def score(self, frame: np.ndarray) -> float:
        return self.details(frame)["overall"]


# ---------------------------------------------------------------------------
# Session-level helpers
# ---------------------------------------------------------------------------


def score_frames(
    paths: list[str | Path],
    model: QualityModel | None = None,
) -> list[dict]:
    """Score FITS ``paths``; the model is fit on the frames first.

    Returns a list of ``{"path", "overall", "trailing", "cloud",
    "focus"}`` dicts in input order.
    """
    paths = [Path(p) for p in paths]
    frames = []
    for p in paths:
        with fits.open(str(p)) as hdul:
            frames.append(np.asarray(hdul[0].data, dtype=np.float64))
    model = model or HeuristicQualityModel()
    model.fit(frames)
    out = []
    for p, f in zip(paths, frames):
        d = model.details(f)
        out.append({"path": str(p), **{k: float(v) for k, v in d.items()}})
    return out


def export_training_data(
    session_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Export labelled-training-data scaffolding for a future CNN.

    For every light frame in ``session_dir`` writes a stretched PNG
    thumbnail plus ``labels.csv`` with columns::

        frame, trailing, cloud, focus, overall, label

    ``label`` starts empty — that column is the human's job (1 = keep,
    0 = reject).  Once labelled, this directory trains the small CNN
    that will one day implement :class:`QualityModel` for real; the
    heuristic scores ship as ready-made input features.
    """
    session_dir = Path(session_dir)
    output_dir = Path(output_dir)
    thumbs = output_dir / "thumbnails"
    thumbs.mkdir(parents=True, exist_ok=True)
    lights = sorted((session_dir / "lights").glob("*.fits"))
    model = HeuristicQualityModel()
    frames = []
    for p in lights:
        with fits.open(str(p)) as hdul:
            frames.append(np.asarray(hdul[0].data, dtype=np.float64))
    model.fit(frames)
    rows = []
    for p, f in zip(lights, frames):
        d = model.details(f)
        png_name = p.stem + ".png"
        (thumbs / png_name).write_bytes(stretch_png(f))
        rows.append({
            "frame": png_name,
            "trailing": f"{d['trailing']:.4f}",
            "cloud": f"{d['cloud']:.4f}",
            "focus": f"{d['focus']:.4f}",
            "overall": f"{d['overall']:.4f}",
            "label": "",
        })
    with open(output_dir / "labels.csv", "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["frame", "trailing", "cloud", "focus",
                            "overall", "label"])
        writer.writeheader()
        writer.writerows(rows)
    readme = (
        "# Training data\n\n"
        f"Exported from {session_dir.name}: {len(rows)} light frames.\n\n"
        "Fill in the `label` column (1 = keep, 0 = reject), then train a\n"
        "small CNN (e.g. 3 conv blocks on 128x128 thumbnails, binary\n"
        "cross-entropy) to predict it. The heuristic columns are free\n"
        "input features. Implement `astrocapture.quality.QualityModel`\n"
        "with the trained weights and the pipeline picks it up.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return output_dir
