"""imaging.py: star detection, shift estimation, shifting, PNG stretch."""

import io

import numpy as np
import pytest
from PIL import Image

from astrocapture.imaging import (
    detect_star_centroids,
    estimate_shift,
    shift_image,
    stretch_png,
)


def starfield(shape=(128, 128), stars=((30.0, 40.0), (80.0, 20.0),
                                      (100.0, 100.0), (50.0, 90.0)),
              background=1000.0, amp=20000.0, sigma=1.6, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    img = np.full(shape, background) + rng.normal(0, 5.0, shape)
    for x, y in stars:
        img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return img


# ---------------------------------------------------------------------------
# detect_star_centroids
# ---------------------------------------------------------------------------


def test_detect_star_centroids_finds_known_stars():
    true = [(30.0, 40.0), (80.0, 20.0), (100.0, 100.0), (50.0, 90.0)]
    found = sorted(detect_star_centroids(starfield(stars=true)))
    assert len(found) == len(true)
    for (fx, fy), (tx, ty) in zip(found, sorted(true)):
        assert fx == pytest.approx(tx, abs=0.1)
        assert fy == pytest.approx(ty, abs=0.1)


def test_detect_star_centroids_rejects_hot_pixels():
    rng = np.random.default_rng(1)
    img = np.full((128, 128), 1000.0) + rng.normal(0, 5.0, (128, 128))
    img[10, 10] = 50000.0
    img[50, 90] = 40000.0
    assert detect_star_centroids(img) == []


def test_detect_star_centroids_empty_frame():
    assert detect_star_centroids(np.full((64, 64), 1000.0)) == []


def test_detect_star_centroids_rejects_non_2d():
    with pytest.raises(ValueError):
        detect_star_centroids(np.zeros((10, 10, 3)))


# ---------------------------------------------------------------------------
# estimate_shift
# ---------------------------------------------------------------------------


def test_estimate_shift_recovers_known_integer_shift():
    base = starfield()
    ref = detect_star_centroids(base)
    moved = shift_image(base, 7.0, -4.0)
    got = estimate_shift(ref, detect_star_centroids(moved))
    # (dx, dy) to *apply* to the moved frame to re-align it: (-7, +4).
    assert got[0] == pytest.approx(-7.0, abs=0.2)
    assert got[1] == pytest.approx(4.0, abs=0.2)


def test_estimate_shift_recovers_subpixel_shift():
    base = starfield()
    ref = detect_star_centroids(base)
    moved = shift_image(base, 2.5, 1.25)
    got = estimate_shift(ref, detect_star_centroids(moved))
    assert got[0] == pytest.approx(-2.5, abs=0.25)
    assert got[1] == pytest.approx(-1.25, abs=0.25)


def test_estimate_shift_failure_returns_zero():
    assert estimate_shift([], [(1.0, 2.0)]) == (0.0, 0.0)
    assert estimate_shift([(1.0, 2.0)], []) == (0.0, 0.0)
    # Disjoint star sets: no plausible translation.
    a = [(10.0, 10.0), (20.0, 20.0), (30.0, 15.0)]
    b = [(100.0, 100.0), (110.0, 105.0), (105.0, 115.0)]
    assert estimate_shift(a, b, max_shift_px=50.0) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# shift_image
# ---------------------------------------------------------------------------


def test_shift_image_integer_matches_roll_interior():
    img = starfield()
    out = shift_image(img, 4.0, 2.0)
    expect = np.roll(np.roll(img, 4, axis=1), 2, axis=0)
    assert out.shape == img.shape
    assert np.allclose(out[4:-4, 4:-4], expect[4:-4, 4:-4])


def test_shift_image_subpixel_moves_centroid():
    yy, xx = np.mgrid[0:64, 0:64]
    img = 1000.0 + 20000.0 * np.exp(-((xx - 30.0) ** 2 + (yy - 40.0) ** 2)
                                    / (2 * 1.6 ** 2))
    out = shift_image(img, 0.5, -0.25)
    (cx, cy) = detect_star_centroids(out, box_radius=12)[0]
    assert cx == pytest.approx(30.5, abs=0.1)
    assert cy == pytest.approx(39.75, abs=0.1)


def test_shift_image_replicates_edges():
    img = np.arange(16.0).reshape(4, 4)
    out = shift_image(img, 10.0, 10.0)  # everything samples off-frame
    assert np.all(np.isfinite(out))
    assert out[0, 0] == pytest.approx(img[0, 0])  # edge replicated


# ---------------------------------------------------------------------------
# stretch_png
# ---------------------------------------------------------------------------


def test_stretch_png_valid_png_and_thumbnail_size():
    rng = np.random.default_rng(2)
    big = (rng.normal(1000, 50, (800, 900))).astype(np.float64)
    big[400, 450] += 50000  # a star so the stretch has range
    png = stretch_png(big, max_px=640)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    im = Image.open(io.BytesIO(png))
    assert max(im.size) <= 640


def test_stretch_png_handles_flat_frame():
    png = stretch_png(np.full((32, 32), 7.0))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
