"""Synthetic camera: image shape/dtype, exposure scaling, FITS round-trip."""

import time

import numpy as np
import pytest
from astropy.io import fits

from astrocapture.drivers.sim import SimCamera


def expose(cam: SimCamera, exptime: float) -> np.ndarray:
    cam.set_exposure_settings(exptime)
    cam.start_exposure()
    deadline = time.monotonic() + exptime + 5
    while not cam.exposure_complete():
        assert time.monotonic() < deadline, "exposure never completed"
        time.sleep(0.01)
    return cam.download_image()


def test_image_shape_dtype():
    cam = SimCamera(width=320, height=240, n_stars=30, seed=3)
    cam.connect()
    img = expose(cam, 0.05)
    assert img.shape == (240, 320)
    assert img.dtype == np.uint16


def test_exposure_scaling():
    # Same seed -> same sky; longer exposure must be brighter overall.
    cam1 = SimCamera(width=256, height=256, n_stars=60, seed=11)
    cam2 = SimCamera(width=256, height=256, n_stars=60, seed=11)
    cam1.connect()
    cam2.connect()
    short = expose(cam1, 0.05).astype(float)
    long = expose(cam2, 0.4).astype(float)
    # Background (~1000 ADU) doesn't scale; starlight does. Compare
    # background-subtracted means.
    assert (long.mean() - 1000) > (short.mean() - 1000) * 1.2
    # Stars present: brightest pixels far above background.
    assert long.max() > long.mean() + 10 * long.std()


def test_deterministic_catalog():
    cam1 = SimCamera(width=128, height=128, n_stars=20, seed=99, read_noise_e=0.0)
    cam2 = SimCamera(width=128, height=128, n_stars=20, seed=99, read_noise_e=0.0)
    cam1.connect()
    cam2.connect()
    # Zero read noise + same seed still differ by photon noise, but the
    # star *positions* (brightest pixels) must match.
    i1 = expose(cam1, 0.2)
    i2 = expose(cam2, 0.2)
    top1 = set(map(tuple, np.argwhere(i1 > np.percentile(i1, 99.9))))
    top2 = set(map(tuple, np.argwhere(i2 > np.percentile(i2, 99.9))))
    assert len(top1 & top2) / max(len(top1), 1) > 0.8


def test_hot_pixels_present():
    cam = SimCamera(width=256, height=256, n_stars=0, seed=5,
                    hot_pixel_fraction=1e-3)
    cam.connect()
    img = expose(cam, 0.5)
    # Hot pixels are far brighter than the ~1000 ADU background.
    assert (img > 5000).sum() > 10


def test_fits_round_trip(tmp_path):
    cam = SimCamera(width=128, height=128, n_stars=20, seed=1)
    cam.connect()
    img = expose(cam, 0.1)
    path = tmp_path / "frame.fits"
    fits.writeto(path, img, overwrite=True)
    with fits.open(path) as hdul:
        back = hdul[0].data
    assert back.shape == img.shape
    assert back.dtype == img.dtype
    np.testing.assert_array_equal(back, img)


def test_dither_shifts_stars():
    cam = SimCamera(width=256, height=256, n_stars=40, seed=21, read_noise_e=0.0)
    cam.connect()
    before = expose(cam, 0.2).astype(float)
    cam.apply_dither(10.0, -6.0)
    after = expose(cam, 0.2).astype(float)
    # Shifted image should match the original shifted by (ox, oy) = (10, -6).
    # Star drawn at (x - ox, y - oy) in `after`  =>  after[i, j] = before[i + oy, j + ox].
    rolled = np.roll(before, shift=(6, -10), axis=(0, 1))
    inner = (slice(20, 236), slice(20, 236))
    corr = np.corrcoef(rolled[inner].ravel(), after[inner].ravel())[0, 1]
    assert corr > 0.95


def test_invalid_exposure_rejected():
    cam = SimCamera()
    with pytest.raises(ValueError):
        cam.set_exposure_settings(0)


def test_cooler_simulated():
    cam = SimCamera()
    assert cam.has_cooler
    cam.set_cooler(-10.0)
    assert cam.get_temperature() == pytest.approx(-10.0)
