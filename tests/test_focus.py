"""measure_hfr on synthetic stars, autofocus on SimFocuser, INDIFocuser."""

import math

import numpy as np
import pytest

from astrocapture.drivers.indi import INDIError
from astrocapture.focus import (
    GAUSSIAN_HFR_FACTOR,
    INDIFocuser,
    SimFocuser,
    assist_mode,
    autofocus,
    measure_hfr,
)


def gaussian_field(shape, stars, background=1000.0):
    """Synthetic image; stars = list of (x, y, sigma, amplitude)."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    img = np.full(shape, background, dtype=float)
    for x, y, sigma, amp in stars:
        img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    return img


# ---------------------------------------------------------------------------
# measure_hfr
# ---------------------------------------------------------------------------


def test_measure_hfr_recovers_known_radius():
    sigma = 2.0
    stars = [
        (40.0, 40.0, sigma, 20000.0),
        (120.0, 90.0, sigma, 15000.0),
        (200.0, 200.0, sigma, 25000.0),
    ]
    hfr = measure_hfr(gaussian_field((256, 256), stars))
    # HFR of a 2-D Gaussian = sigma * sqrt(2 ln 2)
    assert hfr == pytest.approx(GAUSSIAN_HFR_FACTOR * sigma, rel=0.15)


def test_measure_hfr_empty_frame_returns_nan():
    assert math.isnan(measure_hfr(np.full((64, 64), 1000.0)))


def test_measure_hfr_rejects_hot_pixels():
    rng = np.random.default_rng(0)
    img = np.full((128, 128), 1000.0) + rng.normal(0, 5.0, (128, 128))
    img[10, 10] = 50000.0
    img[50, 90] = 40000.0
    img[100, 30] = 60000.0
    # Single-pixel detections are hot pixels, not stars -> no measurement.
    assert math.isnan(measure_hfr(img))


def test_measure_hfr_rejects_non_2d():
    with pytest.raises(ValueError):
        measure_hfr(np.zeros((10, 10, 3)))


# ---------------------------------------------------------------------------
# autofocus
# ---------------------------------------------------------------------------


def test_autofocus_exact_on_noiseless_parabola():
    foc = SimFocuser(best_position=25000, noise=0.0, position=24500)
    best = autofocus(foc, foc.measure, n_positions=7, step=200,
                     settle_s=0, samples=1)
    assert best == 25000
    assert foc.get_position() == 25000  # focuser was moved there


def test_autofocus_lands_near_true_best_with_noise():
    foc = SimFocuser(best_position=25000, noise=0.02, seed=7, position=24500)
    best = autofocus(foc, foc.measure, n_positions=7, step=200,
                     settle_s=0, samples=3)
    assert best == pytest.approx(25000, abs=200)  # within ~1 step
    assert foc.get_position() == best


def test_autofocus_falls_back_with_too_few_valid_measurements():
    foc = SimFocuser(position=25000)
    readings = iter([3.0, 2.0, float("nan"), float("nan"), float("nan")])
    best = autofocus(foc, lambda: next(readings), n_positions=5, step=50,
                     settle_s=0, samples=1)
    # Only 2 finite readings -> no parabola fit; take the measured minimum.
    assert best == 24950
    assert foc.get_position() == 24950


def test_autofocus_needs_at_least_three_positions():
    foc = SimFocuser()
    with pytest.raises(ValueError):
        autofocus(foc, foc.measure, n_positions=2)


# ---------------------------------------------------------------------------
# INDIFocuser
# ---------------------------------------------------------------------------


class FakeINDIClient:
    def __init__(self):
        self._props = {
            ("Focuser", "ABS_FOCUS_POSITION"): {"FOCUS_ABSOLUTE_POSITION": 12345.0}
        }
        self.sent: list = []

    def items_of(self, device, prop):
        return dict(self._props.get((device, prop), {}))

    def send_number(self, device, prop, items):
        self.sent.append((device, prop, items))
        self._props[(device, prop)] = dict(items)

    def wait_for_state(self, device, prop, states=("Ok", "Idle"), timeout=15.0):
        pass


def test_indifocuser_drives_abs_focus_position():
    client = FakeINDIClient()
    foc = INDIFocuser(client, "Focuser")
    assert foc.get_position() == 12345
    foc.move_to(13000)
    assert client.sent[-1] == (
        "Focuser", "ABS_FOCUS_POSITION", {"FOCUS_ABSOLUTE_POSITION": 13000.0},
    )
    assert foc.get_position() == 13000
    foc.move_by(-500)
    assert foc.get_position() == 12500


def test_indifocuser_missing_property_raises():
    client = FakeINDIClient()
    with pytest.raises(INDIError):
        INDIFocuser(client, "NoSuchDevice").get_position()


# ---------------------------------------------------------------------------
# assist_mode
# ---------------------------------------------------------------------------


def test_assist_mode_prints_bahtinov_steps(capsys):
    assist_mode("M51")
    out = capsys.readouterr().out
    assert "Bahtinov" in out
    assert "M51" in out
