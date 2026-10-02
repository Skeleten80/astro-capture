"""Simulator mount: slew math, states, parking."""

import time

import pytest

from astrocapture.drivers.base import MountState
from astrocapture.drivers.sim import SimMount
from astrocapture.util import angular_separation_deg


def make_mount(**kw):
    kw.setdefault("slew_rate_dps", 3600.0)  # fast: tests don't wait on physics
    m = SimMount(**kw)
    m.connect()
    m.unpark()
    return m


def wait_until(fn, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if fn():
            return True
        time.sleep(0.02)
    return fn()


def test_goto_reaches_target():
    m = make_mount()
    m.goto(6.0, 30.0)
    assert m.state == MountState.SLEWING
    assert wait_until(m.slew_complete)
    ra, dec = m.position
    assert ra == pytest.approx(6.0)
    assert dec == pytest.approx(30.0)
    assert m.state in (MountState.IDLE, MountState.TRACKING)


def test_slew_moves_toward_target_monotonically():
    m = make_mount(slew_rate_dps=5.0)
    m.goto(12.0, 0.0)  # 180 deg away in RA
    d0 = angular_separation_deg(*m.position, 12.0, 0.0)
    time.sleep(0.3)
    d1 = angular_separation_deg(*m.position, 12.0, 0.0)
    assert d1 < d0  # got closer
    assert not m.slew_complete()  # 5 deg/s can't cover 180 deg in 0.3 s
    # Expected: ~1.5 deg covered.
    assert d0 - d1 == pytest.approx(1.5, abs=0.6)


def test_slew_time_matches_rate():
    m = make_mount(slew_rate_dps=90.0)
    m.goto(1.0, 0.0)  # from park (0, 90): 90 deg away
    t0 = time.monotonic()
    assert wait_until(m.slew_complete, timeout=10)
    elapsed = time.monotonic() - t0
    # 90 deg at 90 deg/s -> ~1 s (generous bounds for CI jitter).
    assert 0.5 < elapsed < 3.0


def test_ra_wrap_takes_short_path():
    m = make_mount(slew_rate_dps=3600.0)
    m.goto(23.9, 0.0)
    assert wait_until(m.slew_complete)
    m.goto(0.1, 0.0)  # 0.2h = 3 deg across the 0h line, not 357 deg
    assert wait_until(m.slew_complete, timeout=5)
    ra, _ = m.position
    assert ra == pytest.approx(0.1)


def test_park_and_tracking_states():
    m = make_mount()
    m.start_tracking()
    assert m.state == MountState.TRACKING
    m.stop_tracking()
    assert m.state == MountState.IDLE
    m.park()
    assert wait_until(m.slew_complete)
    assert m.state == MountState.PARKED


def test_offset_nudges_position():
    m = make_mount()
    m.goto(6.0, 30.0)
    assert wait_until(m.slew_complete)
    m.offset(900.0, 1800.0)  # +900" RA (= 1 min of RA time), +30' Dec
    ra, dec = m.position
    assert ra == pytest.approx(6.0 + 900 / 54000, abs=1e-6)
    assert dec == pytest.approx(30.5, abs=1e-6)


def test_not_connected_raises():
    m = SimMount()
    with pytest.raises(RuntimeError):
        m.goto(1.0, 1.0)
