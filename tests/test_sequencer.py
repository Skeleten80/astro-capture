"""Sequencer state machine (sim drivers) + FITS header contents."""

import threading
import time
from pathlib import Path

import pytest
from astropy.io import fits

from astrocapture import config
from astrocapture.config import Plan
from astrocapture.drivers.sim import SimCamera, SimMount
from astrocapture.sequencer import SeqState, Sequencer
from astrocapture.session import Session, check_meridian_flip, plate_solve


def make_plan(**over) -> Plan:
    steps = over.pop("steps", [
        {"type": "light", "exposure": 0.1, "count": 2, "gain": 1600,
         "dither_every": 1},
        {"type": "bias", "count": 1, "gain": 1600},
    ])
    return Plan(
        session_name="testseq",
        output_dir="sessions",
        target=config.Target(name="M51", ra_hours=13.5, dec_deg=47.2),
        telescope="Test scope",
        instrument="Test cam",
        mount=config.DriverSpec("sim", {"slew_rate_dps": 3600.0}),
        camera=config.DriverSpec("sim", {"width": 128, "height": 128,
                                         "n_stars": 20, "seed": 1}),
        steps=[config.Step(**s) for s in steps],
    )


def run_plan(tmp_path: Path, **over) -> Sequencer:
    plan = make_plan(**over)
    plan.output_dir = str(tmp_path / "sessions")
    mount = SimMount(**plan.mount.options)
    camera = SimCamera(**plan.camera.options)
    seq = Sequencer(plan, mount, camera)
    seq.run()
    return seq


def test_full_run_reaches_done(tmp_path):
    seq = run_plan(tmp_path)
    assert seq.state == SeqState.DONE
    assert seq.frames_taken == 3
    fits_files = list(seq.session.dir.rglob("*.fits"))
    assert len(fits_files) == 3
    assert (seq.session.dir / "session.log").exists()


def test_fits_headers(tmp_path):
    seq = run_plan(tmp_path)
    light = sorted((seq.session.dir / "lights").glob("*.fits"))[0]
    with fits.open(light) as hdul:
        hdr = hdul[0].header
    assert hdr["OBJECT"] == "M51"
    assert hdr["RA"] == pytest.approx(13.5 * 15.0)
    assert hdr["DEC"] == pytest.approx(47.2)
    assert hdr["EXPTIME"] == pytest.approx(0.1)
    assert hdr["IMAGETYP"].startswith("LIGHT")
    assert hdr["INSTRUME"] == "Test cam"
    assert hdr["TELESCOP"] == "Test scope"
    assert hdr["GAIN"] == 1600
    assert "DATE-OBS" in hdr
    assert hdr["CCD-TEMP"] == pytest.approx(20.0)  # SimCamera default temp
    bias = sorted((seq.session.dir / "bias").glob("*.fits"))[0]
    with fits.open(bias) as hdul:
        assert hdul[0].header["IMAGETYP"].startswith("BIAS")


def test_abort_mid_sequence(tmp_path):
    plan = make_plan(steps=[
        {"type": "light", "exposure": 0.3, "count": 10, "gain": 1600},
    ])
    plan.output_dir = str(tmp_path / "sessions")
    seq = Sequencer(plan, SimMount(slew_rate_dps=3600.0),
                    SimCamera(width=64, height=64, n_stars=5, seed=1))
    t = threading.Thread(target=seq.run)
    t.start()
    time.sleep(0.8)  # let a frame or two complete
    seq.abort()
    t.join(timeout=10)
    assert seq.state == SeqState.ABORTED
    assert 0 < seq.frames_taken < 10


def test_pause_resume(tmp_path):
    plan = make_plan(steps=[
        {"type": "light", "exposure": 0.3, "count": 4, "gain": 1600},
    ])
    plan.output_dir = str(tmp_path / "sessions")
    seq = Sequencer(plan, SimMount(slew_rate_dps=3600.0),
                    SimCamera(width=64, height=64, n_stars=5, seed=1))
    t = threading.Thread(target=seq.run)
    t.start()
    # Keep asking until the sequencer is mid-exposure; pause() only acts
    # in SLEWING/EXPOSING/DITHERING states.
    deadline = time.monotonic() + 5
    while seq.state != SeqState.PAUSED and time.monotonic() < deadline:
        seq.pause()
        time.sleep(0.05)
    assert seq.state == SeqState.PAUSED
    n_while_paused = seq.frames_taken
    time.sleep(0.6)
    assert seq.frames_taken == n_while_paused  # no progress while paused
    seq.resume()
    t.join(timeout=10)
    assert seq.state == SeqState.DONE
    assert seq.frames_taken == 4


def test_dither_offsets_applied(tmp_path):
    seq = run_plan(tmp_path)  # dither_every=1 on the light step
    log_text = (seq.session.dir / "session.log").read_text()
    assert "dithering by" in log_text


def test_meridian_flip_stub(tmp_path):
    import logging
    logger = logging.getLogger("test-flip")
    # Target transiting now (HA ~ 0) -> flip advised.
    assert check_meridian_flip(13.5, 13.5, logger) is True
    # Target 3h from meridian -> no flip.
    assert check_meridian_flip(13.5, 16.5, logger) is False


def test_plate_solve_skipped_without_solver(tmp_path, monkeypatch):
    import logging
    import shutil
    monkeypatch.setattr(shutil, "which", lambda *_: None)
    logger = logging.getLogger("test-platesolve")
    assert plate_solve(tmp_path / "nofile.fits", logger) is None


def test_unknown_driver_raises():
    from astrocapture.drivers import make_camera, make_mount
    with pytest.raises(ValueError, match="Unknown mount driver"):
        make_mount("ascom")
    with pytest.raises(ValueError, match="Unknown camera driver"):
        make_camera("ascom")
