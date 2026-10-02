"""PlateSolver (mocked solve-field) and the recenter loop (fake hardware)."""

import logging
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from astrocapture.platesolve import FakeSolver, PlateSolver, recenter


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeMount:
    """Mount with an optional constant pointing-model error.

    ``goto()`` records the command and sets the *true* pointing to the
    commanded position plus the fixed error, like a real mount with a
    consistent offset between its model and the sky.
    """

    def __init__(self, err_ra_deg: float = 0.0, err_dec_deg: float = 0.0) -> None:
        self.err = (err_ra_deg, err_dec_deg)
        self._pointing = (0.0, 0.0)
        self.gotos: list[tuple[float, float]] = []

    def goto(self, ra_hours: float, dec_deg: float) -> None:
        self.gotos.append((ra_hours, dec_deg))
        self._pointing = (ra_hours * 15.0 + self.err[0], dec_deg + self.err[1])

    def slew_complete(self) -> bool:
        return True

    @property
    def pointing(self) -> tuple[float, float]:
        return self._pointing


class FakeCamera:
    def __init__(self) -> None:
        self.exposures = 0

    def set_exposure_settings(self, exptime_s, gain=0.0, binning=1) -> None:
        self.exptime_s = exptime_s

    def start_exposure(self) -> None:
        self.exposures += 1

    def exposure_complete(self) -> bool:
        return True

    def download_image(self) -> np.ndarray:
        return np.zeros((32, 32), dtype=np.uint16)


def make_session(tmp_path):
    return SimpleNamespace(dir=tmp_path, log=logging.getLogger("test.platesolve"))


def write_frame(path, shape=(64, 64)):
    fits.writeto(path, np.zeros(shape, dtype=np.uint16), overwrite=True)
    return path


# ---------------------------------------------------------------------------
# PlateSolver.solve
# ---------------------------------------------------------------------------


def _mock_solve_field(monkeypatch, run_impl):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/solve-field")
    monkeypatch.setattr(subprocess, "run", run_impl)


def test_solve_parses_wcs(tmp_path, monkeypatch):
    # Build the fake solve-field: writes a .wcs with known CRVAL next to the input.
    def fake_run(cmd, **kwargs):
        from pathlib import Path

        fits_path = Path(cmd[-1])
        w = WCS(naxis=2)
        w.wcs.crpix = [65.0, 65.0]
        w.wcs.crval = [150.0, 20.0]
        w.wcs.cdelt = [-0.01, 0.01]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        fits.PrimaryHDU(data=np.zeros((4, 4)), header=w.to_header()).writeto(
            fits_path.with_name(fits_path.stem + ".wcs"), overwrite=True
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="solved", stderr="")

    _mock_solve_field(monkeypatch, fake_run)
    frame = write_frame(tmp_path / "frame.fits", shape=(128, 128))
    ra, dec = PlateSolver().solve(frame)
    # crpix=(65,65) 1-based == pixel (64,64) 0-based == image center here
    assert ra == pytest.approx(150.0)
    assert dec == pytest.approx(20.0)


def test_solve_falls_back_to_stdout(tmp_path, monkeypatch):
    stdout = (
        "Field 1: solved with index foo\n"
        "Field 1: RA,Dec = (10.68458, 41.26875), pixel scale 1.93 arcsec/pix.\n"
    )
    _mock_solve_field(
        monkeypatch,
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=stdout,
                                                     stderr=""),
    )
    frame = write_frame(tmp_path / "frame.fits")
    assert PlateSolver().solve(frame) == pytest.approx((10.68458, 41.26875))


def test_solve_missing_binary_returns_none(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    frame = write_frame(tmp_path / "frame.fits")
    with caplog.at_level(logging.INFO, logger="astrocapture.platesolve"):
        assert PlateSolver().solve(frame) is None
    assert "plate solve skipped (solve-field not installed)" in caplog.text


def test_solve_nonzero_rc_returns_none(tmp_path, monkeypatch):
    _mock_solve_field(
        monkeypatch,
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="",
                                                     stderr="nope"),
    )
    frame = write_frame(tmp_path / "frame.fits")
    assert PlateSolver().solve(frame) is None


def test_solve_passes_hints_to_solve_field(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    _mock_solve_field(monkeypatch, fake_run)
    frame = write_frame(tmp_path / "frame.fits")
    PlateSolver(radius_deg=3.0).solve(frame, ra_hint_deg=180.0, dec_hint_deg=45.0)
    cmd = seen["cmd"]
    assert "--ra" in cmd and "--dec" in cmd and "--radius" in cmd
    assert cmd[cmd.index("--radius") + 1] == "3.000"


# ---------------------------------------------------------------------------
# recenter
# ---------------------------------------------------------------------------


def test_recenter_converges_with_pointing_error(tmp_path):
    target = (180.0, 45.0)
    mount = FakeMount(err_ra_deg=0.5, err_dec_deg=-0.3)
    mount._pointing = (target[0] + 0.5, target[1] - 0.3)  # where slew landed
    cam = FakeCamera()
    session = make_session(tmp_path)
    solver = FakeSolver(lambda: mount.pointing)  # perfect solve, real error
    ok = recenter(mount, cam, session, *target, tolerance_arcmin=2.0,
                  max_iterations=5, exposure_s=0.01, solver=solver, settle_s=0)
    assert ok is True
    assert solver.calls == 2  # measure the error, correct, verify
    assert cam.exposures == 2
    # Scratch frame + solve-field sidecars are cleaned up.
    assert list(tmp_path.glob("recenter_tmp.*")) == []


def test_recenter_gives_up_after_max_iterations(tmp_path):
    target = (180.0, 45.0)
    mount = FakeMount()  # perfect mount...
    cam = FakeCamera()
    session = make_session(tmp_path)
    # ...but the solver stubbornly reports a 1° offset whatever we do.
    solver = FakeSolver([(181.0, 45.0)])
    ok = recenter(mount, cam, session, *target, tolerance_arcmin=2.0,
                  max_iterations=3, exposure_s=0.01, solver=solver, settle_s=0)
    assert ok is False
    assert solver.calls == 3
    assert len(mount.gotos) == 3


def test_recenter_returns_false_when_solves_fail(tmp_path):
    target = (180.0, 45.0)
    mount = FakeMount()
    cam = FakeCamera()
    session = make_session(tmp_path)
    solver = FakeSolver(lambda: None)  # solve keeps failing
    ok = recenter(mount, cam, session, *target, max_iterations=3,
                  exposure_s=0.01, solver=solver, settle_s=0)
    assert ok is False
    assert solver.calls == 3
    assert mount.gotos == []  # never slew on a failed solve
