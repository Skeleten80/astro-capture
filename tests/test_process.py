"""process.py: combining, masters, calibration, stacking, process_session."""

import numpy as np
import pytest
from astropy.io import fits

from astrocapture.imaging import detect_star_centroids, shift_image
from astrocapture.process import (
    calibrate_light,
    make_master_bias,
    make_master_dark,
    make_master_flat,
    median_combine,
    process_session,
    stack_lights,
)


def write_fits(path, data, exptime=0.0, imagetyp="LIGHT FRAME"):
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = fits.Header()
    hdr["EXPTIME"] = exptime
    hdr["IMAGETYP"] = imagetyp
    hdr["BUNIT"] = "ADU"
    fits.writeto(path, np.asarray(data, dtype=np.float32), hdr, overwrite=True)
    return path


def starfield(shape=(96, 96), stars=((25.0, 30.0), (60.0, 20.0),
                                    (70.0, 70.0), (40.0, 60.0)),
              background=1000.0, amp=20000.0, sigma=1.6, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    img = np.full(shape, background) + rng.normal(0, 5.0, shape)
    for x, y in stars:
        img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return img


# ---------------------------------------------------------------------------
# median_combine
# ---------------------------------------------------------------------------


def test_median_combine_plain_median(tmp_path):
    paths = [write_fits(tmp_path / f"f{i}.fits", np.full((8, 8), 100.0 * (i + 1)))
             for i in range(3)]
    data, info = median_combine(paths, sigma_clip=False)
    assert data.dtype == np.float32
    assert np.allclose(data, 200.0)
    assert info["NCOMBINE"] == 3
    assert info["SIGCLIP"] is False


def test_median_combine_sigma_clip_rejects_outlier_frame(tmp_path):
    rng = np.random.default_rng(3)
    paths = []
    for i in range(5):
        img = 1000.0 + rng.normal(0, 5.0, (16, 16))
        if i == 2:  # one bad frame: cosmic-ray-like blotch
            img[4:8, 4:8] += 40000.0
        paths.append(write_fits(tmp_path / f"f{i}.fits", img))
    data, info = median_combine(paths, sigma_clip=True)
    assert info["CLIPFRAC"] > 0.0
    # The blotch must not survive: median of the survivors ≈ background.
    assert abs(float(data[5, 5]) - 1000.0) < 50.0
    # Sanity: a plain mean *is* corrupted by the single bad frame.
    mean_val = float(np.mean([fits.getdata(str(p))[5, 5] for p in paths]))
    assert mean_val > 5000.0


def test_median_combine_needs_frames():
    with pytest.raises(ValueError):
        median_combine([])


# ---------------------------------------------------------------------------
# masters + calibration math on known synthetic components
# ---------------------------------------------------------------------------

_BIAS_LEVEL = 100.0
_DARK_RATE = 5.0  # ADU/s


def _synthetic_frames(tmp_path, n, shape, exptime, flat, seed):
    """Frames = bias + dark_rate*exptime, scaled by flat, + tiny noise."""
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        img = ((_BIAS_LEVEL + _DARK_RATE * exptime)
               * np.ones(shape) * flat
               + rng.normal(0, 1.0, shape))
        paths.append(write_fits(tmp_path / f"f{i}.fits", img, exptime=exptime))
    return paths


def test_calibration_math_recovers_signal(tmp_path):
    shape = (32, 32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    flat_true = 0.8 + 0.4 * (xx / shape[1])  # 0.8 .. 1.2 gradient
    sig_true = 5000.0  # the "sky" signal we want back

    bias_paths = _synthetic_frames(tmp_path / "b", 5, shape, 0.0,
                                   np.ones(shape), seed=10)
    # Bias frames still carry read noise only; force exact level.
    for p in bias_paths:
        write_fits(p, np.full(shape, _BIAS_LEVEL), exptime=0.0)

    dark_paths = []
    for i in range(5):
        img = np.full(shape, _BIAS_LEVEL + _DARK_RATE * 10.0)
        dark_paths.append(write_fits(tmp_path / f"d{i}.fits", img, exptime=10.0))

    flat_paths = []
    for i in range(5):
        img = ((_BIAS_LEVEL + _DARK_RATE * 1.0)
               + 20000.0 * flat_true)  # flat illumination, no noise
        flat_paths.append(write_fits(tmp_path / f"l{i}.fits", img, exptime=1.0))

    mbias, _ = make_master_bias(bias_paths)
    assert np.allclose(mbias, _BIAS_LEVEL, atol=1e-3)
    mdark, dinfo = make_master_dark(dark_paths, bias=mbias,
                                    light_exposure=10.0)
    assert dinfo["DARKSCAL"] == pytest.approx(1.0)
    assert np.allclose(mdark, _DARK_RATE * 10.0, atol=1e-2)
    # Flats need a dark matched to *their* exposure, not the lights'.
    mdark_flat, _ = make_master_dark(dark_paths, bias=mbias,
                                     light_exposure=1.0)
    mflat, _ = make_master_flat(flat_paths, bias=mbias, dark=mdark_flat)
    assert np.allclose(mflat, flat_true / np.median(flat_true), rtol=1e-3)

    raw = ((_BIAS_LEVEL + _DARK_RATE * 10.0) + sig_true * flat_true)
    cal = calibrate_light(raw, mbias, mdark, mflat)
    # (raw - bias - dark)/flat_normalized = sig_true * median(flat_true)
    assert np.allclose(cal, sig_true * np.median(flat_true), rtol=1e-3)


def test_master_dark_scaling_notes_approximate(tmp_path, capsys):
    paths = [write_fits(tmp_path / f"d{i}.fits",
                        np.full((8, 8), _BIAS_LEVEL + _DARK_RATE * 2.0),
                        exptime=2.0)
             for i in range(3)]
    mdark, info = make_master_dark(paths,
                                   bias=np.full((8, 8), _BIAS_LEVEL),
                                   dark_exposure=2.0, light_exposure=8.0)
    assert info["DARKSCAL"] == pytest.approx(4.0)
    assert np.allclose(mdark, _DARK_RATE * 8.0, atol=1e-2)
    out = capsys.readouterr().out
    assert "approximate" in out


def test_calibrate_light_guards_zero_flat():
    data = np.full((4, 4), 1100.0)
    flat = np.ones((4, 4))
    flat[0, 0] = 0.0  # dead pixel
    cal = calibrate_light(data, master_flat=flat)
    assert np.all(np.isfinite(cal))
    assert cal[1, 1] == pytest.approx(1100.0)


# ---------------------------------------------------------------------------
# stack_lights
# ---------------------------------------------------------------------------


def test_stack_lights_registers_shifted_frames(tmp_path):
    base = starfield()
    shifts_true = [(0.0, 0.0), (6.0, -3.0), (-4.0, 5.0), (2.5, 2.5)]
    paths = []
    for i, (dx, dy) in enumerate(shifts_true):
        paths.append(write_fits(tmp_path / f"l{i:03d}.fits",
                                shift_image(base, dx, dy), exptime=2.0))
    stack, stats = stack_lights(paths)
    assert stats["n_input"] == 4
    assert stats["n_stacked"] == 4
    assert stats["n_rejected"] == 0
    assert stats["registered"] is True
    # Recovered shifts are the *corrections*: expect -shifts_true.
    for (dx, dy), (tx, ty) in zip(stats["shifts"], shifts_true):
        assert dx == pytest.approx(-tx, abs=0.3)
        assert dy == pytest.approx(-ty, abs=0.3)
    # Aligned stack: star centroids match the reference frame's.
    ref_c = sorted(detect_star_centroids(base))
    stk_c = sorted(detect_star_centroids(stack))
    assert len(stk_c) == len(ref_c)
    for (sx, sy), (rx, ry) in zip(stk_c, ref_c):
        assert sx == pytest.approx(rx, abs=0.5)
        assert sy == pytest.approx(ry, abs=0.5)


def test_stack_lights_drops_unregisterable_frame(tmp_path):
    base = starfield()
    rng = np.random.default_rng(9)
    paths = [write_fits(tmp_path / "l000.fits", base, exptime=2.0),
             write_fits(tmp_path / "l001.fits",
                        shift_image(base, 3.0, 1.0), exptime=2.0),
             # Pure noise: no stars, cannot register.
             write_fits(tmp_path / "l002.fits",
                        1000.0 + rng.normal(0, 5.0, base.shape), exptime=2.0)]
    stack, stats = stack_lights(paths)
    assert stats["n_input"] == 3
    assert stats["n_stacked"] == 2
    assert stats["n_rejected"] == 1
    assert stats["shifts"][2] is None


# ---------------------------------------------------------------------------
# process_session
# ---------------------------------------------------------------------------


def _make_session(root, n_light=4, dark_exp=2.0, light_exp=2.0,
                  with_calib=True):
    shape = (64, 64)
    base = starfield(shape=shape)
    rng = np.random.default_rng(11)
    sdir = root / "sess-20250101-120000"
    for i in range(n_light):
        dx, dy = rng.uniform(-4, 4, 2)
        write_fits(sdir / "lights" / f"l{i:03d}.fits",
                   shift_image(base, dx, dy), exptime=light_exp)
    if with_calib:
        for i in range(3):
            write_fits(sdir / "bias" / f"b{i:03d}.fits",
                       np.full(shape, _BIAS_LEVEL), exptime=0.0)
            write_fits(sdir / "darks" / f"d{i:03d}.fits",
                       np.full(shape, _BIAS_LEVEL + _DARK_RATE * dark_exp),
                       exptime=dark_exp)
            write_fits(sdir / "flats" / f"f{i:03d}.fits",
                       np.full(shape, 20000.0), exptime=0.5)
    return sdir


def test_process_session_end_to_end(tmp_path):
    sdir = _make_session(tmp_path)
    out = tmp_path / "out"
    stats = process_session(sdir, out)
    assert stats["n_stacked"] == 4
    assert stats["n_rejected"] == 0
    assert stats["warnings"] == []

    fits_path = out / "stacked.fits"
    assert fits_path.is_file()
    with fits.open(fits_path) as hdul:
        assert hdul[0].data.dtype.kind == "f" and hdul[0].data.dtype.itemsize == 4
        assert hdul[0].data.shape == (64, 64)
        assert hdul[0].header["MBIAS"] == "yes"
        assert hdul[0].header["MDARK"] == "yes"
        assert hdul[0].header["MFLAT"] == "yes"
        assert np.all(np.isfinite(hdul[0].data))

    png = out / "stacked.png"
    assert png.is_file()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    log_text = (out / "process.log").read_text()
    assert "stacked 4/4 lights" in log_text
    assert "master bias" in log_text

    # Stacked image is sane: stars detected, background ≈ calibrated level.
    stk = fits.getdata(str(fits_path))
    assert len(detect_star_centroids(stk)) >= 3


def test_process_session_missing_calibration_stacks_raw(tmp_path):
    sdir = _make_session(tmp_path, with_calib=False)
    out = tmp_path / "out"
    stats = process_session(sdir, out)
    assert stats["n_stacked"] == 4
    assert len(stats["warnings"]) == 3  # bias, dark, flat
    assert (out / "stacked.fits").is_file()
    assert (out / "stacked.png").is_file()
    log_text = (out / "process.log").read_text()
    assert "WARNING" in log_text
    with fits.open(out / "stacked.fits") as hdul:
        assert hdul[0].header["MBIAS"] == "none"


def test_process_session_no_lights(tmp_path):
    sdir = tmp_path / "sess-empty"
    (sdir / "lights").mkdir(parents=True)
    out = tmp_path / "out"
    stats = process_session(sdir, out)
    assert stats["n_lights"] == 0
    assert stats["warnings"]
    assert (out / "process.log").is_file()
    assert not (out / "stacked.fits").exists()


def test_process_session_unusable_flats_warn_and_continue(tmp_path):
    # Flats that calibrate to nothing (flat level == bias level, as with
    # the simulator's non-illuminated "flats") must warn, not crash.
    shape = (32, 32)
    sdir = tmp_path / "sess-flat"
    write_fits(sdir / "lights" / "l000.fits", starfield(shape=shape),
               exptime=2.0)
    write_fits(sdir / "bias" / "b000.fits", np.full(shape, 100.0),
               exptime=0.0)
    write_fits(sdir / "flats" / "f000.fits", np.full(shape, 100.0),
               exptime=0.5)
    out = tmp_path / "out"
    stats = process_session(sdir, out)
    assert stats["n_stacked"] == 1
    assert any("flat" in w for w in stats["warnings"])
    assert (out / "stacked.fits").is_file()
    assert (out / "stacked.png").is_file()
