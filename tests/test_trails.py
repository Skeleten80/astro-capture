"""trails.py: line detection on synthetic frames + process integration."""

import numpy as np
import pytest
from astropy.io import fits

from astrocapture.process import process_session
from astrocapture.trails import detect_trails, has_trail, trail_mask


def starfield(shape=(128, 128), seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    img = np.full(shape, 1000.0) + rng.normal(0, 5.0, shape)
    for x, y in ((25.0, 30.0), (60.0, 20.0), (90.0, 90.0), (40.0, 80.0)):
        img += 20000.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2)
                                / (2 * 1.6 ** 2))
    return img


def add_line(img, angle_deg=45.0, brightness=8000.0, width=2):
    """Draw a bright straight streak across the frame."""
    out = img.copy()
    h, w = img.shape
    theta = np.radians(angle_deg)
    dx, dy = np.cos(theta), np.sin(theta)
    cx, cy = w / 2.0, h / 2.0
    n = np.array([-dy, dx])  # normal
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.abs((xx - cx) * n[0] + (yy - cy) * n[1])
    along = (xx - cx) * dx + (yy - cy) * dy
    out[(dist < width / 2.0) & (np.abs(along) < w * 0.45)] += brightness
    return out


def write_fits(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = fits.Header()
    hdr["EXPTIME"] = 25.0
    fits.writeto(path, np.asarray(data, dtype=np.float32), hdr,
                 overwrite=True)
    return path


def test_diagonal_trail_detected():
    img = add_line(starfield(), angle_deg=45.0)
    trails = detect_trails(img, seed=0)
    assert trails, "bright diagonal trail should be found"
    t = trails[0]
    assert t["length_px"] > 60.0
    assert abs(abs(t["angle_deg"]) - 45.0) < 10.0


def test_vertical_and_shallow_trails_detected():
    for angle in (90.0, 15.0):
        img = add_line(starfield(seed=7), angle_deg=angle)
        assert has_trail(img, seed=1), f"trail at {angle}° missed"


def test_clean_starfield_not_flagged():
    assert not has_trail(starfield(seed=11), seed=2)


def test_short_line_ignored():
    img = starfield(seed=13)
    out = img.copy()
    out[60:75, 60:75] += 8000.0  # 15px blotch, not a trail
    assert not has_trail(out, seed=3)


def test_trail_mask_covers_line():
    img = add_line(starfield(seed=17), angle_deg=30.0)
    trails = detect_trails(img, seed=4)
    mask = trail_mask(img.shape, trails)
    assert mask.sum() > 100  # a real streak's worth of pixels
    assert mask.sum() < 0.2 * img.size  # ...but not half the frame


def test_process_reject_drops_trailed_frame(tmp_path):
    sess = tmp_path / "sess"
    (sess / "lights").mkdir(parents=True)
    write_fits(sess / "lights" / "good_001.fits", starfield(seed=21))
    write_fits(sess / "lights" / "good_002.fits", starfield(seed=22))
    write_fits(sess / "lights" / "good_003.fits", starfield(seed=23))
    write_fits(sess / "lights" / "trailed_004.fits",
               add_line(starfield(seed=24)))
    stats = process_session(sess, tmp_path / "out", trails="reject")
    assert stats["n_trail_rejected"] == 1
    assert stats["n_stacked"] == 3
    log = (tmp_path / "out" / "process.log").read_text()
    assert "trail" in log.lower()


def test_process_mask_keeps_frame_count(tmp_path):
    sess = tmp_path / "sess"
    (sess / "lights").mkdir(parents=True)
    for i in range(3):
        write_fits(sess / "lights" / f"g_{i:03d}.fits", starfield(seed=30 + i))
    write_fits(sess / "lights" / "t_003.fits",
               add_line(starfield(seed=33)))
    stats = process_session(sess, tmp_path / "out", trails="mask")
    assert stats["n_stacked"] == 4  # nothing dropped, trail inpainted
    assert (tmp_path / "out" / "stacked.fits").is_file()


def test_aligned_star_clumps_not_flagged():
    """Regression: star cores that roughly align (gaps between them)
    must not count as a trail — only continuous streaks do."""
    rng = np.random.default_rng(77)
    img = np.full((256, 256), 1000.0) + rng.normal(0, 5.0, (256, 256))
    yy, xx = np.mgrid[0:256, 0:256]
    # Five round "stars" roughly along a vertical line, 40px apart.
    for y in (30.0, 75.0, 120.0, 165.0, 210.0):
        img += 20000.0 * np.exp(-((xx - 128.0) ** 2 + (yy - y) ** 2)
                                / (2 * 1.6 ** 2))
    assert not has_trail(img, seed=5)
