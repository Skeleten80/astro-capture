"""quality.py: heuristic scores, ordering, export."""

import numpy as np
import pytest
from astropy.io import fits

from astrocapture.quality import (
    HeuristicQualityModel,
    QualityModel,
    export_training_data,
    score_frames,
    star_eccentricities,
)


def starfield(shape=(96, 96), stars=((25.0, 30.0), (60.0, 20.0),
                                    (70.0, 70.0), (40.0, 60.0)),
              background=1000.0, amp=20000.0, sigma=1.6, seed=0,
              elongate=1.0):
    """Round stars; elongate>1 stretches them along x (trailing)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    img = np.full(shape, background) + rng.normal(0, 5.0, shape)
    for x, y in stars:
        img += amp * np.exp(-(((xx - x) ** 2) / (2 * (sigma * elongate) ** 2)
                              + ((yy - y) ** 2) / (2 * sigma ** 2)))
    return img


def write_fits(path, data):
    hdr = fits.Header()
    hdr["EXPTIME"] = 25.0
    fits.writeto(path, np.asarray(data, dtype=np.float32), hdr,
                 overwrite=True)
    return path


def test_abc_not_instantiable():
    with pytest.raises(TypeError):
        QualityModel()


def test_sharp_beats_trailed():
    model = HeuristicQualityModel()
    sharp = starfield()
    trailed = starfield(elongate=4.0, seed=1)
    model.fit([sharp, trailed])
    s_sharp = model.score(sharp)
    s_trailed = model.score(trailed)
    assert s_sharp > s_trailed
    assert model.details(trailed)["trailing"] < 0.5


def test_sharp_beats_cloudy():
    model = HeuristicQualityModel()
    sharp = starfield(seed=2)
    rng = np.random.default_rng(9)
    cloudy = (starfield(seed=3) + 4000.0
              + rng.normal(0, 60.0, (96, 96)))
    model.fit([sharp, cloudy])
    assert model.score(sharp) > model.score(cloudy)
    assert model.details(cloudy)["cloud"] < 0.5


def test_eccentricity_round_vs_streak():
    eccs_round = star_eccentricities(starfield(seed=4))
    eccs_streak = star_eccentricities(starfield(seed=4, elongate=5.0))
    assert eccs_round and eccs_streak
    assert float(np.median(eccs_round)) < float(np.median(eccs_streak))


def test_scores_in_unit_range():
    model = HeuristicQualityModel()
    frames = [starfield(seed=i) for i in range(4)]
    model.fit(frames)
    for f in frames:
        s = model.score(f)
        assert 0.0 <= s <= 1.0


def test_unfitted_model_neutral_components():
    model = HeuristicQualityModel()  # no fit()
    d = model.details(starfield(seed=5))
    assert d["cloud"] == 1.0 and d["focus"] == 1.0
    assert 0.0 <= d["overall"] <= 1.0


def test_score_frames_order_and_keys(tmp_path):
    paths = [write_fits(tmp_path / f"f{i}.fits", starfield(seed=i))
             for i in range(3)]
    rows = score_frames(paths)
    assert len(rows) == 3
    for r in rows:
        assert set(r) >= {"path", "overall", "trailing", "cloud", "focus"}


def test_export_training_data(tmp_path):
    sess = tmp_path / "sess"
    (sess / "lights").mkdir(parents=True)
    for i in range(3):
        write_fits(sess / "lights" / f"l_{i:03d}.fits", starfield(seed=i))
    out = export_training_data(sess, tmp_path / "training")
    assert (out / "labels.csv").is_file()
    assert len(list((out / "thumbnails").glob("*.png"))) == 3
    header = (out / "labels.csv").read_text().splitlines()[0]
    assert header == "frame,trailing,cloud,focus,overall,label"
    first = (out / "labels.csv").read_text().splitlines()[1]
    assert first.endswith(",")  # label column starts empty
    assert (out / "README.md").is_file()


def test_process_quality_filter_drops_cloudy_frame(tmp_path):
    from astrocapture.process import process_session
    sess = tmp_path / "sess"
    (sess / "lights").mkdir(parents=True)
    for i in range(3):
        write_fits(sess / "lights" / f"g_{i:03d}.fits", starfield(seed=40 + i))
    rng = np.random.default_rng(9)
    cloudy = starfield(seed=43) + 4000.0 + rng.normal(0, 60.0, (96, 96))
    write_fits(sess / "lights" / "cloudy_003.fits", cloudy)
    stats = process_session(sess, tmp_path / "out", quality_min_score=0.5)
    assert stats["n_quality_rejected"] == 1
    assert stats["n_stacked"] == 3
    log = (tmp_path / "out" / "process.log").read_text()
    assert "REJECTED" in log
