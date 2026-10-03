"""eaa.py: LiveStacker, CLI wiring, dashboard live-stack integration."""

import io
import json
import urllib.request

import numpy as np
import pytest
from PIL import Image

from astrocapture import config
from astrocapture.dash import AstroDash
from astrocapture.eaa import LiveStacker
from astrocapture.imaging import detect_star_centroids, shift_image


def starfield(shape=(96, 96), seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    img = np.full(shape, 1000.0) + rng.normal(0, 5.0, shape)
    for x, y in [(25.0, 30.0), (60.0, 20.0), (70.0, 70.0), (40.0, 60.0)]:
        img += 20000.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2)
                                / (2 * 1.6 ** 2))
    return img


# ---------------------------------------------------------------------------
# LiveStacker
# ---------------------------------------------------------------------------


def test_live_stacker_empty():
    s = LiveStacker()
    assert s.n_frames == 0
    assert s.stack is None
    assert s.snr_db() == 0.0


def test_live_stacker_accumulates_mean():
    s = LiveStacker()
    rng = np.random.default_rng(4)
    frames = [1000.0 + rng.normal(0, 5.0, (32, 32)) for _ in range(4)]
    for f in frames:
        s.add_frame(f)
    assert s.n_frames == 4
    assert s.stack.dtype == np.float32
    assert np.allclose(s.stack, np.mean(frames, axis=0), atol=1e-3)


def test_live_stacker_snr_improves_with_frames():
    # Noiseless base + independent read noise per frame: stacking should
    # improve SNR as sqrt(N) -> +9.0 dB from 1 to 8 frames.
    yy, xx = np.mgrid[0:96, 0:96]
    base = np.full((96, 96), 1000.0)
    for x, y in [(25.0, 30.0), (60.0, 20.0), (70.0, 70.0), (40.0, 60.0)]:
        base += 20000.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2)
                                 / (2 * 1.6 ** 2))
    s = LiveStacker()
    rng = np.random.default_rng(5)
    snrs = []
    for _ in range(8):
        s.add_frame(base + rng.normal(0, 5.0, base.shape))
        snrs.append(s.snr_db())
    assert all(b > a for a, b in zip(snrs, snrs[1:]))  # monotonic
    assert snrs[-1] - snrs[0] == pytest.approx(9.03, abs=1.0)
    assert all(np.isfinite(v) for v in snrs)


def test_live_stacker_registers_to_first_frame_no_drift():
    s = LiveStacker()
    base = starfield()
    ref_c = sorted(detect_star_centroids(base))
    # Each frame is shifted; registration must undo it against frame 0.
    true_shifts = [(0.0, 0.0), (6.0, -2.0), (-3.0, 5.0), (2.5, 2.5)]
    for dx, dy in true_shifts:
        s.add_frame(shift_image(base, dx, dy))
    assert s.n_frames == 4
    # No drift: the stack's stars sit where frame 0's stars are.
    stk_c = sorted(detect_star_centroids(s.stack))
    assert len(stk_c) == len(ref_c)
    for (sx, sy), (rx, ry) in zip(stk_c, ref_c):
        assert sx == pytest.approx(rx, abs=0.5)
        assert sy == pytest.approx(ry, abs=0.5)
    # Last frame's correction undoes its (+2.5, +2.5) offset.
    dx, dy = s.last_shift
    assert dx == pytest.approx(-2.5, abs=0.3)
    assert dy == pytest.approx(-2.5, abs=0.3)


def test_live_stacker_with_masters_calibrates():
    bias = np.full((32, 32), 100.0)
    s = LiveStacker(master_bias=bias)
    s.add_frame(np.full((32, 32), 1100.0))
    assert s.stack is not None
    assert float(np.median(s.stack)) == pytest.approx(1000.0, abs=1.0)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_cli_process_wiring():
    from astrocapture.cli import build_parser, cmd_eaa, cmd_process

    args = build_parser().parse_args(
        ["process", "sessions/foo", "--output", "outdir"])
    assert args.func is cmd_process
    assert args.session_dir == "sessions/foo"
    assert args.output == "outdir"

    args = build_parser().parse_args(
        ["eaa", "--config", "examples/sim_session.yaml", "--frames", "3"])
    assert args.func is cmd_eaa
    assert args.config == "examples/sim_session.yaml"
    assert args.frames == 3


# ---------------------------------------------------------------------------
# Dashboard live-stack integration
# ---------------------------------------------------------------------------

FAST_YAML = """
session:
  name: "eaa-test"
  output_dir: "sessions"
  target: {name: "M51", ra_hours: 13.4979, dec_deg: 47.1953}
  telescope: "Test scope"
  instrument: "Test cam"
mount:
  driver: sim
  slew_rate_dps: 3600.0
camera:
  driver: sim
  width: 128
  height: 128
  n_stars: 20
  seed: 1
sequence:
  - {type: light, exposure: 1, count: 2, gain: 1600}
"""


def _make_dash(tmp_path):
    cfg = tmp_path / "plan.yaml"
    cfg.write_text(FAST_YAML)
    plan = config.load_plan(cfg)
    plan.output_dir = str(tmp_path / "sessions")
    return AstroDash(plan, port=0)


def test_dash_feeds_light_frames_into_live_stack(tmp_path):
    from astrocapture.config import Step

    dash = _make_dash(tmp_path)
    try:
        assert dash.live_png() is None  # no frames yet
        step = Step(type="light", exposure=1.0, count=1, gain=1600.0)
        dash.seq.session.save_frame(starfield(shape=(128, 128)), "light",
                                    step, 13.4979, 47.1953)
        assert dash._live.n_frames == 1
        # Dark frames must NOT enter the live stack.
        step_dark = Step(type="dark", exposure=1.0, count=1)
        dash.seq.session.save_frame(np.full((128, 128), 1000.0), "dark",
                                    step_dark, 13.4979, 47.1953)
        assert dash._live.n_frames == 1
        png = dash.live_png()
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        assert Image.open(io.BytesIO(png)).size[0] <= 640
        live = dash.live_state()
        assert live["frames"] == 1
        assert live["snr_db"] is not None
    finally:
        dash.stop()


def test_dash_live_endpoint(tmp_path):
    from astrocapture.config import Step

    dash = _make_dash(tmp_path)
    dash.start_server()  # server only; no session thread needed
    try:
        base = f"http://127.0.0.1:{dash.port}"
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(base + "/api/live.png", timeout=10)
        assert exc.value.code == 404

        step = Step(type="light", exposure=1.0, count=1)
        dash.seq.session.save_frame(starfield(shape=(128, 128)), "light",
                                    step, 13.4979, 47.1953)
        with urllib.request.urlopen(base + "/api/live.png",
                                    timeout=10) as r:
            assert r.status == 200
            assert r.headers.get_content_type() == "image/png"
            assert r.read(8) == b"\x89PNG\r\n\x1a\n"

        with urllib.request.urlopen(base + "/api/state", timeout=10) as r:
            state = json.loads(r.read().decode("utf-8"))
        assert state["live"]["frames"] == 1
    finally:
        dash.stop()
