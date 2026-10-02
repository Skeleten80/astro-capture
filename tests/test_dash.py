"""Dashboard: HTTP API + page against the simulator."""

import json
import time
import urllib.request
from pathlib import Path

import pytest

from astrocapture import config
from astrocapture.dash import AstroDash

FAST_YAML = """
session:
  name: "dash-test"
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
  - {type: light, exposure: 1, count: 3, gain: 1600, dither_every: 2}
  - {type: bias, count: 1, gain: 1600}
"""

# Slow slew so the pause test can deterministically pause mid-slew
# (pause only surfaces as a PAUSED sequencer state at checkpoints).
SLOW_SLEW_YAML = FAST_YAML.replace("slew_rate_dps: 3600.0", "slew_rate_dps: 2.0")

HTML_IDS = [
    "session-name", "status-pill",
    "target-name", "target-ra", "target-dec",
    "mount-ra", "mount-dec", "mount-state",
    "cam-state", "cam-exposure",
    "progress-bar", "step-list",
    "thumbnail", "log",
    "btn-pause", "btn-resume", "btn-abort",
]

RUNNING = ("idle", "slewing", "exposing", "dithering")
TERMINAL = ("done", "aborted", "error")


def make_dash(tmp_path: Path, yaml_text: str = FAST_YAML, port: int = 0) -> AstroDash:
    cfg = tmp_path / "plan.yaml"
    cfg.write_text(yaml_text)
    plan = config.load_plan(cfg)
    plan.output_dir = str(tmp_path / "sessions")
    # Default driver construction: same make_mount/make_camera path as the CLI.
    return AstroDash(plan, port=port)


@pytest.fixture()
def dash(tmp_path):
    d = make_dash(tmp_path)
    d.start()
    yield d
    d.stop()


def base(d: AstroDash) -> str:
    return f"http://127.0.0.1:{d.port}"


def api_get(d: AstroDash, path: str, timeout: float = 10):
    with urllib.request.urlopen(base(d) + path, timeout=timeout) as r:
        return r.status, r.headers.get_content_type(), r.read()


def api_post(d: AstroDash, path: str, timeout: float = 10):
    req = urllib.request.Request(base(d) + path, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def wait_state(d: AstroDash, want, timeout: float = 15.0) -> str:
    want_set = {want} if isinstance(want, str) else set(want)
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        _, _, body = api_get(d, "/api/state")
        last = json.loads(body)["session"]["status"]
        if last in want_set:
            return last
        time.sleep(0.2)
    pytest.fail(f"timed out waiting for status {sorted(want_set)}; last={last!r}")


def test_state_keys(dash):
    status, ctype, body = api_get(dash, "/api/state")
    assert status == 200
    assert ctype == "application/json"
    s = json.loads(body)
    assert s["ok"] is True

    assert s["session"]["name"] == "dash-test"
    assert s["session"]["status"] in RUNNING + ("paused",) + TERMINAL
    assert Path(s["session"]["dir"]).is_dir()

    assert s["target"]["name"] == "M51"
    assert s["target"]["ra_hours"] == pytest.approx(13.4979)
    assert s["target"]["dec_deg"] == pytest.approx(47.1953)

    plan = s["plan"]
    assert plan["frames_total"] == 4
    assert isinstance(plan["frames_taken"], int)
    assert 0 <= plan["frames_taken"] <= 4
    assert len(plan["steps"]) == 2
    assert isinstance(plan["elapsed_s"], (int, float))
    assert isinstance(plan["eta_s"], (int, float))

    assert s["mount"]["state"] in ("idle", "slewing", "tracking", "parked",
                                   "error", "unknown")
    assert isinstance(s["mount"]["ra_hours"], (int, float))

    assert s["camera"]["state"] in ("idle", "exposing")
    assert "exposure_s" in s["camera"]["last_exposure"]

    assert "latest_frame" in s


def test_thumbnail_png(dash):
    deadline = time.monotonic() + 25
    latest = None
    while time.monotonic() < deadline:
        _, _, body = api_get(dash, "/api/state")
        latest = json.loads(body)["latest_frame"]
        if latest:
            break
        time.sleep(0.3)
    assert latest, "no FITS frame captured in time"
    assert latest.endswith(".fits")

    status, ctype, body = api_get(dash, "/api/thumbnail")
    assert status == 200
    assert ctype == "image/png"
    assert body[:8] == b"\x89PNG\r\n\x1a\n"


def test_log_sse(dash):
    req = urllib.request.Request(
        base(dash) + "/api/log", headers={"Accept": "text/event-stream"})
    with urllib.request.urlopen(req, timeout=15) as r:
        assert r.headers.get_content_type() == "text/event-stream"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            line = r.readline().decode("utf-8", "replace")
            if line.startswith("data:"):
                assert len(line.strip()) > len("data:")
                return
    pytest.fail("SSE stream produced no data: line")


def test_pause_resume(tmp_path):
    d = make_dash(tmp_path, SLOW_SLEW_YAML)
    d.start()
    try:
        wait_state(d, "slewing", timeout=10)

        status, body = api_post(d, "/api/pause")
        assert status == 200 and body["ok"] is True
        wait_state(d, "paused", timeout=10)

        status, body = api_post(d, "/api/resume")
        assert status == 200 and body["ok"] is True
        # After a mid-slew resume the sequencer reports idle until the
        # slew loop's next checkpoint, then slewing/exposing again.
        wait_state(d, ("slewing", "exposing", "idle"), timeout=15)
    finally:
        d.stop()


def test_abort(tmp_path):
    d = make_dash(tmp_path)
    d.start()
    try:
        status, body = api_post(d, "/api/abort")
        assert status == 200 and body["ok"] is True
        assert body["state"] in ("aborted",) + RUNNING + ("paused",)
        wait_state(d, "aborted", timeout=15)
        assert (Path(d.seq.session.dir) / "session.log").exists()
    finally:
        d.stop()


def test_html_ids(dash):
    status, ctype, body = api_get(dash, "/")
    assert status == 200
    assert ctype == "text/html"
    html = body.decode("utf-8")
    for eid in HTML_IDS:
        assert f'id="{eid}"' in html, f"dashboard page missing element id={eid}"


def test_dash_cli_wiring():
    from astrocapture.cli import build_parser, cmd_dash

    args = build_parser().parse_args(
        ["dash", "--config", "examples/sim_session.yaml", "--port", "9999"])
    assert args.func is cmd_dash
    assert args.config == "examples/sim_session.yaml"
    assert args.port == 9999
