"""Sequencer integration: platesolve / autofocus / guiding / watchdog wiring.

All runs use the sim drivers (plus small test doubles); no hardware,
no network, no solve-field binary needed.
"""

import math
import time
from pathlib import Path

import pytest

from astrocapture import config
from astrocapture.config import PlanError
from astrocapture.drivers.base import MountState
from astrocapture.drivers.sim import SimCamera, SimMount
from astrocapture.focus import SimFocuser
from astrocapture.platesolve import FakeSolver
from astrocapture.sequencer import SeqState, Sequencer


# -- helpers ---------------------------------------------------------------


def make_plan(tmp_path: Path, steps=None, **kw) -> config.Plan:
    """Small sim plan; ``kw`` overrides Plan fields (incl. new blocks)."""
    if steps is None:
        steps = [
            {"type": "light", "exposure": 0.1, "count": 2, "gain": 1600,
             "dither_every": 1},
        ]
    fields = dict(
        session_name="intele",
        output_dir=str(tmp_path / "sessions"),
        target=config.Target(name="M51", ra_hours=13.5, dec_deg=47.2),
        telescope="Test scope",
        instrument="Test cam",
        mount=config.DriverSpec("sim", {"slew_rate_dps": 3600.0}),
        camera=config.DriverSpec("sim", {"width": 128, "height": 128,
                                         "n_stars": 30, "seed": 1}),
        steps=[config.Step(**s) for s in steps],
    )
    fields.update(kw)
    return config.Plan(**fields)


def run_seq(plan: config.Plan, **seq_kw):
    mount = SimMount(**plan.mount.options)
    camera = SimCamera(**plan.camera.options)
    seq = Sequencer(plan, mount, camera, **seq_kw)
    seq.run()
    return seq, mount, camera


def wait_parked(mount: SimMount, timeout: float = 10.0) -> MountState:
    """Poll until the sim mount finishes its park slew."""
    deadline = time.monotonic() + timeout
    while mount.state != MountState.PARKED and time.monotonic() < deadline:
        time.sleep(0.05)
    return mount.state


def with_platesolve(plan: config.Plan, **kw) -> FakeSolver:
    """Attach a per-target platesolve block; returns the injected solver."""
    ra_deg = plan.target.ra_hours * 15.0
    dec_deg = plan.target.dec_deg
    solver = FakeSolver(kw.pop("positions", [(ra_deg, dec_deg)]))
    plan.targets = [config.TargetEntry(
        target=plan.target,
        platesolve=config.PlateSolveBlock(enabled=True, **kw),
    )]
    return solver


# -- platesolve ------------------------------------------------------------


def test_platesolve_recenter_called_and_session_completes(tmp_path):
    plan = make_plan(tmp_path)
    solver = with_platesolve(plan, tolerance_arcmin=2.0, max_iterations=3,
                             exposure_s=0.05)
    seq, mount, camera = run_seq(plan, solver=solver)
    assert solver.calls >= 1  # recenter actually ran
    assert seq.state == SeqState.DONE
    assert seq.frames_taken == 2


def test_platesolve_failure_parks_and_stops(tmp_path):
    plan = make_plan(tmp_path,
                     safety=config.SafetyConfig(max_solve_failures=1))
    ra_deg = plan.target.ra_hours * 15.0
    # Solved position 10° off every time: recenter can never converge.
    solver = with_platesolve(plan, positions=[(ra_deg + 10.0, 47.2)],
                             max_iterations=2, exposure_s=0.05)
    seq, mount, camera = run_seq(plan, solver=solver)
    assert seq.state == SeqState.ABORTED
    assert seq.frames_taken == 0  # parked before any imaging
    assert wait_parked(mount) == MountState.PARKED
    log = (seq.session.dir / "session.log").read_text()
    assert "plate solve failed 1 times in a row" in log


def test_multi_target_run_visits_each_target(tmp_path):
    plan = make_plan(tmp_path, steps=[
        {"type": "light", "exposure": 0.1, "count": 1, "gain": 1600},
    ])
    plan.targets = [
        config.TargetEntry(
            target=config.Target(name="M51", ra_hours=13.5, dec_deg=47.2)),
        config.TargetEntry(
            target=config.Target(name="M13", ra_hours=16.695, dec_deg=36.46)),
    ]
    seq, mount, camera = run_seq(plan)
    assert seq.state == SeqState.DONE
    assert seq.frames_taken == 2
    lights = sorted((seq.session.dir / "lights").glob("*.fits"),
                    key=lambda f: f.name)
    names = [f.name.split("_light")[0] for f in lights]
    assert sorted(names) == ["M13", "M51"] and len(names) == 2
    # Frame 1 belongs to the first target, frame 2 to the second.
    by_number = sorted(lights, key=lambda f: int(f.stem.rsplit("_", 1)[1]))
    assert [f.name.split("_light")[0] for f in by_number] == ["M51", "M13"]


# -- autofocus -------------------------------------------------------------


def test_autofocus_with_sim_focuser(tmp_path):
    focuser = SimFocuser(best_position=25000, position=24800,
                         hfr_min=1.8, curvature=2e-6, noise=0.02)
    before = focuser.true_hfr()
    plan = make_plan(
        tmp_path,
        autofocus=config.AutofocusConfig(
            enabled=True, every_minutes=60, focus_exposure_s=0.05,
            n_positions=7, step=150),
    )
    seq, mount, camera = run_seq(plan, focuser=focuser)
    assert seq.state == SeqState.DONE
    assert seq.last_focus_position is not None
    assert abs(seq.last_focus_position - 25000) <= 60
    assert focuser.true_hfr() < before  # focus actually improved
    assert math.isfinite(seq._focus_baseline_hfr)


def test_autofocus_without_focuser_uses_assist_mode(tmp_path, capsys):
    # A non-sim mount driver with no focuser device: the honest path for
    # a stock 6SE (no focus motor) — print the Bahtinov guide, keep going.
    plan = make_plan(
        tmp_path,
        autofocus=config.AutofocusConfig(enabled=True, focuser_device=""),
    )
    plan.mount = config.DriverSpec("custom", {"slew_rate_dps": 3600.0})
    seq, mount, camera = run_seq(plan)
    assert seq.state == SeqState.DONE
    assert seq._focuser is None
    log = (seq.session.dir / "session.log").read_text()
    assert "manual-assist mode" in log
    assert "Bahtinov" in capsys.readouterr().out


# -- guiding ---------------------------------------------------------------


def test_guiding_unreachable_falls_back_to_blind_dither(tmp_path):
    plan = make_plan(tmp_path, guiding=config.GuidingConfig(enabled=True))
    seq, mount, camera = run_seq(plan)
    assert seq.state == SeqState.DONE
    assert seq._phd2 is None  # no client: fell back to unguided
    log = (seq.session.dir / "session.log").read_text()
    assert "continuing unguided" in log
    assert "dithering by" in log  # blind dither path was used


class FlakyGuideClient:
    """PHD2 double: guiding starts, but the star is never re-acquired."""

    def __init__(self):
        self.starts = 0
        self.closed = False

    @property
    def star_lost(self):
        return True

    def clear_star_lost(self):
        pass

    def start_guiding(self, *args, **kwargs):
        self.starts += 1
        return self.starts == 1  # session start OK; resume attempts fail

    def stop_guiding(self):
        return True

    def dither(self, *args, **kwargs):
        return True

    def close(self):
        self.closed = True


def test_guide_lost_exhausting_budget_parks(tmp_path):
    plan = make_plan(
        tmp_path,
        steps=[{"type": "light", "exposure": 0.1, "count": 3, "gain": 1600}],
        guiding=config.GuidingConfig(enabled=True),
        safety=config.SafetyConfig(guide_retry_budget=1),
    )
    fake = FlakyGuideClient()
    seq, mount, camera = run_seq(plan, phd2_client=fake)
    assert seq.state == SeqState.ABORTED
    assert wait_parked(mount) == MountState.PARKED
    assert fake.closed  # client stopped + closed at shutdown
    log = (seq.session.dir / "session.log").read_text()
    assert "guide star lost repeatedly" in log


# -- watchdog --------------------------------------------------------------


class ExplodingCamera(SimCamera):
    def download_image(self):
        raise RuntimeError("simulated camera failure")


def test_exception_mid_run_parks_mount(tmp_path):
    plan = make_plan(tmp_path)
    mount = SimMount(**plan.mount.options)
    camera = ExplodingCamera(**plan.camera.options)
    seq = Sequencer(plan, mount, camera)
    seq.run()
    assert seq.state == SeqState.ABORTED
    assert wait_parked(mount) == MountState.PARKED
    log = (seq.session.dir / "session.log").read_text()
    assert "ALERT" in log and "simulated camera failure" in log


# -- config validation -----------------------------------------------------


BAD_NEW_BLOCKS = """
session:
  name: "bad"
  output_dir: "sessions"
  target: {name: "M51"}
targets:
  - name: "M51"
    platesolve: {enabled: true, tolerance_arcmin: -2.0, max_iterations: 0, exposure_s: 0}
autofocus: {enabled: true, every_minutes: 0, hfr_degradation_trigger: 1.0, n_positions: 2}
guiding: {enabled: true, phd2_port: 99999, dither_pixels: -1}
safety: {max_session_hours: -1, max_solve_failures: 0, guide_retry_budget: -1}
alerts: {webhook_url: "ftp://example.com/hook"}
mount: {driver: sim}
camera: {driver: sim}
sequence:
  - {type: light, exposure: 0.1, count: 1}
"""


def test_config_validation_rejects_bad_new_blocks(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(BAD_NEW_BLOCKS)
    with pytest.raises(PlanError) as excinfo:
        config.load_plan(p)
    msg = str(excinfo.value)
    # Every problem is reported at once, not just the first.
    for key in ("tolerance_arcmin", "max_iterations", "exposure_s",
                "every_minutes", "hfr_degradation_trigger", "n_positions",
                "phd2_port", "dither_pixels",
                "max_session_hours", "max_solve_failures", "guide_retry_budget",
                "webhook_url"):
        assert key in msg, f"missing complaint about {key}"


GOOD_NEW_BLOCKS = """
session:
  name: "good"
  output_dir: "sessions"
  target: {name: "M51"}
targets:
  - name: "M51"
    platesolve: {enabled: true, tolerance_arcmin: 2.0, max_iterations: 5, exposure_s: 5.0}
  - {name: "M13", ra_hours: 16.6950, dec_deg: 36.4600}
autofocus: {enabled: true, every_minutes: 45, hfr_degradation_trigger: 1.3,
            focuser_device: "Focuser", focus_exposure_s: 4.0,
            n_positions: 9, step: 200}
guiding: {enabled: true, phd2_host: "localhost", phd2_port: 4400,
          dither_pixels: 5.0, settle_pixels: 1.5, settle_time_s: 10}
safety: {max_session_hours: 6, max_solve_failures: 2, guide_retry_budget: 2}
alerts: {webhook_url: "https://example.com/hook"}
mount: {driver: sim}
camera: {driver: sim}
sequence:
  - {type: light, exposure: 0.1, count: 1}
"""


def test_config_accepts_all_new_blocks(tmp_path):
    p = tmp_path / "good.yaml"
    p.write_text(GOOD_NEW_BLOCKS)
    plan = config.load_plan(p)
    assert len(plan.targets) == 2
    assert plan.targets[0].platesolve.enabled
    assert plan.targets[0].platesolve.tolerance_arcmin == 2.0
    assert plan.targets[1].target.name == "M13"
    assert not plan.targets[1].platesolve.enabled
    assert plan.autofocus.enabled and plan.autofocus.every_minutes == 45
    assert plan.autofocus.focuser_device == "Focuser"
    assert plan.guiding.enabled and plan.guiding.phd2_port == 4400
    assert plan.safety.max_session_hours == 6
    assert plan.safety.max_solve_failures == 2
    assert plan.alerts.webhook_url == "https://example.com/hook"
    # The legacy single-target field tracks the first entry.
    assert plan.target.name == "M51"
