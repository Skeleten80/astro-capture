"""Night scheduler: config blocks, astro math, target queue, dawn stop."""

from datetime import datetime, timedelta, timezone

import pytest
from astropy.coordinates import EarthLocation, get_body
from astropy.time import Time
import astropy.units as u

from astrocapture import config
from astrocapture.config import PlanError, load_plan
from astrocapture.drivers.base import MountState
from astrocapture.drivers.sim import SimCamera, SimMount
from astrocapture.scheduler import (
    NightScheduler,
    altitude_deg,
    astronomical_dawn_utc,
    moon_separation_deg,
)
from astrocapture.sequencer import SeqState

LAT, LON = 43.3767, -80.9809  # Stratford, Ontario
T0 = datetime(2026, 10, 3, 2, 0, tzinfo=timezone.utc)  # well before dawn


class FakeClock:
    """Injectable clock; ``advance`` jumps it instantly (no sleeping)."""

    def __init__(self, start):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += timedelta(seconds=seconds)


class JumpingClock(FakeClock):
    """After ``jump_after_calls`` reads, report ``jump_to`` (past dawn)."""

    def __init__(self, start, jump_after_calls, jump_to):
        super().__init__(start)
        self.calls = 0
        self.jump_after_calls = jump_after_calls
        self.jump_to = jump_to

    def __call__(self):
        self.calls += 1
        return self.jump_to if self.calls > self.jump_after_calls else self.t


def make_plan(tmp_path, targets, night_over=None, **night_kw):
    """Direct Plan construction: raw ra/dec targets, sim rig, tiny steps."""
    kw = dict(latitude=LAT, longitude=LON, min_alt_deg=0.0,
              min_moon_sep_deg=0.0, time_accel=1000.0)
    kw.update(night_kw)
    if night_over is not None:
        kw.update(night_over)
    entries = [config.TargetEntry(
        target=config.Target(name=name, ra_hours=ra, dec_deg=dec),
        priority=prio) for name, ra, dec, prio in targets]
    return config.Plan(
        session_name="nighttest",
        output_dir=str(tmp_path / "sessions"),
        target=entries[0].target,
        telescope="Test scope",
        instrument="Test cam",
        mount=config.DriverSpec("sim", {"slew_rate_dps": 3600.0}),
        camera=config.DriverSpec("sim", {"width": 128, "height": 128,
                                         "n_stars": 20, "seed": 1}),
        steps=[config.Step(type="light", exposure=0.2, count=2)],
        targets=entries,
        night=config.NightConfig(**kw),
    )


def make_factory(made):
    def factory():
        mount = SimMount(slew_rate_dps=3600.0)
        camera = SimCamera(width=128, height=128, n_stars=20, seed=1)
        made.append(mount)
        return mount, camera
    return factory


# -- config ---------------------------------------------------------------------


def test_night_block_validation(tmp_path):
    base = (
        "session:\n  name: t\n  target: {name: M51, ra_hours: 13.5, dec_deg: 47.2}\n"
        "mount: {driver: sim}\ncamera: {driver: sim}\n"
        "sequence:\n  - {type: light, exposure: 1, count: 1}\n")

    def load(night_block):
        p = tmp_path / "plan.yaml"
        p.write_text(base + night_block)
        return load_plan(p)

    # No night block -> None; summary shows nothing about the night.
    plan = load(base + "")
    assert plan.night is None
    assert "Night" not in config.plan_summary(plan)

    # Missing coordinates are required when the block is present.
    with pytest.raises(PlanError, match="night.latitude is required"):
        load("night:\n  longitude: -80.9\n")
    with pytest.raises(PlanError, match="night.longitude is required"):
        load("night:\n  latitude: 43.3\n")
    # Range checks.
    with pytest.raises(PlanError, match="night.min_alt_deg"):
        load("night:\n  latitude: 43.3\n  longitude: -80.9\n  min_alt_deg: 95\n")
    with pytest.raises(PlanError, match="night.min_moon_sep_deg"):
        load("night:\n  latitude: 43.3\n  longitude: -80.9\n  min_moon_sep_deg: 200\n")
    with pytest.raises(PlanError, match="night.time_accel"):
        load("night:\n  latitude: 43.3\n  longitude: -80.9\n  time_accel: 0.5\n")

    # Valid block: values land on the config, priority lands on targets,
    # and the summary shows both new blocks.
    plan = load(
        "night:\n  latitude: 43.3767\n  longitude: -80.9809\n"
        "  min_alt_deg: 25\n  min_moon_sep_deg: 20\n  time_accel: 60\n"
        "targets:\n  - {name: M51, ra_hours: 13.5, dec_deg: 47.2, priority: 2.5}\n"
        "dew:\n  enabled: true\n")
    assert plan.night.latitude == pytest.approx(43.3767)
    assert plan.night.time_accel == pytest.approx(60.0)
    assert plan.targets[0].priority == pytest.approx(2.5)
    summary = config.plan_summary(plan)
    assert "Dew     : enabled" in summary
    assert "Night   : 43.38°" in summary

    # priority must be positive.
    with pytest.raises(PlanError, match="priority"):
        load("targets:\n  - {name: M51, ra_hours: 13.5, dec_deg: 47.2, priority: 0}\n")


def test_night_scheduler_needs_night_block(tmp_path):
    plan = make_plan(tmp_path, [("a", 12.0, 69.0, 1.0)])
    plan.night = None
    with pytest.raises(ValueError, match="night:"):
        NightScheduler(plan, lambda: (None, None))


# -- astro math -------------------------------------------------------------------


def test_astronomical_dawn_utc():
    dawn = astronomical_dawn_utc(LAT, LON, T0)
    assert dawn.tzinfo is not None
    # The Sun is below -18° just before dawn and above just after.
    loc = EarthLocation(lat=LAT * u.deg, lon=LON * u.deg)
    from astropy.coordinates import AltAz, get_sun
    for dt, below in ((dawn - timedelta(minutes=10), True),
                      (dawn + timedelta(minutes=10), False)):
        alt = get_sun(Time(dt)).transform_to(
            AltAz(obstime=Time(dt), location=loc)).alt.deg
        assert (alt < -18.0) == below
    # Sanity for the chosen test date: dawn is the morning of Oct 4.
    assert (dawn.year, dawn.month, dawn.day) == (2026, 10, 4)


def test_astronomical_dawn_polar_day_raises():
    with pytest.raises(ValueError, match="no astronomical night"):
        astronomical_dawn_utc(78.0, 15.0, datetime(2026, 6, 21))


def test_moon_separation_deg():
    when = datetime(2026, 10, 3, 2, 0, tzinfo=timezone.utc)
    loc = EarthLocation(lat=LAT * u.deg, lon=LON * u.deg)
    moon = get_body("moon", Time(when), loc)
    moon_ra_h, moon_dec = moon.ra.deg / 15.0, moon.dec.deg
    assert moon_separation_deg(moon_ra_h, moon_dec, when, LAT, LON) < 1.0
    anti = moon_separation_deg((moon_ra_h + 12.0) % 24.0, -moon_dec,
                               when, LAT, LON)
    assert anti > 179.0


def test_altitude_deg_circumpolar():
    # M81 (Dec +69) from Stratford never drops below ~22°.
    alt = altitude_deg(148.8882 / 15.0, 69.0653, T0, LAT, LON)
    assert alt > 20.0


# -- scheduling ---------------------------------------------------------------------


def test_target_switching_quota_and_priority(tmp_path):
    # Same sky position => identical altitude/moon factors, so priority
    # alone decides the order.
    plan = make_plan(tmp_path, [("hi", 12.0, 69.0, 2.0),
                                ("lo", 12.0, 69.0, 1.0)])
    made = []
    sched = NightScheduler(plan, make_factory(made), clock=FakeClock(T0))
    assert sched.run() == 0
    assert sched.captured == [2, 2]  # quota (2 lights) filled for both
    assert [r["target"] for r in sched.completed] == ["hi", "lo"]
    assert all(r["status"] == "done" for r in sched.completed)
    for rec in sched.completed:
        lights = list((sched.night_dir / rec["dir"] / "lights").glob("*.fits"))
        assert len(lights) == 2
    assert all(m.state == MountState.PARKED for m in made)


def test_dawn_stops_new_targets_and_parks(tmp_path):
    dawn = astronomical_dawn_utc(LAT, LON, T0)
    clock = JumpingClock(T0, jump_after_calls=2,
                         jump_to=dawn + timedelta(hours=1))
    plan = make_plan(tmp_path, [("first", 12.0, 69.0, 1.0),
                                ("second", 12.0, 69.0, 1.0)])
    made = []
    sched = NightScheduler(plan, make_factory(made), clock=clock)
    assert sched.run() == 0
    # First target completed and parked; second never started.
    assert sched.captured == [2, 0]
    assert len(made) == 1
    assert made[0].state == MountState.PARKED
    assert len(sched.completed) == 1
    run_dirs = [d.name for d in sched.night_dir.iterdir() if d.is_dir()]
    assert not any(n.startswith("02-") for n in run_dirs)
    # Summary file: honestly reports what happened.
    summary = (sched.night_dir / "night_summary.md").read_text()
    assert "first" in summary and "second" in summary
    assert "2 light(s)" in summary
    assert "quota filled" in summary
    assert "no target was started" not in summary


def test_abort_parks_mount_explicitly(tmp_path, monkeypatch):
    """A sequencer that aborts WITHOUT parking: the scheduler must park."""

    class FakeSequencer:
        def __init__(self, plan, mount, camera, session=None):
            self.mount, self.session = mount, session
            self.last_focus_position = None
            self.error = "simulated fault"

        def run(self):
            self.mount.connect()  # like _connect_all, minus the parking
            return SeqState.ABORTED  # and crucially: no park()

    monkeypatch.setattr("astrocapture.scheduler.Sequencer", FakeSequencer)
    dawn = astronomical_dawn_utc(LAT, LON, T0)
    clock = JumpingClock(T0, jump_after_calls=2,
                         jump_to=dawn + timedelta(hours=1))
    plan = make_plan(tmp_path, [("flaky", 12.0, 69.0, 1.0)])
    made = []
    sched = NightScheduler(plan, make_factory(made), clock=clock)
    assert sched.run() == 0
    assert made[0].state == MountState.PARKED  # parked by the scheduler
    assert sched.failures and "flaky" in sched.failures[0]
    summary = (sched.night_dir / "night_summary.md").read_text()
    assert "aborted (simulated fault)" in summary


def test_moon_veto(tmp_path):
    when = datetime(2026, 10, 3, 2, 0, tzinfo=timezone.utc)
    loc = EarthLocation(lat=LAT * u.deg, lon=LON * u.deg)
    moon = get_body("moon", Time(when), loc)
    moon_ra_h, moon_dec = moon.ra.deg / 15.0, moon.dec.deg
    plan = make_plan(tmp_path,
                     [("moonbait", moon_ra_h, moon_dec, 5.0),  # top priority…
                      ("far", 12.0, 69.0, 1.0)],  # …high, far from the Moon
                     night_over={"min_moon_sep_deg": 30.0})
    sched = NightScheduler(plan, lambda: (None, None), clock=FakeClock(when))
    picked = sched.pick_target(when)
    assert picked is not None
    # …but it sits on the Moon, so the far target wins despite priority.
    assert picked[1].target.name == "far"
    # With only the moonbait target, nothing is eligible at all.
    plan2 = make_plan(tmp_path, [("moonbait", moon_ra_h, moon_dec, 1.0)],
                      night_over={"min_moon_sep_deg": 30.0})
    sched2 = NightScheduler(plan2, lambda: (None, None), clock=FakeClock(when))
    assert sched2.pick_target(when) is None
