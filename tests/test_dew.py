"""Dew heater control: Magnus formula, controller law, sim drivers, config."""

import pytest

from astrocapture import config
from astrocapture.config import PlanError, load_plan
from astrocapture.dew import (
    DewController,
    SimHeater,
    SimSensor,
    dew_point_c,
)
from astrocapture.drivers.sim import SimCamera, SimMount
from astrocapture.sequencer import SeqState, Sequencer


# -- Magnus formula ---------------------------------------------------------


def test_magnus_known_values():
    # Reference values from the Magnus formula (Alduchov & Eskridge).
    assert dew_point_c(20.0, 50.0) == pytest.approx(9.26, abs=0.05)
    assert dew_point_c(30.0, 70.0) == pytest.approx(23.93, abs=0.05)
    assert dew_point_c(12.0, 88.0) == pytest.approx(10.08, abs=0.05)


def test_magnus_saturation_and_monotonicity():
    # At 100% RH the dew point is the air temperature itself.
    assert dew_point_c(25.0, 100.0) == pytest.approx(25.0, abs=1e-9)
    # Wetter air at the same temperature => higher dew point.
    assert dew_point_c(20.0, 80.0) > dew_point_c(20.0, 50.0)
    # Colder air at the same RH => lower dew point.
    assert dew_point_c(10.0, 50.0) < dew_point_c(20.0, 50.0)


# -- SimSensor ----------------------------------------------------------------


def test_sim_sensor_series_then_fallback():
    s = SimSensor([(20.0, 50.0), (19.0, 55.0)], fallback=(9.0, 90.0))
    assert s.read() == (20.0, 50.0)
    assert s.read() == (19.0, 55.0)
    assert s.read() == (9.0, 90.0)  # exhausted -> fallback, never crashes
    assert s.read() == (9.0, 90.0)


def test_humid_night_script():
    s = SimSensor.humid_night(start_temp_c=12.0, start_rh_pct=88.0,
                              end_temp_c=9.0, steps=6)
    first = s.read()
    assert first == pytest.approx((12.0, 88.0), abs=0.5)
    series = [first] + [s.read() for _ in range(5)]
    temps = [t for t, _ in series]
    rhs = [r for _, r in series]
    assert temps == sorted(temps, reverse=True)  # cooling
    assert rhs == sorted(rhs)  # ... while RH climbs toward saturation
    assert rhs[-1] == pytest.approx(100.0, abs=1.0)


# -- DewController --------------------------------------------------------------


def fixed_sensor(temp_c, rh_pct):
    return SimSensor([(temp_c, rh_pct)])


def test_control_law_boundaries():
    # Bone dry: duty 0.  (20°C/50% -> Td≈9.3, error=(20-2)-9.3=8.7>8)
    ctl = DewController(fixed_sensor(20.0, 50.0), SimHeater(),
                        margin_c=2.0, aggressiveness=0.5, max_duty=0.9)
    assert ctl.update() == pytest.approx(0.0)

    # Exactly at the margin boundary (error=0): duty = aggr * max_duty.
    temp, rh = 12.0, 80.0
    margin = temp - dew_point_c(temp, rh)  # forces error == 0
    assert margin > 0
    ctl = DewController(fixed_sensor(temp, rh), SimHeater(),
                        margin_c=margin, aggressiveness=0.6, max_duty=0.8)
    assert ctl.update() == pytest.approx(0.6 * 0.8)

    # Deep in the dew zone: proportional region then saturation cap.
    ctl = DewController(fixed_sensor(5.0, 99.0), SimHeater(),
                        margin_c=2.0, aggressiveness=0.5, max_duty=0.9)
    assert ctl.update() == pytest.approx(0.5544, abs=1e-3)
    # Aggressive tuning saturates at max_duty, never above it.
    ctl = DewController(fixed_sensor(5.0, 99.0), SimHeater(),
                        margin_c=2.0, aggressiveness=2.0, max_duty=0.9)
    assert ctl.update() == pytest.approx(0.9)


def test_humid_night_duty_ramps_up():
    sensor = SimSensor.humid_night(steps=12)
    heater = SimHeater()
    ctl = DewController(sensor, heater, margin_c=2.0,
                        aggressiveness=0.5, max_duty=0.9)
    duties = [ctl.update() for _ in range(12)]
    assert duties[-1] > duties[0]
    # Monotonic up to floating-point noise (RH clamps at 100 %).
    assert all(b >= a - 1e-9 for a, b in zip(duties, duties[1:]))
    assert duties[-1] == pytest.approx(0.5625, abs=1e-3)  # deep in margin
    assert heater.get_duty() == pytest.approx(duties[-1])
    assert len(heater.history) == 12


class _BoomSensor(SimSensor):
    def read(self):  # noqa: D102
        raise RuntimeError("sensor unplugged")


class _BoomHeater(SimHeater):
    def set_duty(self, duty):  # noqa: D102
        raise RuntimeError("heater driver fault")


def test_sensor_failure_holds_duty():
    sensor = SimSensor([(15.0, 80.0), (15.0, 80.0)])
    ctl = DewController(sensor, SimHeater(), margin_c=2.0,
                        aggressiveness=0.5, max_duty=0.9)
    d1 = ctl.update()
    d2 = ctl.update()
    assert d2 == pytest.approx(d1)
    ctl.sensor = _BoomSensor()
    assert ctl.update() == pytest.approx(d2)  # no raise, duty held
    assert ctl.last_duty == pytest.approx(d2)


def test_heater_failure_never_raises():
    ctl = DewController(fixed_sensor(10.0, 95.0), _BoomHeater(),
                        margin_c=2.0, aggressiveness=0.5, max_duty=0.9)
    assert ctl.update() == pytest.approx(0.0)  # initial duty held
    assert ctl.last_duty == pytest.approx(0.0)


def test_controller_rejects_bad_params():
    with pytest.raises(ValueError):
        DewController(fixed_sensor(10, 50), SimHeater(), margin_c=0)
    with pytest.raises(ValueError):
        DewController(fixed_sensor(10, 50), SimHeater(), max_duty=1.5)


# -- config block -----------------------------------------------------------------


def _plan_text(dew_block: str) -> str:
    return (
        "session:\n"
        "  name: test\n"
        "  target: {name: M51, ra_hours: 13.5, dec_deg: 47.2}\n"
        "mount: {driver: sim}\n"
        "camera: {driver: sim}\n"
        "sequence:\n"
        "  - {type: light, exposure: 1, count: 1}\n"
        + dew_block
    )


def test_dew_config_valid(tmp_path):
    p = tmp_path / "plan.yaml"
    p.write_text(_plan_text(
        "dew:\n"
        "  enabled: true\n"
        "  margin_c: 3.0\n"
        "  aggressiveness: 0.7\n"
        "  max_duty: 0.8\n"
        "  sensor: {driver: sim}\n"
        "  heater: {driver: sim}\n"))
    plan = load_plan(p)
    assert plan.dew.enabled
    assert plan.dew.margin_c == pytest.approx(3.0)
    assert plan.dew.sensor.driver == "sim"
    assert plan.dew.heater.driver == "sim"


def test_dew_config_defaults_to_disabled(tmp_path):
    p = tmp_path / "plan.yaml"
    p.write_text(_plan_text(""))
    plan = load_plan(p)
    assert not plan.dew.enabled
    assert "Dew     : disabled" in config.plan_summary(plan)


@pytest.mark.parametrize("block", [
    "dew:\n  enabled: true\n  margin_c: 0\n",          # margin must be > 0
    "dew:\n  enabled: true\n  aggressiveness: -1\n",  # aggressiveness > 0
    "dew:\n  enabled: true\n  max_duty: 1.5\n",       # max_duty in (0, 1]
    "dew:\n  enabled: true\n  sensor: {driver: bogus}\n",
    "dew: 42\n",                                      # must be a mapping
])
def test_dew_config_rejected(tmp_path, block):
    p = tmp_path / "plan.yaml"
    p.write_text(_plan_text(block))
    with pytest.raises(PlanError):
        load_plan(p)


# -- sequencer integration ----------------------------------------------------------


def _dew_plan(tmp_path, **dew_over):
    dew_cfg = config.DewConfig(enabled=True, **dew_over)
    plan = config.Plan(
        session_name="dewtest",
        output_dir=str(tmp_path / "sessions"),
        target=config.Target(name="M51", ra_hours=13.5, dec_deg=47.2),
        telescope="Test scope",
        instrument="Test cam",
        mount=config.DriverSpec("sim", {"slew_rate_dps": 3600.0}),
        camera=config.DriverSpec("sim", {"width": 128, "height": 128,
                                         "n_stars": 20, "seed": 1}),
        steps=[config.Step(type="light", exposure=0.1, count=2)],
        dew=dew_cfg,
    )
    return plan


def test_sequencer_runs_dew_per_light_frame(tmp_path):
    plan = _dew_plan(tmp_path)
    seq = Sequencer(plan, SimMount(slew_rate_dps=3600.0),
                    SimCamera(width=128, height=128, n_stars=20, seed=1))
    assert seq.run() == SeqState.DONE
    assert seq.dew is not None
    assert seq.dew_duty is not None and 0.0 <= seq.dew_duty <= 0.9
    assert seq.dew.sensor.reads == 2  # one update per light frame
    log_text = (seq.session.dir / "session.log").read_text()
    assert "dew: controller armed" in log_text
    assert log_text.count("dew: T=") == 2


def test_sequencer_heater_fault_never_aborts_imaging(tmp_path, monkeypatch,
                                                     caplog):
    def boom(self, duty):
        raise RuntimeError("heater driver exploded")
    monkeypatch.setattr(SimHeater, "set_duty", boom)
    plan = _dew_plan(tmp_path)
    seq = Sequencer(plan, SimMount(slew_rate_dps=3600.0),
                    SimCamera(width=128, height=128, n_stars=20, seed=1))
    with caplog.at_level("WARNING", logger="astrocapture.dew"):
        assert seq.run() == SeqState.DONE  # imaging completes regardless
    assert seq.frames_taken == 2
    assert "dew heater failed" in caplog.text


def test_sequencer_dew_setup_failure_continues_without_control(
        tmp_path, monkeypatch):
    monkeypatch.setattr(SimSensor, "humid_night",
                        classmethod(lambda cls, **kw: (_ for _ in ()).throw(
                            RuntimeError("no sensor"))))
    plan = _dew_plan(tmp_path)
    seq = Sequencer(plan, SimMount(slew_rate_dps=3600.0),
                    SimCamera(width=128, height=128, n_stars=20, seed=1))
    assert seq.run() == SeqState.DONE
    assert seq.dew is None  # controller never armed...
    assert seq.frames_taken == 2  # ...but imaging went ahead anyway
