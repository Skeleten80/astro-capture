"""Mathias's real gear: NexStar 6SE + Rebel T7i config and camera profile."""

from pathlib import Path

from astrocapture.config import load_plan
from astrocapture.drivers.dslr import T7I_PROFILE

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "mathias_6se_t7i.yaml"


def test_mathias_config_loads_and_validates():
    plan = load_plan(EXAMPLE)
    assert plan.session_name == "m13-6se-t7i"
    assert plan.target.name == "M13"
    assert plan.mount.driver == "indi"
    assert plan.camera.driver == "indi"
    assert len(plan.steps) == 4


def test_mathias_device_names():
    plan = load_plan(EXAMPLE)
    assert plan.mount.options["device"] == "Celestron GPS"
    assert plan.camera.options["device"] == "Canon DSLR"
    assert plan.camera.options.get("upload_mode") == "client"
    assert "serial_port" in plan.mount.options  # hand-controller USB/serial


def test_altaz_sane_exposures():
    # Alt-az field rotation caps usable subs at ~20-30 s without a wedge.
    plan = load_plan(EXAMPLE)
    assert plan.meridian_flip is False  # fork mount: no meridian flip
    lights = [s for s in plan.steps if s.type == "light"]
    assert lights and all(s.exposure <= 30 for s in lights)
    assert all(s.dither_every > 0 for s in lights)
    assert all(s.gain == 1600 for s in plan.steps if s.type in ("light", "dark"))
    darks = [s for s in plan.steps if s.type == "dark"]
    assert darks and darks[0].exposure == lights[0].exposure  # darks match lights


def test_t7i_profile_covers_setup_checklist():
    assert "T7i" in T7I_PROFILE["model"]
    assert T7I_PROFILE["mode_dial"].startswith("M (Manual)")
    assert "bulb" in T7I_PROFILE["bulb"]
    assert "mirror" in T7I_PROFILE["mirror_lockup"].lower()
    assert "DISABLE" in T7I_PROFILE["auto_power_off"]
    assert "T-ring" in T7I_PROFILE["mechanical"]
    assert T7I_PROFILE["iso_sweet_spot"] == 1600
