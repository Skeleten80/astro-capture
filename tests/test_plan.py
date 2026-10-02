"""Plan loading and validation."""

import textwrap
from pathlib import Path

import pytest

from astrocapture import config
from astrocapture.config import PlanError, load_plan

VALID = textwrap.dedent(
    """
    session:
      name: "test"
      target: {name: "M51", ra_hours: 13.5, dec_deg: 47.2}
      telescope: "Test scope"
      instrument: "Test cam"
    mount: {driver: sim}
    camera: {driver: sim}
    sequence:
      - {type: light, exposure: 10, count: 2, gain: 1600, dither_every: 2}
      - {type: dark, exposure: 10, count: 1, gain: 1600}
      - {type: bias, count: 5}
    """
)


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "plan.yaml"
    p.write_text(text)
    return p


def test_valid_plan_loads(tmp_path):
    plan = load_plan(write(tmp_path, VALID))
    assert plan.session_name == "test"
    assert plan.target.name == "M51"
    assert plan.target.ra_hours == pytest.approx(13.5)
    assert len(plan.steps) == 3
    assert plan.steps[0].dither_every == 2
    assert plan.steps[2].type == "bias"


def test_bad_frame_type_rejected(tmp_path):
    bad = VALID.replace("{type: light,", "{type: lights,", 1)
    with pytest.raises(PlanError, match="type must be one of"):
        load_plan(write(tmp_path, bad))


def test_negative_exposure_rejected(tmp_path):
    bad = VALID.replace("exposure: 10, count: 2", "exposure: -5, count: 2", 1)
    with pytest.raises(PlanError, match="exposure must be > 0"):
        load_plan(write(tmp_path, bad))


def test_ra_out_of_range_rejected(tmp_path):
    bad = VALID.replace("ra_hours: 13.5", "ra_hours: 25.0")
    with pytest.raises(PlanError, match="ra_hours"):
        load_plan(write(tmp_path, bad))


def test_dec_out_of_range_rejected(tmp_path):
    bad = VALID.replace("dec_deg: 47.2", "dec_deg: 91.0")
    with pytest.raises(PlanError, match="dec_deg"):
        load_plan(write(tmp_path, bad))


def test_missing_sequence_rejected(tmp_path):
    bad = VALID.replace("sequence:", "nope:")
    with pytest.raises(PlanError, match="sequence must be a non-empty list"):
        load_plan(write(tmp_path, bad))


def test_missing_mount_driver_rejected(tmp_path):
    bad = VALID.replace("mount: {driver: sim}", "mount: {}")
    with pytest.raises(PlanError, match="mount.*'driver'"):
        load_plan(write(tmp_path, bad))


def test_multiple_errors_all_reported(tmp_path):
    bad = VALID.replace("ra_hours: 13.5", "ra_hours: 99")
    bad = bad.replace("{type: dark,", "{type: bogus,", 1)
    with pytest.raises(PlanError) as excinfo:
        load_plan(write(tmp_path, bad))
    msg = str(excinfo.value)
    assert "ra_hours" in msg and "bogus" in msg


def test_plan_summary_mentions_steps(tmp_path):
    plan = load_plan(write(tmp_path, VALID))
    summary = config.plan_summary(plan)
    assert "M51" in summary and "light" in summary and "bias" in summary
