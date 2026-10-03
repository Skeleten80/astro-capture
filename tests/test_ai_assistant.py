"""ai_assistant.py: LLM path (mocked), rule-based fallback, validation."""

import os

import pytest

from astrocapture import ai_assistant, config


def fake_llm_ok(system: str, user: str) -> str:
    assert "25" in system  # gear constraints are in the prompt
    return """{
      "session": {"name": "m51-ai", "output_dir": "sessions",
                  "telescope": "Celestron NexStar 6SE",
                  "instrument": "Canon EOS Rebel T7i",
                  "target": {"name": "M51"}},
      "mount": {"driver": "indi", "host": "localhost",
                "port": 7624, "device": "Celestron GPS",
                "serial_port": "/dev/ttyUSB0"},
      "camera": {"driver": "indi", "host": "localhost",
                "port": 7624, "device": "Canon DSLR",
                "upload_mode": "client"},
      "sequence": [
        {"type": "light", "exposure": 25, "count": 288, "gain": 1600,
         "dither_every": 3},
        {"type": "dark", "exposure": 25, "count": 15, "gain": 1600},
        {"type": "flat", "exposure": 2, "count": 20, "gain": 1600},
        {"type": "bias", "exposure": 1, "count": 30, "gain": 1600}
      ]
    }"""


def test_llm_path_validates_and_resolves_target():
    plan = ai_assistant.plan_from_text("image M51 for 2 hours",
                                       llm=fake_llm_ok)
    assert plan.target.name == "M51"
    assert plan.target.ra_hours == pytest.approx(202.4695 / 15.0, abs=1e-3)
    assert plan.steps[0].type == "light"
    assert plan.steps[0].count == 288
    assert plan.steps[0].exposure == 25.0


def test_llm_path_rejects_overlong_subs():
    def bad_llm(system, user):
        d = fake_llm_ok(system, user).replace('"exposure": 25',
                                              '"exposure": 300')
        return d

    with pytest.raises(config.PlanError, match="alt-az cap"):
        ai_assistant.plan_from_text("x", llm=bad_llm)


def test_llm_path_rejects_garbage_json():
    with pytest.raises(config.PlanError, match="valid JSON"):
        ai_assistant.plan_from_text("x", llm=lambda s, u: "not json {{")


def test_llm_path_rejects_missing_keys():
    with pytest.raises(config.PlanError, match="missing required keys"):
        ai_assistant.plan_from_text("x", llm=lambda s, u: '{"target": {}}')


def test_llm_fences_stripped():
    plan = ai_assistant.plan_from_text(
        "x", llm=lambda s, u: "```json\n" + fake_llm_ok(s, u) + "\n```")
    assert plan.target.name == "M51"


def test_rule_based_two_hours_m51():
    plan = ai_assistant.plan_from_text("image M51 for 2 hours",
                                       use_llm=False)
    assert plan.target.name == "M51"
    light = plan.steps[0]
    assert light.type == "light" and light.exposure == 25.0
    assert light.count == 288  # 7200 / 25
    assert light.gain == 1600 and light.dither_every == 3
    types = [s.type for s in plan.steps]
    assert types == ["light", "dark", "flat", "bias"]
    dark = plan.steps[1]
    assert dark.exposure == 25.0 and dark.count == 14  # min(15, 288//20)


def test_rule_based_common_name_and_minutes():
    plan = ai_assistant.plan_from_text(
        "photograph the Whirlpool Galaxy for 90 minutes", use_llm=False)
    assert plan.target.name == "M51"
    assert plan.steps[0].count == 216  # 5400 / 25


def test_rule_based_designation_variants():
    for text in ("shoot NGC 7000 tonight", "M 13 for an hour",
                 "Caldwell 14, 30 minutes"):
        plan = ai_assistant.plan_from_text(text, use_llm=False)
        assert plan.target.ra_hours != 0.0 or plan.target.name != "unknown"


def test_rule_based_sub_exposure_clamped():
    plan = ai_assistant.plan_from_text("M51 with 60s subs for an hour",
                                       use_llm=False)
    assert plan.steps[0].exposure == 25.0


def test_rule_based_unknown_target_raises():
    with pytest.raises(config.PlanError, match="could not find a catalog"):
        ai_assistant.plan_from_text("image the Blorpt Nebula",
                                    use_llm=False)


def test_no_api_key_falls_back_to_rules(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # openai may or may not be installed; either way the rule path runs.
    plan = ai_assistant.plan_from_text("M51 for 30 minutes")
    assert plan.target.name == "M51"
    assert plan.steps[0].count == 72


def test_write_plan_yaml_round_trip(tmp_path):
    d = ai_assistant.plan_dict_from_text("M51 for 30 minutes", use_llm=False)
    out = ai_assistant.write_plan_yaml(d, tmp_path / "p.yaml")
    plan = config.load_plan(out)
    assert plan.target.name == "M51"
    assert plan.session_name == "m51-ai"


def test_cli_ask_dry_run(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from astrocapture.cli import main
    rc = main(["ask", "image M51 for 30 minutes", "--dry-run", "--no-llm"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "M51" in out and "dry run" in out


def test_cli_ask_writes_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from astrocapture.cli import main
    rc = main(["ask", "M51 for 30 minutes", "--no-llm",
               "--output", "test-plan.yaml"])
    assert rc == 0
    plan = config.load_plan(tmp_path / "test-plan.yaml")
    assert plan.target.name == "M51"


def test_cli_export_training_data(tmp_path, monkeypatch):
    import numpy as np
    from astropy.io import fits
    sess = tmp_path / "sess"
    (sess / "lights").mkdir(parents=True)
    rng = np.random.default_rng(0)
    img = 1000.0 + rng.normal(0, 5.0, (32, 32))
    hdr = fits.Header()
    hdr["EXPTIME"] = 25.0
    fits.writeto(sess / "lights" / "l_001.fits",
                 img.astype(np.float32), hdr, overwrite=True)
    monkeypatch.chdir(tmp_path)
    from astrocapture.cli import main
    rc = main(["export-training-data", str(sess), "--output", "training"])
    assert rc == 0
    assert (tmp_path / "training" / "labels.csv").is_file()
