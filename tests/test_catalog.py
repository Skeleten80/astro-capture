"""Night-sky catalog: data completeness, lookup, search, tonight ranking,
and plan target-name resolution."""

import math
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest

from astrocapture import catalog
from astrocapture.catalog import (
    UnknownObjectError,
    load_catalog,
    lookup,
    search,
    tonight_best,
)
from astrocapture.config import PlanError, load_plan


def ang_sep_deg(ra1, dec1, ra2, dec2):
    """Great-circle separation in degrees."""
    ra1, dec1, ra2, dec2 = map(math.radians, (ra1, dec1, ra2, dec2))
    cos_c = (math.sin(dec1) * math.sin(dec2)
             + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_c))))


def messier_numbers():
    return {
        int(m.group(1))
        for o in load_catalog() for i in o["ids"]
        for m in [re.fullmatch(r"M(\d+)", i)] if m
    }


def caldwell_numbers():
    return {
        int(m.group(1))
        for o in load_catalog() for i in o["ids"]
        for m in [re.fullmatch(r"C(\d+)", i)] if m
    }


def test_catalog_loads_with_expected_schema():
    cat = load_catalog()
    assert len(cat) >= 4000
    for o in cat[:50]:
        assert set(o) == {"ids", "name", "ra", "dec", "type", "mag",
                          "size_arcmin", "constellation"}
        assert isinstance(o["ids"], list) and o["ids"]
        assert -90.0 <= o["dec"] <= 90.0
        assert 0.0 <= o["ra"] < 360.0


def test_all_110_messier_present():
    assert messier_numbers() == set(range(1, 111))


def test_all_109_caldwell_present():
    assert caldwell_numbers() == set(range(1, 110))


def test_lookup_m51():
    o = lookup("M51")
    assert "M51" in o["ids"] and "NGC 5194" in o["ids"]
    assert ang_sep_deg(o["ra"], o["dec"], 202.4696, 47.1953) < 0.5


def test_lookup_m13_variant():
    o = lookup("m 13")
    assert "M13" in o["ids"]
    assert ang_sep_deg(o["ra"], o["dec"], 250.4218, 36.4599) < 0.5


def test_lookup_ngc7000():
    o = lookup("NGC 7000")
    assert ang_sep_deg(o["ra"], o["dec"], 314.8214, 44.5266) < 0.5
    assert o["type"] == "nebula"


def test_lookup_name_variants_resolve_to_same_object():
    ids = {tuple(lookup(v)["ids"]) for v in
           ("M51", "M 51", "m51", "NGC7000", "NGC 7000", "ngc 7000")}
    assert len(ids) == 2  # M51 variants -> one object, NGC 7000 variants -> one
    assert tuple(lookup("Caldwell 14")["ids"]) == tuple(lookup("C14")["ids"])


def test_lookup_unknown_raises():
    with pytest.raises(UnknownObjectError):
        lookup("Xyzzy 999")
    with pytest.raises(UnknownObjectError):
        lookup("")


def test_search_filters():
    bright_galaxies = search(type="galaxy", max_mag=10)
    assert bright_galaxies, "expected bright galaxies"
    assert all(o["type"] == "galaxy" and o["mag"] <= 10 for o in bright_galaxies)

    orion = search(constellation="Ori")
    assert orion and all(o["constellation"] == "Ori" for o in orion)
    assert any("M42" in o["ids"] for o in orion)

    big = search(min_size_arcmin=100)
    assert big and all(o["size_arcmin"] >= 100 for o in big)

    assert search(type="not_a_type") == []
    assert search(max_mag=1.0) == [] or all(
        o["mag"] is not None and o["mag"] <= 1.0 for o in search(max_mag=1.0))


def test_tonight_best_ranked_and_valid():
    when = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
    best = tonight_best(43.38, -80.96, when=when, min_alt_deg=30, limit=10)
    assert 0 < len(best) <= 10
    keys = {"name", "ids", "type", "mag", "constellation", "peak_alt_deg",
            "peak_time_utc", "hours_above", "rise_utc", "set_utc", "score"}
    prev = (float("inf"), float("inf"))
    for o in best:
        assert keys <= set(o)
        assert 30.0 <= o["peak_alt_deg"] <= 90.0
        assert o["hours_above"] >= 0
        key = (o["score"], o["hours_above"])
        assert key <= prev, "results must be sorted by descending score"
        prev = key


def test_tonight_best_defaults_to_now():
    best = tonight_best(43.38, -80.96, limit=3)
    assert len(best) == 3


def write_plan(tmp_path: Path, target_yaml: str) -> Path:
    text = textwrap.dedent(f"""\
        session:
          name: "cat-test"
          target: {target_yaml}
          telescope: "Test scope"
          instrument: "Test cam"
        mount: {{driver: sim}}
        camera: {{driver: sim}}
        sequence:
          - {{type: light, exposure: 10, count: 1}}
        """)
    p = tmp_path / "plan.yaml"
    p.write_text(text)
    return p


def test_plan_target_name_resolves(tmp_path):
    plan = load_plan(write_plan(tmp_path, '{name: "M51"}'))
    assert plan.target.ra_hours == pytest.approx(202.4696 / 15.0, abs=1e-3)
    assert plan.target.dec_deg == pytest.approx(47.1953, abs=1e-3)
    assert "M51" in plan.target.resolved_ids
    assert "NGC 5194" in plan.target.resolved_ids


def test_plan_explicit_coords_untouched(tmp_path):
    plan = load_plan(write_plan(
        tmp_path, '{name: "M51", ra_hours: 13.5, dec_deg: 47.2}'))
    assert plan.target.ra_hours == pytest.approx(13.5)
    assert plan.target.dec_deg == pytest.approx(47.2)
    assert "M51" in plan.target.resolved_ids


def test_plan_unknown_name_without_coords_rejected(tmp_path):
    with pytest.raises(PlanError, match="not found in the night-sky catalog"):
        load_plan(write_plan(tmp_path, '{name: "Xyzzy 999"}'))
