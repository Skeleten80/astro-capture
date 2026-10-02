"""Night-sky catalog: vendored deep-sky objects with name lookup and search.

The data lives in ``astrocapture/data/catalog.json`` (built by
``tools/build_catalog.py`` from OpenNGC, CC-BY-SA-4.0 — see
``astrocapture/data/ATTRIBUTION.txt``).  It is loaded once and cached in
memory; coordinates are J2000 RA/Dec in decimal degrees.

``lookup("M51")`` resolves a target name to a catalog record;
``search(...)`` filters the catalog; ``tonight_best(...)`` ranks what is
well placed tonight for a given site (needs astropy, already a
dependency).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "catalog.json"

_CATALOG: list[dict] | None = None
_INDEX: dict[str, dict] | None = None


class UnknownObjectError(LookupError):
    """Raised when a target name is not in the night-sky catalog."""


def load_catalog() -> list[dict]:
    """Return the full catalog (list of dicts), loading and caching it."""
    global _CATALOG, _INDEX
    if _CATALOG is None:
        try:
            raw = DATA_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise RuntimeError(
                f"catalog data not found at {DATA_PATH}; "
                "run tools/build_catalog.py to generate it"
            ) from None
        _CATALOG = json.loads(raw)
        _INDEX = {}
        for obj in _CATALOG:
            for ident in obj["ids"]:
                _INDEX.setdefault(_normalize(ident), obj)
    return _CATALOG


def _normalize(name: str) -> str:
    """Aggressively normalize a designation for lookup.

    "M 51" -> "m51", "NGC7000" -> "ngc7000", "Caldwell 14" -> "c14".
    """
    key = re.sub(r"[\s._\-]+", "", name.strip().lower())
    key = re.sub(r"^caldwell(\d+)$", r"c\1", key)
    key = re.sub(r"^messier(\d+)$", r"m\1", key)
    return key


def lookup(name: str) -> dict:
    """Resolve *name* to a catalog record (a copy).

    Raises :class:`UnknownObjectError` if the name is not catalogued.
    """
    load_catalog()
    assert _INDEX is not None
    try:
        obj = _INDEX[_normalize(name)]
    except KeyError:
        raise UnknownObjectError(
            f"unknown object {name!r}: not in the night-sky catalog"
        ) from None
    return {**obj, "ids": list(obj["ids"])}


def search(
    type: str | None = None,  # noqa: A002 - matches the public API name
    max_mag: float | None = None,
    constellation: str | None = None,
    min_size_arcmin: float | None = None,
) -> list[dict]:
    """Filter the catalog; every criterion left as None is ignored."""
    want_const = constellation.strip().upper() if constellation else None
    out = []
    for obj in load_catalog():
        if type is not None and obj["type"] != type:
            continue
        if max_mag is not None and (obj["mag"] is None or obj["mag"] > max_mag):
            continue
        if want_const is not None and (obj["constellation"] or "").upper() != want_const:
            continue
        if min_size_arcmin is not None and (
            obj["size_arcmin"] is None or obj["size_arcmin"] < min_size_arcmin
        ):
            continue
        out.append(obj)
    return out


def _coerce_utc(when: datetime | None) -> datetime:
    if when is None:
        return datetime.now(timezone.utc)
    if when.tzinfo is None:
        return when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc)


def tonight_best(
    lat: float,
    lon: float,
    when: datetime | None = None,
    min_alt_deg: float = 30.0,
    limit: int = 20,
) -> list[dict]:
    """Rank the night's best-placed catalog objects for a site.

    For every object the peak altitude and the hours spent above
    *min_alt_deg* are computed over a 24 h window centred on *when*
    (naive datetimes are taken as UTC; default is now).  Objects are
    ranked by peak altitude first, then by time above the threshold.

    Returns at most *limit* dicts with ``name``, ``ids``, ``type``,
    ``mag``, ``constellation``, ``peak_alt_deg``, ``peak_time_utc``,
    ``hours_above``, ``rise_utc`` / ``set_utc`` (first/last time above
    the threshold, ISO UTC) and ``score`` (== peak altitude).
    """
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time
    import astropy.units as u
    import numpy as np

    when = _coerce_utc(when)
    objs = load_catalog()
    loc = EarthLocation(lat=float(lat) * u.deg, lon=float(lon) * u.deg)
    coords = SkyCoord(
        ra=np.array([o["ra"] for o in objs])[None, :] * u.deg,
        dec=np.array([o["dec"] for o in objs])[None, :] * u.deg,
        frame="icrs",
    )
    step_min = 10
    times = Time(when) + np.arange(-12 * 60, 12 * 60 + 1, step_min) * u.min
    # (1, N) coords against (M, 1) times broadcast to (M, N) altitudes.
    alt = coords.transform_to(
        AltAz(obstime=times[:, None], location=loc)).alt.deg
    n = len(objs)
    idx = np.arange(n)
    peak_idx = np.argmax(alt, axis=0)
    peak_alt = alt[peak_idx, idx]
    above = alt >= min_alt_deg
    hours_above = above.sum(axis=0) * (step_min / 60.0)
    # Round before ranking so the returned dicts sort exactly as displayed.
    score = np.round(peak_alt, 2)
    hours_r = np.round(hours_above, 2)

    order = np.lexsort((-hours_r, -score))
    order = [i for i in order if peak_alt[i] >= min_alt_deg][: max(limit, 0)]

    iso = [t.iso for t in times]
    results = []
    for i in order:
        col = np.flatnonzero(above[:, i])
        results.append({
            "name": objs[i]["name"],
            "ids": list(objs[i]["ids"]),
            "type": objs[i]["type"],
            "mag": objs[i]["mag"],
            "constellation": objs[i]["constellation"],
            "peak_alt_deg": round(float(peak_alt[i]), 1),
            "peak_time_utc": iso[int(peak_idx[i])],
            "hours_above": float(hours_r[i]),
            "rise_utc": iso[int(col[0])] if len(col) else None,
            "set_utc": iso[int(col[-1])] if len(col) else None,
            "score": float(score[i]),
        })
    return results
