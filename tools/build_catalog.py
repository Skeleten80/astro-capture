#!/usr/bin/env python3
"""One-shot builder for the vendored night-sky catalog.

Downloads the OpenNGC deep-sky database (NGC + IC + addendum), merges
Messier/Caldwell cross-references, trims to a sensible amateur set and
writes ``astrocapture/data/catalog.json``.

This is a *build* tool, not a runtime dependency: run it when you want to
refresh the vendored data::

    python3 tools/build_catalog.py [--mag-limit 14.0]

Data sources and licensing:

* OpenNGC (https://github.com/mattiaverga/OpenNGC) by Mattia Verga and
  contributors — ``database_files/NGC.csv`` + ``database_files/addendum.csv``.
  Released under **CC-BY-SA-4.0**; the derived ``catalog.json`` carries the
  same license (see ``astrocapture/data/ATTRIBUTION.txt``).
* Caldwell cross-checks verified against the Caldwell catalogue table on
  Wikipedia (NGC 869/884 = C14 per OpenNGC's own notes).

Known upstream quirks corrected here (with rationale in comments):

* OpenNGC lists the M102 row with ``M=101`` (a typo) and type ``Dup``; the
  standard identification M102 = NGC 5866 (Spindle Galaxy) is applied.
* OpenNGC's addendum carries the non-NGC/IC Caldwell objects as
  ``C009`` (Cave Nebula, C9), ``C041`` (Hyades, C41) and ``C099``
  (Coalsack, C99); these get their ``C<n>`` designations from the name.
* Caldwell 14 (Double Cluster) is NGC 869 *and* NGC 884 — both rows get
  the ``C14`` designation (per OpenNGC's own notes on those rows).
* Caldwell 49 is the Rosette Nebula, NGC 2237 (Wikipedia); OpenNGC tags
  the ``C 049`` identifier on the NGC 2238 knot instead, so the
  designation is moved to the NGC 2237 row.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "astrocapture" / "data" / "catalog.json"

OPENNGC_TAG = "master"  # latest stable snapshot on the default branch
OPENNGC_BASE = (
    f"https://raw.githubusercontent.com/mattiaverga/OpenNGC/{OPENNGC_TAG}"
    "/database_files"
)
OPENNGC_FILES = ("NGC.csv", "addendum.csv")

# Columns the build depends on; fail loudly if upstream renames them.
REQUIRED_COLUMNS = {
    "Name", "Type", "RA", "Dec", "Const", "MajAx",
    "B-Mag", "V-Mag", "M", "Identifiers", "Common names",
}

# OpenNGC type codes we keep for an amateur deep-sky set.
KEPT_TYPES = {
    "G", "GPair", "GTrpl", "GGroup",      # galaxies
    "OCl", "*Ass",                        # open clusters / associations
    "GCl",                                # globular clusters
    "PN",                                 # planetary nebulae
    "SNR",                                # supernova remnants
    "Neb", "EmN", "RfN", "DrkN", "HII", "Cl+N",  # nebulae
    "Other",                              # catalogued, type unclear
    "**",                                 # double star (M40 = Winnecke 4)
}

TYPE_MAP = {
    "G": "galaxy", "GPair": "galaxy", "GTrpl": "galaxy", "GGroup": "galaxy",
    "OCl": "open_cluster", "*Ass": "open_cluster",
    "GCl": "globular_cluster",
    "PN": "planetary_nebula",
    "SNR": "supernova_remnant",
    "Neb": "nebula", "EmN": "nebula", "RfN": "nebula",
    "DrkN": "nebula", "HII": "nebula", "Cl+N": "nebula",
    "Other": "other",
    "**": "star",
}

# Caldwell designations the OpenNGC "Identifiers" column does not carry.
# C9/C41/C99 live in the addendum under C0nn names; C14 is a pair.
CALDWELL_BY_NAME = {"C009": 9, "C041": 41, "C099": 99}
CALDWELL_14_ROWS = ("NGC0869", "NGC0884")  # Double Cluster, per OpenNGC notes
# Wikipedia: C49 = Rosette Nebula = NGC 2237 (OpenNGC tags the NGC 2238 knot).
CALDWELL_49_ROW = "NGC2237"
CALDWELL_49_WRONG_ROW = "NGC2238"


def die(msg: str) -> "sys.NoReturn":
    print(f"build_catalog: FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    try:
        subprocess.run(
            ["curl", "-fL", "--retry", "2", "--max-time", "300",
             "-o", str(dest), url],
            check=True, capture_output=True, text=True,
        )
    except FileNotFoundError:
        die("curl is not installed")
    except subprocess.CalledProcessError as exc:
        die(f"download failed for {url}:\n{exc.stderr.strip()}")
    if not dest.stat().st_size:
        die(f"downloaded empty file for {url}")


def parse_ra_deg(raw: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        h, m, s = raw.split(":")
        return (float(h) + float(m) / 60.0 + float(s) / 3600.0) * 15.0
    except ValueError:
        return None


def parse_dec_deg(raw: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        sign = -1.0 if raw.startswith("-") else 1.0
        d, m, s = raw.lstrip("+-").split(":")
        return sign * (float(d) + float(m) / 60.0 + float(s) / 3600.0)
    except ValueError:
        return None


def fnum(raw: str) -> float | None:
    try:
        return float(raw.strip())
    except (ValueError, AttributeError):
        return None


def canon_id(token: str) -> str:
    """'NGC0224' -> 'NGC 224'; 'M040' -> 'M40'; leaves odd tokens alone."""
    t = re.sub(r"\s+", " ", token.strip())
    compact = t.replace(" ", "")
    m = re.fullmatch(r"M0*(\d+)", compact, re.IGNORECASE)
    if m:
        return f"M{m.group(1)}"
    m = re.fullmatch(r"C0*(\d+)", compact, re.IGNORECASE)
    if m:
        return f"C{m.group(1)}"
    m = re.fullmatch(r"([A-Za-z]+?)0*(\d+[A-Za-z]*)", compact)
    if m:
        return f"{m.group(1).upper()} {m.group(2)}"
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description="Build astrocapture/data/catalog.json")
    ap.add_argument("--mag-limit", type=float, default=14.0,
                    help="V-mag (fallback B-mag) cut for non Messier/Caldwell objects")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="build_catalog_") as tmp:
        tmpdir = Path(tmp)
        rows: list[dict] = []
        for fname in OPENNGC_FILES:
            dest = tmpdir / fname
            download(f"{OPENNGC_BASE}/{fname}", dest)
            with dest.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh, delimiter=";")
                cols = set(reader.fieldnames or [])
                missing = REQUIRED_COLUMNS - cols
                if missing:
                    die(f"{fname}: upstream columns changed; missing {sorted(missing)} "
                        f"(got {sorted(cols)})")
                n = 0
                for rec in reader:
                    rec["_file"] = fname
                    rows.append(rec)
                    n += 1
            print(f"  {fname}: {n} rows")
    print(f"total upstream rows: {len(rows)}")

    by_name: dict[str, dict] = {r["Name"]: r for r in rows}

    # ---- Messier mapping ------------------------------------------------
    messier: dict[int, list[str]] = {}  # M number -> OpenNGC Name keys
    for r in rows:
        m = r["M"].strip()
        if m:
            messier.setdefault(int(m), []).append(r["Name"])
    # Upstream quirk: the M102 row carries M="101" (typo) and type Dup.
    # Standard identification: M102 = NGC 5866 (Spindle Galaxy).
    if 102 not in messier:
        if "NGC5866" not in by_name:
            die("cannot apply M102 fix: NGC5866 missing from OpenNGC data")
        messier.setdefault(102, []).append("NGC5866")
        print("  applied M102 -> NGC5866 fix (upstream M column typo)")
    missing_m = [n for n in range(1, 111) if n not in messier]
    if missing_m:
        die(f"Messier coverage incomplete; missing M numbers: {missing_m}")
    print(f"  Messier objects: {len(messier)} (M1..M110 all present)")

    # ---- Caldwell mapping -----------------------------------------------
    caldwell: dict[int, list[str]] = {}  # C number -> OpenNGC Name keys
    for r in rows:
        for ident in r["Identifiers"].split(","):
            m = re.fullmatch(r"\s*[Cc]\s*0*(\d+)\s*", ident)
            if m:
                caldwell.setdefault(int(m.group(1)), []).append(r["Name"])
    for name, cnum in CALDWELL_BY_NAME.items():
        if name in by_name:
            caldwell.setdefault(cnum, []).append(name)
    for name in CALDWELL_14_ROWS:
        if name not in by_name:
            die(f"Caldwell 14 mapping broken: {name} missing from OpenNGC data")
        caldwell.setdefault(14, []).append(name)
    # C49: move the designation from the NGC 2238 knot to NGC 2237 (Wikipedia).
    if CALDWELL_49_ROW not in by_name:
        die("Caldwell 49 mapping broken: NGC2237 missing from OpenNGC data")
    caldwell.setdefault(49, []).append(CALDWELL_49_ROW)
    for key, holders in caldwell.items():
        if CALDWELL_49_WRONG_ROW in holders and key == 49:
            holders.remove(CALDWELL_49_WRONG_ROW)
    missing_c = [n for n in range(1, 110) if n not in caldwell]
    if missing_c:
        die(f"Caldwell coverage incomplete; missing C numbers: {missing_c}")
    print(f"  Caldwell objects: {len(caldwell)} (C1..C109 all present)")

    # invert: OpenNGC Name -> designations (deduped)
    messier_of: dict[str, int] = {}
    for mnum, names in messier.items():
        for nm in dict.fromkeys(names):
            messier_of[nm] = mnum
    caldwell_of: dict[str, list[int]] = {}
    for cnum, names in caldwell.items():
        for nm in dict.fromkeys(names):
            caldwell_of.setdefault(nm, []).append(cnum)
    for nm in caldwell_of:
        caldwell_of[nm] = sorted(set(caldwell_of[nm]))

    # ---- trim to the amateur set ----------------------------------------
    kept: list[dict] = []
    skipped_no_coords = 0
    for r in rows:
        otype = r["Type"].strip()
        if otype not in KEPT_TYPES:
            continue
        ra = parse_ra_deg(r["RA"])
        dec = parse_dec_deg(r["Dec"])
        if ra is None or dec is None:
            skipped_no_coords += 1
            continue
        name = r["Name"]
        is_mc = name in messier_of or name in caldwell_of
        mag = fnum(r["V-Mag"])
        if mag is None:
            mag = fnum(r["B-Mag"])
        if not is_mc and (mag is None or mag > args.mag_limit):
            continue

        ids: list[str] = []
        if name in messier_of:
            ids.append(f"M{messier_of[name]}")
        for cnum in sorted(caldwell_of.get(name, [])):
            ids.append(f"C{cnum}")
        primary = canon_id(name)
        if primary not in ids:
            ids.append(primary)
        for ident in r["Identifiers"].split(","):
            ident = ident.strip()
            if not ident:
                continue
            if re.fullmatch(r"\s*[Cc]\s*0*(\d+)\s*", ident):
                continue  # already added as C<n>
            cid = canon_id(ident)
            if cid not in ids and len(ids) < 12:
                ids.append(cid)

        common = [c.strip() for c in r["Common names"].split(",") if c.strip()]
        kept.append({
            "ids": ids,
            "name": common[0] if common else primary,
            "ra": round(ra, 4),
            "dec": round(dec, 4),
            "type": TYPE_MAP[otype],
            "mag": round(mag, 2) if mag is not None else None,
            "size_arcmin": (lambda v: round(v, 2) if v is not None else None)(
                fnum(r["MajAx"])),
            "constellation": r["Const"].strip() or None,
        })

    kept.sort(key=lambda o: (o["mag"] is None, o["mag"] or 0.0, o["name"]))
    n_m = sum(1 for o in kept if any(i.startswith("M") and i[1:].isdigit() for i in o["ids"]))
    n_c = sum(1 for o in kept if any(re.fullmatch(r"C\d+", i) for i in o["ids"]))
    print(f"kept {len(kept)} objects "
          f"({n_m} Messier, {n_c} Caldwell; {skipped_no_coords} skipped, no coords)")
    if len(kept) < 4000:
        die(f"catalog too small ({len(kept)} < 4000); check the magnitude cut")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(kept, indent=1) + "\n", encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({size_kb:.0f} KiB)")

    attrib = OUT.parent / "ATTRIBUTION.txt"
    attrib.write_text(
        "Night-sky catalog data sources\n"
        "================================\n\n"
        "OpenNGC — open deep-sky object database\n"
        f"  {OPENNGC_BASE}/NGC.csv and addendum.csv (tag/branch: {OPENNGC_TAG})\n"
        "  © Mattia Verga and contributors — CC-BY-SA-4.0\n"
        "  https://github.com/mattiaverga/OpenNGC\n\n"
        "Caldwell designations cross-checked against the Caldwell catalogue\n"
        "table on Wikipedia (https://en.wikipedia.org/wiki/Caldwell_catalogue).\n\n"
        "This derived catalog.json is therefore distributed under CC-BY-SA-4.0.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
