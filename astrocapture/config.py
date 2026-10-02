"""Imaging plan definition, YAML loading, and validation.

A plan is a YAML document like::

    session:
      name: "m51-lrgb"
      output_dir: "sessions"
      target: {name: "M51", ra_hours: 13.4979, dec_deg: 47.1953}
      telescope: "Askar 103 APO"
      instrument: "Canon EOS Ra"
    mount:
      driver: sim
    camera:
      driver: sim
    sequence:
      - {type: light, filter: L, exposure: 120, count: 12, gain: 1600,
         dither_every: 3}
      - {type: dark, exposure: 120, count: 10, gain: 1600}
      - {type: flat, filter: L, exposure: 2, count: 20, gain: 1600}
      - {type: bias, count: 30, gain: 1600}

``load_plan(path)`` returns a validated ``Plan`` (dataclasses) or raises
``PlanError`` describing every problem found — fail fast, before any
hardware moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

FRAME_TYPES = ("light", "dark", "flat", "bias")


class PlanError(ValueError):
    """Raised when a plan file fails validation, with all problems listed."""


@dataclass
class Target:
    name: str = "unknown"
    ra_hours: float = 0.0
    dec_deg: float = 0.0


@dataclass
class Step:
    type: str
    exposure: float = 1.0
    count: int = 1
    gain: float = 0.0
    binning: int = 1
    filter: str = ""
    dither_every: int = 0


@dataclass
class DriverSpec:
    driver: str
    options: dict = field(default_factory=dict)


@dataclass
class Plan:
    session_name: str
    output_dir: str
    target: Target
    telescope: str
    instrument: str
    mount: DriverSpec
    camera: DriverSpec
    steps: list[Step]
    cooler_temp_c: float | None = None
    meridian_flip: bool = True
    plate_solve: bool = False


def load_plan(path: str | Path) -> Plan:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise PlanError(f"{path}: plan must be a YAML mapping")
    errors: list[str] = []

    def err(msg: str) -> None:
        errors.append(msg)

    sess = raw.get("session", {})
    target_raw = sess.get("target", {}) or {}
    target = Target(
        name=str(target_raw.get("name", "unknown")),
        ra_hours=_num(target_raw.get("ra_hours", 0.0), "ra_hours", err),
        dec_deg=_num(target_raw.get("dec_deg", 0.0), "dec_deg", err),
    )
    if not (0.0 <= target.ra_hours < 24.0):
        err(f"target.ra_hours {target.ra_hours} out of range [0, 24)")
    if not (-90.0 <= target.dec_deg <= 90.0):
        err(f"target.dec_deg {target.dec_deg} out of range [-90, 90]")

    mount = _driver_spec(raw.get("mount"), "mount", err)
    camera = _driver_spec(raw.get("camera"), "camera", err)

    steps: list[Step] = []
    seq = raw.get("sequence")
    if not isinstance(seq, list) or not seq:
        err("sequence must be a non-empty list of steps")
    else:
        for i, s in enumerate(seq):
            steps.append(_step(s, i, err))

    if errors:
        raise PlanError(f"{path}: invalid plan:\n  - " + "\n  - ".join(errors))

    return Plan(
        session_name=str(sess.get("name", "session")),
        output_dir=str(sess.get("output_dir", "sessions")),
        target=target,
        telescope=str(sess.get("telescope", "")),
        instrument=str(sess.get("instrument", "")),
        mount=mount,
        camera=camera,
        steps=steps,
        cooler_temp_c=sess.get("cooler_temp_c"),
        meridian_flip=bool(sess.get("meridian_flip", True)),
        plate_solve=bool(sess.get("plate_solve", False)),
    )


def _num(value, name: str, err) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        err(f"{name} must be a number, got {value!r}")
        return 0.0


def _driver_spec(raw, where: str, err) -> DriverSpec:
    if not isinstance(raw, dict) or "driver" not in raw:
        err(f"{where}: needs a mapping with a 'driver' key")
        return DriverSpec(driver="sim")
    driver = str(raw["driver"])
    options = {k: v for k, v in raw.items() if k != "driver"}
    return DriverSpec(driver=driver, options=options)


def _step(raw, i: int, err) -> Step:
    if not isinstance(raw, dict):
        err(f"sequence[{i}]: step must be a mapping")
        return Step(type="light")
    ftype = str(raw.get("type", ""))
    if ftype not in FRAME_TYPES:
        err(f"sequence[{i}]: type must be one of {FRAME_TYPES}, got {ftype!r}")
    exposure = _num(raw.get("exposure", 1.0), f"sequence[{i}].exposure", err)
    if exposure <= 0:
        err(f"sequence[{i}]: exposure must be > 0")
    count = raw.get("count", 1)
    if not isinstance(count, int) or count < 1:
        err(f"sequence[{i}]: count must be a positive integer")
        count = 1
    dither_every = raw.get("dither_every", 0)
    if not isinstance(dither_every, int) or dither_every < 0:
        err(f"sequence[{i}]: dither_every must be a non-negative integer")
        dither_every = 0
    binning = raw.get("binning", 1)
    if binning not in (1, 2, 3, 4):
        err(f"sequence[{i}]: binning must be 1-4")
        binning = 1
    return Step(
        type=ftype,
        exposure=exposure,
        count=count,
        gain=_num(raw.get("gain", 0.0), f"sequence[{i}].gain", err),
        binning=binning,
        filter=str(raw.get("filter", "")),
        dither_every=dither_every,
    )


def plan_summary(plan: Plan) -> str:
    lines = [
        f"Session : {plan.session_name}",
        f"Target  : {plan.target.name} "
        f"(RA {plan.target.ra_hours:.4f}h, Dec {plan.target.dec_deg:+.4f}°)",
        f"Mount   : {plan.mount.driver} {plan.mount.options}",
        f"Camera  : {plan.camera.driver} {plan.camera.options}",
        "Steps   :",
    ]
    total = 0
    for s in plan.steps:
        frames = s.count * (s.exposure if s.type != "bias" else 0)
        total += frames
        lines.append(
            f"  - {s.count:3d}x {s.type:<5} {s.exposure:>7.1f}s"
            + (f" filter={s.filter}" if s.filter else "")
            + (f" gain={s.gain:g}" if s.gain else "")
            + (f" dither/ {s.dither_every}" if s.dither_every else "")
        )
    h, rem = divmod(total, 3600)
    lines.append(f"Est. integration: {int(h)}h {rem/60:.0f}m (+ slews/overhead)")
    return "\n".join(lines)
