"""Imaging plan definition, YAML loading, and validation.

A plan is a YAML document like::

    session:
      name: "m51-lrgb"
      output_dir: "sessions"
      target: {name: "M51", ra_hours: 13.4979, dec_deg: 47.1953}
      telescope: "Askar 103 APO"
      instrument: "Canon EOS Ra"

The target may also be given by name only — ``target: {name: "M51"}`` —
in which case RA/Dec are resolved from the vendored night-sky catalog at
plan load time and the matched designations are recorded on the target.

A multi-target night uses the optional top-level ``targets:`` list
instead; each entry resolves like ``session.target`` and carries its own
``platesolve`` block.  The sequencer slews to each target in turn and
runs the full step sequence on each.

Optional autonomous-imaging blocks (all off by default; a plan without
them behaves exactly as before)::

    targets:
      - name: "M51"
        platesolve: {enabled: true, tolerance_arcmin: 2.0,
                     max_iterations: 5, exposure_s: 5.0}
    autofocus:
      enabled: false
      every_minutes: 60
      hfr_degradation_trigger: 1.25
      focuser_device: "Focuser"   # "" -> manual-assist mode (stock 6SE)
      focus_exposure_s: 5.0
    guiding:
      enabled: false
      phd2_host: localhost
      phd2_port: 4400
      dither_pixels: 5.0
      settle_pixels: 1.5
      settle_time_s: 10
    safety:
      max_session_hours: null    # wall-clock limit; null = no limit
      max_solve_failures: 3
      guide_retry_budget: 3
    alerts:
      webhook_url: ""            # http(s) webhook for watchdog alerts

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


from astrocapture import catalog


@dataclass
class Target:
    name: str = "unknown"
    ra_hours: float = 0.0
    dec_deg: float = 0.0
    # Designations resolved from the night-sky catalog when the target was
    # given by name (e.g. ["M51", "NGC 5194"]); empty for raw ra/dec targets.
    resolved_ids: list[str] = field(default_factory=list)


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
class PlateSolveBlock:
    """Per-target closed-loop platesolve settings (sequencer.recenter)."""
    enabled: bool = False
    tolerance_arcmin: float = 2.0
    max_iterations: int = 5
    exposure_s: float = 5.0


@dataclass
class TargetEntry:
    """One target of a (possibly multi-target) plan.

    ``targets:`` is the multi-target spelling of ``session.target``; each
    entry resolves its name through the night-sky catalog exactly like
    the legacy single target and carries its own platesolve block.
    """
    target: Target
    platesolve: PlateSolveBlock = field(default_factory=PlateSolveBlock)


@dataclass
class AutofocusConfig:
    """Autofocus schedule (top-level ``autofocus:`` block).

    When ``enabled`` and a motorized focuser is reachable, the sequencer
    runs a V-curve autofocus before the first light frame and every
    ``every_minutes`` thereafter, plus when the median light-frame HFR
    degrades by ``hfr_degradation_trigger`` vs the post-focus baseline.
    With no focuser device configured the sequencer instead prints the
    manual Bahtinov-mask guide (``assist_mode``) once at session start —
    the honest path for a stock NexStar 6SE, which has no focus motor.
    """
    enabled: bool = False
    every_minutes: float = 60.0
    hfr_degradation_trigger: float = 1.25
    focuser_device: str = ""
    focus_exposure_s: float = 5.0
    n_positions: int = 7
    step: int = 150


@dataclass
class GuidingConfig:
    """PHD2 guiding (top-level ``guiding:`` block)."""
    enabled: bool = False
    phd2_host: str = "localhost"
    phd2_port: int = 4400
    dither_pixels: float = 5.0
    settle_pixels: float = 1.5
    settle_time_s: float = 10.0


@dataclass
class SafetyConfig:
    """Watchdog policy (top-level ``safety:`` block)."""
    max_session_hours: float | None = None
    max_solve_failures: int = 3
    guide_retry_budget: int = 3


@dataclass
class AlertsConfig:
    """Alerting (top-level ``alerts:`` block)."""
    webhook_url: str = ""


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
    # Multi-target list; entry 0 is always ``target`` (legacy single-target
    # plans get a one-entry list with platesolve disabled).
    targets: list[TargetEntry] = field(default_factory=list)
    autofocus: AutofocusConfig = field(default_factory=AutofocusConfig)
    guiding: GuidingConfig = field(default_factory=GuidingConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)

    def __post_init__(self) -> None:
        # Direct construction (tests, API use) skips load_plan: always keep
        # at least the legacy single target in the multi-target list.
        if not self.targets:
            self.targets = [TargetEntry(target=self.target)]


def load_plan(path: str | Path) -> Plan:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise PlanError(f"{path}: plan must be a YAML mapping")
    errors: list[str] = []

    def err(msg: str) -> None:
        errors.append(msg)

    sess = raw.get("session", {})
    target_raw = sess.get("target", {}) or {}
    target = _resolve_target(target_raw, "target", err)

    # Multi-target list (optional). Each entry resolves exactly like the
    # legacy single target above and carries its own platesolve block.
    target_entries: list[TargetEntry] = []
    targets_raw = raw.get("targets")
    if targets_raw is not None:
        if not isinstance(targets_raw, list) or not targets_raw:
            err("targets: must be a non-empty list of target entries")
        else:
            for i, t_raw in enumerate(targets_raw):
                entry = _target_entry(t_raw, i, err)
                if entry is not None:
                    target_entries.append(entry)
    if not target_entries:
        target_entries = [TargetEntry(target=target)]
    # plan.target stays the first target for backwards compatibility.
    target = target_entries[0].target

    mount = _driver_spec(raw.get("mount"), "mount", err)
    camera = _driver_spec(raw.get("camera"), "camera", err)
    autofocus = _autofocus_config(raw.get("autofocus"), err)
    guiding = _guiding_config(raw.get("guiding"), err)
    safety = _safety_config(raw.get("safety"), err)
    alerts = _alerts_config(raw.get("alerts"), err)

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
        targets=target_entries,
        autofocus=autofocus,
        guiding=guiding,
        safety=safety,
        alerts=alerts,
    )


def _resolve_target(target_raw: dict, where: str, err) -> Target:
    """Resolve one target mapping to a Target (name -> catalog, or raw)."""
    name = str(target_raw.get("name", "unknown"))
    resolved_ids: list[str] = []
    if "ra_hours" in target_raw or "dec_deg" in target_raw:
        # Raw coordinates: keep them untouched.  Still record the catalog
        # designations when the name resolves, for convenience.
        ra_hours = _num(target_raw.get("ra_hours", 0.0),
                        f"{where}.ra_hours", err)
        dec_deg = _num(target_raw.get("dec_deg", 0.0),
                       f"{where}.dec_deg", err)
        if name and name != "unknown":
            try:
                resolved_ids = list(catalog.lookup(name)["ids"])
            except catalog.UnknownObjectError:
                pass
    elif name and name != "unknown":
        # Name-only target: resolve RA/Dec from the night-sky catalog.
        try:
            obj = catalog.lookup(name)
        except catalog.UnknownObjectError:
            err(f"{where}.name {name!r} not found in the night-sky catalog "
                "(give ra_hours/dec_deg instead)")
            obj = None
        if obj is None:
            ra_hours, dec_deg = 0.0, 0.0
        else:
            resolved_ids = list(obj["ids"])
            ra_hours = obj["ra"] / 15.0
            dec_deg = obj["dec"]
    else:
        ra_hours = _num(target_raw.get("ra_hours", 0.0),
                        f"{where}.ra_hours", err)
        dec_deg = _num(target_raw.get("dec_deg", 0.0),
                       f"{where}.dec_deg", err)
    tgt = Target(name=name, ra_hours=ra_hours, dec_deg=dec_deg,
                 resolved_ids=resolved_ids)
    if not (0.0 <= tgt.ra_hours < 24.0):
        err(f"{where}.ra_hours {tgt.ra_hours} out of range [0, 24)")
    if not (-90.0 <= tgt.dec_deg <= 90.0):
        err(f"{where}.dec_deg {tgt.dec_deg} out of range [-90, 90]")
    return tgt


def _target_entry(raw, i: int, err) -> TargetEntry | None:
    where = f"targets[{i}]"
    if not isinstance(raw, dict):
        err(f"{where}: entry must be a mapping")
        return None
    target = _resolve_target(raw, where, err)
    ps = _platesolve_block(raw.get("platesolve"), f"{where}.platesolve", err)
    return TargetEntry(target=target, platesolve=ps)


def _int(value, name: str, err, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        err(f"{name} must be an integer, got {value!r}")
        return 0
    if minimum is not None and value < minimum:
        err(f"{name} must be >= {minimum}, got {value}")
    return value


def _platesolve_block(raw, where: str, err) -> PlateSolveBlock:
    cfg = PlateSolveBlock()
    if raw is None:
        return cfg
    if not isinstance(raw, dict):
        err(f"{where}: must be a mapping")
        return cfg
    cfg.enabled = bool(raw.get("enabled", False))
    cfg.tolerance_arcmin = _num(raw.get("tolerance_arcmin", 2.0),
                                f"{where}.tolerance_arcmin", err)
    if cfg.tolerance_arcmin <= 0:
        err(f"{where}.tolerance_arcmin must be > 0, "
            f"got {cfg.tolerance_arcmin}")
    cfg.max_iterations = _int(raw.get("max_iterations", 5),
                              f"{where}.max_iterations", err, minimum=1)
    cfg.exposure_s = _num(raw.get("exposure_s", 5.0),
                          f"{where}.exposure_s", err)
    if cfg.exposure_s <= 0:
        err(f"{where}.exposure_s must be > 0, got {cfg.exposure_s}")
    return cfg


def _autofocus_config(raw, err) -> AutofocusConfig:
    cfg = AutofocusConfig()
    if raw is None:
        return cfg
    if not isinstance(raw, dict):
        err("autofocus: must be a mapping")
        return cfg
    cfg.enabled = bool(raw.get("enabled", False))
    cfg.every_minutes = _num(raw.get("every_minutes", 60.0),
                             "autofocus.every_minutes", err)
    if cfg.every_minutes <= 0:
        err(f"autofocus.every_minutes must be > 0, got {cfg.every_minutes}")
    cfg.hfr_degradation_trigger = _num(
        raw.get("hfr_degradation_trigger", 1.25),
        "autofocus.hfr_degradation_trigger", err)
    if cfg.hfr_degradation_trigger <= 1.0:
        err("autofocus.hfr_degradation_trigger must be > 1.0 "
            f"(it is a degradation ratio), got {cfg.hfr_degradation_trigger}")
    cfg.focuser_device = str(raw.get("focuser_device", "") or "")
    cfg.focus_exposure_s = _num(raw.get("focus_exposure_s", 5.0),
                                "autofocus.focus_exposure_s", err)
    if cfg.focus_exposure_s <= 0:
        err(f"autofocus.focus_exposure_s must be > 0, "
            f"got {cfg.focus_exposure_s}")
    cfg.n_positions = _int(raw.get("n_positions", 7),
                           "autofocus.n_positions", err, minimum=3)
    cfg.step = _int(raw.get("step", 150), "autofocus.step", err, minimum=1)
    return cfg


def _guiding_config(raw, err) -> GuidingConfig:
    cfg = GuidingConfig()
    if raw is None:
        return cfg
    if not isinstance(raw, dict):
        err("guiding: must be a mapping")
        return cfg
    cfg.enabled = bool(raw.get("enabled", False))
    cfg.phd2_host = str(raw.get("phd2_host", "localhost") or "localhost")
    cfg.phd2_port = _int(raw.get("phd2_port", 4400), "guiding.phd2_port", err)
    if not (1 <= cfg.phd2_port <= 65535):
        err(f"guiding.phd2_port must be 1-65535, got {cfg.phd2_port}")
    cfg.dither_pixels = _num(raw.get("dither_pixels", 5.0),
                             "guiding.dither_pixels", err)
    if cfg.dither_pixels <= 0:
        err(f"guiding.dither_pixels must be > 0, got {cfg.dither_pixels}")
    cfg.settle_pixels = _num(raw.get("settle_pixels", 1.5),
                             "guiding.settle_pixels", err)
    if cfg.settle_pixels <= 0:
        err(f"guiding.settle_pixels must be > 0, got {cfg.settle_pixels}")
    cfg.settle_time_s = _num(raw.get("settle_time_s", 10.0),
                             "guiding.settle_time_s", err)
    if cfg.settle_time_s <= 0:
        err(f"guiding.settle_time_s must be > 0, got {cfg.settle_time_s}")
    return cfg


def _safety_config(raw, err) -> SafetyConfig:
    cfg = SafetyConfig()
    if raw is None:
        return cfg
    if not isinstance(raw, dict):
        err("safety: must be a mapping")
        return cfg
    hours = raw.get("max_session_hours")
    if hours is not None:
        hours = _num(hours, "safety.max_session_hours", err)
        if hours <= 0:
            err(f"safety.max_session_hours must be > 0, got {hours}")
        else:
            cfg.max_session_hours = hours
    cfg.max_solve_failures = _int(raw.get("max_solve_failures", 3),
                                  "safety.max_solve_failures", err, minimum=1)
    cfg.guide_retry_budget = _int(raw.get("guide_retry_budget", 3),
                                  "safety.guide_retry_budget", err, minimum=0)
    return cfg


def _alerts_config(raw, err) -> AlertsConfig:
    cfg = AlertsConfig()
    if raw is None:
        return cfg
    if not isinstance(raw, dict):
        err("alerts: must be a mapping")
        return cfg
    url = str(raw.get("webhook_url", "") or "")
    if url and not url.startswith(("http://", "https://")):
        err(f"alerts.webhook_url must be an http(s) URL, got {url!r}")
    else:
        cfg.webhook_url = url
    return cfg


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
        f"Mount   : {plan.mount.driver} {plan.mount.options}",
        f"Camera  : {plan.camera.driver} {plan.camera.options}",
        f"Targets : {len(plan.targets)}",
    ]
    for entry in plan.targets:
        tgt = entry.target
        tgt_ids = ", ".join(tgt.resolved_ids[:4])
        ps = (f" [platesolve: tol={entry.platesolve.tolerance_arcmin:g}', "
              f"iter={entry.platesolve.max_iterations}]"
              if entry.platesolve.enabled else "")
        lines.append(
            f"  - {tgt.name} (RA {tgt.ra_hours:.4f}h, "
            f"Dec {tgt.dec_deg:+.4f}°)"
            + (f" [{tgt_ids}]" if tgt_ids else "") + ps
        )
    lines.append("Steps   :")
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
    af = plan.autofocus
    lines.append("Autofocus: " + (
        f"enabled (every {af.every_minutes:g} min, "
        f"focuser={af.focuser_device or 'auto/sim'}, "
        f"HFR trigger x{af.hfr_degradation_trigger:g})"
        if af.enabled else "disabled"))
    gd = plan.guiding
    lines.append("Guiding : " + (
        f"enabled (PHD2 {gd.phd2_host}:{gd.phd2_port}, "
        f"dither {gd.dither_pixels:g} px)"
        if gd.enabled else "disabled"))
    sf = plan.safety
    lim = (f", session limit {sf.max_session_hours:g}h"
           if sf.max_session_hours else "")
    lines.append(f"Safety  : solve-failures→park at {sf.max_solve_failures}, "
                 f"guide retries {sf.guide_retry_budget}{lim}")
    if plan.alerts.webhook_url:
        lines.append(f"Alerts  : webhook {plan.alerts.webhook_url}")
    return "\n".join(lines)
