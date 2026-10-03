"""Natural-language planning assistant: English in, validated plan YAML out.

Two paths, chosen automatically:

- **LLM path (primary)** — when an ``llm`` callable is given, or when the
  ``openai`` package is importable and ``OPENAI_API_KEY`` is set.  The LLM
  receives a system prompt encoding Mathias's gear constraints and must
  return STRICT JSON matching :data:`PLAN_SCHEMA_DOC`.  The JSON is
  schema-checked *and* run through :func:`astrocapture.config.load_plan`
  (which resolves the target name through the night-sky catalog) before
  anything is accepted — raw LLM output is never trusted.

- **Rule-based path (fallback)** — pure offline parsing: finds a catalog
  designation or common name in the text, a total integration time
  ("2 hours", "90 minutes"), and an optional sub-exposure length.  Used
  automatically when no API key is available, or forced with
  ``use_llm=False``.

``plan_from_text`` returns a validated :class:`~astrocapture.config.Plan`;
``plan_dict_from_text`` returns the raw (validated-shape) dict so callers
can dump it to YAML with :func:`write_plan_yaml`.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import yaml

from astrocapture import catalog, config

# ---------------------------------------------------------------------------
# Gear constraints baked into every plan the assistant produces.
# ---------------------------------------------------------------------------

MAX_SUB_S = 25.0        # NexStar 6SE is alt-az: field rotation caps subs.
DEFAULT_SUB_S = 25.0
DEFAULT_ISO = 1600      # Rebel T7i sweet spot used across the examples.
DEFAULT_DITHER_EVERY = 3

SYSTEM_PROMPT = """\
You are the planning assistant for AstroCapture, an astrophotography
capture program. Convert the user's natural-language request into a
STRICT JSON imaging plan. Output ONLY the JSON object, no markdown
fences, no commentary.

HARD GEAR CONSTRAINTS (the user owns this exact rig):
- Mount: Celestron NexStar 6SE — ALT-AZ fork mount. NEVER set a light
  sub-exposure longer than 25 seconds (field rotation). If the user asks
  for longer subs, clamp to 25 and note it in "notes".
- Camera: Canon EOS Rebel T7i (APS-C DSLR). Use gain/ISO 1600 unless the
  user specifies otherwise.
- Dither every 3 light frames (dither_every: 3) to kill fixed-pattern noise.
- Every plan needs calibration frames: darks matching the light exposure
  (at least 10, ~15 for long sessions), ~20 flats, ~30 biases.
- Target: give ONLY the catalog name, e.g. {"name": "M51"}. Do NOT invent
  RA/Dec — the program resolves names through its night-sky catalog.
  Prefer Messier/Caldwell/NGC/IC designations the user names; if they
  name something obscure, use your best guess at the canonical
  designation.
- "2 hours of data" at 25s subs means 288 light frames
  (7200 / 25). Do this arithmetic yourself.

DRIVERS (copy verbatim unless the user says otherwise):
- mount: {"driver": "indi", "host": "localhost", "port": 7624,
  "device": "Celestron GPS", "serial_port": "/dev/ttyUSB0"}
- camera: {"driver": "indi", "host": "localhost", "port": 7624,
  "device": "Canon DSLR", "upload_mode": "client"}

REQUIRED JSON SHAPE:
{
  "session": {"name": "<slug>-ai", "output_dir": "sessions",
              "telescope": "Celestron NexStar 6SE",
              "instrument": "Canon EOS Rebel T7i",
              "target": {"name": "<CATALOG NAME>"}},
  "mount": {...}, "camera": {...},
  "sequence": [
    {"type": "light", "exposure": 25, "count": <int>, "gain": 1600,
     "dither_every": 3},
    {"type": "dark", "exposure": 25, "count": <int>, "gain": 1600},
    {"type": "flat", "exposure": 2, "count": 20, "gain": 1600},
    {"type": "bias", "exposure": 1, "count": 30, "gain": 1600}
  ],
  "notes": "<one line on assumptions/clamps, or empty>"
}

Rules: "count" values are integers. "exposure" is seconds (float ok).
Flat exposure 2s is a placeholder (sky flats metered at dusk). Bias
exposure is ignored by the program but must be > 0. If the user pastes
`astrocapture tonight` output, prefer its highly-ranked targets.
"""

# Minimal shape check applied to LLM JSON before config.load_plan runs.
_REQUIRED_TOP = ("session", "mount", "camera", "sequence")


# ---------------------------------------------------------------------------
# LLM plumbing (guarded: the openai package is optional)
# ---------------------------------------------------------------------------


def _openai_llm():
    """Build the default LLM callable from the environment, or None.

    Returns ``None`` when the ``openai`` package is missing or
    ``OPENAI_API_KEY`` is unset — the caller then uses the rule-based
    fallback.  The returned callable has signature
    ``(system: str, user: str) -> str``.
    """
    try:
        import openai  # type: ignore[import]
    except ImportError:
        return None
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    client = openai.OpenAI(api_key=api_key)

    def call(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        return resp.choices[0].message.content or ""

    return call


def _strip_fences(text: str) -> str:
    """Remove markdown code fences LLMs love to add despite instructions."""
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", text, re.DOTALL)
    return m.group(1) if m else text


def _check_llm_shape(obj: object) -> dict:
    """Minimal structural check on parsed LLM JSON. Raises ValueError."""
    if not isinstance(obj, dict):
        raise ValueError("LLM did not return a JSON object")
    missing = [k for k in _REQUIRED_TOP if k not in obj]
    if missing:
        raise ValueError(f"LLM plan missing required keys: {missing}")
    if not isinstance(obj["session"], dict) or "target" not in obj["session"]:
        raise ValueError("LLM plan 'session' must be a mapping with a "
                         "'target' (the target nests under 'session')")
    seq = obj["sequence"]
    if not isinstance(seq, list) or not seq:
        raise ValueError("LLM plan 'sequence' must be a non-empty list")
    for i, step in enumerate(seq):
        if not isinstance(step, dict) or "type" not in step:
            raise ValueError(f"LLM plan sequence[{i}] is not a valid step")
        if step["type"] == "light" and float(step.get("exposure", 0)) > MAX_SUB_S:
            raise ValueError(
                f"LLM plan sequence[{i}]: light exposure "
                f"{step['exposure']}s exceeds the 25s alt-az cap — "
                "the model ignored the gear constraints"
            )
    return obj


def _llm_plan_dict(text: str, llm) -> dict:
    raw = _strip_fences(llm(SYSTEM_PROMPT, text))
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM did not return valid JSON ({exc}); first 200 chars: "
            f"{raw[:200]!r}"
        ) from exc
    return _check_llm_shape(obj)


# ---------------------------------------------------------------------------
# Rule-based fallback (offline)
# ---------------------------------------------------------------------------


_DESIGNATION_RES = [
    re.compile(r"\bM\s?(\d{1,3})\b", re.IGNORECASE),            # M51 / M 51
    re.compile(r"\bNGC\s?(\d{1,4}[A-Z]?)\b", re.IGNORECASE),    # NGC 7000
    re.compile(r"\bIC\s?(\d{1,4}[A-Z]?)\b", re.IGNORECASE),     # IC 1805
    re.compile(r"\bCaldwell\s?(\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\bMessier\s?(\d{1,3})\b", re.IGNORECASE),
]

_TIME_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m)\b", re.IGNORECASE
)
_SUB_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:s|sec(?:onds?)?)\b\s*(?:subs?|exposures?)?",
    re.IGNORECASE,
)


def _find_target(text: str) -> str:
    """Best catalog designation found in ``text``. Raises PlanError."""
    # 1. Quoted phrases first: image "Whirlpool Galaxy" tonight.
    for quoted in re.findall(r'"([^"]+)"', text):
        try:
            return catalog.lookup(quoted)["ids"][0]
        except catalog.UnknownObjectError:
            continue
    # 2. Designation patterns.
    for rx in _DESIGNATION_RES:
        m = rx.search(text)
        if m:
            candidate = re.sub(r"\s+", "", m.group(0))
            try:
                return catalog.lookup(candidate)["ids"][0]
            except catalog.UnknownObjectError:
                continue
    # 3. Common names ("Whirlpool Galaxy", "Ring Nebula", ...): substring
    #    match against catalog primary names, longest match wins.
    lowered = text.lower()
    best: tuple[int, str] | None = None
    for obj in catalog.load_catalog():
        name = str(obj.get("name") or "")
        if len(name) < 4:
            continue
        idx = lowered.find(name.lower())
        if idx != -1 and (best is None or len(name) > best[0]):
            best = (len(name), obj["ids"][0])
    if best is not None:
        return best[1]
    raise config.PlanError(
        f"could not find a catalog object in {text!r} — name a Messier, "
        "Caldwell, NGC or IC object (e.g. 'M51', 'NGC 7000')"
    )


def _total_seconds(text: str) -> float:
    """Total integration time requested, default 1 hour."""
    m = _TIME_RE.search(text)
    if not m:
        return 3600.0
    value = float(m.group(1))
    unit = m.group(2).lower()
    if unit.startswith("h"):
        return value * 3600.0
    return value * 60.0


def _sub_exposure(text: str) -> tuple[float, str]:
    """Requested sub length, clamped to the alt-az cap. Returns (s, note)."""
    m = _SUB_RE.search(text)
    note = ""
    if not m:
        return DEFAULT_SUB_S, note
    want = float(m.group(1))
    if want > MAX_SUB_S:
        note = (f"clamped sub-exposure {want:g}s to {MAX_SUB_S:g}s "
                "(alt-az field rotation)")
        return MAX_SUB_S, note
    return max(1.0, want), note


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "target"


def _rule_based_plan_dict(text: str) -> dict:
    target_id = _find_target(text)
    total_s = _total_seconds(text)
    sub_s, note = _sub_exposure(text)
    n_lights = max(1, int(round(total_s / sub_s)))
    n_darks = min(15, max(5, n_lights // 20))
    notes = "; ".join(
        n for n in [
            note,
            f"rule-based parse of {text!r} (no LLM used)",
        ] if n
    )
    return {
        "session": {
            "name": f"{_slug(target_id)}-ai",
            "output_dir": "sessions",
            "telescope": "Celestron NexStar 6SE",
            "instrument": "Canon EOS Rebel T7i",
            "target": {"name": target_id},
        },
        "mount": {
            "driver": "indi",
            "host": "localhost",
            "port": 7624,
            "device": "Celestron GPS",
            "serial_port": "/dev/ttyUSB0",
        },
        "camera": {
            "driver": "indi",
            "host": "localhost",
            "port": 7624,
            "device": "Canon DSLR",
            "upload_mode": "client",
        },
        "sequence": [
            {"type": "light", "exposure": sub_s, "count": n_lights,
             "gain": DEFAULT_ISO, "dither_every": DEFAULT_DITHER_EVERY},
            {"type": "dark", "exposure": sub_s, "count": n_darks,
             "gain": DEFAULT_ISO},
            {"type": "flat", "exposure": 2, "count": 20, "gain": DEFAULT_ISO},
            {"type": "bias", "exposure": 1, "count": 30, "gain": DEFAULT_ISO},
        ],
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_plan_dict(plan_dict: dict) -> config.Plan:
    """Run a plan dict through the real validator via a temp YAML file."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        yaml.safe_dump(plan_dict, f)
        tmp = f.name
    try:
        return config.load_plan(tmp)
    finally:
        Path(tmp).unlink(missing_ok=True)


def plan_dict_from_text(text: str, *, llm=None, use_llm: bool = True) -> dict:
    """English request -> plan dict (shape-checked, not yet validated)."""
    if use_llm:
        llm = _openai_llm() if llm is None else llm
    if use_llm and llm is not None:
        try:
            return _llm_plan_dict(text, llm)
        except (ValueError, RuntimeError) as exc:
            raise config.PlanError(
                f"LLM planning failed: {exc} "
                "(retry, or pass use_llm=False for the rule-based parser)"
            ) from exc
    return _rule_based_plan_dict(text)


def plan_from_text(
    text: str, *, llm=None, use_llm: bool = True
) -> config.Plan:
    """English request -> fully validated :class:`config.Plan`.

    ``llm`` is an optional callable ``(system, user) -> str`` (injected in
    tests); when omitted and ``use_llm`` is true, the OpenAI API is used
    if ``OPENAI_API_KEY`` is set, otherwise the rule-based parser runs.
    """
    return validate_plan_dict(plan_dict_from_text(text, llm=llm, use_llm=use_llm))


def write_plan_yaml(plan_dict: dict, path: str | Path) -> Path:
    """Write a plan dict to ``path`` as YAML. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # "notes" is assistant metadata, not part of the plan schema — keep it
    # as a YAML comment-adjacent key? No: load_plan ignores unknown
    # top-level keys? It does NOT error on them (it reads known keys).
    # Keep "notes" in the file; the validator tolerates it.
    path.write_text(yaml.safe_dump(plan_dict, sort_keys=False),
                    encoding="utf-8")
    return path


def describe_source(text: str, *, llm=None, use_llm: bool = True) -> str:
    """Human-readable note on which planning path will be used."""
    if use_llm:
        active = _openai_llm() if llm is None else llm
        if active is not None:
            return "LLM planning (OpenAI API)"
    return "rule-based planning (offline fallback)"
