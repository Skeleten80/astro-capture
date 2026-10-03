"""Calibration and stacking: masters, calibrated lights, registered stacks.

Pipeline::

    bias frames  -> make_master_bias  -> master bias
    dark frames  -> make_master_dark   -> master dark (bias-subtracted,
                     scaled to the light exposure when it differs)
    flat frames  -> make_master_flat  -> master flat (calibrated,
                     normalized to median 1)
    light frames -> calibrate_light    -> registered, sigma-clipped
                     median combine   -> stacked.fits / stacked.png

HONEST NOTE on registration: :func:`stack_lights` aligns frames by
*translation only* (see :func:`astrocapture.imaging.estimate_shift`).
Field rotation between subs is NOT corrected — fine for short alt-az
subs where the field barely rotates; full rotation alignment (e.g. via
a 2-D FFT / star-triangle matching) is a follow-up.

Only numpy/astropy/Pillow are used (via :mod:`astrocapture.imaging`).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from astropy.io import fits

from astrocapture.imaging import (
    _vote_shift,
    detect_star_centroids,
    estimate_shift,
    shift_image,
    stretch_png,
)

log = logging.getLogger(__name__)

_FLAT_EPS = 1e-6  # divide-by-zero guard for master flats


# ---------------------------------------------------------------------------
# FITS loading
# ---------------------------------------------------------------------------


def _load_fits(path: str | Path) -> tuple[np.ndarray, fits.Header]:
    """Load a FITS file as float64 ``(data, header)``."""
    with fits.open(str(path)) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float64), hdul[0].header


def _exposure_of(header: fits.Header) -> float:
    try:
        return float(header.get("EXPTIME", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Combining
# ---------------------------------------------------------------------------


def median_combine(
    paths: list[str | Path],
    sigma_clip: bool = True,
    clip_sigma: float = 3.0,
) -> tuple[np.ndarray, dict]:
    """Sigma-clipped median stack of the FITS files in ``paths``.

    One clip pass: pixels deviating more than ``clip_sigma`` robust
    standard deviations from the per-pixel median are masked, then the
    median is recomputed over the survivors.  "Robust sigma" here is the
    median absolute deviation scaled to sigma (1.4826 × MAD), so a single
    hot frame can't inflate the very threshold meant to reject it.

    Returns ``(data_float32, info_dict)`` where the info dict records
    ``NCOMBINE``, whether clipping ran, ``clip_sigma``, and the fraction
    of masked pixels.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("median_combine needs at least one frame")
    stack = np.stack([_load_fits(p)[0] for p in paths], axis=0)
    shape = stack.shape[1:]
    if stack.shape[1:] != shape or stack.ndim != 3:
        raise ValueError("median_combine: all frames must share one shape")
    med = np.median(stack, axis=0)
    info: dict = {
        "NCOMBINE": len(paths),
        "SIGCLIP": bool(sigma_clip),
        "CLIPSIG": float(clip_sigma),
        "CLIPFRAC": 0.0,
    }
    if sigma_clip and len(paths) >= 3:
        mad = np.median(np.abs(stack - med), axis=0)
        robust_sigma = 1.4826 * mad
        # Never clip tighter than a small floor: pure-read-noise frames
        # have MAD ≈ 0 and would otherwise mask everything.
        floor = 1e-3 * np.maximum(np.abs(med), 1.0)
        thresh = np.maximum(clip_sigma * robust_sigma, floor)
        mask = np.abs(stack - med) > thresh
        info["CLIPFRAC"] = float(mask.mean())
        masked = np.ma.array(stack, mask=mask)
        med = np.ma.median(masked, axis=0).filled(np.nan)
        # A pixel masked in *every* frame has no survivor: fall back to
        # the plain median there rather than NaN.
        bad = ~np.isfinite(med)
        if bad.any():
            med[bad] = np.median(stack, axis=0)[bad]
    return med.astype(np.float32), info


# ---------------------------------------------------------------------------
# Master frames
# ---------------------------------------------------------------------------


def make_master_bias(paths: list[str | Path]) -> tuple[np.ndarray, dict]:
    """Median-combined master bias (sigma-clipped)."""
    data, info = median_combine(paths, sigma_clip=True)
    info["IMAGETYP"] = "MASTER BIAS"
    log.info("master bias: %d frames", info["NCOMBINE"])
    return data, info


def make_master_dark(
    paths: list[str | Path],
    bias: np.ndarray | None = None,
    dark_exposure: float | None = None,
    light_exposure: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Master dark: combined, bias-subtracted, scaled to ``light_exposure``.

    When the darks' exposure differs from the lights', the master is
    scaled by ``light_exposure / dark_exposure`` and a note is printed:
    **dark scaling is approximate** — dark current is only roughly linear
    with exposure time (and temperature), so matching darks are always
    better than scaled ones.
    """
    data, info = median_combine(paths, sigma_clip=True)
    if bias is not None:
        data = data - np.asarray(bias, dtype=np.float64)
    headers = [_load_fits(p)[1] for p in paths]
    if dark_exposure is None:
        exps = [_exposure_of(h) for h in headers]
        dark_exposure = float(np.median(exps)) if exps else 0.0
    scale = 1.0
    if light_exposure and dark_exposure and dark_exposure > 0:
        scale = light_exposure / dark_exposure
    if abs(scale - 1.0) > 1e-9:
        note = (
            f"NOTE: scaling master dark by {scale:.3f} "
            f"(dark {dark_exposure:g}s -> light {light_exposure:g}s). "
            "Dark scaling is approximate: dark current is only roughly "
            "linear with exposure time (and temperature). Matching darks "
            "are always better than scaled ones."
        )
        print(note)
        log.info(note)
    data = (data * scale).astype(np.float32)
    info["IMAGETYP"] = "MASTER DARK"
    info["DARKEXP"] = float(dark_exposure or 0.0)
    info["DARKSCAL"] = float(scale)
    log.info("master dark: %d frames, scale %.3f", info["NCOMBINE"], scale)
    return data, info


def make_master_flat(
    paths: list[str | Path],
    bias: np.ndarray | None = None,
    dark: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Master flat: calibrated, combined, normalized to median 1.0."""
    frames = []
    for p in paths:
        data, _hdr = _load_fits(p)
        if bias is not None:
            data = data - np.asarray(bias, dtype=np.float64)
        if dark is not None:
            data = data - np.asarray(dark, dtype=np.float64)
        frames.append(data)
    stack = np.stack(frames, axis=0)
    med = np.median(stack, axis=0)
    norm = float(np.median(med))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("make_master_flat: combined flat has no positive median")
    master = (med / norm).astype(np.float32)
    info = {
        "NCOMBINE": len(paths),
        "IMAGETYP": "MASTER FLAT",
        "FLATNORM": norm,
    }
    log.info("master flat: %d frames, normalized by median %.1f ADU",
             len(paths), norm)
    return master, info


# ---------------------------------------------------------------------------
# Light calibration
# ---------------------------------------------------------------------------


def calibrate_light(
    data: np.ndarray,
    master_bias: np.ndarray | None = None,
    master_dark: np.ndarray | None = None,
    master_flat: np.ndarray | None = None,
) -> np.ndarray:
    """Calibrate one light frame: ``(data - bias - dark) / flat``.

    Any master may be ``None`` (skipped).  The master flat is clipped at
    a small epsilon so a dead (zero) flat pixel can never produce inf —
    at worst it leaves that pixel slightly wrong, never NaN.
    """
    out = np.asarray(data, dtype=np.float64)
    if master_bias is not None:
        out = out - np.asarray(master_bias, dtype=np.float64)
    if master_dark is not None:
        out = out - np.asarray(master_dark, dtype=np.float64)
    if master_flat is not None:
        flat = np.asarray(master_flat, dtype=np.float64)
        out = out / np.clip(flat, _FLAT_EPS, None)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------


def stack_lights(
    light_paths: list[str | Path],
    bias: np.ndarray | None = None,
    dark: np.ndarray | None = None,
    flat: np.ndarray | None = None,
    register: bool = True,
    sigma_clip: bool = True,
    max_shift_px: float = 50.0,
) -> tuple[np.ndarray, dict]:
    """Calibrate, register and stack light frames.

    The first frame is the registration reference; every later frame is
    translation-registered to it (frames that fail registration are
    dropped and counted in ``stats["n_rejected"]``).  The surviving
    calibrated frames are sigma-clipped median combined.

    Returns ``(stack_float32, stats)`` with stats ``n_input``,
    ``n_stacked``, ``n_rejected``, ``shifts`` (per-input ``(dx, dy)`` or
    ``None`` for dropped frames) and ``clip_info`` from
    :func:`median_combine`.
    """
    light_paths = [Path(p) for p in light_paths]
    if not light_paths:
        raise ValueError("stack_lights needs at least one light frame")

    calibrated: list[np.ndarray] = []
    shifts: list[tuple[float, float] | None] = []
    ref_centroids: list[tuple[float, float]] | None = None
    n_rejected = 0
    can_register = register

    for i, p in enumerate(light_paths):
        data, _hdr = _load_fits(p)
        cal = calibrate_light(data, bias, dark, flat)
        if i == 0 and can_register:
            ref_centroids = detect_star_centroids(cal)
            if not ref_centroids:
                # No stars in the reference (short/clouded exposure):
                # registration is impossible, stack unregistered.
                log.warning("stack: no stars in reference frame; "
                            "stacking without registration")
                can_register = False
        if can_register and i > 0:
            cents = detect_star_centroids(cal)
            ref_a = (np.asarray(ref_centroids, dtype=float).reshape(-1, 2)
                     if ref_centroids else np.zeros((0, 2)))
            img_a = np.asarray(cents, dtype=float).reshape(-1, 2)
            cand, n_in = _vote_shift(ref_a, img_a, max_shift_px, 2.0)
            if cand is None or n_in < 3:
                # No plausible translation maps this frame onto the
                # reference: drop it rather than stack a wild frame.
                log.warning("stack: %s failed registration, dropped", p.name)
                shifts.append(None)
                n_rejected += 1
                continue
            dx, dy = estimate_shift(ref_a, img_a, max_shift_px=max_shift_px)
            cal = shift_image(cal, dx, dy).astype(np.float32)
            shifts.append((dx, dy))
        else:
            shifts.append((0.0, 0.0))
        calibrated.append(cal)

    if not calibrated:
        raise ValueError("stack_lights: every frame failed registration")

    stack_arr = np.stack(calibrated, axis=0)
    med = np.median(stack_arr, axis=0)
    clip_info: dict = {"SIGCLIP": False}
    if sigma_clip and len(calibrated) >= 3:
        mad = np.median(np.abs(stack_arr - med), axis=0)
        robust_sigma = 1.4826 * mad
        floor = 1e-3 * np.maximum(np.abs(med), 1.0)
        thresh = np.maximum(3.0 * robust_sigma, floor)
        mask = np.abs(stack_arr - med) > thresh
        clip_info = {
            "SIGCLIP": True,
            "CLIPSIG": 3.0,
            "CLIPFRAC": float(mask.mean()),
        }
        masked = np.ma.array(stack_arr, mask=mask)
        med = np.ma.median(masked, axis=0).filled(np.nan)
        bad = ~np.isfinite(med)
        if bad.any():
            med[bad] = np.median(stack_arr, axis=0)[bad]

    stats = {
        "n_input": len(light_paths),
        "n_stacked": len(calibrated),
        "n_rejected": n_rejected,
        "registered": bool(can_register),
        "shifts": shifts,
        "clip_info": clip_info,
    }
    return med.astype(np.float32), stats


# ---------------------------------------------------------------------------
# Whole-session processing
# ---------------------------------------------------------------------------


def _fits_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.fits") if p.is_file())


def process_session(
    session_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    """Calibrate + stack every light frame in a session directory.

    Reads ``<session_dir>/{lights,darks,flats,bias}/*.fits``, builds
    master frames, stacks the lights, and writes into ``output_dir``:

    - ``stacked.fits`` — 32-bit float stack, header noting which masters
      were used (or that none were available),
    - ``stacked.png`` — auto-stretched preview (:func:`stretch_png`),
    - ``process.log`` — masters built, per-frame shifts, rejections.

    Missing calibration frames never crash the run: the stack proceeds
    on raw (or partially calibrated) lights and the warning is recorded
    in ``process.log``.  Returns the stats dict.
    """
    session_dir = Path(session_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def note(msg: str) -> None:
        log_lines.append(msg)
        log.info("process: %s", msg)

    note(f"session: {session_dir}")
    lights = _fits_paths(session_dir / "lights")
    darks = _fits_paths(session_dir / "darks")
    flats = _fits_paths(session_dir / "flats")
    biases = _fits_paths(session_dir / "bias")
    note(
        f"found {len(lights)} lights, {len(darks)} darks, "
        f"{len(flats)} flats, {len(biases)} biases"
    )

    stats: dict = {
        "session_dir": str(session_dir),
        "output_dir": str(output_dir),
        "n_lights": len(lights),
        "warnings": [],
    }
    if not lights:
        msg = "no light frames found — nothing to stack"
        stats["warnings"].append(msg)
        note("WARNING: " + msg)
        _write_log(output_dir, log_lines)
        return stats

    # -- masters ---------------------------------------------------------
    master_bias = None
    if biases:
        master_bias, binfo = make_master_bias(biases)
        stats["master_bias"] = binfo
        note(f"master bias from {binfo['NCOMBINE']} frames")
    else:
        stats["warnings"].append("no bias frames; lights not bias-subtracted")
        note("WARNING: no bias frames; lights not bias-subtracted")

    light_exp = float(np.median([_exposure_of(_load_fits(p)[1]) for p in lights]))
    master_dark = None
    master_dark_for_flats = None
    if darks:
        master_dark, dinfo = make_master_dark(
            darks, bias=master_bias, light_exposure=light_exp or None
        )
        stats["master_dark"] = dinfo
        note(
            f"master dark from {dinfo['NCOMBINE']} frames "
            f"(scale {dinfo['DARKSCAL']:.3f} to light exposure)"
        )
        # Flats get their own exposure-matched dark: subtracting a dark
        # scaled to the *light* exposure from a (usually much shorter)
        # flat would over-subtract.
        master_dark_for_flats = master_dark
        if flats:
            flat_exps = [_exposure_of(_load_fits(p)[1]) for p in flats]
            flat_exp = float(np.median(flat_exps)) if flat_exps else 0.0
            if flat_exp > 0 and abs(flat_exp - dinfo["DARKEXP"]) > 1e-9:
                master_dark_for_flats, finfo_d = make_master_dark(
                    darks, bias=master_bias, light_exposure=flat_exp
                )
                note(
                    f"flat-matched master dark scaled {finfo_d['DARKSCAL']:.3f} "
                    f"to flat exposure"
                )
    else:
        stats["warnings"].append("no dark frames; lights not dark-subtracted")
        note("WARNING: no dark frames; lights not dark-subtracted")

    master_flat = None
    if flats:
        try:
            master_flat, finfo = make_master_flat(
                flats, bias=master_bias, dark=master_dark_for_flats
            )
        except ValueError as exc:
            # Unusable flats (e.g. the simulator's "flats" are just short
            # starfield exposures with no even illumination): treat like
            # missing flats rather than crashing the whole run.
            msg = f"flat frames unusable ({exc}); lights not flat-fielded"
            stats["warnings"].append(msg)
            note("WARNING: " + msg)
        else:
            stats["master_flat"] = finfo
            note(f"master flat from {finfo['NCOMBINE']} frames")
    else:
        stats["warnings"].append("no flat frames; lights not flat-fielded")
        note("WARNING: no flat frames; lights not flat-fielded")

    # -- stack -----------------------------------------------------------
    stack, sstats = stack_lights(
        lights, bias=master_bias, dark=master_dark, flat=master_flat
    )
    stats.update(sstats)
    note(
        f"stacked {sstats['n_stacked']}/{sstats['n_input']} lights "
        f"({sstats['n_rejected']} rejected)"
    )
    for p, sh in zip(lights, sstats["shifts"]):
        note(f"  {p.name}: shift {sh if sh else 'DROPPED (no registration)'}")

    # -- outputs ---------------------------------------------------------
    hdr = fits.Header()
    hdr["IMAGETYP"] = ("STACK", "Sigma-clipped registered stack")
    hdr["NSTACK"] = (sstats["n_stacked"], "Frames in stack")
    hdr["NREJECT"] = (sstats["n_rejected"], "Frames rejected")
    hdr["BITPIX"] = -32
    hdr["BUNIT"] = ("ADU", "Pixel units")
    hdr["CREATOR"] = ("AstroCapture process", "Calibration + stacking")
    hdr["MBIAS"] = ("yes" if master_bias is not None else "none",
                    "Master bias used")
    hdr["MDARK"] = ("yes" if master_dark is not None else "none",
                    "Master dark used")
    hdr["MFLAT"] = ("yes" if master_flat is not None else "none",
                    "Master flat used")
    hdr["REGIST"] = ("translation-only",
                     "Registration model (no rotation correction)")
    fits.writeto(output_dir / "stacked.fits", stack, hdr, overwrite=True)
    note("wrote stacked.fits")

    (output_dir / "stacked.png").write_bytes(stretch_png(stack))
    note("wrote stacked.png")

    _write_log(output_dir, log_lines)
    return stats


def _write_log(output_dir: Path, lines: list[str]) -> None:
    with open(output_dir / "process.log", "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
