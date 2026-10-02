"""Session management: directories, logging, FITS output, and hooks.

A session owns one directory::

    sessions/<name>-YYYYMMDD-HHMMSS/
        session.log      # every action, timestamped
        lights/ flats/ darks/ bias/
            <target>_<type>_<filter>_<nnn>.fits

Hooks (dithering, plate solving, meridian-flip check) live here too so
the sequencer stays a pure state machine.
"""

from __future__ import annotations

import logging
import random
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.io import fits

from astrocapture import config as plan_config
from astrocapture.util import utcnow_iso

FRAME_DIRS = {"light": "lights", "dark": "darks", "flat": "flats", "bias": "bias"}


class Session:
    def __init__(self, plan: plan_config.Plan, root: str | Path | None = None) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = Path(root or plan.output_dir)
        self.dir = base / f"{plan.session_name}-{stamp}"
        for sub in ("lights", "darks", "flats", "bias"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.plan = plan
        self.log = self._setup_logging()
        self.frame_counters: dict[str, int] = {t: 0 for t in FRAME_DIRS}

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"astrocapture.{self.dir.name}")
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(self.dir / "session.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
        logger.addHandler(sh)
        logger.propagate = False
        return logger

    # -- FITS output ----------------------------------------------------
    def save_frame(
        self,
        image: np.ndarray,
        frame_type: str,
        step: plan_config.Step,
        ra_hours: float,
        dec_deg: float,
        temperature_c: float | None = None,
    ) -> Path:
        self.frame_counters[frame_type] += 1
        n = self.frame_counters[frame_type]
        filt = f"_{step.filter}" if step.filter else ""
        name = f"{self.plan.target.name}_{frame_type}{filt}_{n:03d}.fits"
        path = self.dir / FRAME_DIRS[frame_type] / name

        hdr = fits.Header()
        hdr["OBJECT"] = (self.plan.target.name, "Target name")
        hdr["RA"] = (ra_hours * 15.0, "[deg] J2000 Right Ascension")
        hdr["DEC"] = (dec_deg, "[deg] J2000 Declination")
        hdr["EXPTIME"] = (step.exposure, "[s] Exposure time")
        hdr["IMAGETYP"] = (frame_type.upper() + " FRAME", "Frame type")
        hdr["DATE-OBS"] = (utcnow_iso(), "UTC start of exposure (approx)")
        hdr["INSTRUME"] = (self.plan.instrument, "Camera")
        hdr["TELESCOP"] = (self.plan.telescope, "Telescope")
        hdr["FILTER"] = (step.filter or "NONE", "Filter")
        hdr["XBINNING"] = (step.binning, "Binning factor")
        hdr["YBINNING"] = (step.binning, "Binning factor")
        if step.gain:
            hdr["GAIN"] = (step.gain, "Gain / ISO setting")
            hdr["ISO"] = (int(step.gain), "ISO (DSLR gain)")
        if temperature_c is not None:
            hdr["CCD-TEMP"] = (temperature_c, "[degC] Sensor temperature")
        hdr["BUNIT"] = ("ADU", "Pixel units")
        hdr["CREATOR"] = ("AstroCapture 0.1.0", "Capture software")

        fits.writeto(path, image, hdr, overwrite=True)
        self.log.info("saved %s (%s, %.1fs)", path.name, frame_type, step.exposure)
        return path


# -- hooks ---------------------------------------------------------------
class Ditherer:
    """Random small RA/Dec offsets between frames.

    Dithering moves the target on the sensor between exposures so hot
    pixels and fixed-pattern noise don't stack coherently.  Offsets are
    uniform in a square of ``max_arcsec`` per axis.
    """

    def __init__(self, max_arcsec: float = 15.0, seed: int | None = None) -> None:
        self.max_arcsec = max_arcsec
        self._rng = random.Random(seed)

    def next_offset(self) -> tuple[float, float]:
        return (
            self._rng.uniform(-self.max_arcsec, self.max_arcsec),
            self._rng.uniform(-self.max_arcsec, self.max_arcsec),
        )


def plate_solve(
    fits_path: str | Path, logger: logging.Logger, timeout: int = 120
) -> dict | None:
    """Blind plate-solve ``fits_path`` with local astrometry.net if present.

    Returns the solved ``{'ra_deg', 'dec_deg'}`` or ``None`` when skipped
    (no ``solve-field`` binary) or failed.  Never raises: a failed solve
    must not kill a running sequence.
    """
    solver = shutil.which("solve-field")
    if solver is None:
        logger.info("plate-solve skipped: solve-field not installed")
        return None
    try:
        result = subprocess.run(
            [solver, "--no-plots", "--overwrite", str(fits_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("plate-solve failed to run: %s", exc)
        return None
    if result.returncode != 0:
        logger.warning("plate-solve failed (rc=%d)", result.returncode)
        return None
    # astrometry.net writes <basename>.solved alongside a .wcs file.
    wcs_path = Path(str(fits_path).replace(".fits", ".wcs"))
    if not wcs_path.exists():
        logger.warning("plate-solve ran but produced no .wcs file")
        return None
    try:
        from astropy.wcs import WCS

        with fits.open(wcs_path) as hdul:
            w = WCS(hdul[0].header)
        h, wpx = hdul[0].data.shape if hdul[0].data is not None else (512, 512)
        ra, dec = w.all_pix2world(wpx / 2, h / 2, 0)
        logger.info("plate-solve OK: center RA=%.4f° Dec=%+.4f°", ra, dec)
        return {"ra_deg": float(ra), "dec_deg": float(dec)}
    except Exception as exc:  # noqa: BLE001 - solve must never kill a run
        logger.warning("plate-solve: could not read WCS: %s", exc)
        return None


def check_meridian_flip(
    ra_hours: float,
    lst_hours: float,
    logger: logging.Logger,
    flip_margin_min: float = 10.0,
) -> bool:
    """Stub: True if a meridian flip is advisable before continuing.

    Hour angle HA = LST - RA; crossing HA=0 (within the margin) means the
    target transits the meridian soon.  Real flips re-slew, re-center,
    and re-start guiding — the sequencer calls this hook before each
    light step; the actual flip is left for a future guiding-aware pass.
    """
    ha_hours = (lst_hours - ra_hours + 12.0) % 24.0 - 12.0
    ha_min = ha_hours * 60.0
    if abs(ha_min) < flip_margin_min:
        logger.warning(
            "meridian flip advised: target within %.0f min of meridian (HA=%+.1f min) "
            "-- flip not yet automated, pausing sequence",
            flip_margin_min,
            ha_min,
        )
        return True
    logger.debug("meridian check OK (HA=%+.1f min)", ha_min)
    return False
