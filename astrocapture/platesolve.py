"""Plate solving via local astrometry.net, plus closed-loop recentering.

``PlateSolver`` shells out to the ``solve-field`` binary (local
astrometry.net install) and parses the solved field center from the
generated ``.wcs`` file (falling back to the ``RA,Dec = (...)`` line
solve-field prints on success).  It never raises for a missing binary
or a failed solve — a failed solve must not kill a running sequence.

``recenter()`` is the closed-loop helper the sequencer will call after
the initial slew: take a short exposure, solve it, and nudge the mount
until the solved position matches the target within tolerance.

Design note on the Mount interface (``drivers/base.py``): it exposes
``offset()`` as an *open-loop* dither nudge, but no closed-loop jog
primitive that reports completion.  ``recenter()`` therefore issues an
absolute ``mount.goto()`` to the error-corrected coordinates and waits
on ``mount.slew_complete()``.  This is driver-agnostic (works on the sim
and INDI backends alike) and converges even with a constant
pointing-model error, because every iteration re-measures the residual.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

from astropy.io import fits

from astrocapture.util import angular_separation_deg

log = logging.getLogger(__name__)

# solve-field prints e.g.:
#   Field 1: solved with index ...
#   Field 1: RA,Dec = (10.68458, 41.26875), pixel scale 1.93 arcsec/pix.
_SOLVED_LINE = re.compile(
    r"RA,Dec\s*=\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)"
)


class PlateSolver:
    """Blind (or hinted) plate solver wrapping ``solve-field``.

    Parameters
    ----------
    timeout_s:
        Subprocess timeout for one solve-field run.
    radius_deg:
        Search radius passed with ``--radius`` when RA/Dec hints are given.
    logger:
        Logger for solve diagnostics; defaults to the module logger.
    """

    def __init__(
        self,
        timeout_s: float = 120.0,
        radius_deg: float = 5.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.timeout_s = timeout_s
        self.radius_deg = radius_deg
        self.log = logger or log

    # -- public API ------------------------------------------------------
    def solve(
        self,
        fits_path: str | Path,
        ra_hint_deg: float | None = None,
        dec_hint_deg: float | None = None,
    ) -> tuple[float, float] | None:
        """Solve ``fits_path``; return ``(ra_deg, dec_deg)`` of field center.

        Returns ``None`` when solve-field is not installed or the solve
        fails.  Never raises.
        """
        solver = shutil.which("solve-field")
        if solver is None:
            self.log.info("plate solve skipped (solve-field not installed): %s",
                           fits_path)
            return None

        cmd = [solver, "--no-plots", "--overwrite", str(fits_path)]
        if ra_hint_deg is not None and dec_hint_deg is not None:
            cmd += [
                "--ra", f"{ra_hint_deg:.6f}",
                "--dec", f"{dec_hint_deg:.6f}",
                "--radius", f"{self.radius_deg:.3f}",
            ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout_s
            )
        except (subprocess.SubprocessError, OSError) as exc:
            self.log.warning("plate solve failed to run: %s", exc)
            return None
        if result.returncode != 0:
            tail = (result.stderr or "")[-400:]
            self.log.warning("plate solve failed (rc=%d): %s",
                             result.returncode, tail)
            return None

        coords = self._parse_wcs(fits_path)
        if coords is not None:
            self.log.info("plate solve OK: RA=%.4f° Dec=%+.4f°", *coords)
            return coords
        coords = self._parse_stdout(result.stdout or "")
        if coords is not None:
            self.log.info("plate solve OK (from solver output): "
                           "RA=%.4f° Dec=%+.4f°", *coords)
            return coords
        self.log.warning("plate solve ran but no coordinates could be read")
        return None

    # -- parsing ----------------------------------------------------------
    def _wcs_path(self, fits_path: str | Path) -> Path:
        p = Path(fits_path)
        name = p.name
        if name.lower().endswith(".fits"):
            return p.with_name(name[: -len(".fits")] + ".wcs")
        return p.with_suffix(".wcs")

    def _parse_wcs(self, fits_path: str | Path) -> tuple[float, float] | None:
        """Read the field center from solve-field's ``.wcs`` output."""
        wcs_path = self._wcs_path(fits_path)
        if not wcs_path.exists():
            return None
        try:
            from astropy.wcs import WCS

            with fits.open(wcs_path) as hdul:
                w = WCS(hdul[0].header)
            # Image dimensions come from the *input* frame (the .wcs file
            # itself usually carries no pixel data, only the header).
            with fits.open(fits_path) as hdul:
                data = hdul[0].data
            if data is not None:
                ny, nx = data.shape[-2:]
            else:  # pragma: no cover - degenerate input
                return None
            ra, dec = w.all_pix2world(nx / 2.0, ny / 2.0, 0)
            return float(ra), float(dec)
        except Exception as exc:  # noqa: BLE001 - solve must never raise
            self.log.warning("plate solve: could not read WCS: %s", exc)
            return None

    def _parse_stdout(self, stdout: str) -> tuple[float, float] | None:
        """Fallback: solve-field's own ``Field 1: RA,Dec = (ra, dec)`` line.

        (The ``.corr`` file solve-field also writes is the per-star
        match correspondence — it cannot yield the field center without
        the WCS, so the solver's stdout line is the honest fallback.)
        """
        m = _SOLVED_LINE.search(stdout)
        if m is None:
            return None
        return float(m.group(1)), float(m.group(2))


class FakeSolver:
    """Test double for :class:`PlateSolver` (also handy for dry runs).

    ``positions`` is either a list of ``(ra_deg, dec_deg)`` tuples —
    ``solve()`` walks the list and then repeats the last entry — or a
    zero-argument callable returning the next solved position (e.g. one
    that reads a fake mount's pointing, so tests can exercise the real
    converge/correct dynamics of :func:`recenter`).
    """

    def __init__(
        self, positions: list[tuple[float, float]] | Callable[[], tuple[float, float]]
    ) -> None:
        self.positions = positions
        self.calls = 0

    def solve(
        self,
        fits_path: str | Path,
        ra_hint_deg: float | None = None,
        dec_hint_deg: float | None = None,
    ) -> tuple[float, float] | None:
        self.calls += 1
        if callable(self.positions):
            return self.positions()
        idx = min(self.calls - 1, len(self.positions) - 1)
        return self.positions[idx]


# ---------------------------------------------------------------------------
# Closed-loop recentering
# ---------------------------------------------------------------------------


def _wrap_ra_diff_deg(diff_deg: float) -> float:
    """Signed RA difference wrapped to [-180, +180) degrees."""
    return (diff_deg + 540.0) % 360.0 - 180.0


def _take_frame(camera, exposure_s: float, poll_s: float = 0.1):
    """Expose and download one frame, mirroring the sequencer's flow."""
    camera.set_exposure_settings(exposure_s)
    camera.start_exposure()
    while not camera.exposure_complete():
        time.sleep(poll_s)
    return camera.download_image()


def recenter(
    mount,
    camera,
    session,
    target_ra_deg: float,
    target_dec_deg: float,
    tolerance_arcmin: float = 2.0,
    max_iterations: int = 5,
    exposure_s: float = 5.0,
    solver: PlateSolver | FakeSolver | None = None,
    settle_s: float = 1.0,
) -> bool:
    """Iteratively center ``target`` using plate solves.

    Each iteration exposes for ``exposure_s``, solves the frame, and —
    if the solved center is outside ``tolerance_arcmin`` — slews to the
    error-corrected coordinates ``target + (target - solved)`` and waits
    for the slew to settle.  Returns ``True`` as soon as the residual is
    within tolerance, ``False`` after ``max_iterations`` without
    converging (or when solves keep failing).

    ``session`` is used only for its ``.dir`` (scratch space for the
    solve frame + solve-field sidecars, cleaned up afterwards) and its
    ``.log`` logger.  Pass a ``FakeSolver`` in tests; the default is a
    real :class:`PlateSolver`.
    """
    logger = session.log
    solver = solver or PlateSolver()
    target_ra_h = target_ra_deg / 15.0
    tmp_path = Path(session.dir) / "recenter_tmp.fits"

    for iteration in range(1, max_iterations + 1):
        image = _take_frame(camera, exposure_s)
        fits.writeto(tmp_path, image, overwrite=True)
        try:
            solved = solver.solve(
                tmp_path,
                ra_hint_deg=target_ra_deg,
                dec_hint_deg=target_dec_deg,
            )
        finally:
            # Remove the frame and every solve-field sidecar (.wcs, .corr,
            # .solved, ...) so scratch never accumulates in the session.
            for sidecar in Path(session.dir).glob("recenter_tmp.*"):
                try:
                    sidecar.unlink()
                except OSError:
                    pass
        if solved is None:
            logger.warning("recenter: solve failed (iteration %d/%d)",
                           iteration, max_iterations)
            continue

        ra_deg, dec_deg = solved
        sep_arcmin = angular_separation_deg(
            ra_deg / 15.0, dec_deg, target_ra_h, target_dec_deg
        ) * 60.0
        logger.info("recenter: iteration %d solved RA=%.4f° Dec=%+.4f° "
                     "(%.1f' from target)", iteration, ra_deg, dec_deg,
                     sep_arcmin)
        if sep_arcmin <= tolerance_arcmin:
            logger.info("recenter: converged within %.1f'",
                        tolerance_arcmin)
            return True

        # Aim at the target *plus* the measured error, so a constant
        # pointing-model offset cancels on the next iteration.
        d_ra = _wrap_ra_diff_deg(target_ra_deg - ra_deg)
        d_dec = target_dec_deg - dec_deg
        corrected_ra_h = (target_ra_deg + d_ra) / 15.0 % 24.0
        corrected_dec = max(-90.0, min(90.0, target_dec_deg + d_dec))
        logger.info("recenter: corrective slew to RA=%.4fh Dec=%+.4f°",
                     corrected_ra_h, corrected_dec)
        mount.goto(corrected_ra_h, corrected_dec)
        while not mount.slew_complete():
            time.sleep(0.2)
        time.sleep(settle_s)

    logger.warning("recenter: did not converge in %d iterations", max_iterations)
    return False
