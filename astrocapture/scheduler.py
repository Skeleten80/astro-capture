"""Multi-target night scheduler: work a target queue until dawn.

Given a plan with a ``targets:`` list and a ``night:`` block, the
scheduler picks the best currently-observable target, runs it through a
fresh :class:`~astrocapture.sequencer.Sequencer` (one session directory
per target), and repeats until astronomical dawn or until every target
has received its quota of light frames.

Per-target quota is deliberately simple: each target gets the plan's
full light sequence (sum of ``count`` over the ``light`` steps), and
quota is tracked as light frames actually written to that target's
session directory.

Target selection (``pick_target``): among targets with remaining quota,
keep those at least ``min_alt_deg`` above the horizon and at least
``min_moon_sep_deg`` from the Moon, then score::

    score = priority * altitude_factor * moon_factor

with ``altitude_factor = (alt - min_alt) / (90 - min_alt)`` and
``moon_factor = (moon_sep - min_sep) / (180 - min_sep)``.  The highest
score wins; ``None`` means nothing is observable right now.

Waiting is compressed by ``time_accel`` (``night.time_accel``): in sim
mode a multi-hour wait becomes seconds.  Test doubles may implement an
``advance(seconds)`` method on the clock instead of sleeping.

After the night, ``night_summary.md`` is written to the night output
directory: per-target frames and status, failures, watchdog alerts
grepped from the session logs, and focus positions when autofocus ran.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.time import Time
import astropy.units as u

from astrocapture import config as plan_config
from astrocapture.sequencer import SeqState, Sequencer
from astrocapture.session import Session

log = logging.getLogger(__name__)

FRAME_OVERHEAD_S = 60.0  # slew + settle budget added to one exposure
MAX_CONSECUTIVE_FAILURES = 3  # per-target aborts before giving up on it
WAIT_RECHECK_S = 600.0  # re-evaluate target selection at least this often


def astronomical_dawn_utc(
    lat: float, lon: float, day: date | datetime
) -> datetime:
    """Next astronomical dawn (Sun rising past -18° altitude) as a UTC datetime.

    ``day`` is the observing date (a ``date`` or ``datetime``; naive
    datetimes are read as UTC).  The Sun's altitude is sampled every 5
    minutes over the 24 h after local noon and the -18° crossing after
    the darkest moment is linearly interpolated.

    Raises :class:`ValueError` when the Sun never drops below -18°
    around that date (polar day / bright high-latitude summer).
    """
    if isinstance(day, datetime):
        day = day.date() if day.tzinfo is None else \
            day.astimezone(timezone.utc).date()
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
    noon = Time(datetime(day.year, day.month, day.day, 12, 0,
                         tzinfo=timezone.utc))
    grid = noon + np.arange(0, 24 * 60 + 5, 5) * u.min
    alt = get_sun(grid).transform_to(AltAz(obstime=grid, location=loc)).alt.deg
    night = alt < -18.0
    if not night.any():
        raise ValueError(
            f"no astronomical night near {day} at latitude {lat:.2f}° "
            "(Sun never drops below -18°)")
    i_min = int(np.argmin(alt))  # darkest moment: always inside the night
    risen = np.nonzero(~night[i_min:])[0]
    if risen.size == 0:
        raise ValueError(
            f"Sun does not rise within 24 h of {day} at {lat:.2f}°")
    j = i_min + int(risen[0])  # first sample at/above -18° after the minimum
    i0, i1 = j - 1, j
    a0, a1 = alt[i0], alt[i1]
    frac = (-18.0 - a0) / (a1 - a0) if a1 != a0 else 0.0
    t0 = grid[i0].to_datetime(timezone.utc)
    t1 = grid[i1].to_datetime(timezone.utc)
    return t0 + (t1 - t0) * frac


def moon_separation_deg(
    ra_hours: float, dec_deg: float, when: datetime, lat: float, lon: float
) -> float:
    """Angular separation (degrees) between a target and the Moon.

    ``ra_hours`` is in hours (matching :class:`Target`), ``dec_deg`` in
    degrees; ``when`` is timezone-aware (naive is read as UTC).  Uses
    astropy's built-in ephemeris via ``get_body("moon", ...)`` —
    astropy ≥ 8 removed the old ``get_moon`` alias.
    """
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
    moon = get_body("moon", Time(when), loc)
    target = SkyCoord(ra_hours * 15.0 * u.deg, dec_deg * u.deg)
    return float(moon.separation(target).deg)


def altitude_deg(
    ra_hours: float, dec_deg: float, when: datetime, lat: float, lon: float
) -> float:
    """Target altitude in degrees above the horizon at ``when``."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
    target = SkyCoord(ra_hours * 15.0 * u.deg, dec_deg * u.deg)
    altaz = target.transform_to(AltAz(obstime=Time(when), location=loc))
    return float(altaz.alt.deg)


class NightScheduler:
    """Run a plan's target queue until astronomical dawn.

    ``drivers_factory`` is a zero-argument callable returning a fresh
    ``(mount, camera)`` pair per target.  ``clock`` defaults to
    ``datetime.now(timezone.utc)``; pass a fake clock with an
    ``advance(seconds)`` method in tests to skip waiting instantly.
    ``time_accel`` compresses real waiting (sleeps); defaults to the
    plan's ``night.time_accel``.
    """

    def __init__(
        self,
        plan: plan_config.Plan,
        drivers_factory,
        clock=None,
        time_accel: float | None = None,
    ) -> None:
        if plan.night is None:
            raise ValueError(
                "NightScheduler needs a plan with a 'night:' block "
                "(see examples/night_queue.yaml)")
        self.plan = plan
        self.night = plan.night
        self.drivers_factory = drivers_factory
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.time_accel = (self.night.time_accel if time_accel is None
                           else time_accel)
        if self.time_accel < 1:
            raise ValueError(f"time_accel must be >= 1, got {self.time_accel}")
        n = len(plan.targets)
        self.captured = [0] * n          # light frames banked per target
        self._consec_failures = [0] * n  # aborts in a row per target
        self.completed: list[dict] = []  # per-run records, in run order
        self.failures: list[str] = []
        self.alerts: list[str] = []
        self.focus_positions: dict[str, int] = {}
        self.dawn: datetime | None = None
        self.night_dir: Path | None = None

    # -- helpers -----------------------------------------------------------
    def _now(self) -> datetime:
        now = self.clock()
        return now if now.tzinfo is not None else \
            now.replace(tzinfo=timezone.utc)

    def _wait(self, seconds: float) -> None:
        advance = getattr(self.clock, "advance", None)
        if callable(advance):
            advance(seconds)  # test double: jump the fake clock, no sleep
        else:
            time.sleep(max(0.0, seconds) / self.time_accel)

    def quota_lights(self) -> int:
        """Light frames per target: the plan's full light sequence."""
        return sum(s.count for s in self.plan.steps if s.type == "light")

    def _frame_estimate_s(self) -> float:
        """One light frame + overhead: don't start a target past this."""
        longest = max((s.exposure for s in self.plan.steps
                       if s.type == "light"), default=0.0)
        return longest + FRAME_OVERHEAD_S

    def _target_remaining(self, i: int) -> bool:
        return (self.captured[i] < self.quota_lights()
                and self._consec_failures[i] < MAX_CONSECUTIVE_FAILURES)

    # -- selection ----------------------------------------------------------
    def pick_target(self, now: datetime) -> tuple[int, plan_config.TargetEntry] | None:
        """Best currently-observable target, or ``None``.

        Eligibility: remaining quota, altitude >= ``min_alt_deg``, Moon
        separation >= ``min_moon_sep_deg``.  Score =
        ``priority * altitude_factor * moon_factor`` (both factors 0..1
        across their allowed ranges).  Returns ``(index, entry)``.
        """
        lat, lon = self.night.latitude, self.night.longitude
        alt_span = max(1e-6, 90.0 - self.night.min_alt_deg)
        moon_span = max(1e-6, 180.0 - self.night.min_moon_sep_deg)
        best: tuple[int, plan_config.TargetEntry] | None = None
        best_score = 0.0
        for i, entry in enumerate(self.plan.targets):
            if not self._target_remaining(i):
                continue
            tgt = entry.target
            alt = altitude_deg(tgt.ra_hours, tgt.dec_deg, now, lat, lon)
            if alt < self.night.min_alt_deg:
                continue
            sep = moon_separation_deg(tgt.ra_hours, tgt.dec_deg, now,
                                      lat, lon)
            if sep < self.night.min_moon_sep_deg:
                continue
            alt_f = min(1.0, (alt - self.night.min_alt_deg) / alt_span)
            moon_f = min(1.0, (sep - self.night.min_moon_sep_deg) / moon_span)
            score = entry.priority * alt_f * moon_f
            if score > best_score:
                best, best_score = (i, entry), score
        return best

    # -- main loop -----------------------------------------------------------
    def run(self) -> int:
        """Work the queue until dawn. Returns 0 on a clean night."""
        now0 = self._now()
        self.dawn = astronomical_dawn_utc(
            self.night.latitude, self.night.longitude, now0)
        self.night_dir = (
            Path(self.plan.output_dir)
            / f"{self.plan.session_name}-night-{now0.strftime('%Y%m%d-%H%M%S')}"
        )
        self.night_dir.mkdir(parents=True, exist_ok=True)
        log.info("night: %d target(s), dawn %s, output %s",
                 len(self.plan.targets), self.dawn.isoformat(),
                 self.night_dir)
        run_idx = 0
        while True:
            now = self._now()
            assert self.dawn is not None
            if now >= self.dawn:
                log.info("night: astronomical dawn reached — done")
                break
            if not any(self._target_remaining(i)
                       for i in range(len(self.plan.targets))):
                log.info("night: all quotas filled (or targets failed out)")
                break
            picked = self.pick_target(now)
            if picked is None:
                wait_s = min((self.dawn - now).total_seconds(), WAIT_RECHECK_S)
                if wait_s <= 0:
                    break
                log.info("night: no eligible target — waiting %.0f s",
                         wait_s)
                self._wait(wait_s)
                continue
            if (self.dawn - now).total_seconds() < self._frame_estimate_s():
                log.info("night: less than one frame + overhead remains "
                         "before dawn — not starting a new target")
                break
            run_idx += 1
            self._run_target(picked[0], picked[1], run_idx)
        assert self.night_dir is not None
        self._write_summary()
        return 0

    def _run_target(self, i: int, entry: plan_config.TargetEntry,
                    run_idx: int) -> None:
        tgt = entry.target
        slug = "".join(c if c.isalnum() else "-" for c in tgt.name
                       ).strip("-") or "target"
        sub = dataclasses.replace(
            self.plan,
            target=tgt,
            session_name=f"{self.plan.session_name}-{slug}",
            targets=[entry],
        )
        assert self.night_dir is not None
        mount, camera = self.drivers_factory()
        session = Session(sub, root=self.night_dir / f"{run_idx:02d}-{slug}")
        seq = Sequencer(sub, mount, camera, session=session)
        log.info("night: starting %s (%d/%d lights banked)",
                 tgt.name, self.captured[i], self.quota_lights())
        final = seq.run()
        new_lights = len(list((session.dir / "lights").glob("*.fits")))
        self.captured[i] += new_lights
        if seq.last_focus_position is not None:
            self.focus_positions[tgt.name] = seq.last_focus_position
        self.alerts.extend(self._grep_alerts(tgt.name, session.dir))
        if final == SeqState.DONE:
            self._consec_failures[i] = 0
            status = "done"
        else:
            self._consec_failures[i] += 1
            status = f"aborted ({seq.error})" if seq.error else "aborted"
            self.failures.append(f"{tgt.name}: {status}")
            # The sequencer parks only on DONE; park explicitly after an
            # abort so the rig is never left tracking unattended.
            try:
                mount.park()
                log.info("night: parked mount after abort on %s", tgt.name)
            except Exception as exc:  # noqa: BLE001 - best effort
                log.warning("night: park after abort failed: %s", exc)
        self.completed.append({"target": tgt.name, "lights": new_lights,
                               "status": status,
                               "dir": str(session.dir.relative_to(
                                   self.night_dir))})
        log.info("night: %s finished: %s, +%d lights (%d/%d banked)",
                 tgt.name, status, new_lights, self.captured[i],
                 self.quota_lights())

    @staticmethod
    def _grep_alerts(target_name: str, session_dir: Path) -> list[str]:
        """Watchdog alerts are logged as ``ALERT:`` lines in session.log."""
        found: list[str] = []
        log_path = session_dir / "session.log"
        if not log_path.exists():
            return found
        for line in log_path.read_text().splitlines():
            if "ALERT:" in line:
                found.append(f"{target_name}: {line.split('ALERT:', 1)[1].strip()}")
        return found

    # -- summary --------------------------------------------------------------
    def _write_summary(self) -> None:
        assert self.night_dir is not None and self.dawn is not None
        quota = self.quota_lights()
        lines = [
            f"# Night summary — {self.plan.session_name}",
            "",
            f"- Site: {self.night.latitude:.4f}°, {self.night.longitude:.4f}°",
            f"- Astronomical dawn (UTC): {self.dawn.isoformat()}",
            f"- Light quota per target: {quota} frame(s)",
            f"- Targets with remaining quota at dawn: "
            f"{sum(1 for i in range(len(self.plan.targets)) if self._target_remaining(i))}",
            "",
            "## Targets",
            "",
        ]
        for i, entry in enumerate(self.plan.targets):
            done = "quota filled" if self.captured[i] >= quota else \
                (f"{quota - self.captured[i]} light(s) short of quota"
                 if self._consec_failures[i] < MAX_CONSECUTIVE_FAILURES
                 else "FAILED OUT (3 consecutive aborts)")
            lines.append(f"- **{entry.target.name}**: {self.captured[i]} light(s), {done}")
        lines += ["", "## Runs (in order)", ""]
        if self.completed:
            for rec in self.completed:
                lines.append(f"- {rec['target']}: {rec['lights']} light(s), "
                             f"{rec['status']} (`{rec['dir']}`)")
        else:
            lines.append("- no target was started")
        lines += ["", "## Failures", ""]
        lines += [f"- {f}" for f in self.failures] or ["- none"]
        lines += ["", "## Watchdog alerts", ""]
        lines += [f"- {a}" for a in self.alerts] or ["- none"]
        lines += ["", "## Focus", ""]
        if self.focus_positions:
            for name, pos in self.focus_positions.items():
                lines.append(f"- {name}: last autofocus position {pos}")
        else:
            lines.append("- autofocus did not run (disabled or no focuser); "
                         "see per-target session.log files for HFR notes")
        path = self.night_dir / "night_summary.md"
        path.write_text("\n".join(lines) + "\n")
        log.info("night: summary written to %s", path)
