"""Session watchdog: decide when a problem is bad enough to park the rig.

The sequencer (a follow-up task wires this in) calls the ``note_*``
methods as events happen during a run.  The watchdog keeps counters
and a retry budget, and when a failure mode crosses its threshold it
parks the mount, stops the sequence, and alerts — exactly once per
trip, so a flapping failure can't spam park commands.

Alerting is pluggable: pass any ``alert(message)`` callable.
``AlertLog`` (the default) just logs; ``WebhookAlert`` POSTs JSON to a
webhook URL (Slack/Discord/ntfy/...) using only the stdlib, and never
raises — a dead webhook must not take down the safety path.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
import urllib.request

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------


class AlertLog:
    """Default alerter: log the message (visible in session.log)."""

    def alert(self, message: str) -> None:
        log.warning("ALERT: %s", message)


class WebhookAlert:
    """POST ``{"text": message}`` to a webhook URL. Stdlib only.

    Failures (DNS, refused connection, timeout, bad status) are logged
    and suppressed — alerting must never raise into the safety path.
    """

    def __init__(self, url: str, timeout_s: float = 10.0) -> None:
        self.url = url
        self.timeout_s = timeout_s

    def alert(self, message: str) -> None:
        try:
            req = urllib.request.Request(
                self.url,
                data=json.dumps({"text": message}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                resp.read()
        except Exception as exc:  # noqa: BLE001 - alerting never raises
            log.warning("webhook alert to %s failed (suppressed): %s",
                        self.url, exc)


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------


class Watchdog:
    """Failure-policy state machine for an unattended session.

    Parameters
    ----------
    park_mount:
        Zero-arg callable that parks the mount (e.g. ``mount.park``).
    stop_sequence:
        Zero-arg callable that halts the run (e.g. ``sequencer.abort``).
    alert:
        ``alert(message)`` callable; defaults to :class:`AlertLog`.
    release_camera:
        Optional zero-arg callable to release the camera
        (e.g. ``camera.disconnect``) on an exception trip.
    max_solve_failures:
        Consecutive plate-solve failures before parking (default 3).
    guide_retry_budget:
        How many ``"retry"`` answers ``note_guide_lost()`` gives before
        switching to ``"park"`` (default 3).
    max_session_seconds:
        Optional wall-clock session limit; ``check_session_limits()``
        stops the sequence (gracefully — no park) past it.
    clock:
        Time source (``time.monotonic``); injectable for tests.
    """

    RETRY = "retry"
    PARK = "park"

    def __init__(
        self,
        park_mount,
        stop_sequence,
        alert=None,
        release_camera=None,
        max_solve_failures: int = 3,
        guide_retry_budget: int = 3,
        max_session_seconds: float | None = None,
        clock=time.monotonic,
    ) -> None:
        if max_solve_failures < 1:
            raise ValueError("max_solve_failures must be >= 1")
        if guide_retry_budget < 0:
            raise ValueError("guide_retry_budget must be >= 0")
        self._park_mount = park_mount
        self._stop_sequence = stop_sequence
        self._alert = alert if alert is not None else AlertLog().alert
        self._release_camera = release_camera
        self._max_solve_failures = max_solve_failures
        self._guide_retry_budget = guide_retry_budget
        self._max_session_seconds = max_session_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._solve_failures = 0
        self._guide_retries_left = guide_retry_budget
        self._parked = False
        self._session_start = self._clock()

    # -- internal trip ----------------------------------------------------
    def _park_once(self) -> None:
        """Park the mount; safe to call repeatedly (parks only once)."""
        with self._lock:
            first = not self._parked
            self._parked = True
        if not first:
            return
        try:
            self._park_mount()
        except Exception as exc:  # noqa: BLE001 - park is best effort
            log.warning("watchdog: park_mount failed: %s", exc)

    def _trip(self, reason: str) -> None:
        """Park once, stop the sequence, alert. The full escalation."""
        self._park_once()
        try:
            self._stop_sequence()
        except Exception as exc:  # noqa: BLE001 - stop is best effort
            log.warning("watchdog: stop_sequence failed: %s", exc)
        self._alert(reason)

    # -- plate solving ------------------------------------------------------
    def note_solve_success(self) -> None:
        """Reset the consecutive-failure counter."""
        with self._lock:
            self._solve_failures = 0

    def note_solve_failure(self) -> None:
        """Count a failed solve; park + stop + alert at the threshold."""
        with self._lock:
            self._solve_failures += 1
            n = self._solve_failures
        if n >= self._max_solve_failures:
            self._trip(
                f"plate solve failed {n} times in a row — "
                "parking mount and stopping sequence"
            )

    # -- guiding --------------------------------------------------------------
    def note_guide_lost(self) -> str:
        """Decide what to do about a lost guide star.

        Returns ``"retry"`` while retry budget remains (the sequencer
        should pause briefly and resume guiding), else escalates to the
        full trip and returns ``"park"``.
        """
        with self._lock:
            if self._guide_retries_left > 0:
                self._guide_retries_left -= 1
                return self.RETRY
        self._trip("guide star lost repeatedly — parking mount and "
                    "stopping sequence")
        return self.PARK

    def note_guide_recovered(self) -> None:
        """Guide star re-acquired; restore the full retry budget."""
        with self._lock:
            self._guide_retries_left = self._guide_retry_budget

    # -- exceptions ------------------------------------------------------------
    def handle_exception(self, exc: BaseException, fatal: bool = False) -> None:
        """Handle an unexpected exception from the sequence.

        Policy: **swallow-after-parking**.  An exception mid-sequence
        usually means unattended hardware in an unknown state — the safe
        move is to park the mount (so it can't track into the pier or
        keep exposing), release the camera, alert with a traceback
        summary, and let the sequence end instead of propagating the
        exception up through equipment-control code.

        Pass ``fatal=True`` only when the *process itself* cannot safely
        continue (e.g. corrupted session state): the mount is still
        parked first, then the exception is re-raised.
        """
        tb_lines = traceback.format_exception(
            type(exc), exc, exc.__traceback__)
        summary = "".join(tb_lines[-6:]).strip()
        log.error("watchdog: exception in sequence:\n%s", summary)
        self._park_once()
        if self._release_camera is not None:
            try:
                self._release_camera()
            except Exception as rel_exc:  # noqa: BLE001 - best effort
                log.warning("watchdog: release_camera failed: %s", rel_exc)
        self._alert(f"sequence exception — mount parked:\n{summary}")
        if fatal:
            raise exc

    # -- session limits ----------------------------------------------------------
    def check_session_limits(self) -> bool:
        """Enforce the wall-clock session limit. Returns True if tripped.

        A tripped limit is a *graceful* stop: the sequence halts and an
        alert goes out, but the mount is NOT parked — the operator (or a
        later policy) decides whether the rig stays tracking.  Parking on
        a mere time limit would be surprising; parking is reserved for
        failure escalation and exceptions.
        """
        if self._max_session_seconds is None:
            return False
        elapsed = self._clock() - self._session_start
        if elapsed < self._max_session_seconds:
            return False
        try:
            self._stop_sequence()
        except Exception as exc:  # noqa: BLE001 - stop is best effort
            log.warning("watchdog: stop_sequence failed: %s", exc)
        self._alert(
            f"session time limit reached "
            f"({elapsed / 3600:.1f}h >= {self._max_session_seconds / 3600:.1f}h) "
            "— stopping sequence"
        )
        return True
