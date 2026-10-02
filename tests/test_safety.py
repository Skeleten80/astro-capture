"""Watchdog escalation policy and alerting (all fake callbacks)."""

import pytest

from astrocapture.safety import AlertLog, Watchdog, WebhookAlert


class Calls:
    """Records what the watchdog asked for."""

    def __init__(self):
        self.parked = 0
        self.stopped = 0
        self.alerts: list = []

    def park_mount(self):
        self.parked += 1

    def stop_sequence(self):
        self.stopped += 1

    def alert(self, message):
        self.alerts.append(message)


def make_wd(calls, **kw):
    return Watchdog(calls.park_mount, calls.stop_sequence,
                    alert=calls.alert, **kw)


# ---------------------------------------------------------------------------
# plate-solve failures
# ---------------------------------------------------------------------------


def test_consecutive_solve_failures_park_once():
    calls = Calls()
    wd = make_wd(calls, max_solve_failures=3)
    for _ in range(5):
        wd.note_solve_failure()
    assert calls.parked == 1  # parked exactly once despite repeated trips
    assert calls.stopped >= 1
    assert any("plate solve failed" in m for m in calls.alerts)


def test_solve_success_resets_failure_counter():
    calls = Calls()
    wd = make_wd(calls, max_solve_failures=3)
    wd.note_solve_failure()
    wd.note_solve_failure()
    wd.note_solve_success()
    wd.note_solve_failure()
    wd.note_solve_failure()
    assert calls.parked == 0


def test_watchdog_rejects_bad_thresholds():
    calls = Calls()
    with pytest.raises(ValueError):
        Watchdog(calls.park_mount, calls.stop_sequence, max_solve_failures=0)


# ---------------------------------------------------------------------------
# guiding
# ---------------------------------------------------------------------------


def test_guide_lost_retry_budget_exhausts_to_park():
    calls = Calls()
    wd = make_wd(calls, guide_retry_budget=2)
    assert wd.note_guide_lost() == "retry"
    assert wd.note_guide_lost() == "retry"
    assert wd.note_guide_lost() == "park"
    assert calls.parked == 1
    # Further losses escalate but don't re-park.
    assert wd.note_guide_lost() == "park"
    assert calls.parked == 1


def test_guide_recovered_restores_budget():
    calls = Calls()
    wd = make_wd(calls, guide_retry_budget=1)
    assert wd.note_guide_lost() == "retry"
    wd.note_guide_recovered()
    assert wd.note_guide_lost() == "retry"
    assert calls.parked == 0


# ---------------------------------------------------------------------------
# exceptions
# ---------------------------------------------------------------------------


def test_handle_exception_parks_alerts_and_swallows():
    calls = Calls()
    released = []
    wd = Watchdog(calls.park_mount, calls.stop_sequence, alert=calls.alert,
                  release_camera=lambda: released.append(True))
    try:
        raise RuntimeError("camera exploded")
    except RuntimeError as exc:
        wd.handle_exception(exc)  # must not raise: swallow-after-parking
    assert calls.parked == 1
    assert released == [True]
    assert any("camera exploded" in m for m in calls.alerts)


def test_handle_exception_fatal_reraises_after_parking():
    calls = Calls()
    wd = make_wd(calls)
    with pytest.raises(RuntimeError, match="boom"):
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            wd.handle_exception(exc, fatal=True)
    assert calls.parked == 1  # mount is parked before the re-raise


# ---------------------------------------------------------------------------
# session limits + alerting
# ---------------------------------------------------------------------------


def test_check_session_limits_stops_gracefully_without_parking():
    calls = Calls()
    now = [1000.0]
    wd = make_wd(calls, max_session_seconds=60.0, clock=lambda: now[0])
    assert wd.check_session_limits() is False
    now[0] += 61.0
    assert wd.check_session_limits() is True
    assert calls.stopped == 1
    assert calls.parked == 0  # graceful stop: no park on a time limit
    assert any("time limit" in m for m in calls.alerts)


def test_check_session_limits_no_limit_configured():
    calls = Calls()
    wd = make_wd(calls)
    assert wd.check_session_limits() is False
    assert calls.stopped == 0


def test_webhook_alert_failure_never_raises():
    # Nothing listens on port 9 (discard); refused connection must not raise.
    WebhookAlert("http://127.0.0.1:9/hook", timeout_s=2).alert("test message")


def test_alert_log_does_not_raise(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        AlertLog().alert("hello")
    assert "hello" in caplog.text
