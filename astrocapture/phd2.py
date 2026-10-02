"""PHD2 guiding client (JSON-over-TCP, default port 4400).

Protocol (per PHD2's "Event Monitoring" documentation, from knowledge
of the PHD2 API): PHD2 runs a TCP server on localhost:4400.  Every
message in both directions is **one JSON object terminated by a
newline** — requests like::

    {"jsonrpc": "2.0", "id": 7, "method": "dither",
     "params": [10, false, {"pixels": 1.5, "time": 8, "timeout": 40}]}

and replies like ``{"jsonrpc": "2.0", "result": 0, "id": 7}``, plus
asynchronous events such as::

    {"Event": "SettleDone", "Timestamp": 1234.5, "Status": 0, ...}
    {"Event": "GuideStep", "Timestamp": ..., "dx": 0.12, "dy": -0.3, ...}
    {"Event": "StarLost", "Timestamp": ..., "Frame": 418, ...}
    {"Event": "AppState", "Timestamp": ..., "State": "Guiding"}

``PHD2Client`` implements a robust incremental line-based reader on a
background thread: partial TCP chunks are buffered until a full newline
is seen, malformed lines are logged and dropped, and events are
dispatched to a small handler registry while RPC replies wake the
waiting caller by id.

Settle semantics: ``dither()`` / ``start_guiding()`` clear the settle
latch, send the RPC, then wait for the ``SettleDone`` event with
``Status == 0``.  Any protocol problem, RPC error, or timeout is
logged and reported as ``False`` — these methods never raise, so a
guiding hiccup degrades to "dither failed" rather than killing the
sequence.  (``connect()`` *does* raise :class:`PHD2Error`: failing to
reach PHD2 at all is a configuration problem the caller must know
about, mirroring ``INDIClient.connect``.)
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from typing import Any, Callable

log = logging.getLogger(__name__)


class PHD2Error(RuntimeError):
    """PHD2 connection or RPC failure."""


class PHD2Client:
    """Client for PHD2's JSON-over-TCP guiding API."""

    def __init__(self) -> None:
        self._sock: socket.socket | None = None
        self._reader: threading.Thread | None = None
        self._running = False
        self._send_lock = threading.Lock()
        self._id_lock = threading.Lock()
        self._next_id = 1
        self._pending: dict[int, dict[str, Any]] = {}
        self._pending_lock = threading.Lock()
        self._buf = b""
        self._handlers: dict[str, list[Callable[[dict], None]]] = {}

        # -- live state, updated by the background dispatch thread -------
        self.app_state: str | None = None
        self.last_settle: dict | None = None
        self.last_guide_step: dict | None = None
        self._settle_latch = threading.Event()
        self._star_lost = threading.Event()

        self.on("AppState", self._on_app_state)
        self.on("SettleDone", self._on_settle_done)
        self.on("StarLost", self._on_star_lost)
        self.on("GuideStep", self._on_guide_step)

    # -- event plumbing -------------------------------------------------
    def on(self, event: str, handler: Callable[[dict], None]) -> None:
        """Register ``handler(msg)`` for a PHD2 event name."""
        self._handlers.setdefault(event, []).append(handler)

    @property
    def star_lost(self) -> bool:
        """True since the last ``StarLost`` event (safety module polls this)."""
        return self._star_lost.is_set()

    def clear_star_lost(self) -> None:
        self._star_lost.clear()

    def _on_app_state(self, msg: dict) -> None:
        self.app_state = msg.get("State")

    def _on_settle_done(self, msg: dict) -> None:
        self.last_settle = dict(msg)
        self._settle_latch.set()

    def _on_star_lost(self, msg: dict) -> None:
        self._star_lost.set()
        log.warning("PHD2: star lost (frame %s)", msg.get("Frame"))

    def _on_guide_step(self, msg: dict) -> None:
        self.last_guide_step = dict(msg)

    # -- connection ------------------------------------------------------
    def connect(self, host: str = "localhost", port: int = 4400,
                timeout: float = 5.0) -> None:
        """Connect to PHD2. Raises PHD2Error if unreachable."""
        try:
            self._sock = socket.create_connection((host, port), timeout)
        except OSError as exc:
            raise PHD2Error(
                f"PHD2: cannot connect to {host}:{port} — is PHD2 running "
                f"with the server enabled (Tools > Enable Server)?"
            ) from exc
        self._running = True
        self._reader = threading.Thread(
            target=self._read_loop, daemon=True, name="phd2-reader"
        )
        self._reader.start()
        log.info("PHD2: connected to %s:%d", host, port)

    def close(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
            self._sock = None
        if self._reader is not None:
            self._reader.join(timeout=2.0)
            self._reader = None
        self._fail_all_pending()

    # -- RPC --------------------------------------------------------------
    def _rpc(self, method: str, params: list | None = None,
             timeout: float = 10.0) -> Any:
        with self._id_lock:
            rid = self._next_id
            self._next_id += 1
        request: dict[str, Any] = {"jsonrpc": "2.0", "id": rid,
                                   "method": method}
        if params is not None:
            request["params"] = params
        slot: dict[str, Any] = {"event": threading.Event(),
                                "result": None, "error": None}
        with self._pending_lock:
            self._pending[rid] = slot
        try:
            payload = (json.dumps(request) + "\n").encode("utf-8")
            with self._send_lock:
                if self._sock is None:
                    raise PHD2Error("PHD2: not connected")
                self._sock.sendall(payload)
        except OSError as exc:
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise PHD2Error(f"PHD2: send failed for {method!r}: {exc}") from exc
        if not slot["event"].wait(timeout):
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise PHD2Error(f"PHD2: RPC {method!r} timed out after {timeout}s")
        if slot["error"] is not None:
            raise PHD2Error(f"PHD2: RPC {method!r} error: {slot['error']}")
        return slot["result"]

    def _fail_all_pending(self) -> None:
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for slot in pending:
            slot["event"].set()  # waiters wake; missing result -> error path

    # -- reader ------------------------------------------------------------
    def _read_loop(self) -> None:
        try:
            while self._running and self._sock is not None:
                try:
                    chunk = self._sock.recv(65536)
                except OSError:
                    break
                if not chunk:
                    break  # server closed the connection
                self._buf += chunk
                while b"\n" in self._buf:
                    line, self._buf = self._buf.split(b"\n", 1)
                    line = line.strip()
                    if line:
                        self._dispatch(line)
        finally:
            self._running = False
            self._fail_all_pending()

    def _dispatch(self, raw: bytes) -> None:
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            log.warning("PHD2: dropping malformed message %.80r (%s)", raw, exc)
            return
        if not isinstance(msg, dict):
            log.warning("PHD2: dropping non-object message %.80r", raw)
            return
        if "Event" in msg:
            for handler in self._handlers.get(msg["Event"], []):
                try:
                    handler(msg)
                except Exception:  # noqa: BLE001 - a bad handler must not kill the reader
                    log.exception("PHD2: event handler failed for %s",
                                  msg.get("Event"))
        elif "id" in msg:
            with self._pending_lock:
                slot = self._pending.pop(msg["id"], None)
            if slot is not None:
                slot["result"] = msg.get("result")
                slot["error"] = msg.get("error")
                slot["event"].set()
            # else: late/duplicate reply after a timeout — ignore
        else:
            log.debug("PHD2: ignoring message without Event/id: %.80r", raw)

    # -- guiding API ---------------------------------------------------------
    def get_state(self) -> str:
        """Current PHD2 app state, e.g. "Guiding", "Paused", "Looping"."""
        return str(self._rpc("get_app_state", timeout=10.0))

    @property
    def is_guiding(self) -> bool:
        """True if PHD2 reports the "Guiding" state (polls the server)."""
        try:
            return self.get_state() == "Guiding"
        except PHD2Error as exc:
            log.debug("PHD2: is_guiding check failed: %s", exc)
            return False

    def start_guiding(
        self,
        settle_pixels: float = 1.5,
        settle_time_s: float = 10.0,
        timeout_s: float = 60.0,
    ) -> bool:
        """Start guiding and wait for the settle. Never raises."""
        try:
            self._settle_latch.clear()
            self.last_settle = None
            self._rpc(
                "guide",
                [{"pixels": settle_pixels, "time": settle_time_s,
                  "timeout": timeout_s}, False],
                timeout=10.0,
            )
            return self._wait_settle(timeout_s)
        except PHD2Error as exc:
            log.warning("PHD2: start_guiding failed: %s", exc)
            return False

    def stop_guiding(self) -> bool:
        """Stop guiding/capture. Never raises."""
        try:
            self._rpc("stop_capture", timeout=10.0)
            return True
        except PHD2Error as exc:
            log.warning("PHD2: stop_guiding failed: %s", exc)
            return False

    def dither(
        self,
        amount_px: float = 5.0,
        settle_pixels: float = 1.5,
        settle_time_s: float = 10.0,
        timeout_s: float = 60.0,
    ) -> bool:
        """Dither and wait for ``SettleDone`` with ``Status == 0``.

        Sends the ``dither`` RPC (amount, RA-only=false, settle params),
        then blocks up to ``timeout_s`` for the settle event.  Returns
        ``True`` only on a clean settle; timeouts, non-zero settle
        status, and protocol errors all log and return ``False``.
        """
        try:
            self._settle_latch.clear()
            self.last_settle = None
            self._rpc(
                "dither",
                [amount_px, False,
                 {"pixels": settle_pixels, "time": settle_time_s,
                  "timeout": timeout_s}],
                timeout=10.0,
            )
            return self._wait_settle(timeout_s)
        except PHD2Error as exc:
            log.warning("PHD2: dither failed: %s", exc)
            return False

    def _wait_settle(self, timeout_s: float) -> bool:
        if not self._settle_latch.wait(timeout_s):
            log.warning("PHD2: timed out after %.0fs waiting for SettleDone",
                        timeout_s)
            return False
        status = (self.last_settle or {}).get("Status")
        if status != 0:
            log.warning("PHD2: settle failed (SettleDone Status=%r)", status)
            return False
        log.info("PHD2: settle complete")
        return True
