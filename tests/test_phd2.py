"""PHD2Client against a scripted fake PHD2 server (threaded socket)."""

import json
import socket
import threading
import time

import pytest

from astrocapture.phd2 import PHD2Client, PHD2Error


class FakePHD2Server:
    """Minimal scripted PHD2: answers RPCs, emits scripted events."""

    def __init__(self, app_state="Guiding", settle_status=0,
                 send_settle=True, extra_events=()):
        self.app_state = app_state
        self.settle_status = settle_status
        self.send_settle = send_settle
        self.extra_events = list(extra_events)
        self.requests: list = []
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        try:
            self._sock.close()
        except OSError:
            pass

    def _send(self, conn, obj):
        conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))

    def _serve(self):
        try:
            conn, _ = self._sock.accept()
        except OSError:
            return
        with conn:
            for event in self.extra_events:  # e.g. StarLost right on connect
                self._send(conn, event)
            f = conn.makefile("r")
            for line in f:
                try:
                    req = json.loads(line)
                except ValueError:
                    continue
                self.requests.append(req)
                method, rid = req.get("method"), req.get("id")
                if method == "get_app_state":
                    self._send(conn, {"jsonrpc": "2.0", "result": self.app_state,
                                      "id": rid})
                elif method in ("dither", "guide"):
                    self._send(conn, {"jsonrpc": "2.0", "result": 0, "id": rid})
                    if self.send_settle:
                        time.sleep(0.05)
                        self._send(conn, {"Event": "SettleDone", "Timestamp": 1.0,
                                          "Status": self.settle_status})
                elif method == "stop_capture":
                    self._send(conn, {"jsonrpc": "2.0", "result": 0, "id": rid})
                else:
                    self._send(conn, {"jsonrpc": "2.0",
                                      "error": {"code": -32601,
                                                "message": "unknown method"},
                                      "id": rid})


def make_client(server):
    client = PHD2Client()
    client.connect("127.0.0.1", server.port)
    return client


def test_dither_success_on_clean_settle():
    server = FakePHD2Server(settle_status=0).start()
    client = make_client(server)
    try:
        assert client.dither(amount_px=5.0, timeout_s=5.0) is True
        assert client.last_settle["Status"] == 0
        req = next(r for r in server.requests if r["method"] == "dither")
        assert req["params"][0] == 5.0
        assert req["params"][2]["pixels"] == 1.5
    finally:
        client.close()
        server.stop()


def test_dither_false_on_bad_settle_status():
    server = FakePHD2Server(settle_status=2).start()
    client = make_client(server)
    try:
        assert client.dither(timeout_s=5.0) is False
        assert client.last_settle["Status"] == 2
    finally:
        client.close()
        server.stop()


def test_dither_false_on_settle_timeout():
    server = FakePHD2Server(send_settle=False).start()
    client = make_client(server)
    try:
        assert client.dither(timeout_s=0.5) is False
    finally:
        client.close()
        server.stop()


def test_star_lost_flag_set_and_cleared():
    server = FakePHD2Server(extra_events=[
        {"Event": "StarLost", "Timestamp": 2.0, "Frame": 418},
    ]).start()
    client = make_client(server)
    try:
        deadline = time.monotonic() + 5.0
        while not client.star_lost and time.monotonic() < deadline:
            time.sleep(0.05)
        assert client.star_lost is True
        client.clear_star_lost()
        assert client.star_lost is False
    finally:
        client.close()
        server.stop()


def test_get_state_and_is_guiding():
    server = FakePHD2Server(app_state="Guiding").start()
    client = make_client(server)
    try:
        assert client.get_state() == "Guiding"
        assert client.is_guiding is True
    finally:
        client.close()
        server.stop()


def test_connect_refused_raises():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    with pytest.raises(PHD2Error):
        PHD2Client().connect("127.0.0.1", port, timeout=1.0)
