"""Tests for the pure-Python INDI protocol client (drivers/indi.py).

A threaded fake INDI server speaks just enough of the XML protocol
(handshake, def vectors, command acks, slew Busy->Ok, exposure
Busy->Ok + BLOB) to exercise INDIMount / INDICamera end to end with no
real hardware and no compiled INDI bindings.
"""

import base64
import io
import socket
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from astropy.io import fits

from astrocapture.drivers import make_camera, make_mount
from astrocapture.drivers.indi import _split_elements


def _make_fits_bytes() -> tuple[bytes, np.ndarray]:
    data = (np.arange(64, dtype=np.uint16).reshape(8, 8) * 257) % 65535
    buf = io.BytesIO()
    fits.PrimaryHDU(data).writeto(buf)
    return buf.getvalue(), data


def _def(kind: str, device: str, name: str, items: list[tuple[str, str]],
         state: str = "Idle") -> str:
    """Build a defXXXVector element. items: (item_name, text_value)."""
    tag = {"text": "Text", "number": "Number",
           "switch": "Switch", "blob": "BLOB"}[kind]
    inner = "".join(f'<def{tag} name="{n}">{v}</def{tag}>' for n, v in items)
    return (f'<def{tag}Vector device="{device}" name="{name}" '
            f'state="{state}">{inner}</def{tag}Vector>')


class FakeINDIServer(threading.Thread):
    """Minimal scripted INDI server for protocol tests."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._send_lock = threading.Lock()
        self._conn: socket.socket | None = None
        self._stop_event = threading.Event()
        self.last_goto: tuple[float, float] | None = None
        self.last_park_switch: str | None = None
        self.last_serial_port: str | None = None
        self.last_exposure: float | None = None
        self.fits_bytes, self.fits_data = _make_fits_bytes()

    # -- helpers -------------------------------------------------------
    def _send(self, conn: socket.socket, xml: str) -> None:
        with self._send_lock:
            conn.sendall(xml.encode())

    def _defs(self, conn: socket.socket) -> None:
        d = "Fake Mount"
        self._send(conn, _def("switch", d, "CONNECTION",
                        [("CONNECT", "Off"), ("DISCONNECT", "On")]))
        self._send(conn, _def("text", d, "DEVICE_PORT", [("PORT", "")]))
        self._send(conn, _def("number", d, "EQUATORIAL_EOD_COORD",
                        [("RA", "0"), ("DEC", "0")]))
        self._send(conn, _def("switch", d, "ON_COORD_SET",
                        [("SLEW", "On"), ("SYNC", "Off"), ("TRACK", "Off")]))
        self._send(conn, _def("switch", d, "TELESCOPE_PARK",
                        [("PARK", "Off"), ("UNPARK", "On")]))
        self._send(conn, _def("switch", d, "TELESCOPE_TRACK_MODE",
                        [("TRACK_SIDEREAL", "On"), ("TRACK_OFF", "Off")]))
        c = "Fake Camera"
        self._send(conn, _def("switch", c, "CONNECTION",
                        [("CONNECT", "Off"), ("DISCONNECT", "On")]))
        self._send(conn, _def("switch", c, "UPLOAD_MODE",
                        [("UPLOAD_CLIENT", "On"), ("UPLOAD_LOCAL", "Off"),
                         ("UPLOAD_BOTH", "Off")]))
        self._send(conn, _def("number", c, "CCD_EXPOSURE", [("CCD_EXPOSURE", "0")]))
        self._send(conn, _def("number", c, "CCD_GAIN", [("GAIN", "0")]))
        self._send(conn, _def("blob", c, "CCD1", [("CCD1", "")]))

    def _echo_ok(self, el: ET.Element, conn: socket.socket) -> None:
        """Turn a newXXXVector into a setXXXVector with state Ok."""
        core = el.tag[3:]  # strip "new"
        if core.endswith("Vector"):
            core = core[: -len("Vector")]
        inner = "".join(ET.tostring(ch, encoding="unicode") for ch in el)
        # oneSwitch/oneNumber/oneText are valid in both new and set vectors
        self._send(
            conn,
            f'<set{core}Vector device="{el.get("device")}" '
            f'name="{el.get("name")}" state="Ok">{inner}</set{core}Vector>',
        )

    def _delayed(self, fn, delay: float) -> None:
        def run() -> None:
            time.sleep(delay)
            fn()
        threading.Thread(target=run, daemon=True).start()

    def _handle(self, xml: str, conn: socket.socket) -> None:
        el = ET.fromstring(xml)
        tag = el.tag
        device, name = el.get("device"), el.get("name")

        def items() -> dict[str, str]:
            return {ch.get("name", ""): (ch.text or "").strip() for ch in el}

        if tag == "getProperties":
            self._defs(conn)
        elif tag == "enableBLOB":
            pass  # nothing to ack
        elif tag == "newTextVector" and name == "DEVICE_PORT":
            self.last_serial_port = items().get("PORT")
            self._echo_ok(el, conn)
        elif tag == "newNumberVector" and name == "EQUATORIAL_EOD_COORD":
            vals = items()
            self.last_goto = (float(vals["RA"]), float(vals["DEC"]))
            inner = "".join(ET.tostring(ch, encoding="unicode") for ch in el)
            self._send(conn, f'<setNumberVector device="{device}" name="{name}" '
                             f'state="Busy">{inner}</setNumberVector>')
            self._delayed(
                lambda: self._send(
                    conn,
                    f'<setNumberVector device="{device}" name="{name}" '
                    f'state="Ok">{inner}</setNumberVector>'),
                0.3,
            )
        elif tag == "newSwitchVector" and name == "TELESCOPE_PARK":
            on = [n for n, v in items().items() if v == "On"]
            self.last_park_switch = on[0] if on else None
            self._echo_ok(el, conn)
        elif tag == "newNumberVector" and name == "CCD_EXPOSURE":
            self.last_exposure = float(items()["CCD_EXPOSURE"])
            inner = "".join(ET.tostring(ch, encoding="unicode") for ch in el)
            self._send(conn, f'<setNumberVector device="{device}" name="{name}" '
                             f'state="Busy">{inner}</setNumberVector>')

            def finish() -> None:
                b64 = base64.b64encode(self.fits_bytes).decode()
                self._send(conn, f'<setNumberVector device="{device}" name="{name}" '
                                 f'state="Ok">{inner}</setNumberVector>')
                self._send(
                    conn,
                    f'<setBLOBVector device="{device}" name="CCD1" state="Ok">'
                    f'<oneBLOB name="CCD1" size="{len(self.fits_bytes)}" '
                    f'format=".fits">{b64}</oneBLOB></setBLOBVector>')
            self._delayed(finish, 0.2)
        else:
            # Generic ack for everything else (CONNECTION, UPLOAD_MODE,
            # ON_COORD_SET, TRACK_MODE, CCD_GAIN, ...)
            self._echo_ok(el, conn)

    def _serve(self, conn: socket.socket) -> None:
        buffer = ""
        try:
            while not self._stop_event.is_set():
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buffer += chunk.decode()
                elements, buffer = _split_elements(buffer)
                for xml in elements:
                    self._handle(xml, conn)
        except OSError:
            pass
        finally:
            conn.close()

    def run(self) -> None:
        # One TCP connection per INDIClient (mount and camera each open
        # their own, and they stay connected simultaneously), so serve
        # every accepted connection on its own thread.
        self._sock.settimeout(0.5)
        while not self._stop_event.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._serve, args=(conn,),
                             daemon=True).start()
        self._sock.close()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()


@pytest.fixture()
def server():
    srv = FakeINDIServer()
    srv.start()
    time.sleep(0.1)  # let accept() get going
    yield srv
    srv.stop()


def _wait_until(fn, timeout: float = 5.0, poll: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if fn():
            return True
        time.sleep(poll)
    return False


def test_handshake_populates_properties(server):
    mount = make_mount("indi", host="127.0.0.1", port=server.port,
                       device="Fake Mount", serial_port="/dev/ttyUSB0")
    cam = make_camera("indi", host="127.0.0.1", port=server.port,
                      device="Fake Camera")
    try:
        mount.connect()
        cam.connect()
        assert mount._client.has_property("Fake Mount", "EQUATORIAL_EOD_COORD")
        assert mount._client.has_property("Fake Mount", "CONNECTION")
        assert cam._client.has_property("Fake Camera", "CCD_EXPOSURE")
        assert cam._client.has_property("Fake Camera", "CCD1")
        # serial port made it to the driver's DEVICE_PORT text property
        assert _wait_until(lambda: server.last_serial_port == "/dev/ttyUSB0")
        ra, dec = mount.position
        assert ra == pytest.approx(0.0) and dec == pytest.approx(0.0)
    finally:
        mount.disconnect()
        cam.disconnect()


def test_goto_sends_correct_ra_dec(server):
    mount = make_mount("indi", host="127.0.0.1", port=server.port,
                       device="Fake Mount")
    try:
        mount.connect()
        mount.goto(13.5, 47.25)
        assert _wait_until(lambda: server.last_goto == (13.5, 47.25))
    finally:
        mount.disconnect()


def test_slew_completes(server):
    mount = make_mount("indi", host="127.0.0.1", port=server.port,
                       device="Fake Mount")
    try:
        mount.connect()
        mount.goto(10.0, 20.0)
        assert _wait_until(mount.slew_complete, timeout=5.0), \
            "slew never reached Ok/Idle"
    finally:
        mount.disconnect()


def test_park_sends_park_switch(server):
    mount = make_mount("indi", host="127.0.0.1", port=server.port,
                       device="Fake Mount")
    try:
        mount.connect()
        mount.park()
        assert _wait_until(lambda: server.last_park_switch == "PARK")
        mount.unpark()
        assert _wait_until(lambda: server.last_park_switch == "UNPARK")
    finally:
        mount.disconnect()


def test_exposure_downloads_fits_bytes(server):
    cam = make_camera("indi", host="127.0.0.1", port=server.port,
                      device="Fake Camera")
    try:
        cam.connect()
        cam.set_exposure_settings(0.5, gain=1600)
        cam.start_exposure()
        assert _wait_until(lambda: server.last_exposure == 0.5)
        frame = cam.download_image()
        # Real FITS bytes arrived and decode to the exact frame we sent.
        assert cam._last_blob is not None
        assert cam._last_blob[1] == server.fits_bytes
        assert frame.shape == (8, 8)
        assert np.array_equal(frame, server.fits_data)
        assert cam.exposure_complete()
    finally:
        cam.disconnect()
