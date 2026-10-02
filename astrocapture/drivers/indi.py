"""INDI backends (``kind: indi``) — pure-Python INDI protocol client.

Talks to an ``indiserver`` process over TCP by speaking the INDI XML
protocol (v1.7) directly on a socket. There are deliberately **zero
compiled dependencies**: no PyIndi-Client, no SWIG bindings, nothing to
build — so this backend imports and runs on Linux, macOS (Intel and
Apple Silicon) and Windows with the stock ``requirements.txt``.

Recommended architecture: run ``indiserver`` on a small Linux box at the
scope (a Raspberry Pi is the classic choice)::

    indiserver indi_celestron_gps indi_gphoto_cc -p 7624 &

and run AstroCapture on whatever laptop you have, pointing the config at
the Pi's address::

    mount: {driver: indi, host: 192.168.1.50, port: 7624, device: "Celestron GPS"}

Protocol summary (verified against the INDI 1.7 spec, indilib.org):
the client opens a TCP connection and sends
``<getProperties version="1.7"/>``. The server replies with
``defTextVector`` / ``defNumberVector`` / ``defSwitchVector`` /
``defBLOBVector`` elements describing every property. The client
commands the device with ``newTextVector`` / ``newNumberVector`` /
``newSwitchVector``; the server answers with ``setXXXVector`` updates
carrying the property state (Idle/Ok/Busy/Alert). Image frames arrive as
``setBLOBVector`` with a ``oneBLOB`` child holding base64-encoded FITS,
but only after the client sends
``<enableBLOB device="..." name="CCD1">Also</enableBLOB>``.

INDI property names vary between drivers. When first light misbehaves,
run ``indi_getprop`` and compare the real property/switch names against
the ones used below — the known variation points are flagged in the
per-device notes (``_CELESTRON_GPS_NOTES``, ``_GPHOTO_NOTES``).
"""
from __future__ import annotations

import base64
import io
import logging
import re
import socket
import threading
import time
import xml.etree.ElementTree as ET
from xml.sax.saxutils import quoteattr

import numpy as np

from astrocapture.drivers.base import Camera, Mount, MountState

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-device notes — Mathias's gear (verified against INDI docs, Oct 2026).
# Doc, not code: INDI drivers evolve, so treat property names as "verify
# with indi_getprop on first light", not gospel.
# ---------------------------------------------------------------------------

_CELESTRON_GPS_NOTES = """
Celestron NexStar 6SE via ``indi_celestron_gps`` (device name: "Celestron GPS").

- Single fork-arm ALT-AZ GoTo; the driver covers the whole NexStar SE line.
- Physical link: USB cable (mini-USB) or a USB-to-serial adapter into the
  port on the BASE of the NexStar+ hand controller, 9600 baud. On
  Debian/Ubuntu the user must be in the ``dialout`` group or the serial
  device will be permission-denied. Pass the port as
  ``serial_port: /dev/ttyUSB0`` — INDIMount sets the driver's DEVICE_PORT
  text property before CONNECT (find yours with ``ls /dev/ttyUSB*``).
- ALIGN FIRST FROM THE HAND CONTROLLER (SkyAlign / auto two-star) with the
  mount powered on. INDI cannot perform the initial star alignment; if you
  connect unaligned, gotos will miss by degrees.
- Capabilities: goto / slew / sync / parking / pulse-guiding. Tracking modes
  include Alt/Az — look for the switch name under TELESCOPE_TRACK_MODE with
  indi_getprop (expect something like TRACK_ALTAZ rather than TRACK_SIDEREAL;
  the default mode is usually correct for an aligned alt-az mount, so don't
  force sidereal).
- Goto still uses EQUATORIAL_EOD_COORD (the driver converts to alt-az
  internally). Slew completion is read from that property's state, as usual.
- Alt-az caveat: the sky rotates in the frame (field rotation). Without an
  equatorial wedge, keep subs to ~20-30 s depending on target declination.
"""

_GPHOTO_NOTES = """
Canon EOS Rebel T7i (800D) via ``indi_gphoto_cc`` (device name: "Canon DSLR").

- libgphoto2 supports the T7i; INDI wraps it. ISO, shutter speed, and bulb
  exposures are controllable over USB. For >30 s the driver handles bulb
  internally — prefer this over the direct-gphoto2 backend.
- ISO arrives as a text/switch property (INDICamera tries "CCD_ISO", "ISO");
  confirm the real name with indi_getprop if ISO changes don't take.
- Frames arrive as the CCD1 BLOB in FITS (download_image decodes it; falls
  back to the raw uint16 buffer). Keep upload_mode: client so frames stream
  to the capture machine.
- Camera-side setup (see also the T7I profile in drivers/dslr.py): Manual
  (M) mode, manual focus, RAW, auto power-off OFF, mirror lockup ON for
  exposures, long-exposure NR OFF (you take darks). Mechanical: T-ring +
  1.25" nosepiece or SCT T-adapter on the 6SE's rear cell.
"""

# ---------------------------------------------------------------------------
# Incremental XML extraction
# ---------------------------------------------------------------------------

_OPEN_TAG = re.compile(r"<([A-Za-z_][\w.\-]*)")


def _split_elements(buffer: str) -> tuple[list[str], str]:
    """Pull complete top-level XML elements off the front of *buffer*.

    Returns (elements, remainder). Handles self-closing tags and large
    payloads (BLOBs) — the buffer just keeps growing until the matching
    close tag arrives. Base64 never contains '<', so naive close-tag
    matching is safe here.
    """
    elements: list[str] = []
    while True:
        m = _OPEN_TAG.search(buffer)
        if m is None:
            return elements, ""
        tag = m.group(1)
        start = m.start()
        gt = buffer.find(">", m.end())
        if gt == -1:
            return elements, buffer[start:]
        if buffer[gt - 1] == "/":  # self-closing, e.g. <getProperties .../>
            elements.append(buffer[start : gt + 1])
            buffer = buffer[gt + 1 :]
            continue
        close = buffer.find(f"</{tag}>", gt)
        if close == -1:
            return elements, buffer[start:]
        end = close + len(tag) + 3
        elements.append(buffer[start:end])
        buffer = buffer[end:]


# INDI property states
IDLE, OK, BUSY, ALERT = "Idle", "Ok", "Busy", "Alert"


class INDIError(RuntimeError):
    """Raised for INDI-level failures (alert state, missing device/property)."""


# ---------------------------------------------------------------------------
# INDIClient — raw protocol client
# ---------------------------------------------------------------------------


class INDIClient:
    """Minimal INDI v1.7 client: socket + background reader + property store.

    The store is keyed by ``(device, property_name)``; each entry is a dict
    with ``kind`` (text/number/switch/blob), ``state``, ``perm`` and
    ``items`` (``{item_name: value}``). BLOB payloads are kept separately in
    ``_blobs`` as ``(format, bytes)`` with a per-property event so
    ``wait_blob`` only returns *new* frames.
    """

    def __init__(self, host: str = "localhost", port: int = 7624) -> None:
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()
        self._props: dict[tuple[str, str], dict] = {}
        self._blobs: dict[tuple[str, str], tuple[str, bytes]] = {}
        self._blob_events: dict[tuple[str, str], threading.Event] = {}
        self._running = False

    # -- connection ------------------------------------------------------
    def connect(self, timeout: float = 10.0) -> None:
        try:
            self._sock = socket.create_connection((self.host, self.port), timeout)
        except OSError as exc:
            raise INDIError(
                f"INDI: could not connect to {self.host}:{self.port} "
                f"(is indiserver running there?)"
            ) from exc
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._send('<getProperties version="1.7"/>')

    def disconnect(self) -> None:
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

    # -- low-level send ---------------------------------------------------
    def _send(self, xml: str) -> None:
        assert self._sock is not None, "INDIClient not connected"
        log.debug("INDI -> %s", xml[:200])
        self._sock.sendall(xml.encode("utf-8"))

    def send_switch(self, device: str, prop: str, items: dict[str, str]) -> None:
        inner = "".join(
            f'<oneSwitch name={quoteattr(n)}>{v}</oneSwitch>'
            for n, v in items.items()
        )
        self._send(
            f"<newSwitchVector device={quoteattr(device)} "
            f"name={quoteattr(prop)}>{inner}</newSwitchVector>"
        )

    def send_number(self, device: str, prop: str, items: dict[str, float]) -> None:
        inner = "".join(
            f"<oneNumber name={quoteattr(n)}>{v}</oneNumber>"
            for n, v in items.items()
        )
        self._send(
            f"<newNumberVector device={quoteattr(device)} "
            f"name={quoteattr(prop)}>{inner}</newNumberVector>"
        )

    def send_text(self, device: str, prop: str, items: dict[str, str]) -> None:
        inner = "".join(
            f"<oneText name={quoteattr(n)}>{v}</oneText>" for n, v in items.items()
        )
        self._send(
            f"<newTextVector device={quoteattr(device)} "
            f"name={quoteattr(prop)}>{inner}</newTextVector>"
        )

    def enable_blob(self, device: str, name: str = "CCD1", mode: str = "Also") -> None:
        """Ask the server to stream BLOBs for *name* (Also = mixed with XML)."""
        self._send(
            f"<enableBLOB device={quoteattr(device)} "
            f"name={quoteattr(name)}>{mode}</enableBLOB>"
        )

    # -- reader -----------------------------------------------------------
    def _read_loop(self) -> None:
        buffer = ""
        try:
            while self._running and self._sock is not None:
                chunk = self._sock.recv(65536)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")
                elements, buffer = _split_elements(buffer)
                for xml in elements:
                    self._handle(xml)
        except OSError:
            pass  # socket closed by disconnect()
        finally:
            self._running = False

    def _handle(self, xml: str) -> None:
        try:
            el = ET.fromstring(xml)
        except ET.ParseError:
            log.warning("INDI: dropping unparsable element: %.80s", xml)
            return
        tag = el.tag
        if tag in ("message", "switchProtocol"):
            return
        if tag == "delProperty":
            device = el.get("device", "")
            name = el.get("name")
            with self._lock:
                if name:
                    self._props.pop((device, name), None)
                else:
                    for key in [k for k in self._props if k[0] == device]:
                        del self._props[key]
            return
        if not (tag.startswith("def") or tag.startswith("set")):
            return  # getProperties echoes etc.; ignore
        device = el.get("device", "")
        name = el.get("name", "")
        core = tag[3:]  # strip "def"/"set"
        if core.endswith("Vector"):
            core = core[: -len("Vector")]
        kind = {
            "Text": "text", "Number": "number", "Switch": "switch", "BLOB": "blob"
        }.get(core, "")
        if not kind:
            return
        state = el.get("state", IDLE)
        items: dict[str, str | float] = {}
        for child in el:
            iname = child.get("name", "")
            if kind == "number":
                try:
                    items[iname] = float((child.text or "0").strip())
                except ValueError:
                    items[iname] = 0.0
            elif kind == "blob":
                size = int(child.get("size", "0") or 0)
                fmt = child.get("format", "")
                raw = re.sub(r"\s+", "", child.text or "")
                try:
                    data = base64.b64decode(raw)
                except Exception:
                    log.warning("INDI: bad base64 in BLOB %s/%s", device, name)
                    continue
                if size and len(data) != size:
                    log.warning(
                        "INDI: BLOB size mismatch %s/%s: header %d, got %d",
                        device, name, size, len(data),
                    )
                key = (device, name)
                with self._lock:
                    self._blobs[key] = (fmt, data)
                    self._blob_events.setdefault(key, threading.Event()).set()
                items[iname] = fmt
            else:
                items[iname] = (child.text or "").strip()
        with self._lock:
            entry = self._props.get((device, name))
            if entry is None or tag.startswith("def"):
                self._props[(device, name)] = {
                    "kind": kind,
                    "state": state,
                    "perm": el.get("perm", ""),
                    "items": dict(items),
                }
            else:  # setXXXVector: merge items, update state
                entry["state"] = state
                entry["items"].update(items)

    # -- queries / waiters -------------------------------------------------
    def has_device(self, device: str) -> bool:
        with self._lock:
            return any(d == device for d, _ in self._props)

    def has_property(self, device: str, name: str) -> bool:
        with self._lock:
            return (device, name) in self._props

    def state_of(self, device: str, name: str) -> str:
        with self._lock:
            entry = self._props.get((device, name))
            return entry["state"] if entry else ""

    def items_of(self, device: str, name: str) -> dict:
        with self._lock:
            entry = self._props.get((device, name))
            return dict(entry["items"]) if entry else {}

    def wait_for_device(self, device: str, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.has_device(device):
                return
            time.sleep(0.05)
        raise INDIError(f"INDI: device {device!r} not seen on {self.host}:{self.port}")

    def wait_for_state(
        self,
        device: str,
        name: str,
        states: tuple[str, ...] = (OK, IDLE),
        timeout: float = 15.0,
    ) -> None:
        """Wait until a property's state is one of *states*.

        Raises INDIError immediately if the property goes to Alert.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.state_of(device, name)
            if state in states:
                return
            if state == ALERT:
                raise INDIError(f"INDI {device}: property {name} entered ALERT state")
            time.sleep(0.05)
        raise TimeoutError(
            f"INDI {device}: timed out ({timeout}s) waiting for {name} "
            f"to reach {states}; last state={self.state_of(device, name)!r}"
        )

    def clear_blob(self, device: str, name: str) -> None:
        with self._lock:
            self._blobs.pop((device, name), None)
            self._blob_events.setdefault((device, name), threading.Event()).clear()

    def wait_blob(
        self, device: str, name: str, timeout: float = 60.0
    ) -> tuple[str, bytes]:
        """Wait for the next BLOB frame; returns (format, raw_bytes)."""
        key = (device, name)
        with self._lock:
            event = self._blob_events.setdefault(key, threading.Event())
        if not event.wait(timeout):
            raise TimeoutError(f"INDI {device}: no BLOB {name} arrived in {timeout}s")
        with self._lock:
            return self._blobs[key]


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


class _INDIBase:
    def __init__(self, host: str = "localhost", port: int = 7624, device: str = "") -> None:
        if not device:
            raise ValueError("indi backend needs a device name, e.g. 'EQMod Mount'")
        self.host = host
        self.port = port
        self.device_name = device
        self._client = INDIClient(host, port)

    def _connect_device(self) -> None:
        self._client.connect()
        self._client.wait_for_device(self.device_name, timeout=10)

    def _first_item(self, prop: str) -> str:
        items = self._client.items_of(self.device_name, prop)
        if not items:
            raise INDIError(f"INDI {self.device_name}: {prop} has no items")
        return next(iter(items))

    def _set_switch_guarded(self, prop: str, on_name: str) -> None:
        """Flip a switch vector; silently skip if the driver lacks it.

        Many INDI switches are optional (e.g. UPLOAD_MODE, TELESCOPE_SLEW
        on drivers that slew on any coord write). Guessing wrong here
        should not kill a session — the required core properties raise.
        """
        if not self._client.has_property(self.device_name, prop):
            return
        items = self._client.items_of(self.device_name, prop)
        if on_name not in items:
            raise INDIError(
                f"INDI {self.device_name}: {prop} has no switch {on_name!r} "
                f"(has: {sorted(items)})"
            )
        self._client.send_switch(
            self.device_name, prop,
            {n: ("On" if n == on_name else "Off") for n in items},
        )

    def _disconnect_device(self) -> None:
        try:
            self._set_switch_guarded("CONNECTION", "DISCONNECT")
        finally:
            self._client.disconnect()


class INDIMount(_INDIBase, Mount):
    """Equatorial mount via INDI (e.g. ``indi_eqmod_telescope``).

    Also drives alt-az GoTo mounts such as the Celestron NexStar 6SE via
    ``indi_celestron_gps`` — see ``_CELESTRON_GPS_NOTES`` above for the
    alignment and serial-port gotchas. Pass ``serial_port="/dev/ttyUSB0"``
    to set the driver's CONNECTION-tab port before connecting.
    """

    def __init__(self, serial_port: str = "", **kwargs) -> None:
        _INDIBase.__init__(self, **kwargs)
        self._serial_port = serial_port
        self._tracking_enabled = True

    def connect(self) -> None:
        self._connect_device()
        if self._serial_port and self._client.has_property(
            self.device_name, "DEVICE_PORT"
        ):
            # Standard INDI Connection-tab text property; not every serial
            # driver exposes it under this exact name — verify with
            # indi_getprop if the mount won't connect.
            self._client.send_text(
                self.device_name, "DEVICE_PORT",
                {self._first_item("DEVICE_PORT"): self._serial_port},
            )
        self._set_switch_guarded("CONNECTION", "CONNECT")
        self._client.wait_for_state(self.device_name, "CONNECTION", (OK,))

    def disconnect(self) -> None:
        self._disconnect_device()

    def unpark(self) -> None:
        self._set_switch_guarded("TELESCOPE_PARK", "UNPARK")
        self._client.wait_for_state(self.device_name, "TELESCOPE_PARK", (OK,))

    def park(self) -> None:
        self._set_switch_guarded("TELESCOPE_PARK", "PARK")
        self._client.wait_for_state(self.device_name, "TELESCOPE_PARK", (OK,),
                                    timeout=120)

    def goto(self, ra_hours: float, dec_deg: float) -> None:
        # Slew (not sync): ON_COORD_SET on most drivers, TELESCOPE_SLEW on
        # some. Both are optional — drivers like indi_celestron_gps slew on
        # any coord write, so missing switches are skipped, not fatal.
        for cand in ("ON_COORD_SET", "TELESCOPE_SLEW"):
            if self._client.has_property(self.device_name, cand):
                items = self._client.items_of(self.device_name, cand)
                if "SLEW" in items:
                    self._set_switch_guarded(cand, "SLEW")
                break
        # Item names are RA/DEC on every telescope driver seen so far, but
        # read them from the def vector rather than assuming.
        names = list(self._client.items_of(self.device_name,
                                           "EQUATORIAL_EOD_COORD"))
        if len(names) < 2:
            raise INDIError(
                f"INDI {self.device_name}: EQUATORIAL_EOD_COORD has "
                f"unexpected items {names} — check with indi_getprop"
            )
        self._client.send_number(
            self.device_name, "EQUATORIAL_EOD_COORD",
            {names[0]: ra_hours, names[1]: dec_deg},
        )

    def slew_complete(self) -> bool:
        return self._client.state_of(
            self.device_name, "EQUATORIAL_EOD_COORD"
        ) in (OK, IDLE)

    def start_tracking(self) -> None:
        # Alt-az note: the correct tracking mode for an aligned alt-az
        # mount is usually already the driver default — only force
        # sidereal if the driver actually exposes this property.
        self._set_switch_guarded("TELESCOPE_TRACK_MODE", "TRACK_SIDEREAL")
        self._tracking_enabled = True

    def stop_tracking(self) -> None:
        self._set_switch_guarded("TELESCOPE_TRACK_MODE", "TRACK_OFF")
        self._tracking_enabled = False

    @property
    def state(self) -> MountState:
        s = self._client.state_of(self.device_name, "EQUATORIAL_EOD_COORD")
        if s == BUSY:
            return MountState.SLEWING
        if s == ALERT:
            return MountState.ERROR
        if self._client.has_property(self.device_name, "TELESCOPE_PARK"):
            items = self._client.items_of(self.device_name, "TELESCOPE_PARK")
            if items.get("PARK") == "On":
                return MountState.PARKED
        return MountState.TRACKING if self._tracking_enabled else MountState.IDLE

    @property
    def position(self) -> tuple[float, float]:
        items = self._client.items_of(self.device_name, "EQUATORIAL_EOD_COORD")
        names = list(items)
        if len(names) < 2:
            raise INDIError(
                f"INDI {self.device_name}: cannot read position from "
                f"EQUATORIAL_EOD_COORD items {names}"
            )
        return float(items[names[0]]), float(items[names[1]])


class INDICamera(_INDIBase, Camera):
    """Camera via INDI (e.g. ``indi_gphoto_cc`` for DSLRs, ``indi_asi_ccd``).

    For DSLR specifics (Canon Rebel T7i via ``indi_gphoto_cc``) see
    ``_GPHOTO_NOTES`` above and the ``T7I_PROFILE`` in ``drivers/dslr.py``.
    """

    def __init__(self, upload_mode: str = "client", blob_property: str = "CCD1",
                 **kwargs) -> None:
        _INDIBase.__init__(self, **kwargs)
        self._upload_mode = upload_mode
        self._blob_property = blob_property
        self._exptime = 1.0
        self._cooler_supported = False
        self._last_blob: tuple[str, bytes] | None = None  # test hook

    def connect(self) -> None:
        self._connect_device()
        self._set_switch_guarded("CONNECTION", "CONNECT")
        self._client.wait_for_state(self.device_name, "CONNECTION", (OK,))
        self._set_switch_guarded(
            "UPLOAD_MODE", f"UPLOAD_{self._upload_mode.upper()}"
        )
        # Without this the server never sends setBLOBVector frames.
        self._client.enable_blob(self.device_name, self._blob_property, "Also")
        self._cooler_supported = self._client.has_property(
            self.device_name, "CCD_COOLER"
        )

    def disconnect(self) -> None:
        self._disconnect_device()

    def set_exposure_settings(
        self, exptime_s: float, gain: float = 0.0, binning: int = 1
    ) -> None:
        if exptime_s <= 0:
            raise ValueError("exposure time must be positive")
        self._exptime = exptime_s
        # Gain: try common property names across drivers.
        for prop in ("CCD_GAIN", "GAIN"):
            if self._client.has_property(self.device_name, prop):
                self._client.send_number(
                    self.device_name, prop,
                    {self._first_item(prop): gain},
                )
                break
        # DSLRs via gphoto: ISO lives in a text property. Driver naming
        # varies ("CCD_ISO", "ISO", sometimes a switch) — verify with
        # indi_getprop if ISO changes don't take.
        if gain:
            for prop in ("CCD_ISO", "ISO"):
                if self._client.has_property(self.device_name, prop):
                    self._client.send_text(
                        self.device_name, prop,
                        {self._first_item(prop): str(int(gain))},
                    )
                    break
        if binning != 1 and self._client.has_property(
            self.device_name, "CCD_BINNING"
        ):
            names = list(
                self._client.items_of(self.device_name, "CCD_BINNING")
            )
            self._client.send_number(
                self.device_name, "CCD_BINNING",
                {n: float(binning) for n in names[:2] or names},
            )

    def start_exposure(self) -> None:
        # Drop any stale frame so wait_blob only returns the new one.
        self._client.clear_blob(self.device_name, self._blob_property)
        self._client.send_number(
            self.device_name, "CCD_EXPOSURE",
            {self._first_item("CCD_EXPOSURE"): self._exptime},
        )

    def exposure_complete(self) -> bool:
        return self._client.state_of(
            self.device_name, "CCD_EXPOSURE"
        ) != BUSY

    def abort_exposure(self) -> None:
        self._set_switch_guarded("CCD_ABORT_EXPOSURE", "ABORT")

    def download_image(self) -> np.ndarray:
        fmt, data = self._client.wait_blob(
            self.device_name, self._blob_property, timeout=self._exptime + 60
        )
        self._last_blob = (fmt, data)
        if fmt.lower() in (".fits", "fits", ".fit", "fit"):
            from astropy.io import fits

            with fits.open(io.BytesIO(data)) as hdul:
                return np.asarray(hdul[0].data)
        # Fallback: raw frame buffer (uint16, row-major).
        arr = np.frombuffer(data, dtype=np.uint16)
        frame = self._client.items_of(self.device_name, "CCD_FRAME")
        w = int(frame.get("WIDTH", 0) or 0)
        h = int(frame.get("HEIGHT", 0) or 0)
        if w and h and arr.size >= w * h:
            return arr[: w * h].reshape(h, w)
        return arr

    @property
    def has_cooler(self) -> bool:
        return self._cooler_supported

    def set_cooler(self, temperature_c: float) -> None:
        if not self._cooler_supported:
            raise NotImplementedError("INDI camera has no cooler switch")
        self._set_switch_guarded("CCD_COOLER", "COOLER_ON")
        self._client.send_number(
            self.device_name, "CCD_TEMPERATURE",
            {self._first_item("CCD_TEMPERATURE"): temperature_c},
        )

    def get_temperature(self) -> float | None:
        if self._client.has_property(self.device_name, "CCD_TEMPERATURE"):
            items = self._client.items_of(self.device_name, "CCD_TEMPERATURE")
            return float(next(iter(items.values())))
        return None
