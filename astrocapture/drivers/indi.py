"""INDI backends (``kind: indi``).

Talks to an ``indiserver`` process over TCP using the ``PyIndi-Client``
package (``pip install PyIndi-Client``).  The import is guarded: this
module raises a clear error at *construction* time if the package is
missing, so ``import astrocapture`` always works.

Typical setup on Ubuntu::

    sudo apt install indi-bin
    indiserver indi_eqmod_telescope indi_gphoto_cc -p 7624 &

then in the plan config::

    mount: {driver: indi, host: localhost, port: 7624, device: "EQMod Mount"}
    camera: {driver: indi, host: localhost, port: 7624, device: "Canon DSLR"}

INDI property names vary between drivers. When first light misbehaves,
run ``indi_getprop`` and compare the real property/switch names against
the ones used below — the known variation points are flagged in the
per-device notes (``_CELESTRON_GPS_NOTES``, ``_GPHOTO_NOTES``).
"""
from __future__ import annotations

import time

import numpy as np

from astrocapture.drivers.base import Camera, Mount, MountState

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

try:  # optional dependency, guarded
    import PyIndi  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - needs PyIndi installed
    raise ImportError(
        "The INDI backend needs the PyIndi-Client package: "
        "pip install PyIndi-Client (and a running indiserver)."
    ) from exc


class _INDIBase:
    def __init__(self, host: str = "localhost", port: int = 7624, device: str = "") -> None:
        if not device:
            raise ValueError("indi backend needs a device name, e.g. 'EQMod Mount'")
        self.host = host
        self.port = port
        self.device_name = device
        self._indiclient = PyIndi.BaseClient()
        self._device: PyIndi.Device | None = None

    # -- plumbing -------------------------------------------------------
    def _connect_client(self) -> None:
        self._indiclient.setServer(self.host, self.port)
        if not self._indiclient.connectServer():
            raise RuntimeError(f"INDI: could not connect to {self.host}:{self.port}")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            dev = self._indiclient.getDevice(self.device_name)
            if dev is not None:
                self._device = dev
                return
            time.sleep(0.2)
        raise RuntimeError(
            f"INDI: device {self.device_name!r} not found on {self.host}:{self.port}"
        )

    def _number(self, prop: str) -> PyIndi.PropertyNumber:
        vec = self._device.getNumber(prop)
        if vec is None:
            raise RuntimeError(f"INDI {self.device_name}: no number property {prop!r}")
        return vec

    def _switch(self, prop: str) -> PyIndi.PropertySwitch:
        vec = self._device.getSwitch(prop)
        if vec is None:
            raise RuntimeError(f"INDI {self.device_name}: no switch property {prop!r}")
        return vec

    def _text(self, prop: str) -> PyIndi.PropertyText:
        vec = self._device.getText(prop)
        if vec is None:
            raise RuntimeError(f"INDI {self.device_name}: no text property {prop!r}")
        return vec

    def _blob(self, prop: str) -> PyIndi.PropertyBlob:
        vec = self._device.getBLOB(prop)
        if vec is None:
            raise RuntimeError(f"INDI {self.device_name}: no BLOB property {prop!r}")
        return vec

    def _wait_ok(self, prop_getter, timeout: float = 15.0, poll: float = 0.2) -> None:
        """Wait until a property's state is OK/IDLE (not ALERT/BUSY)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = prop_getter().getState()
            if state in (PyIndi.IPS_OK, PyIndi.IPS_IDLE):
                return
            if state == PyIndi.IPS_ALERT:
                raise RuntimeError(
                    f"INDI {self.device_name}: property entered ALERT state"
                )
            time.sleep(poll)
        raise TimeoutError(f"INDI {self.device_name}: timed out waiting for property")

    def _set_switch(self, prop: str, on_name: str, timeout: float = 15.0) -> None:
        vec = self._switch(prop)
        found = False
        for sw in vec:
            sw.s = PyIndi.ISS_ON if sw.name == on_name else PyIndi.ISS_OFF
            found = found or sw.name == on_name
        if not found:
            raise ValueError(f"INDI {self.device_name}: {prop} has no switch {on_name!r}")
        self._indiclient.sendNewSwitch(vec)
        self._wait_ok(lambda: self._switch(prop), timeout)


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
        self._connect_client()
        if self._serial_port:
            # Standard INDI Connection-tab text property; not every serial
            # driver exposes it under this exact name — verify with
            # indi_getprop if the mount won't connect.
            try:
                vec = self._text("DEVICE_PORT")
                vec[0].text = self._serial_port
                self._indiclient.sendNewText(vec)
            except RuntimeError:
                pass  # driver has no DEVICE_PORT; configure it by hand
        self._set_switch("CONNECTION", "CONNECT")

    def disconnect(self) -> None:
        try:
            self._set_switch("CONNECTION", "DISCONNECT", timeout=5)
        finally:
            self._indiclient.disconnectServer()

    def unpark(self) -> None:
        self._set_switch("TELESCOPE_PARK", "UNPARK")

    def park(self) -> None:
        self._set_switch("TELESCOPE_PARK", "PARK", timeout=120)

    def goto(self, ra_hours: float, dec_deg: float) -> None:
        coord = self._number("EQUATORIAL_EOD_COORD")
        coord[0].value = ra_hours
        coord[1].value = dec_deg
        # Tracking mode: slew (not sync) — find the slew switch by name.
        for cand in ("TELESCOPE_SLEW", "ON_COORD_SET"):
            try:
                self._set_switch(cand, "SLEW", timeout=5)
                break
            except (RuntimeError, ValueError):
                continue
        self._indiclient.sendNewNumber(coord)

    def slew_complete(self) -> bool:
        state = self._number("EQUATORIAL_EOD_COORD").getState()
        return state in (PyIndi.IPS_OK, PyIndi.IPS_IDLE)

    def start_tracking(self) -> None:
        self._set_switch("TELESCOPE_TRACK_MODE", "TRACK_SIDEREAL", timeout=5)
        self._tracking_enabled = True

    def stop_tracking(self) -> None:
        try:
            self._set_switch("TELESCOPE_TRACK_MODE", "TRACK_OFF", timeout=5)
        except (RuntimeError, ValueError):
            pass
        self._tracking_enabled = False

    @property
    def state(self) -> MountState:
        coord = self._number("EQUATORIAL_EOD_COORD")
        s = coord.getState()
        if s == PyIndi.IPS_BUSY:
            return MountState.SLEWING
        if s == PyIndi.IPS_ALERT:
            return MountState.ERROR
        try:
            park = self._switch("TELESCOPE_PARK")
            if any(sw.name == "PARK" and sw.s == PyIndi.ISS_ON for sw in park):
                return MountState.PARKED
        except RuntimeError:
            pass
        return MountState.TRACKING if self._tracking_enabled else MountState.IDLE

    @property
    def position(self) -> tuple[float, float]:
        coord = self._number("EQUATORIAL_EOD_COORD")
        return float(coord[0].value), float(coord[1].value)


class INDICamera(_INDIBase, Camera):
    """Camera via INDI (e.g. ``indi_gphoto_cc`` for DSLRs, ``indi_asi_ccd``).

    For DSLR specifics (Canon Rebel T7i via ``indi_gphoto_cc``) see
    ``_GPHOTO_NOTES`` above and the ``T7I_PROFILE`` in ``drivers/dslr.py``.
    """

    def __init__(self, upload_mode: str = "client", **kwargs) -> None:
        _INDIBase.__init__(self, **kwargs)
        self._upload_mode = upload_mode
        self._exptime = 1.0
        self._cooler_supported = False

    def connect(self) -> None:
        self._connect_client()
        self._set_switch("CONNECTION", "CONNECT")
        try:
            self._set_switch("UPLOAD_MODE", f"UPLOAD_{self._upload_mode.upper()}", timeout=5)
        except (RuntimeError, ValueError):
            pass  # driver without upload-mode switch; server default applies
        try:
            self._switch("CCD_COOLER")
            self._cooler_supported = True
        except RuntimeError:
            self._cooler_supported = False

    def disconnect(self) -> None:
        try:
            self._set_switch("CONNECTION", "DISCONNECT", timeout=5)
        finally:
            self._indiclient.disconnectServer()

    def set_exposure_settings(
        self, exptime_s: float, gain: float = 0.0, binning: int = 1
    ) -> None:
        if exptime_s <= 0:
            raise ValueError("exposure time must be positive")
        self._exptime = exptime_s
        # Gain: try common property names across drivers.
        for prop in ("CCD_GAIN", "GAIN"):
            try:
                vec = self._number(prop)
                vec[0].value = gain
                self._indiclient.sendNewNumber(vec)
                break
            except RuntimeError:
                continue
        # DSLRs via gphoto: ISO lives in a text/switch property.
        if gain:
            for prop in ("CCD_ISO", "ISO"):
                try:
                    txt = self._text(prop)
                    txt[0].text = str(int(gain))
                    self._indiclient.sendNewText(txt)
                    break
                except RuntimeError:
                    continue
        if binning != 1:
            try:
                vec = self._number("CCD_BINNING")
                vec[0].value = binning
                vec[1].value = binning
                self._indiclient.sendNewNumber(vec)
            except RuntimeError:
                pass

    def start_exposure(self) -> None:
        vec = self._number("CCD_EXPOSURE")
        vec[0].value = self._exptime
        self._indiclient.sendNewNumber(vec)

    def exposure_complete(self) -> bool:
        return self._number("CCD_EXPOSURE").getState() != PyIndi.IPS_BUSY

    def abort_exposure(self) -> None:
        self._set_switch("CCD_ABORT_EXPOSURE", "ABORT", timeout=5)

    def download_image(self) -> np.ndarray:
        blob_vec = self._blob("CCD1")
        self._wait_ok(lambda: blob_vec, timeout=self._exptime + 60)
        blob = blob_vec[0]
        fmt = blob.format.lower()
        data = bytes(blob.blob)
        if fmt in (".fits", "fits", ".fit", "fit"):
            import io

            from astropy.io import fits

            with fits.open(io.BytesIO(data)) as hdul:
                return np.asarray(hdul[0].data)
        # Fallback: raw frame buffer (uint16, row-major).
        w = int(self._number("CCD_FRAME")[0].value or 0)
        h = int(self._number("CCD_FRAME")[1].value or 0)
        arr = np.frombuffer(data, dtype=np.uint16)
        if w and h and arr.size >= w * h:
            return arr[: w * h].reshape(h, w)
        return arr

    @property
    def has_cooler(self) -> bool:
        return self._cooler_supported

    def set_cooler(self, temperature_c: float) -> None:
        if not self._cooler_supported:
            raise NotImplementedError("INDI camera has no cooler switch")
        self._set_switch("CCD_COOLER", "COOLER_ON", timeout=5)
        vec = self._number("CCD_TEMPERATURE")
        vec[0].value = temperature_c
        self._indiclient.sendNewNumber(vec)

    def get_temperature(self) -> float | None:
        try:
            return float(self._number("CCD_TEMPERATURE")[0].value)
        except RuntimeError:
            return None
