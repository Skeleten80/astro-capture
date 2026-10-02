"""DSLR-over-USB sketch via ``gphoto2`` (``kind: dslr``).

**Preferred route on Linux:** run your DSLR through INDI's
``indi_gphoto_cc`` driver and use the ``indi`` camera backend instead —
it handles bulb exposures, live view, and FITS wrapping for you, and
AstroCapture's sequencer treats it like any other camera.  This module
exists for two reasons:

1. Direct gphoto2 control without an INDI server in the loop.
2. A documented starting point if you ever want to bypass INDI.

The import is guarded the same way as the INDI backend: the module always
imports, but constructing ``GPhotoCamera`` without ``gphoto2`` installed
raises a clear error.
"""

from __future__ import annotations

import io
import time

import numpy as np

from astrocapture.drivers.base import Camera

try:  # optional dependency, guarded at construction time
    import gphoto2 as gp  # type: ignore[import]
except ImportError:  # pragma: no cover - needs gphoto2 installed
    gp = None  # type: ignore[assignment]

_GPHOTO2_MISSING = (
    "The dslr backend needs the gphoto2 Python bindings: "
    "pip install gphoto2 (plus libgphoto2 system package). "
    "Tip: for Canon/Nikon/Sony on Linux, the indi backend with "
    "'indi_gphoto_cc' is the preferred route."
)

# ---------------------------------------------------------------------------
# Camera profile: Canon EOS Rebel T7i (EOS 800D).
# Checklist for putting this body on a telescope (also applies when driving
# it through indi_gphoto_cc — the INDI backend):
# ---------------------------------------------------------------------------
T7I_PROFILE = {
    "model": "Canon EOS Rebel T7i / EOS 800D",
    "mode_dial": "M (Manual) — required. The camera must NOT be in a scene "
    "or auto mode or gphoto2/INDI cannot set shutter and ISO.",
    "file_format": "RAW (CR2). Set Image Quality to RAW only — no RAW+JPEG, "
    "so every capture is one file and downloads stay fast.",
    "bulb": "For exposures > 30 s use bulb: gphoto2 "
    "`--set-config shutterspeed=bulb` then `--set-config bulb=1` + "
    "`--wait-event=<seconds>s --set-config bulb=0` (or the eosremoterelease "
    "path). Via INDI, indi_gphoto_cc handles bulb internally.",
    "mirror_lockup": "Enable mirror lockup (Custom Function) for exposures: "
    "first shutter press locks the mirror, second starts the exposure — "
    "kills mirror-slap vibration on a 1500 mm f/10 SCT.",
    "auto_power_off": "DISABLE (or set to max). A body that sleeps mid-"
    "sequence drops the USB session and the run aborts.",
    "long_exposure_nr": "OFF — in-camera dark subtraction doubles every "
    "exposure time. Take real dark frames instead (the sequencer does).",
    "focus": "Manual focus, via Live View at 10x on a bright star. Tape the "
    "focus ring once set; refocus if temperature swings > ~5 C.",
    "mechanical": "T-ring (Canon EF) + 1.25\" nosepiece, or an SCT T-adapter "
    "threaded directly to the 6SE's rear cell. The T7i body hangs off the "
    "visual back — keep the diagonal OUT of the train and check balance.",
    "iso_sweet_spot": 1600,
}


def _require_gphoto2() -> None:
    if gp is None:
        raise ImportError(_GPHOTO2_MISSING)


class GPhotoCamera(Camera):
    """Canon/Nikon/Sony DSLR (or mirrorless) over USB via libgphoto2.

    For the Canon Rebel T7i specifically, see ``T7I_PROFILE`` above —
    the same checklist applies when the body is driven through
    ``indi_gphoto_cc`` instead of this backend.

    Exposure flow: set ``shutterspeed`` to ``bulb`` for exptime > 30 s
    (bulb capture holds the shutter open), otherwise pick the closest
    supported shutter speed.  ``download_image`` pulls the last capture
    and decodes it to a numpy array (raw via rawpy when available,
    JPEG thumbnail otherwise).
    """

    def __init__(self, iso: int = 1600) -> None:
        _require_gphoto2()
        self._iso = iso
        self._exptime = 1.0
        self._cam: gp.Camera | None = None
        self._last_file: tuple[str, str] | None = None

    # -- Camera API -----------------------------------------------------
    def connect(self) -> None:
        self._cam = gp.Camera()
        self._cam.init()
        self._set_config("iso", str(self._iso))

    def disconnect(self) -> None:
        if self._cam is not None:
            self._cam.exit()
            self._cam = None

    def set_exposure_settings(
        self, exptime_s: float, gain: float = 0.0, binning: int = 1
    ) -> None:
        if exptime_s <= 0:
            raise ValueError("exposure time must be positive")
        self._exptime = exptime_s
        if gain:
            self._iso = int(gain)
            self._set_config("iso", str(self._iso))
        if binning != 1:
            # DSLRs don't bin on-sensor; note it and continue unbinned.
            pass

    def start_exposure(self) -> None:
        assert self._cam is not None, "GPhotoCamera: not connected"
        if self._exptime > 30:
            # Bulb path: hold the shutter with a timed capture.
            self._set_config("shutterspeed", "bulb")
            # NOTE: exact bulb API differs per model; remote-release
            # via 'eosremoterelease'/'bulb' capture is the portable route.
            file_path = self._cam.capture(gp.GP_CAPTURE_IMAGE)
        else:
            self._set_config("shutterspeed", self._nearest_shutter(self._exptime))
            file_path = self._cam.capture(gp.GP_CAPTURE_IMAGE)
        self._last_file = (file_path.folder, file_path.name)
        self._exposure_start = time.monotonic()

    def exposure_complete(self) -> bool:
        # gphoto2 capture() blocks until the shot is taken, so by the
        # time start_exposure() returns the exposure is done (except the
        # bulb path above, which still needs per-model work — see note).
        return True

    def abort_exposure(self) -> None:
        # Bulb abort is model-specific; not implemented in this sketch.
        raise NotImplementedError("GPhotoCamera: bulb abort not implemented")

    def download_image(self) -> np.ndarray:
        assert self._cam is not None, "GPhotoCamera: not connected"
        assert self._last_file is not None, "GPhotoCamera: nothing captured yet"
        folder, name = self._last_file
        cam_file = self._cam.file_get(folder, name, gp.GP_FILE_TYPE_NORMAL)
        data = bytes(memoryview(cam_file.get_data_and_size()))
        try:
            import rawpy  # type: ignore[import]

            with rawpy.imread(io.BytesIO(data)) as raw:
                return raw.postprocess().mean(axis=2)
        except ImportError:
            pass
        from PIL import Image  # Pillow fallback: decode embedded JPEG

        with Image.open(io.BytesIO(data)) as im:
            return np.asarray(im.convert("L"), dtype=np.float64)

    # -- internals ------------------------------------------------------
    def _set_config(self, name: str, value: str) -> None:
        assert self._cam is not None
        try:
            cfg = self._cam.get_config()
            node = cfg.get_child_by_name(name)
            node.set_value(value)
            self._cam.set_config(cfg)
        except gp.GPhoto2Error:
            # Not every body exposes every setting; keep going.
            pass

    @staticmethod
    def _nearest_shutter(exptime: float) -> str:
        # Common gphoto2 shutter-speed strings, seconds.
        speeds = [
            (1 / 8000, "1/8000"), (1 / 4000, "1/4000"), (1 / 2000, "1/2000"),
            (1 / 1000, "1/1000"), (1 / 500, "1/500"), (1 / 250, "1/250"),
            (1 / 125, "1/125"), (1 / 60, "1/60"), (1 / 30, "1/30"),
            (1 / 15, "1/15"), (1 / 8, "1/8"), (1 / 4, "1/4"),
            (0.5, "1/2"), (1.0, "1"), (2.0, "2"), (4.0, "4"),
            (8.0, "8"), (15.0, "15"), (30.0, "30"),
        ]
        return min(speeds, key=lambda s: abs(s[0] - exptime))[1]
