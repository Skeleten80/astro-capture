"""Simulator backends: no hardware required.

``SimMount`` slews toward its target at a configurable rate (deg/sec)
and otherwise behaves like a real mount: states, parking, tracking.

``SimCamera`` synthesizes a starfield per exposure: stars placed at
fixed sky positions (seeded), Gaussian PSFs, photon shot noise and
read noise, hot pixels, and signal that scales with exposure time —
so a stack of frames actually behaves like a stack of frames.
"""

from __future__ import annotations

import math
import time

import numpy as np

from astrocapture.drivers.base import Camera, Mount, MountState
from astrocapture.util import radec_to_vec, vec_to_radec


class SimMount(Mount):
    def __init__(
        self,
        slew_rate_dps: float = 3.0,
        park_ra_hours: float = 0.0,
        park_dec_deg: float = 90.0,
    ) -> None:
        self.slew_rate_dps = slew_rate_dps
        self._ra = park_ra_hours
        self._dec = park_dec_deg
        self._target: tuple[float, float] | None = None
        self._state = MountState.PARKED
        self._connected = False
        self._tracking = False

    # -- Mount API ------------------------------------------------------
    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def unpark(self) -> None:
        self._require_connected()
        self._state = MountState.IDLE

    def park(self) -> None:
        self._require_connected()
        self._parking = True
        self._target = (0.0, 90.0)
        self._tracking = False
        self._state = MountState.SLEWING

    def goto(self, ra_hours: float, dec_deg: float) -> None:
        self._require_connected()
        self._target = (ra_hours % 24.0, max(-90.0, min(90.0, dec_deg)))
        self._state = MountState.SLEWING

    def slew_complete(self) -> bool:
        self._tick()
        return self._target is None

    def start_tracking(self) -> None:
        self._require_connected()
        self._tracking = True
        if self._state == MountState.IDLE:
            self._state = MountState.TRACKING

    def stop_tracking(self) -> None:
        self._tracking = False
        if self._state == MountState.TRACKING:
            self._state = MountState.IDLE

    @property
    def state(self) -> MountState:
        self._tick()
        return self._state

    @property
    def position(self) -> tuple[float, float]:
        self._tick()
        return self._ra, self._dec

    def offset(self, d_ra_arcsec: float, d_dec_arcsec: float) -> None:
        ra, dec = self.position
        self._ra = (ra + d_ra_arcsec / 54000.0) % 24.0
        self._dec = max(-90.0, min(90.0, dec + d_dec_arcsec / 3600.0))

    # -- internals ------------------------------------------------------
    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("SimMount: not connected")

    def _tick(self) -> None:
        """Advance the simulated position along the great circle to target."""
        if self._target is None:
            return
        now = time.monotonic()
        last = getattr(self, "_last_tick", now)
        self._last_tick = now
        dt = now - last

        v0 = radec_to_vec(self._ra, self._dec)
        v1 = radec_to_vec(*self._target)
        cos_c = float(np.clip(np.dot(v0, v1), -1.0, 1.0))
        dist = math.degrees(math.acos(cos_c))
        step = self.slew_rate_dps * dt
        if dist <= step or dist == 0.0:
            self._ra, self._dec = self._target
            self._target = None
            if getattr(self, "_parking", False):
                self._parking = False
                self._state = MountState.PARKED
            else:
                self._state = MountState.TRACKING if self._tracking else MountState.IDLE
            return
        # Slerp: constant angular rate along the great circle.
        ang = math.radians(dist)
        t = step / dist
        sin_ang = math.sin(ang)
        v = (math.sin((1 - t) * ang) * v0 + math.sin(t * ang) * v1) / sin_ang
        self._ra, self._dec = vec_to_radec(v)


class SimCamera(Camera):
    """Synthetic starfield camera.

    The sky is a fixed catalog of stars (seeded RNG) on a virtual sensor.
    Pointing drift (e.g. from dithering) shifts the catalog relative to
    the sensor; exposure time scales the counts.
    """

    def __init__(
        self,
        width: int = 1024,
        height: int = 1024,
        n_stars: int = 400,
        read_noise_e: float = 5.0,
        hot_pixel_fraction: float = 2e-4,
        seed: int = 42,
    ) -> None:
        self.width = width
        self.height = height
        self.n_stars = n_stars
        self.read_noise_e = read_noise_e
        self.hot_pixel_fraction = hot_pixel_fraction
        self._rng = np.random.default_rng(seed)
        self._catalog = self._make_catalog(seed)
        self._hot = self._make_hot_pixels()
        self._exptime = 1.0
        self._gain = 0.0
        self._binning = 1
        self._exposure_start: float | None = None
        self._connected = False
        self._dither_offset = (0.0, 0.0)  # pixels of pointing shift

    # -- Camera API -----------------------------------------------------
    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def set_exposure_settings(
        self, exptime_s: float, gain: float = 0.0, binning: int = 1
    ) -> None:
        if exptime_s <= 0:
            raise ValueError("exposure time must be positive")
        self._exptime = exptime_s
        self._gain = gain
        self._binning = binning

    def start_exposure(self) -> None:
        if not self._connected:
            raise RuntimeError("SimCamera: not connected")
        self._exposure_start = time.monotonic()

    def exposure_complete(self) -> bool:
        if self._exposure_start is None:
            return False
        return time.monotonic() - self._exposure_start >= self._exptime

    def abort_exposure(self) -> None:
        self._exposure_start = None

    def download_image(self) -> np.ndarray:
        if self._exposure_start is None:
            raise RuntimeError("SimCamera: no exposure in progress")
        self._exposure_start = None
        return self._render()

    @property
    def has_cooler(self) -> bool:
        return True

    def set_cooler(self, temperature_c: float) -> None:
        self._cooler_setpoint = temperature_c  # simulated; instantly reached

    def get_temperature(self) -> float:
        return getattr(self, "_cooler_setpoint", 20.0)

    # -- helpers ----------------------------------------------------------
    def apply_dither(self, dx_px: float, dy_px: float) -> None:
        """Shift pointing by (dx, dy) pixels for the next frames."""
        ox, oy = self._dither_offset
        self._dither_offset = (ox + dx_px, oy + dy_px)

    def _make_catalog(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        xs = rng.uniform(0, self.width, self.n_stars)
        ys = rng.uniform(0, self.height, self.n_stars)
        mags = rng.uniform(8.0, 16.0, self.n_stars)  # instrumental magnitudes
        return np.column_stack([xs, ys, mags])

    def _make_hot_pixels(self) -> np.ndarray:
        n = int(self.width * self.height * self.hot_pixel_fraction)
        xs = self._rng.integers(0, self.width, n)
        ys = self._rng.integers(0, self.height, n)
        return np.column_stack([xs, ys])

    def _render(self) -> np.ndarray:
        rng = self._rng
        img = np.full((self.height, self.width), 1000.0)  # sky background ADU
        img += rng.normal(0, 3.0, img.shape)  # sky glow variation

        ox, oy = self._dither_offset
        scale = self._exptime  # counts scale with exposure time
        yy, xx = np.mgrid[0 : self.height, 0 : self.width]
        for x, y, mag in self._catalog:
            flux = 10 ** ((14.0 - mag) / 2.5) * 500.0 * scale
            sx, sy = x - ox, y - oy
            if -20 < sx < self.width + 20 and -20 < sy < self.height + 20:
                # Poisson-ish noise on the star, gaussian PSF sigma=1.6 px
                n_phot = rng.poisson(max(flux, 1e-9))
                img += n_phot * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * 1.6**2))

        img += rng.normal(0, self.read_noise_e, img.shape)  # read noise
        for x, y in self._hot:  # hot pixels: bright, exposure-scaled
            img[y, x] += 8000.0 * scale

        # 16-bit ADC like a real sensor
        return np.clip(img, 0, 65535).astype(np.uint16)

    # -- test helper: expected slew time math is in SimMount --------------
    @property
    def last_settings(self) -> tuple[float, float, int]:
        return self._exptime, self._gain, self._binning
