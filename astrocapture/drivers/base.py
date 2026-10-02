"""Abstract driver interfaces every backend implements.

The sequencer only ever talks to these two classes, so a new backend
(ASCOM Alpaca, a vendor SDK, …) is just a new file in this package that
implements the same methods.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod

import numpy as np


class MountState(enum.Enum):
    IDLE = "idle"
    SLEWING = "slewing"
    TRACKING = "tracking"
    PARKED = "parked"
    ERROR = "error"


class Mount(ABC):
    """Equatorial (or Alt-Az) mount driver."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def unpark(self) -> None: ...

    @abstractmethod
    def park(self) -> None: ...

    @abstractmethod
    def goto(self, ra_hours: float, dec_deg: float) -> None:
        """Start a slew to the given J2000 coordinates. Returns immediately."""

    @abstractmethod
    def slew_complete(self) -> bool:
        """True once the mount has settled on target."""

    @abstractmethod
    def start_tracking(self) -> None: ...

    @abstractmethod
    def stop_tracking(self) -> None: ...

    @property
    @abstractmethod
    def state(self) -> MountState: ...

    @property
    @abstractmethod
    def position(self) -> tuple[float, float]:
        """Current (ra_hours, dec_deg)."""

    def offset(self, d_ra_arcsec: float, d_dec_arcsec: float) -> None:
        """Nudge the target by the given offsets (dithering).

        Default implementation does a fresh goto to the offset target;
        drivers that support guide-rate pulses can override.
        """
        ra, dec = self.position
        self.goto(ra + d_ra_arcsec / 54000.0, dec + d_dec_arcsec / 3600.0)


class Camera(ABC):
    """Imaging camera driver."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def set_exposure_settings(
        self, exptime_s: float, gain: float = 0.0, binning: int = 1
    ) -> None: ...

    @abstractmethod
    def start_exposure(self) -> None: ...

    @abstractmethod
    def exposure_complete(self) -> bool: ...

    @abstractmethod
    def abort_exposure(self) -> None: ...

    @abstractmethod
    def download_image(self) -> np.ndarray:
        """Return the exposed frame as a 2-D float or uint16 numpy array."""

    # --- cooler: optional; base implementation is a no-op -----------------
    @property
    def has_cooler(self) -> bool:
        return False

    def set_cooler(self, temperature_c: float) -> None:
        raise NotImplementedError(f"{type(self).__name__} has no cooler")

    def get_temperature(self) -> float | None:
        return None
