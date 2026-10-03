"""Dew heater control: keep the corrector plate / objective above the dew point.

When the optics cool to the dew point, water condenses on them and the
night is over.  A dew strap (a resistive heater band around the dew
shield) driven by a PWM controller adds just enough heat to stay a few
degrees above the dew point without injecting tube currents.

Software side (this module):

* :func:`dew_point_c` — Magnus-formula dew point from air temperature
  and relative humidity.
* :class:`TempHumSensor` / :class:`DewHeater` — abstract sensor and
  heater interfaces, with ``sim`` implementations for testing and
  thin INDI stubs for real hardware.
* :class:`DewController` — a proportional controller that ramps the
  heater duty up as the air temperature approaches
  ``dew_point + margin``.  It never raises: sensor/heater faults are
  logged and the last duty is held, so a flaky dew controller can never
  abort imaging.

Hardware side (NOT in this repo — you must buy/build it):

* a 12 V dew strap sized for the telescope,
* a PWM dew controller (commercial, or a MOSFET + Arduino-style board),
* optionally a temperature/humidity sensor at the scope.

The sequencer calls ``controller.update()`` once per light frame (the
documented cadence: dew evolves on ~10-minute timescales, so per-frame
updates on minute-scale exposures track it with negligible overhead).
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable

log = logging.getLogger(__name__)

# Magnus-formula constants (Alduchov & Eskridge 1996; the widely used
# parameterization, accurate to ~0.1 °C over -40..+50 °C):
#
#   gamma(T, RH) = ln(RH/100) + b*T / (c + T)
#   Td           = c*gamma / (b - gamma)
MAGNUS_B = 17.625
MAGNUS_C = 243.04  # °C


def dew_point_c(temp_c: float, rh_pct: float) -> float:
    """Dew point in °C from air temperature (°C) and relative humidity (%).

    Magnus formula with the Alduchov & Eskridge constants documented
    above.  Sanity: ``dew_point_c(20.0, 50.0)`` ≈ 9.3 °C.
    """
    rh = min(100.0, max(0.1, rh_pct))  # clamp: sensors glitch, math must not
    gamma = math.log(rh / 100.0) + MAGNUS_B * temp_c / (MAGNUS_C + temp_c)
    return MAGNUS_C * gamma / (MAGNUS_B - gamma)


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------


class TempHumSensor(ABC):
    """Ambient temperature/humidity sensor at the scope."""

    @abstractmethod
    def read(self) -> tuple[float, float]:
        """Return ``(temp_c, rh_pct)``. May raise on hardware faults."""


class SimSensor(TempHumSensor):
    """Scriptable stand-in: plays back a ``(temp_c, rh_pct)`` series.

    When the series is exhausted, ``read()`` keeps returning the
    ``fallback`` value, so a short script never crashes a long run.
    """

    def __init__(
        self,
        readings: Iterable[tuple[float, float]] | None = None,
        fallback: tuple[float, float] = (10.0, 85.0),
    ) -> None:
        self._readings = [tuple(map(float, r)) for r in (readings or [])]
        self._i = 0
        self.fallback = (float(fallback[0]), float(fallback[1]))
        self.reads = 0

    def read(self) -> tuple[float, float]:
        self.reads += 1
        if self._i < len(self._readings):
            value = self._readings[self._i]
            self._i += 1
            return value
        return self.fallback

    @classmethod
    def humid_night(
        cls,
        start_temp_c: float = 12.0,
        start_rh_pct: float = 88.0,
        end_temp_c: float = 9.0,
        steps: int = 48,
    ) -> "SimSensor":
        """A scripted humid night: temperature falls toward the dew point.

        Absolute humidity is held constant (fixed dew point from the
        starting conditions), so relative humidity climbs toward 100 %
        as the air cools — the classic dew-forming scenario.  One
        ``read()`` per controller update; ``steps`` of them total.
        """
        dewpoint = dew_point_c(start_temp_c, start_rh_pct)
        # Constant absolute humidity <=> constant gamma at the dew point.
        gamma_d = MAGNUS_B * dewpoint / (MAGNUS_C + dewpoint)
        series: list[tuple[float, float]] = []
        for k in range(max(1, steps)):
            frac = k / max(1, steps - 1)
            temp = start_temp_c + (end_temp_c - start_temp_c) * frac
            rh = 100.0 * math.exp(
                gamma_d - MAGNUS_B * temp / (MAGNUS_C + temp)
            )
            series.append((temp, min(100.0, rh)))
        return cls(series)


class INDIWeatherSensor(TempHumSensor):
    """Read ambient temp/RH from an INDI weather device.

    UNTESTED AGAINST HARDWARE — INDI weather drivers name their
    properties differently (``indi_weather``, DIY Arduino stations,
    …).  Verify every property/item name below with ``indi_getprop``
    on first light and pass the real ones as constructor arguments.

    The INDI import happens lazily on first ``read()`` (never at
    module import time), so importing this module can never fail for
    lack of hardware or backend code.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 7624,
        device: str = "Weather",
        temp_property: str = "WEATHER_TEMPERATURE",
        temp_item: str = "TEMPERATURE",
        rh_property: str = "WEATHER_HUMIDITY",
        rh_item: str = "HUMIDITY",
        timeout: float = 10.0,
    ) -> None:
        self._host = host
        self._port = port
        self._device = device
        self._temp_property = temp_property
        self._temp_item = temp_item
        self._rh_property = rh_property
        self._rh_item = rh_item
        self._timeout = timeout
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from astrocapture.drivers.indi import INDIClient
        except ImportError as exc:
            raise RuntimeError(
                "INDIWeatherSensor: the INDI backend is unavailable "
                f"({exc})"
            ) from exc
        client = INDIClient(host=self._host, port=self._port)
        client.connect()
        client.wait_for_device(self._device, timeout=self._timeout)
        self._client = client
        return client

    def read(self) -> tuple[float, float]:
        client = self._ensure_client()
        temp = float(client.items_of(self._device, self._temp_property)
                     [self._temp_item])
        rh = float(client.items_of(self._device, self._rh_property)
                   [self._rh_item])
        return temp, rh


# ---------------------------------------------------------------------------
# Heaters
# ---------------------------------------------------------------------------


class DewHeater(ABC):
    """Dew strap heater with PWM duty control."""

    @abstractmethod
    def set_duty(self, duty: float) -> None:
        """Set heater power, ``duty`` in 0..1. May raise on hardware faults."""

    @abstractmethod
    def get_duty(self) -> float:
        """Last commanded duty (0..1)."""


class SimHeater(DewHeater):
    """Records every commanded duty; ``history`` shows the whole night."""

    def __init__(self) -> None:
        self._duty = 0.0
        self.history: list[float] = []

    def set_duty(self, duty: float) -> None:
        self._duty = min(1.0, max(0.0, duty))
        self.history.append(self._duty)

    def get_duty(self) -> float:
        return self._duty


class INDIDewHeater(DewHeater):
    """Drive a dew heater through INDI (e.g. an Arduino focuser aux output).

    UNTESTED AGAINST HARDWARE — dew-controller INDI drivers are
    typically DIY, so the property/item names below are parameters, not
    promises.  The default assumes a number property carrying 0–100 %
    power; verify with ``indi_getprop`` and adjust.

    Like :class:`INDIWeatherSensor`, the INDI import is lazy so this
    module always imports cleanly without hardware.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 7624,
        device: str = "Dew Heater",
        power_property: str = "DEW_HEATER",
        power_item: str = "DUTY",
        timeout: float = 10.0,
    ) -> None:
        self._host = host
        self._port = port
        self._device = device
        self._power_property = power_property
        self._power_item = power_item
        self._timeout = timeout
        self._client = None
        self._duty = 0.0  # local echo; readback is driver-specific

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from astrocapture.drivers.indi import INDIClient
        except ImportError as exc:
            raise RuntimeError(
                f"INDIDewHeater: the INDI backend is unavailable ({exc})"
            ) from exc
        client = INDIClient(host=self._host, port=self._port)
        client.connect()
        client.wait_for_device(self._device, timeout=self._timeout)
        self._client = client
        return client

    def set_duty(self, duty: float) -> None:
        duty = min(1.0, max(0.0, duty))
        client = self._ensure_client()
        client.send_number(self._device, self._power_property,
                           {self._power_item: duty * 100.0})
        self._duty = duty

    def get_duty(self) -> float:
        return self._duty


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class DewController:
    """Proportional dew heater controller.

    Control law — let ``Td`` be the dew point, ``m`` the safety margin,
    ``error = (T - m) - Td`` (how far the air is above the danger zone),
    and ``spread`` the comfort band (°C) over which the heater ramps::

        duty = max_duty * clamp(aggressiveness * (1 - error / spread), 0, 1)

    * ``error >= spread`` (air comfortably dry): duty 0.
    * ``error = 0`` (air exactly at ``dew_point + margin``): duty =
      ``aggressiveness * max_duty``.
    * ``error < 0`` (inside the margin or below the dew point): duty
      rises linearly and saturates at ``max_duty``.

    ``aggressiveness`` therefore sets how hard the strap works when the
    air first reaches the margin boundary (1.0 = full ``max_duty`` right
    at the boundary); ``max_duty`` caps the output so a misbehaving
    sensor can't cook the optics.

    ``update()`` never raises: any sensor or heater exception is logged
    as a warning and the previous duty is held (and returned), so dew
    control can never abort imaging.
    """

    def __init__(
        self,
        sensor: TempHumSensor,
        heater: DewHeater,
        margin_c: float = 2.0,
        aggressiveness: float = 0.5,
        max_duty: float = 0.9,
        spread_c: float = 8.0,
    ) -> None:
        if margin_c <= 0:
            raise ValueError(f"margin_c must be > 0, got {margin_c}")
        if aggressiveness <= 0:
            raise ValueError(f"aggressiveness must be > 0, got {aggressiveness}")
        if not (0.0 < max_duty <= 1.0):
            raise ValueError(f"max_duty must be in (0, 1], got {max_duty}")
        if spread_c <= 0:
            raise ValueError(f"spread_c must be > 0, got {spread_c}")
        self.sensor = sensor
        self.heater = heater
        self.margin_c = margin_c
        self.aggressiveness = aggressiveness
        self.max_duty = max_duty
        self.spread_c = spread_c
        # Last known state (NaN until the first successful update).
        self.last_temp_c = float("nan")
        self.last_rh_pct = float("nan")
        self.last_dewpoint_c = float("nan")
        self.last_error_c = float("nan")
        self.last_duty = 0.0

    def update(self) -> float:
        """One control step. Returns the commanded duty (0..1).

        Never raises: on any sensor/heater failure a warning is logged
        and the previous duty is held.
        """
        try:
            temp_c, rh_pct = self.sensor.read()
            if not (math.isfinite(temp_c) and math.isfinite(rh_pct)):
                raise ValueError(
                    f"non-finite sensor reading: temp={temp_c}, rh={rh_pct}"
                )
            dewpoint = dew_point_c(temp_c, rh_pct)
            error = (temp_c - self.margin_c) - dewpoint
        except Exception as exc:  # noqa: BLE001 - dew control never aborts
            log.warning("dew controller update failed (%s) — holding duty %.2f",
                        exc, self.last_duty)
            return self.last_duty
        # The environment was measured successfully; record it even if the
        # heater misbehaves below.
        self.last_temp_c = temp_c
        self.last_rh_pct = rh_pct
        self.last_dewpoint_c = dewpoint
        self.last_error_c = error
        frac = self.aggressiveness * (1.0 - error / self.spread_c)
        duty = min(1.0, max(0.0, frac)) * self.max_duty
        try:
            self.heater.set_duty(duty)
        except Exception as exc:  # noqa: BLE001 - heater fault holds duty
            log.warning("dew heater failed (%s) — holding duty %.2f",
                        exc, self.last_duty)
            return self.last_duty
        self.last_duty = duty
        return duty
