"""Small coordinate/time helpers shared by drivers and session tools."""

from __future__ import annotations

import math
from datetime import datetime, timezone


def angular_separation_deg(
    ra1_h: float, dec1_d: float, ra2_h: float, dec2_d: float
) -> float:
    """Great-circle distance between two J2000 positions, in degrees."""
    r1, d1 = math.radians(ra1_h * 15.0), math.radians(dec1_d)
    r2, d2 = math.radians(ra2_h * 15.0), math.radians(dec2_d)
    cos_c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    cos_c = max(-1.0, min(1.0, cos_c))
    return math.degrees(math.acos(cos_c))


def utcnow_iso() -> str:
    """Current UTC time as an ISO-8601 string (for DATE-OBS)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def hours_to_hms(hours: float) -> str:
    h = hours % 24.0
    hh = int(h)
    mm = int((h - hh) * 60)
    ss = (h - hh - mm / 60.0) * 3600.0
    return f"{hh:02d}:{mm:02d}:{ss:05.2f}"


def deg_to_dms(deg: float) -> str:
    sign = "+" if deg >= 0 else "-"
    a = abs(deg)
    dd = int(a)
    mm = int((a - dd) * 60)
    ss = (a - dd - mm / 60.0) * 3600.0
    return f"{sign}{dd:02d}:{mm:02d}:{ss:05.2f}"


def radec_to_vec(ra_hours: float, dec_deg: float):
    """J2000 position as a unit vector (for great-circle math)."""
    import numpy as np

    ra, dec = math.radians(ra_hours * 15.0), math.radians(dec_deg)
    return np.array(
        [math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)]
    )


def vec_to_radec(vec) -> tuple[float, float]:
    """Unit vector back to (ra_hours, dec_deg)."""
    import numpy as np

    x, y, z = (float(v) for v in vec)
    dec = math.degrees(math.asin(max(-1.0, min(1.0, z))))
    ra = math.degrees(math.atan2(y, x)) / 15.0 % 24.0
    return ra, dec
