"""AstroCapture — telescope + camera control and capture sequencing.

Hardware is never touched directly: everything goes through the driver
abstractions in :mod:`astrocapture.drivers`.  Real drivers talk INDI;
the ``sim`` drivers let the whole pipeline run with no gear at all.
"""

__version__ = "0.1.0"

from astrocapture.drivers.base import Camera, Mount, MountState  # noqa: F401
