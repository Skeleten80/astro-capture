"""Driver registry: ``make_mount(kind, **kw)`` / ``make_camera(kind, **kw)``.

``kind`` is one of ``"sim"``, ``"indi"`` (mount + camera) or ``"dslr"``
(camera only).  The ``indi`` and ``dslr`` modules guard their optional
third-party imports so *this* package imports cleanly on any machine —
a clear error is raised only if you actually select a backend whose
dependency is missing.
"""

from __future__ import annotations

from typing import Any

from astrocapture.drivers.base import Camera, Mount

_MOUNT_BUILDERS: dict[str, Any] = {}
_CAMERA_BUILDERS: dict[str, Any] = {}


def _register() -> None:
    from astrocapture.drivers import sim as sim_drivers

    _MOUNT_BUILDERS["sim"] = sim_drivers.SimMount
    _CAMERA_BUILDERS["sim"] = sim_drivers.SimCamera

    try:
        from astrocapture.drivers import indi as indi_drivers

        _MOUNT_BUILDERS["indi"] = indi_drivers.INDIMount
        _CAMERA_BUILDERS["indi"] = indi_drivers.INDICamera
    except ImportError:  # pragma: no cover - optional dependency
        pass

    try:
        from astrocapture.drivers import dslr as dslr_drivers

        _CAMERA_BUILDERS["dslr"] = dslr_drivers.GPhotoCamera
    except ImportError:  # pragma: no cover - optional dependency
        pass


def available_drivers() -> dict[str, list[str]]:
    """Return {'mounts': [...], 'cameras': [...]} for the current machine."""
    _register()
    return {
        "mounts": sorted(_MOUNT_BUILDERS),
        "cameras": sorted(_CAMERA_BUILDERS),
    }


def make_mount(kind: str, **kwargs: Any) -> Mount:
    _register()
    try:
        return _MOUNT_BUILDERS[kind](**kwargs)
    except KeyError:
        raise ValueError(
            f"Unknown mount driver {kind!r}. Available: {sorted(_MOUNT_BUILDERS)}"
        ) from None


def make_camera(kind: str, **kwargs: Any) -> Camera:
    _register()
    try:
        return _CAMERA_BUILDERS[kind](**kwargs)
    except KeyError:
        raise ValueError(
            f"Unknown camera driver {kind!r}. Available: {sorted(_CAMERA_BUILDERS)}"
        ) from None
