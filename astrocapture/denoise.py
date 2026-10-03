"""Denoising for stacked images: learned (ONNX) or classical fallback.

HONEST NOTE: this module ships **no trained astro denoise model**.
:func:`OnnxDenoiser` is the slot a community model plugs into (any
single-input/single-output float32 ONNX image model); with no model file
the pipeline uses :class:`ClassicalDenoiser`, a real edge-aware
bilateral filter in pure numpy that measurably improves SNR out of the
box.  Nothing here pretends a download happened.

On Apple Silicon the ONNX backend accepts
``["CoreMLExecutionProvider", "CPUExecutionProvider"]`` as providers —
the same pattern as Mathias's car-logger vision stack — dispatching
eligible ops to the Neural Engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from astrocapture.imaging import shift_image


class Denoiser(ABC):
    """Anything that maps a noisy 2-D image to a cleaner one."""

    @abstractmethod
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Return the denoised image (same shape, float32)."""
        ...


# ---------------------------------------------------------------------------
# Classical fallback: edge-aware bilateral filter (numpy only)
# ---------------------------------------------------------------------------


class ClassicalDenoiser(Denoiser):
    """Vectorized bilateral filter: spatial x range Gaussian weighting.

    For each pixel, a weighted mean over a ``(2*radius+1)`` window where
    the weight is ``exp(-d_spatial^2 / 2σs^2) * exp(-d_range^2 / 2σr^2)``.
    Flat regions get averaged (noise falls); edges keep their range
    weight near zero across the edge (detail survives).  ``sigma_range``
    defaults to 3x the robust noise estimate (MAD-based) so the filter
    adapts to the frame instead of needing hand tuning.
    """

    def __init__(
        self,
        radius: int = 2,
        sigma_spatial: float = 1.5,
        sigma_range: float | None = None,
    ) -> None:
        if radius < 1:
            raise ValueError("radius must be >= 1")
        self.radius = radius
        self.sigma_spatial = float(sigma_spatial)
        self.sigma_range = sigma_range

    def _estimate_sigma(self, img: np.ndarray) -> float:
        if self.sigma_range is not None:
            return float(self.sigma_range)
        med = float(np.median(img))
        mad = float(np.median(np.abs(img - med)))
        return max(1e-6, 3.0 * 1.4826 * mad)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=np.float64)
        if img.ndim != 2:
            raise ValueError(
                f"ClassicalDenoiser needs a 2-D image, got shape {img.shape}"
            )
        sigma_r = self._estimate_sigma(img)
        r = self.radius
        # Spatial weights for each integer offset in the window.
        offsets = [(dy, dx) for dy in range(-r, r + 1)
                   for dx in range(-r, r + 1)]
        acc = np.zeros_like(img)
        wsum = np.zeros_like(img)
        for dy, dx in offsets:
            w_spatial = float(np.exp(
                -(dx * dx + dy * dy) / (2.0 * self.sigma_spatial ** 2)))
            shifted = shift_image(img, float(dx), float(dy))
            diff = shifted - img
            w_range = np.exp(-(diff * diff) / (2.0 * sigma_r ** 2))
            w = w_spatial * w_range
            acc += w * shifted
            wsum += w
        out = acc / np.maximum(wsum, 1e-12)
        return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Learned backend: user-supplied ONNX model (optional dependency)
# ---------------------------------------------------------------------------


class OnnxDenoiser(Denoiser):
    """Run a user-supplied ONNX denoise model via onnxruntime.

    ``model_path``: ``.onnx`` file (single image input, single image
    output, float32).  ``providers``: execution providers in preference
    order, e.g. ``["CoreMLExecutionProvider", "CPUExecutionProvider"]``
    on Apple Silicon; unavailable providers are skipped with a warning
    and CPU is always the final fallback.

    Raises ``ImportError`` with install instructions when onnxruntime is
    not installed — the classical fallback needs nothing.
    """

    def __init__(
        self,
        model_path: str | Path,
        providers: list[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "OnnxDenoiser needs the 'onnxruntime' package "
                "(pip install onnxruntime); without it, use "
                "ClassicalDenoiser, which has no dependencies."
            ) from exc
        self._ort = ort
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        available = set(ort.get_available_providers())
        wanted = providers or ["CPUExecutionProvider"]
        use = [p for p in wanted if p in available]
        skipped = [p for p in wanted if p not in available]
        if skipped:
            print(f"OnnxDenoiser: providers not available, skipping: "
                  f"{skipped} (have: {sorted(available)})")
        if "CPUExecutionProvider" not in use:
            use.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(str(model_path), providers=use)
        self.providers = use
        self._input_name = self.session.get_inputs()[0].name
        self._input_shape = self.session.get_inputs()[0].shape
        print(f"OnnxDenoiser: {model_path.name} providers={use}")

    def _prepare(self, image: np.ndarray) -> tuple[np.ndarray, tuple]:
        """Normalize to the model's expected input layout; remember how."""
        img = np.asarray(image, dtype=np.float32)
        shape = [int(d) if isinstance(d, int) else -1
                 for d in self._input_shape]
        # Common layouts: (1,1,H,W), (1,H,W,1), (1,H,W), (H,W).
        if len(shape) == 4 and shape[1] == 1:
            return img[None, None, :, :], ("nchw", img.shape)
        if len(shape) == 4 and shape[3] == 1:
            return img[None, :, :, None], ("nhwc", img.shape)
        if len(shape) == 3:
            return img[None, :, :], ("chw", img.shape)
        return img, ("hw", img.shape)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        blob, (layout, orig_shape) = self._prepare(image)
        out = self.session.run(None, {self._input_name: blob})[0]
        arr = np.asarray(out, dtype=np.float32)
        if layout == "nchw":
            arr = arr[0, 0]
        elif layout == "nhwc":
            arr = arr[0, :, :, 0]
        elif layout == "chw":
            arr = arr[0]
        if arr.shape != orig_shape:
            raise RuntimeError(
                f"ONNX model changed the image shape {orig_shape} -> "
                f"{arr.shape}; denoise models must be shape-preserving"
            )
        return arr.astype(np.float32)


def make_denoiser(spec: bool | dict | None) -> Denoiser | None:
    """Build a denoiser from a process-config value.

    - ``None``/``False``/``"off"`` → None (no denoising)
    - ``True``/``"classical"`` → :class:`ClassicalDenoiser`
    - ``{"model": "path.onnx", "providers": [...]}`` → :class:`OnnxDenoiser`
    """
    if spec is None or spec is False or spec == "off":
        return None
    if spec is True or spec == "classical":
        return ClassicalDenoiser()
    if isinstance(spec, dict):
        model = spec.get("model") or spec.get("model_path")
        if model:
            return OnnxDenoiser(model, providers=spec.get("providers"))
        return ClassicalDenoiser()
    raise ValueError(f"unrecognized denoise spec: {spec!r}")
