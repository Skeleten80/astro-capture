"""denoise.py: classical fallback improves SNR; ONNX errors are clear."""

import numpy as np
import pytest

from astrocapture.denoise import (
    ClassicalDenoiser,
    Denoiser,
    OnnxDenoiser,
    make_denoiser,
)


def noisy_gradient(shape=(64, 64), noise=8.0, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    clean = 1000.0 + 3.0 * xx + 2.0 * yy
    return clean, clean + rng.normal(0, noise, shape)


def snr(clean, img):
    err = img - clean
    return 10.0 * np.log10(np.mean(clean ** 2) / np.mean(err ** 2))


def test_abc_not_instantiable():
    with pytest.raises(TypeError):
        Denoiser()


def test_classical_denoiser_improves_snr():
    clean, noisy = noisy_gradient()
    out = ClassicalDenoiser().denoise(noisy)
    assert snr(clean, out) > snr(clean, noisy) + 1.0  # measurable gain


def test_classical_denoiser_preserves_shape_and_edges():
    rng = np.random.default_rng(1)
    img = np.full((48, 48), 1000.0)
    img[:, 24:] = 5000.0  # hard vertical edge
    noisy = img + rng.normal(0, 10.0, img.shape)
    out = ClassicalDenoiser().denoise(noisy)
    assert out.shape == noisy.shape and out.dtype == np.float32
    # Edge survives: column means still jump at x=24.
    assert float(out[:, 30].mean() - out[:, 18].mean()) > 2000.0


def test_classical_denoiser_rejects_2d_only():
    with pytest.raises(ValueError, match="2-D"):
        ClassicalDenoiser().denoise(np.zeros((4, 4, 4)))


def test_onnx_missing_dependency_clear_error():
    try:
        import onnxruntime  # noqa: F401
        pytest.skip("onnxruntime is installed here")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="onnxruntime"):
        OnnxDenoiser("whatever.onnx")


def test_onnx_missing_model_file(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    with pytest.raises(FileNotFoundError, match="not found"):
        OnnxDenoiser(tmp_path / "nope.onnx")


def test_make_denoiser_specs():
    assert make_denoiser(None) is None
    assert make_denoiser(False) is None
    assert make_denoiser("off") is None
    assert isinstance(make_denoiser(True), ClassicalDenoiser)
    assert isinstance(make_denoiser("classical"), ClassicalDenoiser)
    assert isinstance(make_denoiser({}), ClassicalDenoiser)
    with pytest.raises(ValueError, match="unrecognized"):
        make_denoiser("fancy")


def test_process_denoise_writes_outputs(tmp_path):
    from astropy.io import fits as _fits

    from astrocapture.process import process_session
    sess = tmp_path / "sess"
    (sess / "lights").mkdir(parents=True)
    rng = np.random.default_rng(2)
    for i in range(3):
        yy, xx = np.mgrid[0:48, 0:48]
        img = 1000.0 + 2.0 * xx + rng.normal(0, 8.0, (48, 48))
        hdr = _fits.Header()
        hdr["EXPTIME"] = 25.0
        p = sess / "lights" / f"l_{i:03d}.fits"
        _fits.writeto(p, img.astype(np.float32), hdr, overwrite=True)
    stats = process_session(sess, tmp_path / "out", denoise=True)
    assert stats["denoised"] is True
    assert stats["denoise_backend"] == "ClassicalDenoiser"
    assert (tmp_path / "out" / "stacked_denoised.fits").is_file()
    assert (tmp_path / "out" / "stacked_denoised.png").is_file()


def test_process_bad_denoise_spec_fails_fast(tmp_path):
    from astrocapture.process import process_session
    with pytest.raises(ValueError, match="unrecognized"):
        process_session(tmp_path / "sess", tmp_path / "out",
                        denoise="fancy")
