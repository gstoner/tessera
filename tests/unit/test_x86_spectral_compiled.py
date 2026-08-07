"""Typed compound Schedule→Tile spectral packages on x86 AVX-512.

The native image owns framing, windowing, overlap-add, pointwise complex work,
and digest-keyed persistent workspace; runtime never re-enters Graph metadata.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op_name, operands, kwargs):
    from tessera.compiler.scheduled_spectral import lower_scheduled_spectral

    names = [f"a{i}" for i in range(len(operands))]
    scheduled = lower_scheduled_spectral(
        target="x86",
        op_name=op_name,
        input_shapes=tuple(tuple(int(dim) for dim in value.shape) for value in operands),
        axis=int(kwargs.get("axis", -1)),
        hop=kwargs.get("hop"),
        input_signature=kwargs.get("input_signature"),
        shape_bounds=kwargs.get("shape_bounds"),
        storage=kwargs.get("storage", "f32"),
        normalization=kwargs.get("normalization", "backward"),
    ).to_metadata()
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_spectral_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "scheduled_spectral": scheduled,
    })


def _stft_ref(x, win, hop):
    wl = win.shape[-1]
    return np.stack(
        [np.fft.rfft(x[..., s:s + wl] * win, axis=-1)
         for s in range(0, max(1, x.shape[-1] - wl + 1), hop)], axis=-2)


_TOL = dict(atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("shape_pre", [(), (3,)])
def test_dct(shape_pre):
    rt = _rt_or_skip()
    rng = np.random.default_rng(1 + len(shape_pre))
    x = rng.standard_normal(shape_pre + (8,)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.dct", (x,), {"axis": -1}), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_spectral_compiled"
    y = np.concatenate([x, np.flip(x, -1)], -1).astype(np.complex64)
    ref = np.real(np.fft.fft(y, axis=-1)[..., :8]).astype(np.float32)
    np.testing.assert_allclose(np.asarray(res["output"]), ref, **_TOL)


def test_spectral_conv():
    rt = _rt_or_skip()
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 12)).astype(np.float32)
    w = rng.standard_normal((3, 5)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.spectral_conv", (x, w), {}), (x, w))
    assert res["ok"] is True, res.get("reason")
    n = x.shape[-1] + w.shape[-1] - 1
    nfft = 1 << int(np.ceil(np.log2(n)))
    ref = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(w, nfft), nfft)[..., :n]
    np.testing.assert_allclose(np.asarray(res["output"]), ref, **_TOL)


def test_spectral_filter():
    rt = _rt_or_skip()
    rng = np.random.default_rng(3)
    Xf = (rng.standard_normal((2, 5)) + 1j * rng.standard_normal((2, 5))
          ).astype(np.complex64)
    Hf = (rng.standard_normal((2, 5)) + 1j * rng.standard_normal((2, 5))
          ).astype(np.complex64)
    res = rt.launch(_art(rt, "tessera.spectral_filter", (Xf, Hf), {}), (Xf, Hf))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.complex64),
                               (Xf * Hf).astype(np.complex64), atol=1e-4)


def test_stft_istft():
    rt = _rt_or_skip()
    rng = np.random.default_rng(4)
    sig = rng.standard_normal((32,)).astype(np.float32)
    win = np.hanning(8).astype(np.float32)
    s = rt.launch(_art(rt, "tessera.stft", (sig, win), {"hop": 4}), (sig, win))
    assert s["ok"] is True, s.get("reason")
    sref = _stft_ref(sig, win, 4)
    out = np.asarray(s["output"])
    assert out.shape == sref.shape
    np.testing.assert_allclose(out.astype(np.complex64),
                               sref.astype(np.complex64), **_TOL)
    i = rt.launch(_art(rt, "tessera.istft", (out, win), {"hop": 4}), (out, win))
    assert i["ok"] is True, i.get("reason")
    # round-trip recovers the windowed-overlap region (COLA on hann/hop=4)
    rec = np.asarray(i["output"])
    assert rec.shape[-1] == (sref.shape[-2] - 1) * 4 + 8


def test_ragged_prime_bluestein_package():
    rt = _rt_or_skip()
    rng = np.random.default_rng(9)
    x = rng.standard_normal((2, 19)).astype(np.float32)
    d = rt.launch(_art(rt, "tessera.dct", (x,), {}), (x,))
    assert d["ok"] is True, d.get("reason")
    mirrored = np.concatenate([x, np.flip(x, -1)], -1).astype(np.complex64)
    reference = np.fft.fft(mirrored, axis=-1)[..., :19].real.astype(np.float32)
    np.testing.assert_allclose(np.asarray(d["output"]), reference, **_TOL)

    signal = rng.standard_normal((2, 47)).astype(np.float32)
    window = np.hanning(19).astype(np.float32)
    result = rt.launch(
        _art(rt, "tessera.stft", (signal, window), {"hop": 7}),
        (signal, window),
    )
    assert result["ok"] is True, result.get("reason")
    np.testing.assert_allclose(
        np.asarray(result["output"]), _stft_ref(signal, window, 7), **_TOL
    )


def test_bounded_dynamic_specialization_rebuilds_exact_artifact():
    rt = _rt_or_skip()
    seed = np.arange(16, dtype=np.float32).reshape(2, 8)
    artifact = _art(
        rt,
        "tessera.dct",
        (seed,),
        {
            "input_signature": ((None, 8),),
            "shape_bounds": ((4, 8),),
        },
    )
    x = np.arange(24, dtype=np.float32).reshape(3, 8)
    result = rt.launch(artifact, (x,))
    assert result["ok"] is True, result.get("reason")
    mirrored = np.concatenate([x, np.flip(x, -1)], -1).astype(np.complex64)
    reference = np.fft.fft(mirrored, axis=-1)[..., :8].real.astype(np.float32)
    np.testing.assert_allclose(np.asarray(result["output"]), reference, **_TOL)


def test_arbitrary_axis_dct_conv_and_stft_istft():
    rt = _rt_or_skip()
    rng = np.random.default_rng(13)

    x = rng.standard_normal((8, 3)).astype(np.float32)
    dct = rt.launch(_art(rt, "tessera.dct", (x,), {"axis": 0}), (x,))
    assert dct["ok"] is True, dct.get("reason")
    mirrored = np.concatenate([x, np.flip(x, 0)], axis=0).astype(np.complex64)
    dct_ref = np.fft.fft(mirrored, axis=0)[:8].real.astype(np.float32)
    np.testing.assert_allclose(np.asarray(dct["output"]), dct_ref, **_TOL)

    signal = rng.standard_normal((12, 3)).astype(np.float32)
    kernel = rng.standard_normal((5, 3)).astype(np.float32)
    conv = rt.launch(
        _art(rt, "tessera.spectral_conv", (signal, kernel), {"axis": 0}),
        (signal, kernel),
    )
    assert conv["ok"] is True, conv.get("reason")
    conv_ref = np.stack(
        [np.convolve(signal[:, column], kernel[:, column]) for column in range(3)],
        axis=1,
    ).astype(np.float32)
    np.testing.assert_allclose(np.asarray(conv["output"]), conv_ref, **_TOL)

    wave = rng.standard_normal((32, 2)).astype(np.float32)
    window = np.hanning(8).astype(np.float32)
    stft_artifact = _art(rt, "tessera.stft", (wave, window), {"axis": 0, "hop": 4})
    stft = rt.launch(stft_artifact, (wave, window))
    assert stft["ok"] is True, stft.get("reason")
    stft_ref = np.moveaxis(_stft_ref(np.moveaxis(wave, 0, -1), window, 4), (-2, -1), (0, 1))
    np.testing.assert_allclose(np.asarray(stft["output"]), stft_ref, **_TOL)

    spectra = np.asarray(stft["output"])
    istft = rt.launch(
        _art(rt, "tessera.istft", (spectra, window), {"axis": 1, "hop": 4}),
        (spectra, window),
    )
    assert istft["ok"] is True, istft.get("reason")
    assert np.asarray(istft["output"]).shape == wave.shape


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_reduced_storage_dct_uses_f32_accumulation(storage):
    rt = _rt_or_skip()
    dtype = np.float16
    if storage == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    x = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(3, 8).astype(dtype)
    result = rt.launch(_art(rt, "tessera.dct", (x,), {"storage": storage}), (x,))
    assert result["ok"] is True, result.get("reason")
    assert str(np.asarray(result["output"]).dtype) == str(np.dtype(dtype))
    source = x.astype(np.float32)
    mirrored = np.concatenate([source, np.flip(source, -1)], -1).astype(np.complex64)
    reference = np.fft.fft(mirrored, axis=-1)[..., :8].real.astype(np.float32)
    np.testing.assert_allclose(
        np.asarray(result["output"]).astype(np.float32), reference,
        atol=3e-2, rtol=3e-2,
    )


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_reduced_storage_conv_stft_and_istft_native_package(storage):
    rt = _rt_or_skip()
    dtype = np.float16
    if storage == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(41)
    x = rng.standard_normal((2, 12)).astype(dtype)
    kernel = rng.standard_normal((2, 5)).astype(dtype)
    conv = rt.launch(
        _art(rt, "tessera.spectral_conv", (x, kernel), {"storage": storage}),
        (x, kernel),
    )
    assert conv["ok"] is True, conv.get("reason")
    conv_ref = np.stack([
        np.convolve(x[row].astype(np.float32), kernel[row].astype(np.float32))
        for row in range(2)
    ])
    np.testing.assert_allclose(
        np.asarray(conv["output"]).astype(np.float32), conv_ref,
        atol=0.3 if storage == "bf16" else 4e-2, rtol=4e-2,
    )

    signal = rng.standard_normal((24,)).astype(dtype)
    window = np.hanning(8).astype(dtype)
    stft = rt.launch(
        _art(rt, "tessera.stft", (signal, window),
             {"hop": 4, "storage": storage}),
        (signal, window),
    )
    assert stft["ok"] is True, stft.get("reason")
    stft_ref = _stft_ref(signal.astype(np.float32), window.astype(np.float32), 4)
    np.testing.assert_allclose(np.asarray(stft["output"]), stft_ref,
                               atol=5e-2, rtol=5e-2)
    spectra = np.asarray(stft["output"]).astype(np.complex64)
    istft = rt.launch(
        _art(rt, "tessera.istft", (spectra, window),
             {"hop": 4, "storage": storage}),
        (spectra, window),
    )
    assert istft["ok"] is True, istft.get("reason")
    assert str(np.asarray(istft["output"]).dtype) == str(np.dtype(dtype))


@pytest.mark.parametrize("normalization", ["forward", "ortho"])
def test_native_forward_and_ortho_normalization(normalization):
    rt = _rt_or_skip()
    rng = np.random.default_rng(29)
    x = rng.standard_normal((2, 8)).astype(np.float32)
    dct = rt.launch(
        _art(rt, "tessera.dct", (x,), {"normalization": normalization}), (x,)
    )
    assert dct["ok"] is True, dct.get("reason")
    mirrored = np.concatenate([x, np.flip(x, -1)], -1).astype(np.complex64)
    scale = 1.0 / (16.0 if normalization == "forward" else np.sqrt(16.0))
    reference = np.fft.fft(mirrored, axis=-1)[..., :8].real * scale
    np.testing.assert_allclose(np.asarray(dct["output"]), reference, **_TOL)

    signal = rng.standard_normal((24,)).astype(np.float32)
    window = np.hanning(8).astype(np.float32)
    stft = rt.launch(
        _art(rt, "tessera.stft", (signal, window),
             {"hop": 4, "normalization": normalization}),
        (signal, window),
    )
    assert stft["ok"] is True, stft.get("reason")
    scale = 1.0 / (8.0 if normalization == "forward" else np.sqrt(8.0))
    np.testing.assert_allclose(
        np.asarray(stft["output"]), _stft_ref(signal, window, 4) * scale, **_TOL
    )
    spectra = _stft_ref(signal, window, 4).astype(np.complex64)
    backward = rt.launch(
        _art(rt, "tessera.istft", (spectra, window), {"hop": 4}),
        (spectra, window),
    )
    inverse = rt.launch(
        _art(rt, "tessera.istft", (spectra, window),
             {"hop": 4, "normalization": normalization}),
        (spectra, window),
    )
    assert backward["ok"] is True and inverse["ok"] is True
    inverse_scale = 8.0 if normalization == "forward" else np.sqrt(8.0)
    np.testing.assert_allclose(
        np.asarray(inverse["output"]), np.asarray(backward["output"]) * inverse_scale,
        **_TOL,
    )


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_combined_dynamic_axis_reduced_storage_ortho_policy(storage):
    rt = _rt_or_skip()
    dtype = np.float16
    if storage == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(73)
    x = rng.standard_normal((64, 6)).astype(dtype)
    seed = x[:, :4]
    artifact = _art(
        rt,
        "tessera.dct",
        (seed,),
        {
            "axis": 0,
            "storage": storage,
            "normalization": "ortho",
            "input_signature": ((64, None),),
            "shape_bounds": ((64, 8),),
        },
    )
    result = rt.launch(artifact, (x,))
    assert result["ok"] is True, result.get("reason")
    source = x.astype(np.float32)
    mirrored = np.concatenate([source, np.flip(source, 0)], axis=0).astype(np.complex64)
    reference = np.fft.fft(mirrored, axis=0)[:64].real / np.sqrt(128.0)
    np.testing.assert_allclose(
        np.asarray(result["output"]).astype(np.float32), reference,
        atol=0.15 if storage == "bf16" else 4e-2, rtol=4e-2,
    )
