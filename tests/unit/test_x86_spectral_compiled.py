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
        dct_type=kwargs.get("type"),
        input_signature=kwargs.get("input_signature"),
        shape_bounds=kwargs.get("shape_bounds"),
        storage=kwargs.get("storage", "f32"),
        normalization=kwargs.get("normalization", "backward"),
        center=kwargs.get("center", False),
        pad_mode=kwargs.get("pad_mode", "constant"),
        output_length=kwargs.get("length"),
        n_fft=kwargs.get("n_fft"),
        onesided=kwargs.get("onesided", True),
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


def _centered_stft_ref(x, win, hop, pad_mode):
    pad = win.size // 2
    padded = np.pad(
        np.asarray(x, np.float32),
        ((0, 0),) * (np.ndim(x) - 1) + ((pad, pad),),
        mode=pad_mode,
    )
    return _stft_ref(padded, np.asarray(win, np.float32), hop)


def _istft_ref(spectrum, win, hop, *, center, length):
    win = np.asarray(win, np.float32)
    frames = np.fft.irfft(np.asarray(spectrum), n=win.size, axis=-1).astype(np.float32)
    raw_length = (frames.shape[-2] - 1) * hop + win.size
    numerator = np.zeros(frames.shape[:-2] + (raw_length,), np.float32)
    denominator = np.zeros(raw_length, np.float32)
    for frame in range(frames.shape[-2]):
        start = frame * hop
        numerator[..., start:start + win.size] += frames[..., frame, :] * win
        denominator[start:start + win.size] += win * win
    output = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12
    )
    trim = win.size // 2 if center else 0
    return output[..., trim:trim + length]


def _general_stft_ref(x, win, hop, *, n_fft, center, onesided):
    values = np.asarray(x, np.float32)
    window = np.zeros(n_fft, np.float32)
    offset = (n_fft - win.size) // 2
    window[offset:offset + win.size] = np.asarray(win, np.float32)
    if center:
        values = np.pad(values, ((0, 0),) * (values.ndim - 1) +
                        ((n_fft // 2, n_fft // 2),))
    if values.shape[-1] < n_fft:
        values = np.pad(values, ((0, 0),) * (values.ndim - 1) +
                        ((0, n_fft - values.shape[-1]),))
    transform = np.fft.rfft if onesided else np.fft.fft
    return np.stack([
        transform(values[..., start:start + n_fft] * window, axis=-1)
        for start in range(0, values.shape[-1] - n_fft + 1, hop)
    ], axis=-2).astype(np.complex64)


def _general_istft_ref(spectrum, win, hop, *, n_fft, center, length, onesided):
    inverse = np.fft.irfft if onesided else np.fft.ifft
    frames = np.real(inverse(np.asarray(spectrum), n=n_fft, axis=-1)).astype(np.float32)
    window = np.zeros(n_fft, np.float32)
    offset = (n_fft - win.size) // 2
    window[offset:offset + win.size] = np.asarray(win, np.float32)
    raw_length = (frames.shape[-2] - 1) * hop + n_fft
    numerator = np.zeros(frames.shape[:-2] + (raw_length,), np.float32)
    denominator = np.zeros(raw_length, np.float32)
    for frame in range(frames.shape[-2]):
        start = frame * hop
        numerator[..., start:start + n_fft] += frames[..., frame, :] * window
        denominator[start:start + n_fft] += window * window
    output = np.divide(
        numerator, denominator, out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    trim = n_fft // 2 if center else 0
    return output[..., trim:trim + length]


_TOL = dict(atol=2e-3, rtol=2e-3)


def _dct2_ref(x, axis=-1, scale=1.0):
    moved = np.moveaxis(np.asarray(x, np.float64), axis, -1)
    n = moved.shape[-1]
    source = np.arange(n, dtype=np.float64)
    frequency = np.arange(n, dtype=np.float64)
    basis = 2.0 * np.cos(
        np.pi * np.outer(2.0 * source + 1.0, frequency) / float(2 * n)
    )
    return np.moveaxis(moved @ basis, -1, axis) * scale


@pytest.mark.parametrize("shape_pre", [(), (3,)])
def test_dct(shape_pre):
    rt = _rt_or_skip()
    rng = np.random.default_rng(1 + len(shape_pre))
    x = rng.standard_normal(shape_pre + (8,)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.dct", (x,), {"axis": -1}), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_spectral_compiled"
    ref = _dct2_ref(x).astype(np.float32)
    np.testing.assert_allclose(np.asarray(res["output"]), ref, **_TOL)


@pytest.mark.parametrize("dct_type", [1, 3, 4])
def test_extended_dct_types_use_distinct_physical_contracts(dct_type):
    import tessera

    rt = _rt_or_skip()
    x = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(2, 8)
    result = rt.launch(
        _art(rt, "tessera.dct", (x,), {"type": dct_type}), (x,)
    )
    assert result["ok"] is True, result.get("reason")
    reference = tessera.ops.dct(x, type=dct_type)
    np.testing.assert_allclose(np.asarray(result["output"]), reference, **_TOL)


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


def test_scalar_convolution_preserves_length_one_contract():
    rt = _rt_or_skip()
    x = np.array([[2.0], [-3.0], [0.5]], dtype=np.float32)
    w = np.array([[4.0], [2.0], [-8.0]], dtype=np.float32)
    result = rt.launch(
        _art(rt, "tessera.spectral_conv", (x, w), {}), (x, w)
    )
    assert result["ok"] is True, result.get("reason")
    np.testing.assert_array_equal(np.asarray(result["output"]), x * w)


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


def test_unit_window_stft_istft_uses_odd_fallback():
    rt = _rt_or_skip()
    signal = np.array([1.0, -2.0, 3.5, 0.25], dtype=np.float32)
    window = np.ones((1,), dtype=np.float32)
    stft = rt.launch(
        _art(rt, "tessera.stft", (signal, window), {"hop": 1}),
        (signal, window),
    )
    assert stft["ok"] is True, stft.get("reason")
    spectrum = np.asarray(stft["output"])
    np.testing.assert_array_equal(spectrum[:, 0].real, signal)
    np.testing.assert_array_equal(spectrum[:, 0].imag, 0.0)
    istft = rt.launch(
        _art(rt, "tessera.istft", (spectrum, window), {"hop": 1}),
        (spectrum, window),
    )
    assert istft["ok"] is True, istft.get("reason")
    np.testing.assert_array_equal(np.asarray(istft["output"]), signal)


def test_ragged_prime_bluestein_package():
    rt = _rt_or_skip()
    rng = np.random.default_rng(9)
    x = rng.standard_normal((2, 19)).astype(np.float32)
    d = rt.launch(_art(rt, "tessera.dct", (x,), {}), (x,))
    assert d["ok"] is True, d.get("reason")
    reference = _dct2_ref(x).astype(np.float32)
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
    reference = _dct2_ref(x).astype(np.float32)
    np.testing.assert_allclose(np.asarray(result["output"]), reference, **_TOL)


def test_arbitrary_axis_dct_conv_and_stft_istft():
    rt = _rt_or_skip()
    rng = np.random.default_rng(13)

    x = rng.standard_normal((8, 3)).astype(np.float32)
    dct = rt.launch(_art(rt, "tessera.dct", (x,), {"axis": 0}), (x,))
    assert dct["ok"] is True, dct.get("reason")
    dct_ref = _dct2_ref(x, axis=0).astype(np.float32)
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
    reference = _dct2_ref(source).astype(np.float32)
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
    scale = 1.0 / (16.0 if normalization == "forward" else np.sqrt(16.0))
    reference = _dct2_ref(x, scale=scale)
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
    reference = _dct2_ref(source, axis=0, scale=1.0 / np.sqrt(128.0))
    np.testing.assert_allclose(
        np.asarray(result["output"]).astype(np.float32), reference,
        atol=0.15 if storage == "bf16" else 4e-2, rtol=4e-2,
    )


@pytest.mark.parametrize("pad_mode", ["constant", "reflect"])
@pytest.mark.parametrize("storage", ["f32", "f16", "bf16"])
def test_centered_ragged_stft_and_cropped_istft_native_policy(storage, pad_mode):
    rt = _rt_or_skip()
    dtype = np.float32 if storage == "f32" else np.float16
    if storage == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(801)
    signal = rng.normal(size=(2, 46)).astype(dtype)
    window = (np.hanning(18) + 0.25).astype(dtype)
    stft = rt.launch(
        _art(rt, "tessera.stft", (signal, window), {
            "hop": 7, "center": True, "pad_mode": pad_mode, "storage": storage,
        }),
        (signal, window),
    )
    assert stft["ok"] is True, stft.get("reason")
    tolerance = {"f32": 3e-3, "f16": 4e-2, "bf16": 2e-1}[storage]
    np.testing.assert_allclose(
        stft["output"], _centered_stft_ref(signal, window, 7, pad_mode),
        atol=tolerance, rtol=tolerance,
    )
    spectrum = np.asarray(stft["output"], np.complex64)
    istft = rt.launch(
        _art(rt, "tessera.istft", (spectrum, window), {
            "hop": 7, "center": True, "length": 40, "storage": storage,
        }),
        (spectrum, window),
    )
    assert istft["ok"] is True, istft.get("reason")
    np.testing.assert_allclose(
        np.asarray(istft["output"], np.float32),
        _istft_ref(spectrum, window, 7, center=True, length=40),
        atol=tolerance, rtol=tolerance,
    )


def test_centered_policy_executes_true_noncontiguous_stride_abi():
    rt = _rt_or_skip()
    signal = np.arange(92, dtype=np.float32).reshape(46, 2).T
    assert not signal.flags.c_contiguous
    window = np.linspace(0.25, 1.25, 36, dtype=np.float32)[::2]
    assert not window.flags.c_contiguous
    artifact = _art(rt, "tessera.stft", (signal, window), {
        "hop": 7, "center": True, "pad_mode": "constant",
    })
    result = rt.launch(artifact, (signal, window))
    assert result["ok"] is True, result.get("reason")
    np.testing.assert_allclose(
        result["output"], _centered_stft_ref(signal, window, 7, "constant"),
        **_TOL,
    )


def test_centered_cropped_arbitrary_axis_native_policy():
    rt = _rt_or_skip()
    rng = np.random.default_rng(803)
    signal = rng.normal(size=(2, 46, 3)).astype(np.float32)
    window = (np.hanning(18) + 0.25).astype(np.float32)
    stft = rt.launch(
        _art(rt, "tessera.stft", (signal, window), {
            "axis": 1, "hop": 7, "center": True, "pad_mode": "reflect",
        }), (signal, window),
    )
    assert stft["ok"] is True, stft.get("reason")
    moved_signal = np.moveaxis(signal, 1, -1)
    moved_ref = _centered_stft_ref(moved_signal, window, 7, "reflect")
    reference = np.moveaxis(moved_ref, (-2, -1), (1, 2))
    np.testing.assert_allclose(stft["output"], reference, **_TOL)
    spectrum = np.asarray(stft["output"], np.complex64)
    istft = rt.launch(
        _art(rt, "tessera.istft", (spectrum, window), {
            "axis": 2, "hop": 7, "center": True, "length": 40,
        }), (spectrum, window),
    )
    assert istft["ok"] is True, istft.get("reason")
    moved_spectrum = np.moveaxis(spectrum, (1, 2), (-2, -1))
    moved_output = _istft_ref(
        moved_spectrum, window, 7, center=True, length=40
    )
    np.testing.assert_allclose(
        istft["output"], np.moveaxis(moved_output, -1, 1), **_TOL
    )


@pytest.mark.parametrize("onesided", [True, False])
def test_broader_nfft_short_window_full_spectrum_and_strides(onesided):
    rt = _rt_or_skip()
    rng = np.random.default_rng(805 + int(onesided))
    signal = rng.normal(size=(44, 3, 2)).astype(np.float32).transpose(1, 0, 2)
    window = (np.hanning(30).astype(np.float32) + 0.2)[::2]
    assert not signal.flags.c_contiguous and not window.flags.c_contiguous
    stft = rt.launch(
        _art(rt, "tessera.stft", (signal, window), {
            "axis": 1, "hop": 6, "n_fft": 20, "center": True,
            "onesided": onesided,
        }), (signal, window),
    )
    assert stft["ok"] is True, stft.get("reason")
    moved = np.moveaxis(signal, 1, -1)
    reference = _general_stft_ref(
        moved, window, 6, n_fft=20, center=True, onesided=onesided
    )
    reference = np.moveaxis(reference, (-2, -1), (1, 2))
    np.testing.assert_allclose(stft["output"], reference, atol=4e-3, rtol=4e-3)

    spectral_output = np.asarray(stft["output"], np.complex64)
    holder = np.empty(spectral_output.shape[:-1] + (2 * spectral_output.shape[-1],),
                      np.complex64)
    holder[..., ::2] = spectral_output
    spectrum = holder[..., ::2]
    assert not spectrum.flags.c_contiguous
    istft = rt.launch(
        _art(rt, "tessera.istft", (spectrum, window), {
            "axis": 2, "hop": 6, "n_fft": 20, "center": True,
            "length": 38, "onesided": onesided,
        }), (spectrum, window),
    )
    assert istft["ok"] is True, istft.get("reason")
    moved_spectrum = np.moveaxis(spectrum, (1, 2), (-2, -1))
    expected = _general_istft_ref(
        moved_spectrum, window, 6, n_fft=20, center=True, length=38,
        onesided=onesided,
    )
    np.testing.assert_allclose(
        istft["output"], np.moveaxis(expected, -1, 1), atol=5e-3, rtol=5e-3,
    )


def test_per_batch_window_broadcast_executes_in_native_package():
    rt = _rt_or_skip()
    rng = np.random.default_rng(808)
    signal = rng.standard_normal((2, 48, 3)).astype(np.float32)[:, ::2, :]
    windows = np.stack((np.hanning(8), np.hamming(8)), axis=0).astype(
        np.float32
    )[:, None, :]
    kwargs = {
        "axis": 1, "hop": 4, "n_fft": 10, "center": False,
        "onesided": False,
    }
    result = rt.launch(
        _art(rt, "tessera.stft", (signal, windows), kwargs),
        (signal, windows),
    )
    assert result["ok"] is True, result.get("reason")
    actual = np.asarray(result["output"])
    expected = np.empty_like(actual)
    for batch in range(signal.shape[0]):
        for channel in range(signal.shape[2]):
            expected[batch, :, :, channel] = _general_stft_ref(
                signal[batch, :, channel], windows[batch, 0], 4,
                n_fft=10, center=False, onesided=False,
            )
    np.testing.assert_allclose(actual, expected, **_TOL)

    inverse_kwargs = {
        "axis": 2, "hop": 4, "n_fft": 10, "center": False,
        "onesided": False, "length": 22,
    }
    inverse = rt.launch(
        _art(rt, "tessera.istft", (actual, windows), inverse_kwargs),
        (actual, windows),
    )
    assert inverse["ok"] is True, inverse.get("reason")
    inverse_actual = np.asarray(inverse["output"])
    inverse_expected = np.empty_like(inverse_actual)
    for batch in range(signal.shape[0]):
        for channel in range(signal.shape[2]):
            inverse_expected[batch, :, channel] = _general_istft_ref(
                actual[batch, :, :, channel], windows[batch, 0], 4,
                n_fft=10, center=False, length=22, onesided=False,
            )
    np.testing.assert_allclose(inverse_actual, inverse_expected, **_TOL)
