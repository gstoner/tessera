"""Content-addressed TSOL composites in the prebuilt gfx1151 device package.

Framing, windowing, pointwise work, and overlap-add are native HIP work around
digest-bound persistent child FFT plans; no host NumPy composition is allowed.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, op_name, operands, kwargs):
    from tessera.compiler.scheduled_spectral import lower_scheduled_spectral

    names = [f"a{i}" for i in range(len(operands))]
    contract = lower_scheduled_spectral(
        target="rocm",
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
        "target": "rocm", "compiler_path": "rocm_spectral_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "scheduled_spectral": contract,
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


_TOL = dict(atol=1e-2, rtol=1e-2)


def _dct2_ref(x, axis=-1, scale=1.0):
    moved = np.moveaxis(np.asarray(x, np.float64), axis, -1)
    n = moved.shape[-1]
    source = np.arange(n, dtype=np.float64)
    frequency = np.arange(n, dtype=np.float64)
    basis = 2.0 * np.cos(
        np.pi * np.outer(2.0 * source + 1.0, frequency) / float(2 * n)
    )
    return np.moveaxis(moved @ basis, -1, axis) * scale


def test_dct():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 8)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.dct", (x,), {"axis": -1}), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_spectral_compiled"
    ref = _dct2_ref(x).astype(np.float32)
    np.testing.assert_allclose(np.asarray(res["output"]), ref, **_TOL)


@pytest.mark.parametrize("dct_type", [1, 3, 4])
def test_extended_dct_types_use_distinct_physical_contracts(dct_type):
    import tessera

    rt = _rocm_or_skip()
    x = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(2, 8)
    result = rt.launch(
        _art(rt, "tessera.dct", (x,), {"type": dct_type}), (x,)
    )
    assert result["ok"] is True, result.get("reason")
    reference = tessera.ops.dct(x, type=dct_type)
    np.testing.assert_allclose(np.asarray(result["output"]), reference, **_TOL)


def test_spectral_conv():
    rt = _rocm_or_skip()
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
    rt = _rocm_or_skip()
    x = np.array([[2.0], [-3.0], [0.5]], dtype=np.float32)
    w = np.array([[4.0], [2.0], [-8.0]], dtype=np.float32)
    result = rt.launch(
        _art(rt, "tessera.spectral_conv", (x, w), {}), (x, w)
    )
    assert result["ok"] is True, result.get("reason")
    np.testing.assert_array_equal(np.asarray(result["output"]), x * w)


def test_standalone_convolution_export_uses_packed_plans():
    _rocm_or_skip()
    from tessera.compiler.emit import spectral_candidates as candidates

    lib = candidates._amd_composite_lib()
    assert lib is not None
    fft_n = 8
    _, forward = candidates._rocm_plan(fft_n // 2, -1, "a" * 64)
    _, inverse = candidates._rocm_plan(fft_n // 2, 1, "b" * 64)
    x = np.array([[1.0, -2.0, 3.0, 0.5, 4.0],
                  [-1.0, 0.25, 2.0, -3.0, 1.5]], dtype=np.float32)
    w = np.array([[2.0, -1.0, 0.5],
                  [0.5, 1.0, -2.0]], dtype=np.float32)
    output = np.empty((2, 7), dtype=np.float32)
    rc = lib.ts_spectral_conv_hostptr_batch_amd(
        forward, inverse, candidates._cptr(x), x.shape[1],
        candidates._cptr(w), w.shape[1], candidates._cptr(output),
        x.shape[0], fft_n,
    )
    assert rc == 0
    reference = np.stack([np.convolve(x[row], w[row]) for row in range(2)])
    np.testing.assert_allclose(output, reference, **_TOL)


def test_spectral_filter():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(3)
    Xf = (rng.standard_normal((2, 5)) + 1j * rng.standard_normal((2, 5))
          ).astype(np.complex64)
    Hf = (rng.standard_normal((2, 5)) + 1j * rng.standard_normal((2, 5))
          ).astype(np.complex64)
    res = rt.launch(_art(rt, "tessera.spectral_filter", (Xf, Hf), {}), (Xf, Hf))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.complex64),
                               (Xf * Hf).astype(np.complex64), atol=1e-4)


def test_composite_workspace_plan_is_reused_by_artifact_digest():
    rt = _rocm_or_skip()
    from tessera.compiler.emit import spectral_candidates as candidates

    a = np.ones((2, 5), dtype=np.complex64)
    b = np.full((2, 5), 2 + 1j, dtype=np.complex64)
    artifact = _art(rt, "tessera.spectral_filter", (a, b), {})
    contract = artifact.metadata["scheduled_spectral"]
    lib = candidates._amd_composite_lib()
    assert lib is not None
    assert lib.ts_spectral_composite_package_abi_amd() == b"tessera.rocm.spectral_composite.v7"
    assert lib.ts_spectral_composite_arch_amd() == b"gfx1151"

    first = candidates._rocm_composite_plan(contract, lib)
    result = rt.launch(artifact, (a, b))
    second = candidates._rocm_composite_plan(contract, lib)

    assert result["ok"] is True, result.get("reason")
    assert first.value == second.value
    assert lib.ts_spectral_composite_plan_digest_amd(first) == contract[
        "schedule_digest"
    ].encode("ascii")
    assert lib.ts_spectral_composite_plan_workspace_bytes_amd(first) == contract[
        "workspace_bytes"
    ]


def test_stft_istft():
    rt = _rocm_or_skip()
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
    rec = np.asarray(i["output"])
    assert rec.shape[-1] == (sref.shape[-2] - 1) * 4 + 8


def test_unit_window_stft_istft_uses_odd_fallback():
    rt = _rocm_or_skip()
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


def test_ragged_batched_bluestein_composites():
    """Prime transform lengths exercise child Bluestein plans, not radix-only fixtures."""
    rt = _rocm_or_skip()
    rng = np.random.default_rng(5)

    x = rng.standard_normal((2, 17)).astype(np.float32)
    d = rt.launch(_art(rt, "tessera.dct", (x,), {}), (x,))
    assert d["ok"] is True, d.get("reason")
    dref = _dct2_ref(x).astype(np.float32)
    np.testing.assert_allclose(np.asarray(d["output"]), dref, **_TOL)

    signal = rng.standard_normal((2, 43)).astype(np.float32)
    window = np.hanning(17).astype(np.float32)
    s = rt.launch(
        _art(rt, "tessera.stft", (signal, window), {"hop": 6}),
        (signal, window),
    )
    assert s["ok"] is True, s.get("reason")
    sref = _stft_ref(signal, window, 6)
    np.testing.assert_allclose(np.asarray(s["output"]), sref, **_TOL)

    i = rt.launch(
        _art(rt, "tessera.istft", (np.asarray(s["output"]), window), {"hop": 6}),
        (np.asarray(s["output"]), window),
    )
    assert i["ok"] is True, i.get("reason")
    assert np.asarray(i["output"]).shape == (2, 41)


def test_bounded_dynamic_and_arbitrary_axis_dct():
    rt = _rocm_or_skip()
    seed = np.arange(16, dtype=np.float32).reshape(8, 2)
    artifact = _art(
        rt,
        "tessera.dct",
        (seed,),
        {
            "axis": 0,
            "input_signature": ((8, None),),
            "shape_bounds": ((8, 4),),
        },
    )
    x = np.arange(24, dtype=np.float32).reshape(8, 3)
    result = rt.launch(artifact, (x,))
    assert result["ok"] is True, result.get("reason")
    reference = _dct2_ref(x, axis=0).astype(np.float32)
    np.testing.assert_allclose(np.asarray(result["output"]), reference, **_TOL)


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_reduced_storage_dct_uses_f32_accumulation(storage):
    rt = _rocm_or_skip()
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
        atol=4e-2, rtol=4e-2,
    )


@pytest.mark.parametrize("storage", ["f16", "bf16"])
def test_reduced_storage_conv_stft_and_istft_native_package(storage):
    rt = _rocm_or_skip()
    dtype = np.float16
    if storage == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(43)
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
        atol=0.3 if storage == "bf16" else 5e-2, rtol=5e-2,
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
                               atol=6e-2, rtol=6e-2)
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
    rt = _rocm_or_skip()
    rng = np.random.default_rng(31)
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
    rt = _rocm_or_skip()
    dtype = np.float16
    if storage == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(79)
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
        atol=0.15 if storage == "bf16" else 5e-2, rtol=5e-2,
    )


@pytest.mark.parametrize("pad_mode", ["constant", "reflect"])
@pytest.mark.parametrize("storage", ["f32", "f16", "bf16"])
def test_centered_ragged_stft_and_cropped_istft_native_policy(storage, pad_mode):
    rt = _rocm_or_skip()
    dtype = np.float32 if storage == "f32" else np.float16
    if storage == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(802)
    signal = rng.normal(size=(2, 46)).astype(dtype)
    window = (np.hanning(18) + 0.25).astype(dtype)
    stft = rt.launch(
        _art(rt, "tessera.stft", (signal, window), {
            "hop": 7, "center": True, "pad_mode": pad_mode, "storage": storage,
        }),
        (signal, window),
    )
    assert stft["ok"] is True, stft.get("reason")
    tolerance = {"f32": 1e-2, "f16": 5e-2, "bf16": 2.5e-1}[storage]
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


def test_centered_policy_executes_noncontiguous_runtime_storage():
    rt = _rocm_or_skip()
    signal = np.arange(92, dtype=np.float32).reshape(46, 2).T
    assert not signal.flags.c_contiguous
    window = np.ones(18, np.float32)
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
    rt = _rocm_or_skip()
    rng = np.random.default_rng(804)
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
def test_gfx1151_true_stride_broader_full_spectrum_and_broadcast(onesided):
    from tests.unit.test_x86_spectral_compiled import (
        _general_istft_ref, _general_stft_ref,
    )

    rt = _rocm_or_skip()
    rng = np.random.default_rng(809 + int(onesided))
    signal = rng.standard_normal((2, 48, 3)).astype(np.float32)[:, ::2, :]
    window = np.stack((np.hanning(8), np.hamming(8)), axis=0).astype(
        np.float32
    )[:, None, :]
    stft = rt.launch(
        _art(rt, "tessera.stft", (signal, window), {
            "axis": 1, "hop": 4, "n_fft": 10, "onesided": onesided,
        }), (signal, window),
    )
    assert stft["ok"] is True, stft.get("reason")
    actual = np.asarray(stft["output"])
    expected = np.empty_like(actual)
    for batch in range(2):
        for channel in range(3):
            expected[batch, :, :, channel] = _general_stft_ref(
                signal[batch, :, channel], window[batch, 0], 4,
                n_fft=10, center=False, onesided=onesided,
            )
    np.testing.assert_allclose(actual, expected, atol=8e-4, rtol=8e-4)

    istft = rt.launch(
        _art(rt, "tessera.istft", (actual, window), {
            "axis": 2, "hop": 4, "n_fft": 10, "onesided": onesided,
            "length": 22,
        }), (actual, window),
    )
    assert istft["ok"] is True, istft.get("reason")
    expected_inverse = np.empty((2, 22, 3), np.float32)
    for batch in range(2):
        for channel in range(3):
            expected_inverse[batch, :, channel] = _general_istft_ref(
                actual[batch, :, :, channel], window[batch, 0], 4,
                n_fft=10, center=False, length=22, onesided=onesided,
            )
    np.testing.assert_allclose(
        istft["output"], expected_inverse, atol=1e-3, rtol=1e-3,
    )
