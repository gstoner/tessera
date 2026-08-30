"""Exact-SM120 public spectral JVP, Schedule→Tile, and storage-policy proof."""

from __future__ import annotations

import math

import numpy as np
import pytest

import tessera



# Declares the hardware this file needs. The marker is what the PR-lane
# expression deselects and what tests/_support/device_accounting.py counts;
# an unmarked device test is invisible to both.
pytestmark = pytest.mark.hardware_nvidia

@tessera.jit(target="nvidia_sm120", autodiff="jvp", wrt=("x", "window"))
def _stft_product(x, window):
    return tessera.ops.stft(
        x, window, axis=-1, n_fft=16, hop=8, center=False,
        onesided=True, norm="ortho",
    )


@tessera.jit(
    target="nvidia_sm120", autodiff="jvp", wrt=("spectrum", "window")
)
def _istft_product(spectrum, window):
    return tessera.ops.istft(
        spectrum, window, axis=-1, n_fft=16, hop=8, center=False,
        onesided=True, length=56, norm="ortho",
    )


@tessera.jit(
    target="nvidia_sm120", autodiff="jvp", wrt=("xf", "hf")
)
def _spectral_filter_product(xf, hf):
    return tessera.ops.spectral_filter(xf, hf)


@tessera.jit(
    target="nvidia_sm120", autodiff="jvp", wrt=("x", "kernel")
)
def _spectral_conv_product(x, kernel):
    return tessera.ops.spectral_conv(x, kernel, axis=-1, norm="backward")


@tessera.jit(target="nvidia_sm120", autodiff="jvp", wrt=("x",))
def _fft_product(x):
    return tessera.ops.fft(x, axis=-1, norm="ortho")


@tessera.jit(target="nvidia_sm120", autodiff="reverse", wrt=("x", "window"))
def _stft_reverse(x, window):
    return tessera.ops.stft(
        x, window, axis=-1, n_fft=16, hop=8, center=False,
        onesided=True, norm="ortho",
    )


@tessera.jit(
    target="nvidia_sm120", autodiff="reverse", wrt=("spectrum", "window")
)
def _istft_reverse(spectrum, window):
    return tessera.ops.istft(
        spectrum, window, axis=-1, n_fft=16, hop=8, center=False,
        onesided=True, length=56, norm="ortho",
    )


def _require_sm120() -> None:
    from tessera import runtime

    lib = runtime._load_nvidia_fft_runtime()
    if lib is None or lib.tessera_nvidia_spectral_arch() != 120:
        pytest.skip("exact SM120 CUDA spectral package is unavailable")


def test_public_fft_jvp_reaches_cuda_with_bound_operation_metadata():
    _require_sm120()
    rng = np.random.default_rng(1206)
    x = (rng.normal(size=(2, 16)) + 1j * rng.normal(size=(2, 16))).astype(
        np.complex64
    )
    dx = (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)).astype(
        np.complex64
    )
    primal, tangent = _fft_product.native_jvp(x, tangents=(dx,))
    np.testing.assert_allclose(
        primal, np.fft.fft(x, axis=-1, norm="ortho"), rtol=3e-5, atol=3e-5
    )
    np.testing.assert_allclose(
        tangent, np.fft.fft(dx, axis=-1, norm="ortho"), rtol=3e-5, atol=3e-5
    )
    assert _fft_product.last_jvp_execution["family"] == "spectral"


def _dct_reference(x: np.ndarray, dct_type: int, normalization: str) -> np.ndarray:
    values = np.asarray(x, np.float64)
    n = values.shape[-1]
    result = np.empty_like(values)
    for row in range(values.shape[0]):
        for k in range(n):
            if dct_type == 1:
                total = values[row, 0] + (-1.0 if k & 1 else 1.0) * values[row, -1]
                total += sum(
                    2.0 * values[row, j] * math.cos(math.pi * j * k / (n - 1))
                    for j in range(1, n - 1)
                )
                scale_n = 2 * (n - 1)
            else:
                total = sum(
                    2.0 * values[row, j]
                    * math.cos(math.pi * (2 * j + 1) * k / (2 * n))
                    for j in range(n)
                )
                scale_n = 2 * n
            scale = 1.0 if normalization == "backward" else (
                1.0 / scale_n if normalization == "forward" else 1.0 / math.sqrt(scale_n)
            )
            result[row, k] = total * scale
    return result


def test_sm120_schedule_to_tile_admits_digest_bound_numeric_policy():
    from tessera.compiler.scheduled_spectral import lower_scheduled_spectral

    artifact = lower_scheduled_spectral(
        target="nvidia_sm120", op_name="tessera.stft",
        input_shapes=((2, 32), (16,)), axis=-1, hop=8, n_fft=16,
        storage="f16", normalization="ortho",
    )
    assert artifact.target == "nvidia_sm120"
    assert artifact.architecture == "sm120"
    assert artifact.numeric_policy == {"storage": "fp16", "accum": "fp32"}
    assert artifact.native_entry == "tessera_nvidia_stft_policy_broadcast_layout_storage"
    assert artifact.tile_ir.count("tile.spectral_program_kernel") == 1
    assert 'target = "nvidia_sm120"' in artifact.tile_ir
    assert 'numeric_policy = {accum = "fp32", storage = "fp16"}' in artifact.tile_ir


@pytest.mark.parametrize("storage", ("f16", "bf16"))
def test_sm120_low_precision_dct_stft_istft_use_fp32_accumulation(storage):
    from tessera import runtime
    from tessera.compiler.scheduled_spectral import lower_scheduled_spectral

    _require_sm120()
    dtype = np.float16
    if storage == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(1207 if storage == "f16" else 1208)

    dct_input = rng.normal(size=(2, 9)).astype(dtype)
    dct_contract = lower_scheduled_spectral(
        target="nvidia_sm120", op_name="tessera.dct",
        input_shapes=(dct_input.shape,), axis=-1, dct_type=2,
        storage=storage, normalization="ortho",
    ).to_metadata()
    dct_result = runtime.launch(runtime.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_spectral_compiled",
        "executable": True,
        "arg_names": ["x"], "scheduled_spectral": dct_contract,
    }), (dct_input,))
    assert dct_result["ok"] is True, dct_result.get("reason")
    dct_output = dct_result["output"]
    assert str(dct_output.dtype) == str(np.dtype(dtype))
    np.testing.assert_allclose(
        np.asarray(dct_output, np.float32),
        _dct_reference(np.asarray(dct_input, np.float32), 2, "ortho"),
        rtol=2e-2 if storage == "f16" else 8e-2,
        atol=2e-2 if storage == "f16" else 8e-2,
    )

    signal = rng.normal(size=(2, 32)).astype(dtype)
    window = (0.3 + np.hanning(16)).astype(dtype)
    stft_contract = lower_scheduled_spectral(
        target="nvidia_sm120", op_name="tessera.stft",
        input_shapes=(signal.shape, window.shape), axis=-1, hop=8, n_fft=16,
        storage=storage, normalization="ortho",
    ).to_metadata()
    stft_result = runtime.launch(runtime.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_spectral_compiled",
        "executable": True,
        "arg_names": ["x", "window"], "scheduled_spectral": stft_contract,
    }), (signal, window))
    assert stft_result["ok"] is True, stft_result.get("reason")
    spectrum = stft_result["output"]
    expected_frames = np.stack([
        np.asarray(signal, np.float32)[:, offset:offset + 16]
        * np.asarray(window, np.float32)
        for offset in (0, 8, 16)
    ], axis=1)
    expected_spectrum = np.fft.rfft(expected_frames, axis=-1) / math.sqrt(16.0)
    np.testing.assert_allclose(spectrum, expected_spectrum, rtol=2e-3, atol=2e-3)

    istft_contract = lower_scheduled_spectral(
        target="nvidia_sm120", op_name="tessera.istft",
        input_shapes=(spectrum.shape, window.shape), axis=-1, hop=8, n_fft=16,
        output_length=32, storage=storage, normalization="ortho",
    ).to_metadata()
    istft_result = runtime.launch(runtime.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_spectral_compiled",
        "executable": True,
        "arg_names": ["spectrum", "window"], "scheduled_spectral": istft_contract,
    }), (spectrum, window))
    assert istft_result["ok"] is True, istft_result.get("reason")
    reconstructed = istft_result["output"]
    assert str(reconstructed.dtype) == str(np.dtype(dtype))
    tolerance = 3e-2 if storage == "f16" else 1.2e-1
    np.testing.assert_allclose(
        np.asarray(reconstructed, np.float32), np.asarray(signal, np.float32),
        rtol=tolerance, atol=tolerance,
    )


@pytest.mark.parametrize("storage", ("f16", "bf16"))
def test_sm120_low_precision_stft_istft_reverse_preserves_storage_policy(storage):
    _require_sm120()
    dtype = np.float16
    if storage == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(1209 if storage == "f16" else 1210)
    signal = rng.normal(size=(2, 56)).astype(dtype)
    window = (0.4 + np.hanning(16)).astype(dtype)
    spectrum = (
        rng.normal(size=(2, 6, 9)) + 1j * rng.normal(size=(2, 6, 9))
    ).astype(np.complex64)
    spectrum[..., (0, -1)] = spectrum[..., (0, -1)].real
    spectrum_cotangent = (
        rng.normal(size=(2, 6, 9)) + 1j * rng.normal(size=(2, 6, 9))
    ).astype(np.complex64)
    output_cotangent = rng.normal(size=(2, 56)).astype(dtype)

    dx, dwindow_stft = _stft_reverse.native_backward(
        signal, window, out_cotangents=spectrum_cotangent
    )
    dx_ref, dwindow_stft_ref = _stft_reverse.native_backward(
        np.asarray(signal, np.float32), np.asarray(window, np.float32),
        out_cotangents=spectrum_cotangent,
    )
    assert str(dx.dtype) == str(np.dtype(dtype))
    assert str(dwindow_stft.dtype) == str(np.dtype(dtype))

    dspectrum, dwindow_istft = _istft_reverse.native_backward(
        spectrum, window, out_cotangents=output_cotangent
    )
    dspectrum_ref, dwindow_istft_ref = _istft_reverse.native_backward(
        spectrum, np.asarray(window, np.float32),
        out_cotangents=np.asarray(output_cotangent, np.float32),
    )
    assert dspectrum.dtype == np.complex64
    assert str(dwindow_istft.dtype) == str(np.dtype(dtype))

    tolerance = 2e-2 if storage == "f16" else 8e-2
    np.testing.assert_allclose(
        np.asarray(dx, np.float32), dx_ref, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        np.asarray(dwindow_stft, np.float32), dwindow_stft_ref,
        rtol=tolerance, atol=tolerance,
    )
    np.testing.assert_allclose(
        dspectrum, dspectrum_ref, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        np.asarray(dwindow_istft, np.float32), dwindow_istft_ref,
        rtol=tolerance, atol=tolerance,
    )


def test_public_content_addressed_stft_jvp_matches_analytic_product():
    _require_sm120()
    rng = np.random.default_rng(1211)
    x = rng.normal(size=(2, 56)).astype(np.float32)
    window = (0.25 + np.hanning(16)).astype(np.float32)
    dx = rng.normal(size=x.shape).astype(np.float32)
    dwindow = rng.normal(size=window.shape).astype(np.float32)
    primal, tangent = _stft_product.native_jvp(
        x, window, tangents=(dx, dwindow)
    )
    expected_primal = np.fft.rfft(
        np.stack([x[:, at:at + 16] * window for at in range(0, 41, 8)], axis=1),
        axis=-1,
    ) / 4.0
    expected_tangent = np.fft.rfft(
        np.stack([
            dx[:, at:at + 16] * window + x[:, at:at + 16] * dwindow
            for at in range(0, 41, 8)
        ], axis=1), axis=-1,
    ) / 4.0
    np.testing.assert_allclose(primal, expected_primal, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=4e-5, atol=4e-5)
    evidence = _stft_product.last_jvp_execution
    assert evidence["evidence_target"] == "nvidia_sm120"
    assert evidence["family"] == "spectral_compound"
    assert len(evidence["artifact_hash"]) == 64
    assert evidence["schedule_program_digest"] != evidence["tile_program_digest"]


def test_public_content_addressed_istft_jvp_matches_centered_difference():
    _require_sm120()
    rng = np.random.default_rng(1213)
    spectrum = (
        rng.normal(size=(2, 6, 9)) + 1j * rng.normal(size=(2, 6, 9))
    ).astype(np.complex64)
    spectrum[..., (0, -1)] = spectrum[..., (0, -1)].real
    window = (0.35 + np.hanning(16)).astype(np.float32)
    dspectrum = (
        rng.normal(size=spectrum.shape) + 1j * rng.normal(size=spectrum.shape)
    ).astype(np.complex64)
    dspectrum[..., (0, -1)] = dspectrum[..., (0, -1)].real
    dwindow = rng.normal(size=window.shape).astype(np.float32)
    primal, tangent = _istft_product.native_jvp(
        spectrum, window, tangents=(dspectrum, dwindow)
    )
    epsilon = np.float32(1.5e-3)
    plus = tessera.ops.istft(
        spectrum + epsilon * dspectrum, window + epsilon * dwindow,
        n_fft=16, hop=8, length=56, norm="ortho",
    )
    minus = tessera.ops.istft(
        spectrum - epsilon * dspectrum, window - epsilon * dwindow,
        n_fft=16, hop=8, length=56, norm="ortho",
    )
    oracle = (np.asarray(plus) - np.asarray(minus)) / (2.0 * epsilon)
    expected = tessera.ops.istft(
        spectrum, window, n_fft=16, hop=8, length=56, norm="ortho",
    )
    np.testing.assert_allclose(primal, expected, rtol=4e-5, atol=4e-5)
    np.testing.assert_allclose(tangent, oracle, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("storage", ("f16", "bf16"))
def test_public_content_addressed_stft_jvp_preserves_low_precision_policy(storage):
    _require_sm120()
    dtype = np.float16
    if storage == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(1223 if storage == "f16" else 1229)
    x = rng.normal(size=(2, 56)).astype(dtype)
    window = (0.25 + np.hanning(16)).astype(dtype)
    dx = rng.normal(size=x.shape).astype(dtype)
    dwindow = rng.normal(size=window.shape).astype(dtype)
    primal, tangent = _stft_product.native_jvp(
        x, window, tangents=(dx, dwindow)
    )
    frames = np.stack([
        np.asarray(x[:, at:at + 16], np.float32)
        * np.asarray(window, np.float32)
        for at in range(0, 41, 8)
    ], axis=1)
    tangent_frames = np.stack([
        np.asarray(dx[:, at:at + 16], np.float32)
        * np.asarray(window, np.float32)
        + np.asarray(x[:, at:at + 16], np.float32)
        * np.asarray(dwindow, np.float32)
        for at in range(0, 41, 8)
    ], axis=1)
    expected_primal = np.fft.rfft(frames, axis=-1) / 4.0
    expected_tangent = np.fft.rfft(tangent_frames, axis=-1) / 4.0
    tolerance = 3.0e-3 if storage == "f16" else 1.5e-2
    np.testing.assert_allclose(
        primal, expected_primal, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        tangent, expected_tangent, rtol=tolerance, atol=tolerance
    )


def test_public_spectral_filter_jvp_uses_cuda_binary_accumulator():
    _require_sm120()
    rng = np.random.default_rng(1231)
    xf = (rng.normal(size=(3, 17)) + 1j * rng.normal(size=(3, 17))).astype(
        np.complex64
    )
    hf = (rng.normal(size=xf.shape) + 1j * rng.normal(size=xf.shape)).astype(
        np.complex64
    )
    dxf = (rng.normal(size=xf.shape) + 1j * rng.normal(size=xf.shape)).astype(
        np.complex64
    )
    dhf = (rng.normal(size=xf.shape) + 1j * rng.normal(size=xf.shape)).astype(
        np.complex64
    )
    primal, tangent = _spectral_filter_product.native_jvp(
        xf, hf, tangents=(dxf, dhf)
    )
    np.testing.assert_allclose(primal, xf * hf, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        tangent, dxf * hf + xf * dhf, rtol=3e-6, atol=3e-6
    )
    evidence = _spectral_filter_product.last_jvp_execution
    assert evidence["evidence_target"] == "nvidia_sm120"
    assert evidence["family"] == "spectral_compound"


def test_public_spectral_conv_jvp_matches_bilinear_oracle():
    _require_sm120()
    rng = np.random.default_rng(1237)
    x = rng.normal(size=(2, 19)).astype(np.float32)
    kernel = rng.normal(size=(2, 7)).astype(np.float32)
    dx = rng.normal(size=x.shape).astype(np.float32)
    dkernel = rng.normal(size=kernel.shape).astype(np.float32)
    primal, tangent = _spectral_conv_product.native_jvp(
        x, kernel, tangents=(dx, dkernel)
    )
    expected_primal = np.stack([
        np.convolve(x[row], kernel[row], mode="full") for row in range(2)
    ])
    expected_tangent = np.stack([
        np.convolve(dx[row], kernel[row], mode="full")
        + np.convolve(x[row], dkernel[row], mode="full")
        for row in range(2)
    ])
    np.testing.assert_allclose(primal, expected_primal, rtol=4e-5, atol=4e-5)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=6e-5, atol=6e-5)


def test_stft_jvp_vjp_adjoint_law_on_exact_sm120():
    _require_sm120()
    rng = np.random.default_rng(1217)
    x = rng.normal(size=(2, 56)).astype(np.float32)
    window = (0.3 + np.hanning(16)).astype(np.float32)
    dx = rng.normal(size=x.shape).astype(np.float32)
    dwindow = rng.normal(size=window.shape).astype(np.float32)
    _, tangent = _stft_product.native_jvp(x, window, tangents=(dx, dwindow))
    cotangent = (
        rng.normal(size=tangent.shape) + 1j * rng.normal(size=tangent.shape)
    ).astype(np.complex64)
    gx, gw = _stft_reverse.native_backward(
        x, window, out_cotangents=cotangent
    )
    lhs = float(np.real(np.vdot(cotangent, tangent)))
    rhs = float(np.vdot(gx, dx) + np.vdot(gw, dwindow))
    np.testing.assert_allclose(lhs, rhs, rtol=4e-4, atol=4e-4)
