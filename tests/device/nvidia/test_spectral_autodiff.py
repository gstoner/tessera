"""SuperBear proof for NVIDIA compound-spectral reverse products."""

from __future__ import annotations

import numpy as np
import pytest

import tessera


@tessera.jit(target="nvidia_sm120", autodiff="reverse", wrt=("x", "weight"))
def _spectral_conv(x, weight):
    return tessera.ops.spectral_conv(x, weight)


@tessera.jit(target="nvidia_sm120", autodiff="reverse", wrt=("x", "weight"))
def _spectral_filter(x, weight):
    return tessera.ops.spectral_filter(x, weight)


def _require_fft():
    from tessera import runtime
    if runtime._load_nvidia_fft_runtime() is None:
        pytest.skip("NVIDIA CUDA FFT package is unavailable")


def test_spectral_convolution_vjp_matches_direct_correlation(monkeypatch):
    from tessera import runtime
    _require_fft()
    calls = []
    native = runtime._nvidia_fftexec

    def counted(op_name, value, kwargs):
        calls.append(op_name)
        return native(op_name, value, kwargs)

    monkeypatch.setattr(runtime, "_nvidia_fftexec", counted)
    rng = np.random.default_rng(307)
    x = rng.normal(size=(3, 13)).astype(np.float32)
    weight = rng.normal(size=(3, 6)).astype(np.float32)
    dy = rng.normal(size=(3, 18)).astype(np.float32)
    dx, dw = _spectral_conv.native_backward(
        x, weight, out_cotangents=dy
    )
    expected_dx = np.stack([np.correlate(dy[i], weight[i], mode="valid")
                            for i in range(3)]).astype(np.float32)
    expected_dw = np.stack([np.correlate(dy[i], x[i], mode="valid")
                            for i in range(3)]).astype(np.float32)
    np.testing.assert_allclose(dx, expected_dx, rtol=4e-5, atol=4e-5)
    np.testing.assert_allclose(dw, expected_dw, rtol=4e-5, atol=4e-5)
    assert "tessera.rfft" in calls and "tessera.irfft" in calls


def test_spectral_filter_vjp_uses_complex_adjoint():
    from types import SimpleNamespace
    from tessera import runtime
    from tessera.compiler.native_spectral_vjp import build_native_spectral_vjp_package

    _require_fft()
    rng = np.random.default_rng(311)
    values = [
        (rng.normal(size=(2, 17)) + 1j * rng.normal(size=(2, 17))).astype(np.complex64)
        for _ in range(3)
    ]
    x, weight, dy = values
    # complex64 is still planned-gated as a public Graph storage dtype on
    # NVIDIA. Prove the typed physical package directly without weakening that
    # legality gate or misreporting public Graph execution.
    package = build_native_spectral_vjp_package(
        source_graph_ir='module attributes {tessera.frontend.authority = "tracer"} {}',
        source=SimpleNamespace(
            op_name="tessera.spectral_filter", kwargs={"axis": -1}
        ),
        target="nvidia_sm120", ordered_inputs=(x, weight),
        arg_names=("x", "weight"), out_cotangent=dy,
    )
    result = runtime.launch(
        runtime.RuntimeArtifact(metadata=package.runtime_metadata()),
        (dy, x, weight),
    )
    assert result["ok"] is True, result.get("reason")
    dx, dw = result["output"]
    np.testing.assert_allclose(dx, dy * np.conj(weight), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(dw, dy * np.conj(x), rtol=2e-6, atol=2e-6)
