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


@tessera.jit(target="nvidia_sm120", autodiff="reverse", wrt=("x", "window"))
def _stft(x, window):
    return tessera.ops.stft(
        x, window, axis=-1, n_fft=16, hop=8, center=False,
        onesided=True, norm="backward",
    )


@tessera.jit(
    target="nvidia_sm120", autodiff="reverse", wrt=("spectrum", "window")
)
def _istft(spectrum, window):
    return tessera.ops.istft(
        spectrum, window, axis=-1, n_fft=16, hop=8, center=False,
        onesided=True, length=56, norm="backward",
    )


@tessera.jit(target="nvidia_sm120", autodiff="reverse", wrt=("x", "window"))
def _stft_full_broadcast_axis1(x, window):
    return tessera.ops.stft(
        x, window, axis=1, n_fft=10, hop=4, center=False,
        onesided=False, norm="ortho",
    )


@tessera.jit(target="nvidia_sm120", autodiff="reverse", wrt=("x", "window"))
def _stft_full_reflect_axis1(x, window):
    return tessera.ops.stft(
        x, window, axis=1, n_fft=20, hop=6, center=True,
        pad_mode="reflect", onesided=False, norm="ortho",
    )


@tessera.jit(
    target="nvidia_sm120", autodiff="reverse", wrt=("spectrum", "window")
)
def _istft_full_broadcast_axis2(spectrum, window):
    return tessera.ops.istft(
        spectrum, window, axis=2, n_fft=10, hop=4, center=False,
        onesided=False, length=22, norm="ortho",
    )


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


@pytest.mark.parametrize("kind", ("stft", "istft"))
def test_stft_istft_vjp_matches_independent_reference(kind):
    from tessera.autodiff import vjp

    _require_fft()
    rng = np.random.default_rng(120 if kind == "stft" else 121)
    window = (0.25 + np.hanning(16)).astype(np.float32)
    if kind == "stft":
        primal = rng.standard_normal(56).astype(np.float32)
        dy = (rng.standard_normal((6, 9)) +
              1j * rng.standard_normal((6, 9))).astype(np.complex64)
        actual = _stft.native_backward(primal, window, out_cotangents=dy)
        expected = vjp._VJPS["stft"](
            dy, primal, window, axis=-1, n_fft=16, hop=8,
            center=False, onesided=True, norm="backward",
        )
        function = _stft
    else:
        primal = (rng.standard_normal((6, 9)) +
                  1j * rng.standard_normal((6, 9))).astype(np.complex64)
        dy = rng.standard_normal(56).astype(np.float32)
        actual = _istft.native_backward(primal, window, out_cotangents=dy)
        expected = vjp._VJPS["istft"](
            dy, primal, window, axis=-1, n_fft=16, hop=8,
            center=False, onesided=True, length=56, norm="backward",
        )
        function = _istft
    np.testing.assert_allclose(actual[0], expected[0], rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(actual[1], expected[1], rtol=5e-5, atol=5e-5)
    proof = function.last_backward_execution
    assert proof["compiler_path"] == "nvidia_sm120_spectral_backward_compiled"
    assert proof["target_consumer"] == "nvidia.sm120_spectral_backward"
    assert proof["physical_attestation"]["device_arch"] == "sm_120"


@pytest.mark.parametrize("kind", ("stft", "istft"))
def test_generalized_full_spectrum_broadcast_strided_vjp(kind):
    from tessera.autodiff import vjp

    _require_fft()
    rng = np.random.default_rng(122 if kind == "stft" else 123)
    window_storage = (0.2 + rng.random((2, 1, 16))).astype(np.float32)
    window = window_storage[..., ::2]
    assert not window.flags.c_contiguous
    if kind == "stft":
        primal_storage = rng.standard_normal((2, 48, 3)).astype(np.float32)
        primal = primal_storage[:, ::2, :]
        dy_storage = np.empty((2, 4, 20, 3), np.complex64)
        dy = dy_storage[:, :, ::2, :]
        dy[...] = (rng.standard_normal(dy.shape) +
                   1j * rng.standard_normal(dy.shape)).astype(np.complex64)
        actual = _stft_full_broadcast_axis1.native_backward(
            primal, window, out_cotangents=dy
        )
        expected = vjp._VJPS["stft"](
            dy, primal, window, axis=1, n_fft=10, hop=4, center=False,
            onesided=False, norm="ortho",
        )
        function = _stft_full_broadcast_axis1
    else:
        primal_storage = np.empty((2, 4, 20, 3), np.complex64)
        primal = primal_storage[:, :, ::2, :]
        primal[...] = (rng.standard_normal(primal.shape) +
                       1j * rng.standard_normal(primal.shape)).astype(np.complex64)
        dy_storage = rng.standard_normal((2, 44, 3)).astype(np.float32)
        dy = dy_storage[:, ::2, :]
        actual = _istft_full_broadcast_axis2.native_backward(
            primal, window, out_cotangents=dy
        )
        expected = vjp._VJPS["istft"](
            dy, primal, window, axis=2, n_fft=10, hop=4, center=False,
            onesided=False, length=22, norm="ortho",
        )
        function = _istft_full_broadcast_axis2
    np.testing.assert_allclose(actual[0], expected[0], rtol=8e-5, atol=8e-5)
    np.testing.assert_allclose(actual[1], expected[1], rtol=8e-5, atol=8e-5)
    proof = function.last_backward_execution
    assert proof["execution_certificate"]["artifact_identities"][
        "tile_program_digest"
    ]


def test_native_stft_forward_and_adjoint_satisfy_inner_product_identity():
    from tessera import runtime

    _require_fft()
    rng = np.random.default_rng(124)
    x = rng.standard_normal((2, 46, 3)).astype(np.float32)
    window = (0.2 + rng.random((3, 15))).astype(np.float32)
    kwargs = {
        "axis": 1, "n_fft": 20, "hop": 6, "center": True,
        "pad_mode": "reflect", "onesided": False,
        "normalization": "ortho",
    }
    artifact = runtime.RuntimeArtifact(metadata={
        "target": "nvidia_sm120",
        "compiler_path": "nvidia_spectral_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": ["x", "window"],
        "output_name": "output",
        "ops": [{
            "op_name": "tessera.stft", "result": "output",
            "operands": ["x", "window"], "kwargs": kwargs,
        }],
    })
    launched = runtime.launch(artifact, (x, window))
    assert launched["ok"] is True, launched.get("reason")
    spectrum = np.asarray(launched["output"])
    dy = (rng.standard_normal(spectrum.shape) +
          1j * rng.standard_normal(spectrum.shape)).astype(np.complex64)
    dx, _ = _stft_full_reflect_axis1.native_backward(
        x, window, out_cotangents=dy
    )
    lhs = float(np.vdot(dy, spectrum).real)
    rhs = float(np.vdot(dx, x).real)
    np.testing.assert_allclose(lhs, rhs, rtol=8e-6, atol=8e-5)
