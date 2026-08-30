"""SuperBear proof for the canonical CUDA FFT plan/workspace ABI."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest



# Declares the hardware this file needs. The marker is what the PR-lane
# expression deselects and what tests/_support/device_accounting.py counts;
# an unmarked device test is invisible to both.
pytestmark = pytest.mark.hardware_nvidia

def _runtime_or_skip():
    from tessera import runtime

    lib = runtime._load_nvidia_fft_runtime()
    if lib is None:
        pytest.skip("libtessera_nvidia_fft.so not built")
    return runtime, lib


def test_versioned_abi_and_explicit_workspace_contract():
    runtime, lib = _runtime_or_skip()
    assert lib.tessera_nvidia_fft_package_abi() == (
        b"tessera.nvidia.cuda_fft_workspace.v2")
    plan = ctypes.c_void_p()
    workspace_bytes = ctypes.c_size_t()
    assert lib.tessera_nvidia_fft_plan_create_c2c_f32(
        3, 257, ctypes.byref(plan), ctypes.byref(workspace_bytes)) == 0
    assert plan.value is not None
    workspace = ctypes.c_void_p()
    assert lib.tessera_nvidia_fft_workspace_alloc(
        workspace_bytes.value, ctypes.byref(workspace)) == 0
    assert workspace.value is not None
    x = np.ones((3, 257), np.complex64)
    out = np.empty_like(x)
    pointer = ctypes.POINTER(ctypes.c_float)
    if workspace_bytes.value:
        assert lib.tessera_nvidia_fft_execute_c2c_f32(
            plan, x.view(np.float32).ctypes.data_as(pointer),
            out.view(np.float32).ctypes.data_as(pointer), workspace,
            workspace_bytes.value - 1, 0) == 1
    assert lib.tessera_nvidia_fft_workspace_free(workspace) == 0
    assert lib.tessera_nvidia_fft_plan_destroy(plan) == 0


@pytest.mark.parametrize("batch,length", ((1, 4), (3, 16), (2, 100), (2, 257)))
def test_forward_and_normalized_inverse_match_numpy(batch, length):
    runtime, _ = _runtime_or_skip()
    generator = np.random.default_rng(batch * 1000 + length)
    x = (generator.standard_normal((batch, length)) +
         1j * generator.standard_normal((batch, length))).astype(np.complex64)
    forward = runtime._nvidia_fft_c2c_rows(x, False, np)
    np.testing.assert_allclose(
        forward, np.fft.fft(x, axis=-1).astype(np.complex64),
        rtol=2e-5, atol=2e-5)
    inverse = runtime._nvidia_fft_c2c_rows(x, True, np)
    np.testing.assert_allclose(
        inverse, np.fft.ifft(x, axis=-1).astype(np.complex64),
        rtol=2e-5, atol=2e-5)


def test_plan_and_workspace_are_reused_by_shape():
    runtime, _ = _runtime_or_skip()
    runtime._clear_nvidia_fft_plan_cache()
    x = np.arange(96, dtype=np.float32).reshape(3, 32).astype(np.complex64)
    first = runtime._nvidia_fft_c2c_rows(x, False, np)
    package = runtime._nvidia_fft_plans[("c2c", 3, 32)]
    second = runtime._nvidia_fft_c2c_rows(x, False, np)
    assert runtime._nvidia_fft_plans[("c2c", 3, 32)] is package
    np.testing.assert_array_equal(first, second)


def test_plan_cache_is_bounded_and_releases_evicted_shapes(monkeypatch):
    runtime, _ = _runtime_or_skip()
    runtime._clear_nvidia_fft_plan_cache()
    monkeypatch.setattr(runtime, "_NVIDIA_FFT_PLAN_CACHE_LIMIT", 2)
    try:
        for length in (8, 16, 32):
            runtime._nvidia_fft_c2c_rows(
                np.ones((1, length), dtype=np.complex64), False, np)
        assert len(runtime._nvidia_fft_plans) == 2
        assert ("c2c", 1, 8) not in runtime._nvidia_fft_plans
    finally:
        runtime._clear_nvidia_fft_plan_cache()


def test_fft_consumer_handles_nonleading_axis_and_padding():
    runtime, _ = _runtime_or_skip()
    generator = np.random.default_rng(22)
    x = (generator.standard_normal((2, 5, 3)) +
         1j * generator.standard_normal((2, 5, 3))).astype(np.complex64)
    artifact = runtime.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_fft_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["x"], "output_name": "output",
        "ops": [{"op_name": "tessera.fft", "result": "output",
                 "operands": ["x"], "kwargs": {"axis": 1, "n": 8}}],
    })
    launched = runtime.launch(artifact, (x,))
    assert launched["ok"] is True, launched.get("reason")
    assert launched["compiler_path"] == "nvidia_fft_compiled"
    actual = launched["output"]
    expected = np.fft.fft(x, n=8, axis=1).astype(np.complex64)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("batch,length", ((1, 5), (3, 16), (2, 101), (2, 257)))
def test_native_real_round_trip_matches_numpy(batch, length):
    runtime, _ = _runtime_or_skip()
    generator = np.random.default_rng(batch * 2000 + length)
    x = generator.standard_normal((batch, length)).astype(np.float32)
    spectrum = runtime._nvidia_fft_real_rows(x, False, None, np)
    np.testing.assert_allclose(
        spectrum, np.fft.rfft(x, axis=-1).astype(np.complex64),
        rtol=2e-5, atol=2e-5)
    restored = runtime._nvidia_fft_real_rows(spectrum, True, length, np)
    np.testing.assert_allclose(restored, x, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("op_name,length", (("tessera.rfft", 17),
                                               ("tessera.irfft", 18)))
def test_real_fft_runtime_consumer(op_name, length):
    runtime, _ = _runtime_or_skip()
    rng = np.random.default_rng(length)
    if op_name == "tessera.rfft":
        x = rng.standard_normal((2, 7)).astype(np.float32)
        expected = np.fft.rfft(x, n=length, axis=-1).astype(np.complex64)
    else:
        x = (rng.standard_normal((2, length // 2 + 1)) +
             1j * rng.standard_normal((2, length // 2 + 1))).astype(np.complex64)
        expected = np.fft.irfft(x, n=length, axis=-1).astype(np.float32)
    artifact = runtime.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_fft_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["x"], "output_name": "output",
        "ops": [{"op_name": op_name, "result": "output", "operands": ["x"],
                 "kwargs": {"axis": -1, "n": length}}],
    })
    result = runtime.launch(artifact, (x,))
    assert result["ok"] is True, result.get("reason")
    actual = result["output"]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("op_name", (
    "tessera.fft", "tessera.ifft", "tessera.rfft", "tessera.irfft"))
@pytest.mark.parametrize("normalization", ("forward", "ortho"))
def test_fft_runtime_honors_normalization_modes(op_name, normalization):
    runtime, _ = _runtime_or_skip()
    length = 18
    rng = np.random.default_rng(length)
    if op_name == "tessera.rfft":
        x = rng.standard_normal((2, length)).astype(np.float32)
        expected = np.fft.rfft(x, axis=-1, norm=normalization).astype(np.complex64)
    elif op_name == "tessera.irfft":
        x = (rng.standard_normal((2, length // 2 + 1)) +
             1j * rng.standard_normal((2, length // 2 + 1))).astype(np.complex64)
        expected = np.fft.irfft(
            x, n=length, axis=-1, norm=normalization).astype(np.float32)
    else:
        x = (rng.standard_normal((2, length)) +
             1j * rng.standard_normal((2, length))).astype(np.complex64)
        transform = np.fft.ifft if op_name == "tessera.ifft" else np.fft.fft
        expected = transform(x, axis=-1, norm=normalization).astype(np.complex64)
    artifact = runtime.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_fft_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["x"], "output_name": "output",
        "ops": [{"op_name": op_name, "result": "output", "operands": ["x"],
                 "kwargs": {"axis": -1, "n": length,
                            "normalization": normalization}}],
    })
    result = runtime.launch(artifact, (x,))
    assert result["ok"] is True, result.get("reason")
    actual = result["output"]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_normalized_spectral_convolution_matches_numpy():
    runtime, _ = _runtime_or_skip()
    rng = np.random.default_rng(83)
    x = rng.standard_normal(11).astype(np.float32)
    w = rng.standard_normal(5).astype(np.float32)
    for normalization in ("forward", "ortho"):
        artifact = runtime.RuntimeArtifact(metadata={
            "target": "nvidia_sm120",
            "compiler_path": "nvidia_spectral_compiled",
            "executable": True, "execution_kind": "native_gpu",
            "arg_names": ["x", "w"],
            "ops": [{"op_name": "tessera.spectral_conv", "result": "output",
                     "operands": ["x", "w"],
                     "kwargs": {"normalization": normalization}}],
        })
        result = runtime.launch(artifact, (x, w))
        assert result["ok"] is True, result.get("reason")
        actual = np.asarray(result["output"])
        n = x.size + w.size - 1
        nfft = 1 << int(np.ceil(np.log2(n)))
        expected = np.fft.irfft(
            np.fft.rfft(x, nfft, norm=normalization) *
            np.fft.rfft(w, nfft, norm=normalization),
            nfft, norm=normalization)[:n].astype(np.float32)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_fft_runtime_rejects_unknown_normalization():
    runtime, _ = _runtime_or_skip()
    x = np.ones((1, 8), dtype=np.complex64)
    artifact = runtime.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_fft_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["x"], "output_name": "output",
        "ops": [{"op_name": "tessera.fft", "result": "output",
                 "operands": ["x"], "kwargs": {"normalization": "invalid"}}],
    })
    result = runtime.launch(artifact, (x,))
    assert result["ok"] is False
    assert "normalization must be backward, forward, or ortho" in result["reason"]


@pytest.mark.parametrize("op_name", ("tessera.dct", "tessera.stft",
                                      "tessera.istft", "tessera.spectral_conv",
                                      "tessera.spectral_filter"))
def test_nvidia_spectral_consumers_route_through_native_fft(op_name, monkeypatch):
    runtime, _ = _runtime_or_skip()
    calls = []
    native = runtime._nvidia_fftexec

    def counted(sub_op, x, kwargs):
        calls.append(sub_op)
        return native(sub_op, x, kwargs)

    monkeypatch.setattr(runtime, "_nvidia_fftexec", counted)
    rng = np.random.default_rng(71)
    if op_name == "tessera.dct":
        operands, kwargs = [rng.standard_normal(16).astype(np.float32)], {"type": 2}
    elif op_name == "tessera.stft":
        operands = [rng.standard_normal(32).astype(np.float32), np.hanning(8).astype(np.float32)]
        kwargs = {"hop": 4}
    elif op_name == "tessera.istft":
        window = np.hanning(8).astype(np.float32)
        frames = np.stack([np.fft.rfft(rng.standard_normal(8)).astype(np.complex64)
                           for _ in range(4)])
        operands, kwargs = [frames, window], {"hop": 4}
    elif op_name == "tessera.spectral_conv":
        operands = [rng.standard_normal(11).astype(np.float32),
                    rng.standard_normal(5).astype(np.float32)]
        kwargs = {}
    else:
        operands = [(rng.standard_normal(9) + 1j*rng.standard_normal(9)).astype(np.complex64),
                    (rng.standard_normal(9) + 1j*rng.standard_normal(9)).astype(np.complex64)]
        kwargs = {}
    artifact = runtime.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_spectral_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": [f"x{i}" for i in range(len(operands))],
        "ops": [{"op_name": op_name, "result": "o",
                 "operands": [f"x{i}" for i in range(len(operands))],
                 "kwargs": kwargs}],
    })
    result = runtime.launch(artifact, tuple(operands))
    assert result["ok"] is True, result.get("reason")
    actual = np.asarray(result["output"])
    if op_name == "tessera.dct":
        x = operands[0]
        n = x.shape[-1]
        expected = 2.0 * np.stack([
            np.sum(x * np.cos(np.pi * (np.arange(n) + 0.5) * k / n))
            for k in range(n)
        ]).astype(np.float32)
    elif op_name == "tessera.stft":
        x, window = operands
        expected = np.stack([
            np.fft.rfft(x[start:start + window.size] * window)
            for start in range(0, x.size - window.size + 1, kwargs["hop"])
        ]).astype(np.complex64)
    elif op_name == "tessera.istft":
        spectra, window = operands
        frames = np.fft.irfft(spectra, n=window.size, axis=-1)
        expected = np.zeros((spectra.shape[-2] - 1) * kwargs["hop"] + window.size)
        weight = np.zeros_like(expected)
        for index, frame in enumerate(frames):
            start = index * kwargs["hop"]
            expected[start:start + window.size] += frame * window
            weight[start:start + window.size] += window * window
        expected = (expected / np.maximum(weight, 1e-12)).astype(np.float32)
    elif op_name == "tessera.spectral_conv":
        expected = np.convolve(operands[0], operands[1], mode="full").astype(np.float32)
    else:
        expected = (operands[0] * operands[1]).astype(np.complex64)
    np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=4e-5)
    if op_name in {"tessera.dct", "tessera.stft", "tessera.istft"}:
        # These two composites are now one target-owned CUDA policy package;
        # they must not reconstruct framing/OLA through the Python FFT helper.
        assert not calls
        _, lib = _runtime_or_skip()
        assert lib.tessera_nvidia_spectral_package_abi() == (
            b"tessera.nvidia.spectral_policy.v1")
        assert lib.tessera_nvidia_spectral_arch() == 120
    elif op_name != "tessera.spectral_filter":
        assert calls
