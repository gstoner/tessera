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
        b"tessera.nvidia.cuda_fft_workspace.v4")
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
    runtime, lib = _runtime_or_skip()
    runtime._clear_nvidia_fft_plan_cache()
    x = np.arange(96, dtype=np.float32).reshape(3, 32).astype(np.complex64)
    first = runtime._nvidia_fft_c2c_rows(x, False, np)
    key = (runtime._nvidia_fft_device(lib), "c2c", 3, 32)  # v3: device-scoped
    package = runtime._nvidia_fft_plans[key]
    second = runtime._nvidia_fft_c2c_rows(x, False, np)
    assert runtime._nvidia_fft_plans[key] is package
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
    native_conv = runtime._nvidia_native_spectral_conv
    native_conv_served = []

    def spied_native_conv(*args):
        result = native_conv(*args)
        native_conv_served.append(result is not None)
        return result

    monkeypatch.setattr(runtime, "_nvidia_native_spectral_conv", spied_native_conv)
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
    elif op_name == "tessera.spectral_conv":
        # One native batched convolution, not three host-staged transforms.
        assert not calls
        assert native_conv_served == [True]
    elif op_name != "tessera.spectral_filter":
        assert calls


def test_cached_plans_are_keyed_by_the_live_cuda_device():
    runtime, lib = _runtime_or_skip()
    device = ctypes.c_int(-1)
    assert lib.tessera_nvidia_fft_current_device(ctypes.byref(device)) == 0
    assert device.value >= 0
    runtime._clear_nvidia_fft_plan_cache()
    x = np.ones((2, 64), np.complex64)
    np.testing.assert_allclose(
        runtime._nvidia_fft_c2c_rows(x, False, np), np.fft.fft(x, axis=-1),
        rtol=2e-5, atol=2e-5)
    assert list(runtime._nvidia_fft_plans) == [(device.value, "c2c", 2, 64)]
    real = np.ones((2, 64), np.float32)
    runtime._nvidia_fft_real_rows(real, False, None, np)
    assert (device.value, "r2c", 2, 64) in runtime._nvidia_fft_plans


_GET_DEVICE_SHIM = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
/* 0 pass through; 1 the query fails; 2 the query succeeds naming device 7. */
static int mode = 0;
void tessera_test_set_get_device_mode(int value) { mode = value; }
int cudaGetDevice(int *device) {
  static int (*real)(int *) = 0;
  if (mode == 1) return 999; /* cudaErrorUnknown */
  if (mode == 2) { *device = 7; return 0; }
  if (!real) real = (int (*)(int *))dlsym(RTLD_NEXT, "cudaGetDevice");
  return real(device);
}
"""

_STATUS_PROBE = r"""
import ctypes, json
import numpy as np
from tessera import runtime
lib = runtime._load_nvidia_fft_runtime()
assert lib is not None
set_mode = ctypes.CDLL(None).tessera_test_set_get_device_mode
plan, size = ctypes.c_void_p(), ctypes.c_size_t()
assert lib.tessera_nvidia_fft_plan_create_c2c_f32(1, 16, ctypes.byref(plan), ctypes.byref(size)) == 0
workspace = ctypes.c_void_p()
assert lib.tessera_nvidia_fft_workspace_alloc(size.value, ctypes.byref(workspace)) == 0
x = np.ones((1, 16), np.complex64)
out = np.empty_like(x)
pointer = ctypes.POINTER(ctypes.c_float)
def execute():
    return lib.tessera_nvidia_fft_execute_c2c_f32(
        plan, x.view(np.float32).ctypes.data_as(pointer),
        out.view(np.float32).ctypes.data_as(pointer), workspace, size.value, 0)
statuses = {}
for label, mode in (("ok", 0), ("query_failure", 1), ("other_device", 2), ("ok_again", 0)):
    set_mode(mode)
    statuses[label] = execute()
set_mode(0)
lib.tessera_nvidia_fft_workspace_free(workspace)
lib.tessera_nvidia_fft_plan_destroy(plan)
print(json.dumps(statuses))
"""


def test_device_query_failure_is_an_execution_error_not_a_mismatch(tmp_path):
    """Codex review on #841: a failed cudaGetDevice must not read as status 4.

    A preloaded shim intercepts the library's dynamic cudaGetDevice so both
    branches run on real hardware: the query failing (status 3, a CUDA error
    like any other during execution) and the query naming another device
    (status 4, the refusal) -- the second being unreachable on a one-GPU box
    otherwise.
    """
    import json
    import os
    import shutil
    import subprocess
    import sys

    runtime, lib = _runtime_or_skip()
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is needed to build the cudaGetDevice shim")
    source = tmp_path / "shim.c"
    source.write_text(_GET_DEVICE_SHIM)
    shim = tmp_path / "libgetdevice_shim.so"
    subprocess.run([compiler, "-shared", "-fPIC", str(source), "-o", str(shim), "-ldl"],
                   check=True, capture_output=True)
    env = dict(os.environ, LD_PRELOAD=str(shim))
    result = subprocess.run([sys.executable, "-c", _STATUS_PROBE], env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    statuses = json.loads(result.stdout.strip().splitlines()[-1])
    assert statuses == {"ok": 0, "query_failure": 3, "other_device": 4, "ok_again": 0}


def _cudart():
    """The CUDA runtime the FFT library already loaded (same soname)."""
    cudart = ctypes.CDLL("libcudart.so.13")
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    cudart.cudaDeviceSynchronize.argtypes = []
    return cudart


def _bind_device_entry_points(lib):
    lib.tessera_nvidia_fft_execute_c2c_device_f32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    for name in ("tessera_nvidia_fft_execute_r2c_device_f32",
                 "tessera_nvidia_fft_execute_c2r_device_f32"):
        getattr(lib, name).argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_void_p]


@pytest.mark.parametrize("kind", ("c2c", "c2c_inverse", "r2c", "c2r"))
def test_device_pointer_entry_points_match_numpy(kind):
    """ROCm-parity device-resident execution: device buffers, no staging."""
    runtime, lib = _runtime_or_skip()
    if not hasattr(lib, "tessera_nvidia_fft_execute_c2c_device_f32"):
        pytest.skip("library predates the device-pointer entry points")
    _bind_device_entry_points(lib)
    cudart = _cudart()
    batch, length = 3, 1024
    rng = np.random.default_rng(97)
    if kind.startswith("c2c"):
        host_in = (rng.standard_normal((batch, length)) +
                   1j * rng.standard_normal((batch, length))).astype(np.complex64)
        host_out = np.empty_like(host_in)
        create = lib.tessera_nvidia_fft_plan_create_c2c_f32
        expected = (np.fft.ifft if kind == "c2c_inverse" else np.fft.fft)(host_in, axis=-1)
    elif kind == "r2c":
        host_in = rng.standard_normal((batch, length)).astype(np.float32)
        host_out = np.empty((batch, length // 2 + 1), np.complex64)
        create = lib.tessera_nvidia_fft_plan_create_r2c_f32
        expected = np.fft.rfft(host_in, axis=-1)
    else:
        real = rng.standard_normal((batch, length)).astype(np.float32)
        host_in = np.fft.rfft(real, axis=-1).astype(np.complex64)
        host_out = np.empty((batch, length), np.float32)
        create = lib.tessera_nvidia_fft_plan_create_c2r_f32
        expected = real
    plan, size = ctypes.c_void_p(), ctypes.c_size_t()
    assert create(batch, length, ctypes.byref(plan), ctypes.byref(size)) == 0
    workspace = ctypes.c_void_p()
    assert lib.tessera_nvidia_fft_workspace_alloc(size.value, ctypes.byref(workspace)) == 0
    device_in, device_out = ctypes.c_void_p(), ctypes.c_void_p()
    try:
        assert cudart.cudaMalloc(ctypes.byref(device_in), host_in.nbytes) == 0
        assert cudart.cudaMalloc(ctypes.byref(device_out), host_out.nbytes) == 0
        assert cudart.cudaMemcpy(device_in, host_in.ctypes.data, host_in.nbytes, 1) == 0
        if kind.startswith("c2c"):
            rc = lib.tessera_nvidia_fft_execute_c2c_device_f32(
                plan, device_in, device_out, workspace, size.value,
                int(kind == "c2c_inverse"), None)
        elif kind == "r2c":
            rc = lib.tessera_nvidia_fft_execute_r2c_device_f32(
                plan, device_in, device_out, workspace, size.value, None)
        else:
            rc = lib.tessera_nvidia_fft_execute_c2r_device_f32(
                plan, device_in, device_out, workspace, size.value, None)
        assert rc == 0
        assert cudart.cudaDeviceSynchronize() == 0  # the entry points do not sync
        assert cudart.cudaMemcpy(host_out.ctypes.data, device_out, host_out.nbytes, 2) == 0
        np.testing.assert_allclose(host_out, expected, rtol=2e-4, atol=2e-4)
    finally:
        for pointer in (device_in, device_out):
            if pointer.value:
                cudart.cudaFree(pointer)
        lib.tessera_nvidia_fft_workspace_free(workspace)
        lib.tessera_nvidia_fft_plan_destroy(plan)



def _composite_conv_reference(x, w, normalization):
    n = x.shape[-1] + w.shape[-1] - 1
    nfft = 1 << int(np.ceil(np.log2(n)))
    return np.fft.irfft(np.fft.rfft(x, nfft, norm=normalization) *
                        np.fft.rfft(w, nfft, norm=normalization),
                        nfft, norm=normalization)[..., :n]


@pytest.mark.parametrize("normalization", ("backward", "forward", "ortho"))
@pytest.mark.parametrize("x_shape,w_shape", (
    ((13,), (5,)),                 # 1-D, odd lengths
    ((4, 3, 257), (4, 3, 33)),     # batched, one kernel per row
    ((6, 1000), (1, 64)),          # one kernel row broadcast to all rows
    ((1, 300), (5, 17)),           # outside the native envelope: host composite
))
def test_native_spectral_convolution_matches_composite(normalization, x_shape, w_shape):
    runtime, lib = _runtime_or_skip()
    if not hasattr(lib, "tessera_nvidia_spectral_conv_f32"):
        pytest.skip("library predates tessera_nvidia_spectral_conv_f32")
    rng = np.random.default_rng(len(x_shape) * 100 + x_shape[-1])
    x = rng.standard_normal(x_shape).astype(np.float32)
    w = rng.standard_normal(w_shape).astype(np.float32)
    native = runtime._nvidia_native_spectral_conv([x, w], {"normalization": normalization}, np)
    if w_shape == (5, 17):
        assert native is None  # the batch broadcast is left to the composite
    else:
        assert native is not None
        np.testing.assert_allclose(native, _composite_conv_reference(x, w, normalization),
                                   rtol=5e-5, atol=5e-5 * max(1.0, float(np.max(np.abs(native)))))
    artifact = runtime.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_spectral_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["x", "w"], "output_name": "o",
        "ops": [{"op_name": "tessera.spectral_conv", "result": "o",
                 "operands": ["x", "w"], "kwargs": {"normalization": normalization}}],
    })
    result = runtime.launch(artifact, (x, w))
    assert result["ok"] is True, result.get("reason")
    expected = _composite_conv_reference(x, w, normalization)
    np.testing.assert_allclose(np.asarray(result["output"]), expected, rtol=5e-5,
                               atol=5e-5 * max(1.0, float(np.max(np.abs(expected)))))


# Every layout entry point is called with a null shape or stride descriptor
# (Codex review on #842: the compact-layout fast path indexed `strides` before
# any check). Each must return its invalid-argument status, not crash. Run in a
# child so a segfault is a failed assertion rather than a dead pytest worker.
_NULL_DESCRIPTOR_PROBE = r"""
import ctypes, json, sys
lib = ctypes.CDLL(sys.argv[1])
P, I, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
buf = ctypes.create_string_buffer(1 << 16)
b = ctypes.cast(buf, P)
L2 = (ctypes.c_int64 * 2)
L3 = (ctypes.c_int64 * 3)
shape2, strides2 = L2(2, 64), L2(64, 1)
shape3, strides3 = L3(2, 7, 9), L3(63, 9, 1)
wshape, wstrides = (ctypes.c_int64 * 1)(16), (ctypes.c_int64 * 1)(1)
results = {}
for label, sh2, st2, sh3, st3 in (("strides", shape2, None, shape3, None),
                                  ("shape", None, strides2, None, strides3)):
    calls = {
        "dct_f32": lambda: lib.tessera_nvidia_dct_policy_layout_f32(
            None, b, b, I(2), sh2, st2, I(1), I(2), F(1.0)),
        "stft_f32": lambda: lib.tessera_nvidia_stft_policy_broadcast_layout_f32(
            None, b, b, b, I(2), sh2, st2, I(1), I(1), wshape, wstrides,
            I(16), I(8), I(7), F(1.0), I(0), I(0), I(1)),
        "stft_jvp_f32": lambda: lib.tessera_nvidia_stft_jvp_broadcast_layout_f32(
            None, b, b, b, b, b, b, I(2), sh2, st2, I(1), I(1), wshape,
            wstrides, I(16), I(8), I(7), F(1.0), I(0), I(0), I(1)),
        "istft_f32": lambda: lib.tessera_nvidia_istft_policy_broadcast_layout_f32(
            None, b, b, b, I(3), sh3, st3, I(2), I(1), wshape, wstrides,
            I(16), I(8), F(1.0), I(0), I(64), I(1)),
        "istft_jvp_f32": lambda: lib.tessera_nvidia_istft_jvp_broadcast_layout_f32(
            None, b, b, b, b, b, b, I(3), sh3, st3, I(2), I(1), wshape,
            wstrides, I(16), I(8), F(1.0), I(0), I(64), I(1)),
        "streaming_f32": lambda: lib.tessera_nvidia_streaming_stft_broadcast_layout_f32(
            None, b, b, b, b, b, I(2), sh2, st2, I(1), I(0), I(1), wshape,
            wstrides, I(16), I(8), I(7), F(1.0), I(1)),
        "stft_backward_f32": lambda: lib.tessera_nvidia_stft_backward_broadcast_layout_f32(
            None, b, b, b, b, b, I(2), sh2, st2, I(1), I(3), sh3, st3, I(1),
            wshape, wstrides, I(16), I(8), F(1.0), I(0), I(0), I(1)),
        "istft_backward_f32": lambda: lib.tessera_nvidia_istft_backward_broadcast_layout_f32(
            None, b, b, b, b, b, I(2), sh2, st2, I(1), I(3), sh3, st3, I(1),
            I(2), I(1), wshape, wstrides, I(16), I(8), F(1.0), I(0), I(1)),
        "istft_storage": lambda: lib.tessera_nvidia_istft_policy_broadcast_layout_storage(
            None, b, b, b, I(3), sh3, st3, I(2), I(1), wshape, wstrides,
            I(16), I(8), I(1), F(1.0), I(0), I(64), I(1)),
        "istft_jvp_storage": lambda: lib.tessera_nvidia_istft_jvp_broadcast_layout_storage(
            None, b, b, b, b, b, b, I(3), sh3, st3, I(2), I(1), wshape,
            wstrides, I(16), I(8), I(1), F(1.0), I(0), I(64), I(1)),
        "istft_backward_storage": lambda: lib.tessera_nvidia_istft_backward_broadcast_layout_storage(
            None, b, b, b, b, b, I(2), sh2, st2, I(1), I(3), sh3, st3, I(1),
            I(2), I(1), wshape, wstrides, I(16), I(8), I(1), F(1.0), I(0), I(1)),
    }
    for name, call in calls.items():
        results[f"{name}/{label}"] = call()
print(json.dumps(results))
"""

_NULL_DESCRIPTOR_STATUS = {
    "dct_f32": 290, "stft_f32": 300, "stft_jvp_f32": 360, "istft_f32": 310,
    "istft_jvp_f32": 350, "streaming_f32": 316, "stft_backward_f32": 320,
    "istft_backward_f32": 330, "istft_storage": 344, "istft_jvp_storage": 356,
    "istft_backward_storage": 366,
}


def test_layout_entry_points_refuse_null_descriptors():
    import json
    import subprocess
    import sys

    _, lib = _runtime_or_skip()
    result = subprocess.run([sys.executable, "-c", _NULL_DESCRIPTOR_PROBE, lib._name],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (
        f"probe died (rc={result.returncode}): {result.stderr[-2000:]}")
    statuses = json.loads(result.stdout.strip().splitlines()[-1])
    expected = {f"{name}/{label}": status
                for name, status in _NULL_DESCRIPTOR_STATUS.items()
                for label in ("strides", "shape")}
    assert statuses == expected
