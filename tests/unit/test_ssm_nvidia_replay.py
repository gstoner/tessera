"""CUDA ReplaySSM output-only decode contract.

The NVIDIA factory must retain the reference handle as its semantic fallback;
on an sm_120 CUDA host it exercises the fused one-launch reconstruction, while
off-device these tests still validate the exact same state-handle ABI.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

import tessera
from tessera import runtime as rt
from tessera.cache import SSMStateHandle
from tessera.compiler.emit import nvidia_cuda


def _cuda_host_ready() -> bool:
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not os.path.isfile(nvcc) or shutil.which("nvidia-smi") is None:
        return False
    try:
        return subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=5, check=False,
        ).returncode == 0
    except OSError:
        return False


def _decode(handle, delta, x, b, c):
    B, S, D = x.shape
    out = np.zeros((B, S, D))
    for t in range(S):
        out[:, t, :] = handle.step(delta[:, t, :], x[:, t, :],
                                   b[:, t, :], c[:, t, :])
    return out


def test_nvidia_replay_factory_wires_scalar_decode_path():
    h = rt.nvidia_ssm_replay_state_handle(1, 4, 3, -np.ones(4), capacity=8)
    assert isinstance(h, SSMStateHandle)
    assert h.backend == "nvidia_sm120_replay_device"
    assert getattr(h, "_device") is not None


def test_nvidia_replay_decode_matches_eager_or_falls_back():
    """The CUDA kernel runs when available; otherwise the factory declines.

    Both routes must preserve the ReplaySSM identity.  This remains host-safe so
    the core state ABI is continuously checked outside the CUDA machine too.
    """
    rng = np.random.default_rng(417)
    B, S, D, N = 2, 13, 5, 4
    x = rng.standard_normal((B, S, D))
    a = -np.abs(rng.standard_normal(D))
    b = rng.standard_normal((B, S, N))
    c = rng.standard_normal((B, S, N))
    delta = np.abs(rng.standard_normal((B, S, D))) * 0.5
    eager = np.asarray(tessera.ops.selective_ssm(x, a, b, c, delta))
    handle = rt.nvidia_ssm_replay_state_handle(B, D, N, a, capacity=32)
    np.testing.assert_allclose(_decode(handle, delta, x, b, c), eager,
                               rtol=5e-4, atol=5e-4)


def test_cuda_replay_source_keeps_state_read_only():
    source = nvidia_cuda._synthesize_ssm_replay_decode_cuda()
    assert "ssm_replay_k" in source
    assert "const float*s0" in source
    assert "float*y" in source
    assert "cudaMemcpy(hy,y" in source


def test_cuda_replay_shape_validation_precedes_compilation():
    with np.testing.assert_raises_regex(ValueError, r"matching \[M,B,D\]"):
        nvidia_cuda.run_ssm_replay_decode_f32(
            np.zeros((1, 1, 2), np.float32), np.zeros((1, 2, 1), np.float32),
            np.zeros((1, 1, 3), np.float32), np.zeros((1, 2, 3), np.float32),
            np.zeros((1, 3), np.float32), np.zeros(2, np.float32),
        )


def test_cuda_replay_long_decode_flush_and_rollback_match_reference():
    rng = np.random.default_rng(818)
    B, D, N, T, L = 2, 4, 3, 41, 7
    a = -np.abs(rng.standard_normal(D))
    delta = np.abs(rng.standard_normal((T, B, D))) * .2
    x, b, c = (rng.standard_normal((T, B, q)) for q in (D, N, N))
    gpu = rt.nvidia_ssm_replay_state_handle(B, D, N, a, capacity=L)
    ref = SSMStateHandle(B, D, N, a, capacity=L)
    for t in range(T):
        np.testing.assert_allclose(gpu.step(delta[t], x[t], b[t], c[t]),
                                   ref.step(delta[t], x[t], b[t], c[t]),
                                   rtol=2e-4, atol=2e-4)
    for t in range(4):
        gpu.append(delta[t], x[t], b[t]); ref.append(delta[t], x[t], b[t])
    gpu.rollback(2); ref.rollback(2)
    np.testing.assert_allclose(gpu.read_output(c[0]), ref.read_output(c[0]),
                               rtol=2e-4, atol=2e-4)


def test_cuda_replay_block_submit_matches_ordered_steps():
    rng = np.random.default_rng(119)
    T, B, D, N = 6, 2, 4, 3
    a = -np.abs(rng.standard_normal(D))
    d = np.abs(rng.standard_normal((T, B, D))) * .2
    x, b, c = (rng.standard_normal((T, B, q)) for q in (D, N, N))
    gpu = rt.nvidia_ssm_replay_state_handle(B, D, N, a, capacity=16)
    ref = SSMStateHandle(B, D, N, a, capacity=16)
    got = gpu.step_block(d, x, b, c)
    want = np.stack([ref.step(d[i], x[i], b[i], c[i]) for i in range(T)])
    np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)


def test_cuda_replay_async_submit_wait_matches_ordered_steps():
    rng = np.random.default_rng(201)
    T, B, D, N = 4, 1, 3, 2
    a = -np.abs(rng.standard_normal(D)); d = np.abs(rng.standard_normal((T,B,D))) *.2
    x, b, c = (rng.standard_normal((T, B, q)) for q in (D, N, N))
    gpu = rt.nvidia_ssm_replay_state_handle(B,D,N,a,capacity=8)
    ref = SSMStateHandle(B,D,N,a,capacity=8)
    future = gpu.submit_block_async(d,x,b,c)
    assert future.device_buffer.shape == (T, B, D)
    assert future.device_buffer.dtype == "float32"
    got = future.wait()
    want = np.stack([ref.step(d[i],x[i],b[i],c[i]) for i in range(T)])
    np.testing.assert_allclose(got,want,rtol=2e-4,atol=2e-4)


@pytest.mark.skipif(not _cuda_host_ready(), reason="CUDA toolkit or GPU unavailable")
def test_cuda_replay_kernel_executes_on_available_nvidia_host():
    rng = np.random.default_rng(73)
    M, B, D, N = 5, 2, 4, 3
    delta = np.abs(rng.standard_normal((M, B, D))).astype(np.float32) * 0.25
    x = rng.standard_normal((M, B, D)).astype(np.float32)
    b = rng.standard_normal((M, B, N)).astype(np.float32)
    s0 = rng.standard_normal((B, D, N)).astype(np.float32)
    c = rng.standard_normal((B, N)).astype(np.float32)
    a = (-np.abs(rng.standard_normal(D))).astype(np.float32)
    out = nvidia_cuda.run_ssm_replay_decode_f32(delta, x, b, s0, c, a)
    reference = SSMStateHandle(batch=B, num_channels=D, state_dim=N, a=a,
                               capacity=M + 1)
    reference._s0 = s0.astype(np.float64)
    for i in range(M):
        reference.append(delta[i], x[i], b[i])
    np.testing.assert_allclose(out, reference.read_output(c), rtol=5e-5, atol=5e-5)
