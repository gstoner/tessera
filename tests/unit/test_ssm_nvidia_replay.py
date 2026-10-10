"""CUDA ReplaySSM output-only decode contract.

The NVIDIA factory must retain the reference handle as its semantic fallback;
on an sm_120 CUDA host it exercises the fused one-launch reconstruction, while
off-device these tests still validate the exact same state-handle ABI.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera
from tessera import runtime as rt
from tessera.cache import SSMStateHandle
from tessera.compiler.emit import nvidia_cuda


def _decode(handle, delta, x, b, c):
    B, S, D = x.shape
    out = np.zeros((B, S, D))
    for t in range(S):
        out[:, t, :] = handle.step(delta[:, t, :], x[:, t, :],
                                   b[:, t, :], c[:, t, :])
    return out


def test_nvidia_replay_factory_wires_scalar_decode_path(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("CUDA device constructor unavailable")
    monkeypatch.setattr(nvidia_cuda, "NvidiaReplayDeviceState", unavailable)
    h = rt.nvidia_ssm_replay_state_handle(1, 4, 3, -np.ones(4), capacity=8)
    assert isinstance(h, SSMStateHandle)
    assert h.backend == "nvidia_sm120_replay_device"
    descriptor = h._state_descriptor
    assert descriptor.workspace.lifetime == "session"
    assert descriptor.workspace.initialization == "preserve"
    assert descriptor.ordering.synchronization[-1] == "teardown_drains_pending"
    assert descriptor.workspace.bytes > descriptor.checkpoint_bytes
    assert descriptor.pinned_host_bytes > 0
    assert getattr(h, "_device") is None


def test_nvidia_replay_factory_rejects_single_async_slot():
    with pytest.raises(ValueError, match="at least two slots"):
        rt.nvidia_ssm_replay_state_handle(
            1, 4, 3, -np.ones(4), capacity=8, async_slots=1)


def test_nvidia_replay_descriptor_rejects_out_of_ring_span_before_submission():
    h = rt.nvidia_ssm_replay_state_handle(
        1, 4, 3, -np.ones(4), capacity=8, async_slots=2)
    with pytest.raises(ValueError, match="contained in the persistent ring"):
        h._state_descriptor.validate_span(start=7, tokens=2)


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


def test_cuda_replay_async_source_owns_an_ordered_slot_ring():
    source = nvidia_cuda._synthesize_ssm_replay_device_cuda()
    assert "S*slots" in source
    assert "cudaEvent_t beg,ev" in source
    assert "cudaEventQuery(z.ev)" in source
    assert "cudaEventElapsedTime(ms,z.beg,z.ev)" in source
    assert "cudaStreamWaitEvent" in source
    assert "cudaEventRecord(z.ev,st)" in source
    assert "extern \"C\" void* dp" in source


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
