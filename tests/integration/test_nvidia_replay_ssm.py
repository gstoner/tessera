"""Native NVIDIA ReplaySSM serving integration proofs."""
from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import nvidia_cuda_host_ready
from tessera import runtime as rt
from tessera.cache import SSMStateHandle
from tessera.compiler.emit import nvidia_cuda


@pytest.mark.skipif(not nvidia_cuda_host_ready(), reason="CUDA toolkit or GPU unavailable")
@pytest.mark.hardware_nvidia
def test_cuda_replay_async_ring_reports_backpressure():
    rng = np.random.default_rng(1210); batch, channels, state = 1, 2, 2
    gpu = rt.nvidia_ssm_replay_state_handle(batch, channels, state, -np.ones(channels), capacity=8, async_slots=2)
    def inputs():
        return (np.abs(rng.standard_normal((1, batch, channels))) * .1, rng.standard_normal((1, batch, channels)), rng.standard_normal((1, batch, state)), rng.standard_normal((1, batch, state)))
    first = gpu.submit_block_async(*inputs()); second = gpu.submit_block_async(*inputs())
    with pytest.raises(RuntimeError, match="ring is full"): gpu.submit_block_async(*inputs())
    first.wait(); third = gpu.submit_block_async(*inputs()); second.wait(); third.wait()


@pytest.mark.skipif(not nvidia_cuda_host_ready(), reason="CUDA toolkit or GPU unavailable")
@pytest.mark.hardware_nvidia
def test_cuda_replay_kernel_executes_on_available_nvidia_host():
    rng = np.random.default_rng(73); steps, batch, channels, state = 5, 2, 4, 3
    delta = np.abs(rng.standard_normal((steps, batch, channels))).astype(np.float32) * 0.25
    x = rng.standard_normal((steps, batch, channels)).astype(np.float32); b = rng.standard_normal((steps, batch, state)).astype(np.float32); s0 = rng.standard_normal((batch, channels, state)).astype(np.float32); c = rng.standard_normal((batch, state)).astype(np.float32); a = (-np.abs(rng.standard_normal(channels))).astype(np.float32)
    out = nvidia_cuda.run_ssm_replay_decode_f32(delta, x, b, s0, c, a)
    reference = SSMStateHandle(batch=batch, num_channels=channels, state_dim=state, a=a, capacity=steps + 1); reference._s0 = s0.astype(np.float64)
    for i in range(steps): reference.append(delta[i], x[i], b[i])
    np.testing.assert_allclose(out, reference.read_output(c), rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize("batch,channels,state", [
    (1, 3, 2), (2, 7, 5), (4, 8, 3),
])
@pytest.mark.skipif(not nvidia_cuda_host_ready(),
                    reason="CUDA toolkit or GPU unavailable")
@pytest.mark.hardware_nvidia
def test_cuda_replay_wide_transition_matrix(batch, channels, state):
    """Long decode, forced flush, rollback, reset, and block submit stay exact."""
    rng = np.random.default_rng(1700 + batch * 100 + channels * 10 + state)
    steps, capacity = 29, 9
    a = -np.abs(rng.standard_normal(channels))
    delta = np.abs(rng.standard_normal((steps, batch, channels))) * .15
    x = rng.standard_normal((steps, batch, channels))
    b = rng.standard_normal((steps, batch, state))
    c = rng.standard_normal((steps, batch, state))
    gpu = rt.nvidia_ssm_replay_state_handle(
        batch, channels, state, a, capacity=capacity, async_slots=3)
    ref = SSMStateHandle(batch, channels, state, a, capacity=capacity)
    try:
        # Cross several actual flush boundaries.
        for i in range(19):
            np.testing.assert_allclose(
                gpu.step(delta[i], x[i], b[i], c[i]),
                ref.step(delta[i], x[i], b[i], c[i]),
                rtol=2e-4, atol=2e-4)
        # Reject a speculative suffix by rewinding only the replay cursor.
        for i in range(19, 22):
            gpu.append(delta[i], x[i], b[i], auto_flush=False)
            ref.append(delta[i], x[i], b[i], auto_flush=False)
        gpu.rollback(2)
        ref.rollback(2)
        np.testing.assert_allclose(
            gpu.read_output(c[22]), ref.read_output(c[22]),
            rtol=2e-4, atol=2e-4)
        # One native block must have the same ordered transition semantics.
        got = gpu.step_block(delta[22:26], x[22:26], b[22:26], c[22:26])
        want = np.stack([ref.step(delta[i], x[i], b[i], c[i])
                         for i in range(22, 26)])
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)
        gpu.reset()
        ref.reset()
        np.testing.assert_allclose(
            gpu.step(delta[28], x[28], b[28], c[28]),
            ref.step(delta[28], x[28], b[28], c[28]),
            rtol=2e-4, atol=2e-4)
    finally:
        if gpu._device is not None:
            gpu._device.close()


@pytest.mark.skipif(not nvidia_cuda_host_ready(),
                    reason="CUDA toolkit or GPU unavailable")
@pytest.mark.hardware_nvidia
def test_cuda_replay_rejected_async_work_does_not_advance_committed_cursor():
    rng = np.random.default_rng(1811)
    batch, channels, state = 1, 4, 3
    gpu = rt.nvidia_ssm_replay_state_handle(
        batch, channels, state, -np.ones(channels),
        capacity=12, async_slots=2)

    def inputs():
        return (
            np.abs(rng.standard_normal((2, batch, channels))) * .1,
            rng.standard_normal((2, batch, channels)),
            rng.standard_normal((2, batch, state)),
            rng.standard_normal((2, batch, state)),
        )

    try:
        first = gpu.submit_block_async(*inputs())
        second = gpu.submit_block_async(*inputs())
        count_before = gpu.count
        with pytest.raises(RuntimeError, match="ring is full"):
            gpu.submit_block_async(*inputs())
        assert gpu.count == count_before
        first.wait()
        second.wait()
    finally:
        if gpu._device is not None:
            gpu._device.close()
