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
