"""Exact CUDA replay ownership and asynchronous numerical proof."""
import numpy as np
import pytest
from tests.unit._nvidia_testutil import nvidia_cuda_host_ready
from tessera import runtime as rt
from tessera.cache import SSMStateHandle


@pytest.mark.hardware_nvidia
@pytest.mark.skipif(not nvidia_cuda_host_ready(), reason="CUDA toolkit or GPU unavailable")
def test_nvidia_replay_factory_binds_scalar_device_state():
    h = rt.nvidia_ssm_replay_state_handle(1, 4, 3, -np.ones(4), capacity=8)
    assert h.backend == "nvidia_sm120_replay_device"
    assert h._device is not None

@pytest.mark.hardware_nvidia
@pytest.mark.skipif(not nvidia_cuda_host_ready(), reason="CUDA toolkit or GPU unavailable")
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
    assert future.event.elapsed_ms() > 0
    got = future.wait()
    want = np.stack([ref.step(d[i],x[i],b[i],c[i]) for i in range(T)])
    np.testing.assert_allclose(got,want,rtol=2e-4,atol=2e-4)


@pytest.mark.hardware_nvidia
@pytest.mark.skipif(not nvidia_cuda_host_ready(), reason="CUDA toolkit or GPU unavailable")
def test_cuda_replay_multi_slot_ring_and_device_consumer_protocol():
    rng = np.random.default_rng(1209)
    B, D, N = 1, 4, 3
    a = -np.abs(rng.standard_normal(D))
    gpu = rt.nvidia_ssm_replay_state_handle(
        B, D, N, a, capacity=12, async_slots=2)
    ref = SSMStateHandle(B, D, N, a, capacity=12)

    futures = []
    expected = []
    for T in (2, 3):
        d = np.abs(rng.standard_normal((T, B, D))) * .2
        x = rng.standard_normal((T, B, D))
        b = rng.standard_normal((T, B, N))
        c = rng.standard_normal((T, B, N))
        futures.append(gpu.submit_block_async(d, x, b, c))
        expected.append(np.stack([
            ref.step(d[i], x[i], b[i], c[i]) for i in range(T)]))

    iface = futures[0].device_buffer.__cuda_array_interface__
    assert iface["shape"] == (2, B, D)
    assert iface["typestr"] == "<f4"
    assert iface["data"][0] != 0
    assert iface["stream"] != 0
    futures[0].event.wait()
    np.testing.assert_allclose(
        futures[0].wait(), expected[0], rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(
        futures[1].wait(), expected[1], rtol=2e-4, atol=2e-4)

