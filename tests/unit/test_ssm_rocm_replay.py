"""ROCM-REPLAY-1 persistent gfx1151 serving contract and exact-device proof."""
from __future__ import annotations

import math
import os
import shutil

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.cache import SSMStateHandle
from tessera.compiler.emit import rocm_hip
from tessera.speculative import advance_ssm


def _rocm_ready() -> bool:
    hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
    return os.path.isfile(hipcc) and rt._rocm_chip() == "gfx1151"


def _inputs(seed: int, tokens: int, batch: int, channels: int, state_dim: int):
    rng = np.random.default_rng(seed)
    delta = np.abs(rng.standard_normal((tokens, batch, channels))) * .2
    x = rng.standard_normal((tokens, batch, channels))
    b = rng.standard_normal((tokens, batch, state_dim))
    c = rng.standard_normal((tokens, batch, state_dim))
    return delta, x, b, c


def test_rocm_replay_source_has_persistent_checkpoint_and_slot_ring():
    source = rocm_hip._synthesize_ssm_replay_device_hip()
    assert "float*d,*x,*b,*s0,*c,*a,*y" in source
    assert "__global__ void replay_out" in source
    assert "__global__ void replay_gram" in source
    assert "q.gram[(long long)i*q.B+bi]" in source
    assert "__global__ void replay_flush" in source
    assert "hipStreamCreateWithFlags" in source
    assert "hipHostMalloc" in source
    assert "hipEventElapsedTime" in source
    assert "hipStreamWaitEvent" in source
    assert "extern \"C\" void* dp" in source


def test_rocm_replay_factory_and_async_slot_guard():
    handle = rt.rocm_ssm_replay_state_handle(
        1, 4, 3, -np.ones(4), capacity=8)
    assert isinstance(handle, SSMStateHandle)
    assert handle.backend == "rocm_gfx1151_replay_device"
    if _rocm_ready():
        assert handle._device is not None
    with pytest.raises(ValueError, match="at least two slots"):
        rt.rocm_ssm_replay_state_handle(
            1, 4, 3, -np.ones(4), capacity=8, async_slots=1)


@pytest.mark.skipif(not _rocm_ready(), reason="requires gfx1151 and hipcc")
def test_rocm_replay_long_decode_flush_rollback_and_reset():
    B, D, N, T, L = 2, 5, 4, 43, 7
    rng = np.random.default_rng(818)
    a = -np.abs(rng.standard_normal(D))
    delta, x, b, c = _inputs(819, T, B, D, N)
    gpu = rt.rocm_ssm_replay_state_handle(B, D, N, a, capacity=L)
    ref = SSMStateHandle(B, D, N, a, capacity=L)
    assert gpu._device is not None
    for token in range(T):
        np.testing.assert_allclose(
            gpu.step(delta[token], x[token], b[token], c[token]),
            ref.step(delta[token], x[token], b[token], c[token]),
            rtol=3e-4, atol=3e-4)
    for token in range(4):
        gpu.append(delta[token], x[token], b[token])
        ref.append(delta[token], x[token], b[token])
    gpu.rollback(2); ref.rollback(2)
    np.testing.assert_allclose(gpu.read_output(c[0]), ref.read_output(c[0]),
                               rtol=3e-4, atol=3e-4)
    gpu.reset(); ref.reset()
    gpu.append(delta[0], x[0], b[0]); ref.append(delta[0], x[0], b[0])
    np.testing.assert_allclose(gpu.read_output(c[0]), ref.read_output(c[0]),
                               rtol=3e-4, atol=3e-4)


@pytest.mark.skipif(not _rocm_ready(), reason="requires gfx1151 and hipcc")
def test_rocm_summary_baseline_writes_state_each_token():
    B, D, N, T = 1, 4, 3, 9
    a = -np.linspace(.2, 1.0, D)
    delta, x, b, c = _inputs(991, T, B, D, N)
    device = rocm_hip.RocmReplayDeviceState(
        np.zeros((B, D, N)), a, capacity=1)
    ref = SSMStateHandle(B, D, N, a, capacity=1)
    for token in range(T):
        out, device_ms = device.summary_step(
            delta[token], x[token], b[token], c[token])
        expected = ref.step(delta[token], x[token], b[token], c[token])
        ref.flush()
        # HIP event timing is valid at zero for work below the device timer's
        # reporting resolution; reject only invalid/negative measurements.
        assert math.isfinite(device_ms) and device_ms >= 0
        np.testing.assert_allclose(out, expected, rtol=3e-4, atol=3e-4)


@pytest.mark.skipif(not _rocm_ready(), reason="requires gfx1151 and hipcc")
def test_rocm_replay_block_and_speculative_rejection_match_reference():
    T, B, D, N = 6, 1, 4, 3
    a = -np.arange(1, D + 1, dtype=np.float64) / D
    delta, x, b, c = _inputs(1201, T, B, D, N)
    gpu = rt.rocm_ssm_replay_state_handle(
        B, D, N, a, capacity=16, spec_window=2)
    ref = SSMStateHandle(B, D, N, a, capacity=16, spec_window=2)
    got = gpu.step_block(delta[:3], x[:3], b[:3], c[:3])
    want = np.stack([
        ref.step(delta[i], x[i], b[i], c[i]) for i in range(3)])
    np.testing.assert_allclose(got, want, rtol=3e-4, atol=3e-4)
    for i in range(3, 6):
        gpu.append(delta[i], x[i], b[i], auto_flush=False)
        ref.append(delta[i], x[i], b[i], auto_flush=False)
    advance_ssm(gpu, 1, num_drafts=3)
    advance_ssm(ref, 1, num_drafts=3)
    np.testing.assert_allclose(gpu.read_output(c[-1]), ref.read_output(c[-1]),
                               rtol=3e-4, atol=3e-4)
    fork = gpu.clone()
    fork.append(delta[-1], x[-1], b[-1], auto_flush=False)
    assert fork.count == gpu.count + 1
    np.testing.assert_allclose(gpu.read_output(c[0]), ref.read_output(c[0]),
                               rtol=3e-4, atol=3e-4)


@pytest.mark.skipif(not _rocm_ready(), reason="requires gfx1151 and hipcc")
def test_rocm_replay_async_order_backpressure_and_device_lease():
    B, D, N = 1, 4, 3
    a = -np.ones(D)
    gpu = rt.rocm_ssm_replay_state_handle(
        B, D, N, a, capacity=16, async_slots=2)
    ref = SSMStateHandle(B, D, N, a, capacity=16)
    futures, expected = [], []
    for seed, tokens in ((1301, 2), (1302, 3)):
        delta, x, b, c = _inputs(seed, tokens, B, D, N)
        futures.append(gpu.submit_block_async(delta, x, b, c))
        expected.append(np.stack([
            ref.step(delta[i], x[i], b[i], c[i]) for i in range(tokens)]))
    delta, x, b, c = _inputs(1303, 1, B, D, N)
    with pytest.raises(RuntimeError, match="ring is full"):
        gpu.submit_block_async(delta, x, b, c)
    iface = futures[0].device_buffer.__hip_array_interface__
    assert iface["shape"] == (2, B, D)
    assert iface["data"][0] != 0 and iface["stream"] != 0
    futures[0].event.wait()
    elapsed_ms = futures[0].event.elapsed_ms()
    assert math.isfinite(elapsed_ms) and elapsed_ms >= 0
    np.testing.assert_allclose(futures[0].wait(), expected[0],
                               rtol=3e-4, atol=3e-4)
    third = gpu.submit_block_async(delta, x, b, c)
    np.testing.assert_allclose(futures[1].wait(), expected[1],
                               rtol=3e-4, atol=3e-4)
    third.wait()

    # Device-consumer path: establish cross-stream order, then retire the lease
    # without downloading it. The slot becomes reusable only after that stream.
    delta, x, b, c = _inputs(1304, 1, B, D, N)
    leased = gpu.submit_block_async(delta, x, b, c)
    hip = rt._load_hip_for_launch()
    assert hip is not None
    stream = rt.ctypes.c_void_p()
    assert hip.hipStreamCreate(rt.ctypes.byref(stream)) == 0
    leased.event.wait_on(int(stream.value))
    leased.release(stream=int(stream.value))
    assert hip.hipStreamSynchronize(stream) == 0
    assert hip.hipStreamDestroy(stream) == 0
    with pytest.raises(RuntimeError, match="already consumed"):
        _ = leased.device_buffer.__hip_array_interface__
