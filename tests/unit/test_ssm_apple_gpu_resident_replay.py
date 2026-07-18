from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.cache import SSMStateHandle


def _inputs(seed: int, tokens: int, B: int, D: int, N: int):
    rng = np.random.default_rng(seed)
    delta = np.abs(rng.standard_normal((tokens, B, D))) * 0.2
    x = rng.standard_normal((tokens, B, D))
    b = rng.standard_normal((tokens, B, N))
    c = rng.standard_normal((tokens, B, N))
    return delta, x, b, c


def test_resident_replay_rejects_invalid_ring_size():
    with pytest.raises(ValueError, match="async_slots"):
        rt.apple_gpu_resident_ssm_replay_state_handle(
            1, 4, 3, -np.ones(4), async_slots=0)


def test_resident_replay_symbols_exported():
    runtime = rt._load_apple_gpu_runtime()
    assert hasattr(runtime, "ts_enc_commit_async")
    assert hasattr(runtime, "ts_enc_wait_destroy")
    assert hasattr(runtime, "tessera_apple_gpu_ssm_replay_decode_dev_f32_enc")


@pytest.mark.hardware_apple_gpu
def test_resident_replay_step_forced_flush_and_rollback_match_oracle():
    from tests._support.apple import require_apple_metal
    require_apple_metal()
    B, D, N, T = 1, 5, 3, 9
    a = -np.linspace(0.1, 0.7, D)
    delta, x, b, c = _inputs(5101, T, B, D, N)
    gpu = rt.apple_gpu_resident_ssm_replay_state_handle(
        B, D, N, a, capacity=4, async_slots=2)
    ref = SSMStateHandle(B, D, N, a, capacity=4)
    assert gpu.resident_inputs
    try:
        for i in range(6):
            np.testing.assert_allclose(
                gpu.step(delta[i], x[i], b[i], c[i]),
                ref.step(delta[i], x[i], b[i], c[i]),
                rtol=3e-4, atol=3e-4)
        assert gpu.last_flush_execution == "resident_write_after_reference_fold"
        gpu.append(delta[6], x[6], b[6]); ref.append(delta[6], x[6], b[6])
        gpu.append(delta[7], x[7], b[7]); ref.append(delta[7], x[7], b[7])
        gpu.rollback(1); ref.rollback(1)
        np.testing.assert_allclose(
            gpu.read_output(c[7]), ref.read_output(c[7]), rtol=3e-4, atol=3e-4)
    finally:
        gpu.close()
    assert gpu.lifecycle_telemetry()["closed"] is True


@pytest.mark.hardware_apple_gpu
def test_resident_replay_ordered_ring_backpressure_and_cleanup():
    from tests._support.apple import require_apple_metal
    require_apple_metal()
    B, D, N = 1, 4, 3
    a = -np.linspace(0.2, 0.8, D)
    gpu = rt.apple_gpu_resident_ssm_replay_state_handle(
        B, D, N, a, capacity=16, async_slots=2)
    ref = SSMStateHandle(B, D, N, a, capacity=16)
    futures = []
    expected = []
    try:
        for seed, tokens in ((5201, 2), (5202, 3)):
            delta, x, b, c = _inputs(seed, tokens, B, D, N)
            futures.append(gpu.submit_block_async(delta, x, b, c))
            expected.append(np.stack([
                ref.step(delta[i], x[i], b[i], c[i]) for i in range(tokens)]))
        before = gpu.count
        delta, x, b, c = _inputs(5203, 1, B, D, N)
        with pytest.raises(RuntimeError, match="ring is full"):
            gpu.submit_block_async(delta, x, b, c)
        assert gpu.count == before
        with pytest.raises(RuntimeError, match="pending"):
            gpu.flush()
        assert futures[0].device_buffer.mtl_buffer() != 0
        np.testing.assert_allclose(
            futures[0].wait(), expected[0], rtol=3e-4, atol=3e-4)
        third = gpu.submit_block_async(delta, x, b, c)
        expected_third = ref.step(delta[0], x[0], b[0], c[0])[None]
        np.testing.assert_allclose(
            futures[1].wait(), expected[1], rtol=3e-4, atol=3e-4)
        np.testing.assert_allclose(
            third.wait(), expected_third, rtol=3e-4, atol=3e-4)
        assert gpu.last_submission_telemetry["submission_mode"] == \
            "ordered_async_command_buffer"
        assert gpu.lifecycle_telemetry()["leased_slots"] == 0
        gpu.flush()
        assert gpu.count == 0
    finally:
        gpu.close()
    lifecycle = gpu.lifecycle_telemetry()
    assert lifecycle["pending_submissions"] == 0
    assert lifecycle["leased_slots"] == 0
    assert lifecycle["closed"] is True


@pytest.mark.hardware_apple_gpu
def test_resident_replay_partial_speculative_rejection_commits_prefix_only():
    from tests._support.apple import require_apple_metal
    require_apple_metal()
    B, D, N, T = 1, 4, 3, 5
    a = -np.ones(D) * 0.4
    delta, x, b, c = _inputs(5301, T, B, D, N)
    gpu = rt.apple_gpu_resident_ssm_replay_state_handle(
        B, D, N, a, capacity=16, spec_window=2, async_slots=2)
    ref = SSMStateHandle(B, D, N, a, capacity=16, spec_window=2)
    try:
        out = gpu.submit_block_async(delta, x, b, c).wait()
        want = np.stack([ref.step(delta[i], x[i], b[i], c[i]) for i in range(T)])
        np.testing.assert_allclose(out, want, rtol=3e-4, atol=3e-4)
        gpu.rollback(3); ref.rollback(3)
        assert gpu.count == ref.count == 2
        np.testing.assert_allclose(
            gpu.read_output(c[1]), ref.read_output(c[1]), rtol=3e-4, atol=3e-4)
    finally:
        gpu.close()
