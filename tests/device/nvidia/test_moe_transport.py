from __future__ import annotations

import numpy as np
import pytest

from tests._support.nvidia import assert_native_gpu, nvidia_mma_runtime_available
from tessera.stdlib import moe


def _art(rt, op, names):
    return rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120",
        "compiler_path": "nvidia_moe_transport_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": names,
        "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names, "kwargs": {}}],
    })


def _plan(seed=1):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, 4, (12, 2), dtype=np.int64)
    weights = rng.random((12, 2), dtype=np.float32)
    weights /= weights.sum(1, keepdims=True)
    return moe.plan_dispatch(ids, weights, 4, capacity=5)


@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="requires nvcc and NVIDIA GPU")
@pytest.mark.hardware_nvidia
def test_dispatch_and_combine_match_oracles():
    from tessera import runtime as rt

    rng = np.random.default_rng(3)
    x = rng.standard_normal((12, 9)).astype(np.float32)
    plan = _plan(3)
    dispatched = rt.launch(_art(rt, "tessera.moe_dispatch", ["x", "plan"]), (x, plan))
    assert_native_gpu(dispatched)
    np.testing.assert_array_equal(dispatched["output"], moe.dispatch(x, plan))
    partials = dispatched["output"] * rng.uniform(.8, 1.2, dispatched["output"].shape).astype(np.float32)
    combined = rt.launch(_art(rt, "tessera.moe_combine", ["partials", "plan"]), (partials, plan))
    assert_native_gpu(combined)
    np.testing.assert_allclose(combined["output"], moe.combine(partials, plan), rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="requires nvcc and NVIDIA GPU")
@pytest.mark.hardware_nvidia
def test_grouped_gemm_matches_per_expert_oracle():
    from tessera import runtime as rt
    from tessera.compiler.grouped_layout import reference_grouped_gemm

    rng = np.random.default_rng(8)
    sizes = np.array([2, 0, 3, 1], np.int64)
    x = rng.standard_normal((6, 7)).astype(np.float32)
    weights = rng.standard_normal((4, 7, 5)).astype(np.float32)
    result = rt.launch(
        _art(rt, "tessera.grouped_gemm", ["x", "weights", "group_sizes"]),
        {"x": x, "weights": weights, "group_sizes": sizes},
    )
    assert_native_gpu(result)
    np.testing.assert_allclose(
        result["output"], reference_grouped_gemm(x, weights, sizes), rtol=2e-5, atol=2e-5,
    )


@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="requires nvcc and NVIDIA GPU")
@pytest.mark.hardware_nvidia
def test_moe_device_timing_retains_resident_batches():
    from tessera.compiler.emit.nvidia_cuda import measure_moe_dispatch_device

    x = np.arange(17 * 31, dtype=np.float32).reshape(17, 31)
    token_ids = np.arange(23, dtype=np.int32) % 17
    batch_medians = []
    latency = measure_moe_dispatch_device(
        x, token_ids, reps=10, batches=3, batch_medians=batch_medians)
    assert latency > 0
    assert len(batch_medians) == 3
    assert all(sample > 0 for sample in batch_medians)


@pytest.mark.skipif(not nvidia_mma_runtime_available(), reason="requires nvcc and NVIDIA GPU")
@pytest.mark.hardware_nvidia
def test_grouped_swiglu_collapsed_route_matches_legacy_decomposition():
    from tessera.compiler.emit.nvidia_cuda import (
        run_grouped_swiglu_f32, run_grouped_swiglu_legacy_f32,
    )
    rng = np.random.default_rng(20260722)
    groups = np.array([5, 0, 7, 4], np.int64)
    x = (rng.standard_normal((16, 13)) * 0.2).astype(np.float32)
    wg = (rng.standard_normal((4, 13, 11)) * 0.15).astype(np.float32)
    wu = (rng.standard_normal((4, 13, 11)) * 0.15).astype(np.float32)
    wd = (rng.standard_normal((4, 11, 13)) * 0.15).astype(np.float32)
    collapsed = run_grouped_swiglu_f32(x, wg, wu, wd, groups)
    legacy = run_grouped_swiglu_legacy_f32(x, wg, wu, wd, groups)
    np.testing.assert_allclose(collapsed, legacy, rtol=2e-5, atol=2e-5)
