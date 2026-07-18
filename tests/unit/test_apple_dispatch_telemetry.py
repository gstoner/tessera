"""APPLE-GEMM-1 capture-gated dispatch telemetry proof."""
from __future__ import annotations

import ctypes

import pytest


@pytest.mark.hardware_apple_gpu
def test_capture_gate_records_owned_mps_and_mtl4_command_buffers():
    import numpy as np
    from tessera import runtime as rt

    from tessera._apple_gpu_dispatch import (
        bind_registered,
        clear_dispatch_telemetry,
        read_dispatch_telemetry,
        read_profiling_capabilities,
        set_dispatch_telemetry_enabled,
    )
    from tests._support.apple import require_apple_metal

    require_apple_metal()
    a = np.ones((64, 64), dtype=np.float16)
    u16 = ctypes.POINTER(ctypes.c_uint16)
    fp32 = ctypes.POINTER(ctypes.c_float)
    mps_out = np.empty((64, 64), dtype=np.uint16)
    mtl4_out = np.empty((64, 64), dtype=np.float32)
    mps = bind_registered("tessera_apple_gpu_mps_matmul_f16_status")
    mtl4 = bind_registered("tessera_apple_gpu_mtl4_matmul2d_f16")
    assert mps is not None and mtl4 is not None

    try:
        assert set_dispatch_telemetry_enabled(False)
        clear_dispatch_telemetry()
        assert mps(
            a.view(np.uint16).ctypes.data_as(u16),
            a.view(np.uint16).ctypes.data_as(u16),
            mps_out.ctypes.data_as(u16), 64, 64, 64) == 1
        disabled = read_dispatch_telemetry()
        assert disabled["capture_enabled"] is False
        assert disabled["device_time_ns"] is None

        assert set_dispatch_telemetry_enabled(True)
        clear_dispatch_telemetry()
        assert mps(
            a.view(np.uint16).ctypes.data_as(u16),
            a.view(np.uint16).ctypes.data_as(u16),
            mps_out.ctypes.data_as(u16), 64, 64, 64) == 1
        mps_record = read_dispatch_telemetry()
        assert mps_record["device_time_ns"] > 0
        assert mps_record["timing_source"] in {
            "metal_kernel_interval", "metal_command_buffer_interval"}
        assert mps_record["counter_timestamp_delta"] is None

        clear_dispatch_telemetry()
        x = np.ones((64, 64), dtype=np.float32)
        softmax = rt._apple_gpu_dispatch_mpsgraph_softmax(x, np)
        mpsgraph_record = read_dispatch_telemetry()
        assert mpsgraph_record["device_time_ns"] > 0
        assert mpsgraph_record["timing_source"] in {
            "metal_kernel_interval", "metal_command_buffer_interval"}
        np.testing.assert_allclose(softmax, 1.0 / 64.0, rtol=1e-5, atol=1e-6)

        clear_dispatch_telemetry()
        msl_softmax = rt._apple_gpu_dispatch_softmax(
            "tessera.softmax", [x], {}, np)
        msl_record = read_dispatch_telemetry()
        assert msl_record["device_time_ns"] > 0
        assert msl_record["resources"]["thread_execution_width"] > 0
        assert msl_record["resources"]["simdgroups_per_threadgroup"] > 0
        assert msl_record["resources"]["occupancy"] is None
        assert msl_record["resources"]["spill_count"] is None
        np.testing.assert_allclose(msl_softmax, 1.0 / 64.0, rtol=1e-5, atol=1e-6)

        profiling = read_profiling_capabilities()
        assert profiling["capabilities"]["pipeline_limits"] is True
        assert profiling["capabilities"]["occupancy"] is False
        assert profiling["capabilities"]["spill_count"] is False

        clear_dispatch_telemetry()
        promoted_x = np.ones((128, 257), dtype=np.float32)
        promoted = rt._apple_gpu_dispatch_softmax(
            "tessera.softmax", [promoted_x], {}, np)
        promoted_record = read_dispatch_telemetry()
        assert promoted_record["device_time_ns"] > 0
        # Apple7's retained end-to-end winner is MPSGraph, whose framework
        # pipeline does not expose the MSL pipeline-limit record.
        assert promoted_record["resources"] is None
        np.testing.assert_allclose(promoted, 1.0 / 257.0, rtol=1e-5, atol=1e-6)

        clear_dispatch_telemetry()
        assert mtl4(
            a.view(np.uint16).ctypes.data_as(u16),
            a.view(np.uint16).ctypes.data_as(u16),
            mtl4_out.ctypes.data_as(fp32), 64, 64, 64) == 1
        mtl4_record = read_dispatch_telemetry()
        assert mtl4_record["device_time_ns"] > 0
        assert mtl4_record["timing_source"] == "metal4_timestamp_heap"
        assert mtl4_record["counter_sampling_supported"] is True
        assert mtl4_record["counter_timestamp_delta"] > 0
        assert mtl4_record["resources"]["threadgroup"] == [128, 1, 1]
        assert mtl4_record["resources"]["thread_execution_width"] > 0
        assert mtl4_record["resources"]["max_total_threads_per_threadgroup"] >= 128
        assert mtl4_record["resources"]["static_threadgroup_memory_bytes"] >= 0
        np.testing.assert_allclose(mtl4_out, 64.0, rtol=0, atol=0)

        # The heap remains cached, but disabling capture must stop timestamp
        # writes and clear the record so production dispatch pays no sampling cost.
        assert set_dispatch_telemetry_enabled(False)
        assert mtl4(
            a.view(np.uint16).ctypes.data_as(u16),
            a.view(np.uint16).ctypes.data_as(u16),
            mtl4_out.ctypes.data_as(fp32), 64, 64, 64) == 1
        disabled_after_capture = read_dispatch_telemetry()
        assert disabled_after_capture["capture_enabled"] is False
        assert disabled_after_capture["device_time_ns"] is None
        assert disabled_after_capture["counter_timestamp_delta"] is None
    finally:
        set_dispatch_telemetry_enabled(False)


@pytest.mark.hardware_apple_gpu
def test_epilogue_routes_retain_live_pipeline_and_dynamic_memory_evidence():
    import numpy as np
    from tessera._apple_gpu_dispatch import (
        clear_dispatch_telemetry,
        read_dispatch_telemetry,
        set_dispatch_telemetry_enabled,
    )
    from tessera.compiler.fusion import FusedRegion, run_fused_region
    from tests._support.apple import require_apple_metal

    require_apple_metal()
    rng = np.random.default_rng(617)
    try:
        assert set_dispatch_telemetry_enabled(True)

        # Pointwise epilogues select the cooperative-matrix route.
        a = rng.standard_normal((64, 64)).astype(np.float16)
        b = rng.standard_normal((64, 64)).astype(np.float16)
        clear_dispatch_telemetry()
        out, execution = run_fused_region(FusedRegion(("bias", "silu")), a, b,
                                          np.zeros(64, dtype=np.float16))
        coopmat = read_dispatch_telemetry()
        assert execution == "metal_runtime"
        assert coopmat["device_time_ns"] > 0
        assert coopmat["resources"]["threadgroup"] in ([128, 1, 1], [256, 1, 1])
        assert coopmat["resources"]["thread_execution_width"] > 0
        np.testing.assert_allclose(
            out.astype(np.float32),
            FusedRegion(("bias", "silu")).reference(
                a, b, np.zeros(64, dtype=np.float16)),
            rtol=3e-2, atol=3e-2,
        )

        # A ragged wide reduction selects the tiled scalar route and must
        # retain its N*f32 dynamic threadgroup scratch, not merely static PSO
        # memory.
        wide_b = rng.standard_normal((64, 2049)).astype(np.float32)
        clear_dispatch_telemetry()
        wide_out, execution = run_fused_region(
            FusedRegion((), reduction="softmax"), a.astype(np.float32), wide_b)
        tiled = read_dispatch_telemetry()
        assert execution == "metal_runtime"
        assert tiled["device_time_ns"] > 0
        assert tiled["resources"]["threadgroup"] == [32, 1, 1]
        assert tiled["resources"]["threadgroup_memory_bytes"] >= 2049 * 4
        np.testing.assert_allclose(
            wide_out,
            FusedRegion((), reduction="softmax").reference(
                a.astype(np.float32), wide_b),
            rtol=1e-4, atol=1e-4,
        )
    finally:
        set_dispatch_telemetry_enabled(False)


@pytest.mark.hardware_apple_gpu
def test_flash_attention_status_proves_native_resources_and_envelope_negative():
    import numpy as np
    from tessera._apple_gpu_dispatch import (
        bind_registered,
        clear_dispatch_telemetry,
        read_dispatch_telemetry,
        set_dispatch_telemetry_enabled,
    )
    from tests._support.apple import require_apple_metal

    require_apple_metal()
    rng = np.random.default_rng(1807)

    def reference(q, k, v, *, causal):
        scores = np.einsum("bqd,bkd->bqk", q.astype(np.float32),
                           k.astype(np.float32)) * (q.shape[-1] ** -0.5)
        if causal:
            mask = np.arange(q.shape[1])[:, None] < np.arange(k.shape[1])[None, :]
            scores = np.where(mask[None, :, :], -np.inf, scores)
        scores -= np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        return np.einsum("bqk,bkd->bqd", weights, v.astype(np.float32))

    try:
        assert set_dispatch_telemetry_enabled(True)
        for dtype, symbol_name in (
                (np.float32, "tessera_apple_gpu_flash_attn_f32_status"),
                (np.float16, "tessera_apple_gpu_flash_attn_f16_status")):
            bsz, sq, sk, dim = 2, 17, 19, 128
            q = np.ascontiguousarray(rng.standard_normal((bsz, sq, dim)).astype(dtype))
            k = np.ascontiguousarray(rng.standard_normal((bsz, sk, dim)).astype(dtype))
            v = np.ascontiguousarray(rng.standard_normal((bsz, sk, dim)).astype(dtype))
            out = np.empty((bsz, sq, dim), dtype=dtype)
            symbol = bind_registered(symbol_name)
            assert symbol is not None
            ptr = (ctypes.POINTER(ctypes.c_float) if dtype == np.float32
                   else ctypes.POINTER(ctypes.c_uint16))
            view = (lambda array: array if dtype == np.float32
                    else array.view(np.uint16))
            clear_dispatch_telemetry()
            assert symbol(
                view(q).ctypes.data_as(ptr), view(k).ctypes.data_as(ptr),
                view(v).ctypes.data_as(ptr), view(out).ctypes.data_as(ptr),
                bsz, sq, sk, dim, dim ** -0.5, 1) == 1
            record = read_dispatch_telemetry()
            assert record["device_time_ns"] > 0
            assert record["resources"]["threadgroup"] == [17, 2, 1]
            assert record["resources"]["thread_execution_width"] > 0
            np.testing.assert_allclose(
                out.astype(np.float32), reference(q, k, v, causal=True),
                rtol=4e-2 if dtype == np.float16 else 2e-3,
                atol=4e-2 if dtype == np.float16 else 2e-3)

        # The public fallback-capable ABI computes a reference for D > 256;
        # the status ABI must reject that envelope instead of labeling it GPU.
        too_wide = np.zeros((1, 2, 257), dtype=np.float32)
        wide_kv = np.zeros((1, 3, 257), dtype=np.float32)
        wide_out = np.empty_like(too_wide)
        f32 = bind_registered("tessera_apple_gpu_flash_attn_f32_status")
        fp = ctypes.POINTER(ctypes.c_float)
        clear_dispatch_telemetry()
        assert f32(
            too_wide.ctypes.data_as(fp), wide_kv.ctypes.data_as(fp),
            wide_kv.ctypes.data_as(fp), wide_out.ctypes.data_as(fp),
            1, 2, 3, 257, 257 ** -0.5, 0) == 0
        assert read_dispatch_telemetry()["device_time_ns"] is None
    finally:
        set_dispatch_telemetry_enabled(False)


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("dtype,suffix", [("f32", "f32"), ("f16", "f16")])
@pytest.mark.parametrize("q_heads,kv_heads", [(4, 4), (4, 2), (4, 1)])
def test_attention_variant_status_covers_mha_gqa_mqa_bias_window_softcap(
        dtype, suffix, q_heads, kv_heads):
    import numpy as np
    from tessera._apple_gpu_dispatch import (
        bind_registered,
        clear_dispatch_telemetry,
        read_dispatch_telemetry,
        set_dispatch_telemetry_enabled,
    )
    from tests._support.apple import require_apple_metal

    require_apple_metal()
    storage = np.float32 if dtype == "f32" else np.float16
    rng = np.random.default_rng(2300 + kv_heads)
    outer, sq, sk, dim = 1, 5, 37, 64
    bq = outer * q_heads
    bkv = outer * kv_heads
    q = np.ascontiguousarray(rng.normal(size=(bq, sq, dim)).astype(storage))
    k = np.ascontiguousarray(rng.normal(size=(bkv, sk, dim)).astype(storage))
    v = np.ascontiguousarray(rng.normal(size=(bkv, sk, dim)).astype(storage))
    bias = np.ascontiguousarray(
        rng.normal(scale=0.2, size=(bq, sq, sk)).astype(storage))
    out = np.empty_like(q)
    scale, window, softcap = dim ** -0.5, 11, 3.0

    ref = np.empty_like(q, dtype=np.float32)
    group = q_heads // kv_heads
    offset = sk - sq
    for h in range(q_heads):
        kvh = h // group
        scores = np.einsum(
            "qd,kd->qk", q[h].astype(np.float32),
            k[kvh].astype(np.float32)) * scale
        scores += bias[h].astype(np.float32)
        scores = softcap * np.tanh(scores / softcap)
        qpos = np.arange(sq)[:, None] + offset
        kpos = np.arange(sk)[None, :]
        mask = (kpos > qpos) | (kpos <= qpos - window)
        scores = np.where(mask, -np.inf, scores)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        ref[h] = weights @ v[kvh].astype(np.float32)

    symbol = bind_registered(
        f"tessera_apple_gpu_flash_attn_variant_{suffix}_status")
    assert symbol is not None
    ptr = (ctypes.POINTER(ctypes.c_float) if dtype == "f32"
           else ctypes.POINTER(ctypes.c_uint16))
    view = (lambda a: a if dtype == "f32" else a.view(np.uint16))
    try:
        assert set_dispatch_telemetry_enabled(True)
        clear_dispatch_telemetry()
        assert symbol(
            view(q).ctypes.data_as(ptr), view(k).ctypes.data_as(ptr),
            view(v).ctypes.data_as(ptr), view(bias).ctypes.data_as(ptr),
            view(out).ctypes.data_as(ptr), bq, q_heads, kv_heads, sq, sk, dim,
            scale, 1, window, softcap) == 1
        record = read_dispatch_telemetry()
        assert record["device_time_ns"] > 0
        assert record["resources"]["threadgroup"] == [sq, bq, 1]
        assert record["resources"]["thread_execution_width"] > 0
        np.testing.assert_allclose(
            out.astype(np.float32), ref,
            rtol=4e-2 if dtype == "f16" else 2e-3,
            atol=4e-2 if dtype == "f16" else 2e-3)
    finally:
        set_dispatch_telemetry_enabled(False)


@pytest.mark.hardware_apple_gpu
def test_attention_variant_long_context_and_explicit_fallback_negatives():
    import numpy as np
    from tessera._apple_gpu_dispatch import (
        bind_registered,
        clear_dispatch_telemetry,
        read_dispatch_telemetry,
        set_dispatch_telemetry_enabled,
    )
    from tests._support.apple import require_apple_metal

    require_apple_metal()
    rng = np.random.default_rng(2401)
    q_heads, kv_heads, sq, sk, dim = 4, 1, 3, 1025, 64
    q = np.ascontiguousarray(rng.normal(size=(q_heads, sq, dim)).astype(np.float32))
    k = np.ascontiguousarray(rng.normal(size=(kv_heads, sk, dim)).astype(np.float32))
    v = np.ascontiguousarray(rng.normal(size=(kv_heads, sk, dim)).astype(np.float32))
    out = np.empty_like(q)
    fp = ctypes.POINTER(ctypes.c_float)
    symbol = bind_registered("tessera_apple_gpu_flash_attn_variant_f32_status")
    assert symbol is not None
    try:
        assert set_dispatch_telemetry_enabled(True)
        clear_dispatch_telemetry()
        assert symbol(
            q.ctypes.data_as(fp), k.ctypes.data_as(fp), v.ctypes.data_as(fp),
            None, out.ctypes.data_as(fp), q_heads, q_heads, kv_heads, sq, sk,
            dim, dim ** -0.5, 1, 128, 0.0) == 1
        assert read_dispatch_telemetry()["device_time_ns"] > 0
        assert np.isfinite(out).all()

        # Invalid GQA grouping, head dimension, and window must fail before a
        # command buffer is submitted; none may inherit the prior GPU interval.
        for bad_qh, bad_kvh, bad_dim, bad_window in (
                (3, 2, dim, 128),
                (q_heads, kv_heads, 257, 128),
                (q_heads, kv_heads, dim, -1)):
            clear_dispatch_telemetry()
            assert symbol(
                q.ctypes.data_as(fp), k.ctypes.data_as(fp), v.ctypes.data_as(fp),
                None, out.ctypes.data_as(fp), bad_qh, bad_qh, bad_kvh, sq, sk,
                bad_dim, dim ** -0.5, 1, bad_window, 0.0) == 0
            assert read_dispatch_telemetry()["device_time_ns"] is None
    finally:
        set_dispatch_telemetry_enabled(False)
