"""Exact-device proof for APPLE-ATTN-FWD-1 resident candidates."""
from __future__ import annotations

import numpy as np
import pytest


def _reference(q, k, v, bias, *, q_heads, kv_heads, causal, window, softcap):
    bq, sq, dim = q.shape
    sk = k.shape[1]
    outer = bq // q_heads
    group = q_heads // kv_heads
    offset = max(sk - sq, 0)
    out = np.empty((bq, sq, dim), dtype=np.float32)
    for batch in range(outer):
        for head in range(q_heads):
            qh = batch * q_heads + head
            kh = batch * kv_heads + head // group
            score = q[qh].astype(np.float32) @ k[kh].astype(np.float32).T
            score *= dim ** -0.5
            score += bias[qh].astype(np.float32)
            if softcap:
                score = softcap * np.tanh(score / softcap)
            qpos = np.arange(sq)[:, None] + offset
            kpos = np.arange(sk)[None, :]
            mask = kpos > qpos if causal else np.zeros((sq, sk), dtype=bool)
            if window:
                mask |= kpos <= qpos - window
            score = np.where(mask, -np.inf, score)
            weight = np.exp(score - score.max(axis=-1, keepdims=True))
            weight /= weight.sum(axis=-1, keepdims=True)
            out[qh] = weight @ v[kh].astype(np.float32)
    return out


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("route", ["resident", "cooperative"])
@pytest.mark.parametrize("dtype", ["f32", "f16", "bf16"])
def test_resident_attention_variants_are_native_timed_and_match_oracle(
        route, dtype):
    from tessera import apple_gpu_batched as agpu
    from tessera._apple_gpu_dispatch import (
        clear_dispatch_telemetry,
        read_dispatch_telemetry,
        set_dispatch_telemetry_enabled,
    )
    from tests._support.apple import require_apple_metal

    require_apple_metal()
    if dtype == "bf16":
        ml_dtypes = pytest.importorskip("ml_dtypes")
        storage = ml_dtypes.bfloat16
        if not agpu.bf16_session_available():
            pytest.skip("native MPSGraph bf16 storage unavailable")
    else:
        storage = np.float32 if dtype == "f32" else np.float16

    rng = np.random.default_rng(2700 + len(dtype) + len(route))
    outer, q_heads, kv_heads, sq, sk, dim = 2, 4, 2, 5, 19, 64
    bq, bkv = outer * q_heads, outer * kv_heads
    q = np.ascontiguousarray((rng.normal(size=(bq, sq, dim)) * .2).astype(storage))
    k = np.ascontiguousarray((rng.normal(size=(bkv, sk, dim)) * .2).astype(storage))
    v = np.ascontiguousarray((rng.normal(size=(bkv, sk, dim)) * .2).astype(storage))
    bias = np.ascontiguousarray(
        (rng.normal(size=(bq, sq, sk)) * .1).astype(storage))
    reference = _reference(
        q, k, v, bias, q_heads=q_heads, kv_heads=kv_heads, causal=True,
        window=9, softcap=3.0)
    qd = agpu.device_tensor(q)
    kd = agpu.device_tensor(k)
    vd = agpu.device_tensor(v)
    bd = agpu.device_tensor(bias)
    out = None
    try:
        assert set_dispatch_telemetry_enabled(True)
        clear_dispatch_telemetry()
        with agpu.batched_session() as session:
            fn = (agpu.flash_attn_variant_enc if route == "resident"
                  else agpu.flash_attn_cooperative_enc)
            out = fn(
                session, qd, kd, vd, bd, dtype=dtype, B=bq,
                q_heads=q_heads, kv_heads=kv_heads, Sq=sq, Sk=sk, D=dim,
                causal=True, window_size=9, logit_softcap=3.0)
        record = read_dispatch_telemetry()
        assert record["device_time_ns"] > 0
        assert record["timing_source"] in {
            "metal_kernel_interval", "metal_command_buffer_interval"}
        assert record["resources"]["thread_execution_width"] > 0
        actual = out.download(np.dtype(storage), q.shape)
        tolerance = 5e-2 if dtype != "f32" else 3e-3
        np.testing.assert_allclose(
            actual.astype(np.float32), reference, rtol=tolerance,
            atol=tolerance)
    finally:
        set_dispatch_telemetry_enabled(False)
        if out is not None:
            out.free()
        bd.free()
        vd.free()
        kd.free()
        qd.free()


def test_cooperative_attention_rejects_unsupported_head_dimension_before_abi():
    from tessera import apple_gpu_batched as agpu

    dummy = agpu.DeviceTensor(handle=1, nbytes=4)
    with pytest.raises(ValueError, match="1 <= D <= 256"):
        agpu.flash_attn_cooperative_enc(
            1, dummy, dummy, dummy, None, dtype="f32", B=4, q_heads=4,
            kv_heads=1, Sq=1, Sk=1, D=257)
