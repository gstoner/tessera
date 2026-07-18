"""Exact-device correctness for the APPLE-ATTN-BWD-1 native candidates."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from tests._support.apple import require_apple_metal


def _reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, do: np.ndarray,
               *, scale: float, causal: bool) -> tuple[np.ndarray, ...]:
    dq = np.zeros_like(q, dtype=np.float32)
    dk = np.zeros_like(k, dtype=np.float32)
    dv = np.zeros_like(v, dtype=np.float32)
    for b in range(q.shape[0]):
        scores = (q[b] @ k[b].T) * scale
        if causal:
            scores = np.where(np.triu(np.ones(scores.shape, dtype=bool), 1),
                              -np.inf, scores)
        prob = np.exp(scores - scores.max(axis=-1, keepdims=True))
        prob /= prob.sum(axis=-1, keepdims=True)
        dp = do[b] @ v[b].T
        ds = prob * (dp - (prob * dp).sum(axis=-1, keepdims=True))
        dq[b] = scale * (ds @ k[b])
        dk[b] = scale * (ds.T @ q[b])
        dv[b] = prob.T @ do[b]
    return dq, dk, dv


def _variant_reference(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, do: np.ndarray,
    *, q_heads: int, kv_heads: int, scale: float, causal: bool,
    bias: np.ndarray | None, window: int, softcap: float,
) -> tuple[np.ndarray, ...]:
    dq = np.zeros_like(q, dtype=np.float32)
    dk = np.zeros_like(k, dtype=np.float32)
    dv = np.zeros_like(v, dtype=np.float32)
    sq, sk = q.shape[1], k.shape[1]
    group = q_heads // kv_heads
    for qb in range(q.shape[0]):
        outer, q_head = divmod(qb, q_heads)
        kvb = outer * kv_heads + q_head // group
        raw = (q[qb] @ k[kvb].T) * scale
        if bias is not None:
            raw = raw + bias[qb]
        keep = np.ones((sq, sk), dtype=bool)
        for qi in range(sq):
            qpos = qi + max(sk - sq, 0)
            if causal:
                keep[qi, qpos + 1:] = False
            if window:
                if causal:
                    keep[qi, :max(qpos - window + 1, 0)] = False
                else:
                    half = window // 2
                    keep[qi, :max(qpos - half, 0)] = False
                    keep[qi, min(qpos + half + 1, sk):] = False
        score = (softcap * np.tanh(raw / softcap)
                 if softcap > 0 else raw)
        score = np.where(keep, score, -np.inf)
        probability = np.exp(score - score.max(axis=-1, keepdims=True))
        probability /= probability.sum(axis=-1, keepdims=True)
        dp = do[qb] @ v[kvb].T
        ds = probability * (
            dp - (probability * dp).sum(axis=-1, keepdims=True))
        if softcap > 0:
            tanh_raw = np.tanh(raw / softcap)
            ds *= 1.0 - tanh_raw * tanh_raw
        dq[qb] = scale * (ds @ k[kvb])
        dk[kvb] += scale * (ds.T @ q[qb])
        dv[kvb] += probability.T @ do[qb]
    return dq, dk, dv


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("route", [0, 1, 2], ids=[
    "serial_recompute", "atomic", "split_reduced"])
@pytest.mark.parametrize("shape,causal", [((1, 3, 5, 8), False),
                                            ((2, 5, 7, 16), True),
                                            ((1, 17, 19, 128), True)])
def test_native_dq_dk_dv_matches_shared_f32_oracle(shape, causal, route):
    from tessera._apple_gpu_dispatch import (
        bind_registered, clear_dispatch_telemetry, read_dispatch_telemetry,
        set_dispatch_telemetry_enabled,
    )

    require_apple_metal()
    b, sq, sk, d = shape
    rng = np.random.default_rng(711 + sq + sk + d + int(causal))
    q = (rng.normal(size=(b, sq, d)) * .2).astype(np.float32)
    k = (rng.normal(size=(b, sk, d)) * .2).astype(np.float32)
    v = (rng.normal(size=(b, sk, d)) * .2).astype(np.float32)
    do = (rng.normal(size=(b, sq, d)) * .2).astype(np.float32)
    expected = _reference(q, k, v, do, scale=d ** -.5, causal=causal)
    fn = bind_registered("tessera_apple_gpu_flash_attn_bwd_route_f32_status")
    assert fn is not None
    pointer = ctypes.POINTER(ctypes.c_float)

    def run() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dq, dk, dv = np.zeros_like(q), np.zeros_like(k), np.zeros_like(v)
        assert fn(q.ctypes.data_as(pointer), k.ctypes.data_as(pointer),
                  v.ctypes.data_as(pointer), do.ctypes.data_as(pointer),
                  dq.ctypes.data_as(pointer), dk.ctypes.data_as(pointer),
                  dv.ctypes.data_as(pointer), b, sq, sk, d, d ** -.5,
                  int(causal), route) == 1
        return dq, dk, dv

    assert set_dispatch_telemetry_enabled(True)
    try:
        clear_dispatch_telemetry()
        got = run()
        telemetry = read_dispatch_telemetry()
        assert telemetry["device_time_ns"] > 0
        assert telemetry["resources"]["thread_execution_width"] > 0
        for actual, reference in zip(got, expected):
            np.testing.assert_allclose(actual, reference, rtol=5e-4, atol=5e-5)
        repeated = run()
        if route != 1:
            for actual, again in zip(got, repeated):
                np.testing.assert_array_equal(actual, again)
        else:
            for actual, reference in zip(repeated, expected):
                np.testing.assert_allclose(
                    actual, reference, rtol=5e-4, atol=5e-5)
    finally:
        set_dispatch_telemetry_enabled(False)


@pytest.mark.hardware_apple_gpu
def test_unknown_native_route_fails_without_device_interval():
    from tessera._apple_gpu_dispatch import (
        bind_registered, clear_dispatch_telemetry, read_dispatch_telemetry,
        set_dispatch_telemetry_enabled,
    )

    require_apple_metal()
    fn = bind_registered("tessera_apple_gpu_flash_attn_bwd_route_f32_status")
    assert fn is not None
    arrays = [np.zeros((1, 1, 4), dtype=np.float32) for _ in range(7)]
    pointer = ctypes.POINTER(ctypes.c_float)
    assert set_dispatch_telemetry_enabled(True)
    try:
        clear_dispatch_telemetry()
        status = fn(*(array.ctypes.data_as(pointer) for array in arrays),
                    1, 1, 1, 4, .5, 0, 3)
        assert status == 0
        assert read_dispatch_telemetry()["device_time_ns"] is None
    finally:
        set_dispatch_telemetry_enabled(False)


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("route", [0, 1, 2], ids=[
    "serial_recompute", "atomic", "split_reduced"])
@pytest.mark.parametrize(
    "outer,q_heads,kv_heads,sq,sk,dim,causal,window,softcap,with_bias",
    [
        (1, 4, 2, 5, 7, 8, True, 4, 2.5, True),
        (1, 4, 1, 4, 9, 16, False, 5, 0.0, False),
        (2, 2, 2, 3, 7, 8, True, 0, 1.5, True),
    ],
    ids=["gqa_bias_causal_window_softcap", "mqa_window", "mha_right_aligned"],
)
def test_native_variant_backward_matches_shared_oracle(
    route, outer, q_heads, kv_heads, sq, sk, dim, causal, window, softcap,
    with_bias,
):
    from tessera._apple_gpu_dispatch import bind_registered

    require_apple_metal()
    rng = np.random.default_rng(
        1801 + route + q_heads + kv_heads + sq + sk + dim)
    b = outer * q_heads
    kv_outer = outer * kv_heads
    q = (rng.normal(size=(b, sq, dim)) * .2).astype(np.float32)
    k = (rng.normal(size=(kv_outer, sk, dim)) * .2).astype(np.float32)
    v = (rng.normal(size=(kv_outer, sk, dim)) * .2).astype(np.float32)
    do = (rng.normal(size=(b, sq, dim)) * .2).astype(np.float32)
    bias = ((rng.normal(size=(b, sq, sk)) * .05).astype(np.float32)
            if with_bias else None)
    scale = dim ** -.5
    expected = _variant_reference(
        q, k, v, do, q_heads=q_heads, kv_heads=kv_heads, scale=scale,
        causal=causal, bias=bias, window=window, softcap=softcap)
    fn = bind_registered(
        "tessera_apple_gpu_flash_attn_bwd_variant_f32_status")
    assert fn is not None
    pointer = ctypes.POINTER(ctypes.c_float)
    dq, dk, dv = np.empty_like(q), np.empty_like(k), np.empty_like(v)
    bias_pointer = None if bias is None else bias.ctypes.data_as(pointer)
    status = fn(
        q.ctypes.data_as(pointer), k.ctypes.data_as(pointer),
        v.ctypes.data_as(pointer), do.ctypes.data_as(pointer), bias_pointer,
        dq.ctypes.data_as(pointer), dk.ctypes.data_as(pointer),
        dv.ctypes.data_as(pointer), b, q_heads, kv_heads, sq, sk, dim, scale,
        int(causal), window, softcap, route)
    assert status == 1
    for actual, reference in zip((dq, dk, dv), expected):
        np.testing.assert_allclose(actual, reference, rtol=8e-4, atol=8e-5)


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("route", [0, 1, 2], ids=[
    "serial_recompute", "atomic", "split_reduced"])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_native_two_byte_storage_backward_keeps_f32_accumulation(dtype, route):
    from tessera._apple_gpu_dispatch import bind_registered

    require_apple_metal()
    ml_dtypes = pytest.importorskip("ml_dtypes")
    rng = np.random.default_rng(2603 + route + int(dtype == "bf16"))
    q_heads, kv_heads, sq, sk, dim = 4, 2, 5, 9, 16
    q0 = (rng.normal(size=(q_heads, sq, dim)) * .2).astype(np.float32)
    k0 = (rng.normal(size=(kv_heads, sk, dim)) * .2).astype(np.float32)
    v0 = (rng.normal(size=(kv_heads, sk, dim)) * .2).astype(np.float32)
    do0 = (rng.normal(size=(q_heads, sq, dim)) * .2).astype(np.float32)
    bias0 = (rng.normal(size=(q_heads, sq, sk)) * .05).astype(np.float32)
    storage_type = np.float16 if dtype == "f16" else ml_dtypes.bfloat16

    def store(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        typed = np.ascontiguousarray(value.astype(storage_type))
        return typed.view(np.uint16), typed.astype(np.float32)

    q, qf = store(q0)
    k, kf = store(k0)
    v, vf = store(v0)
    do, dof = store(do0)
    bias, biasf = store(bias0)
    scale = dim ** -.5
    expected = _variant_reference(
        qf, kf, vf, dof, q_heads=q_heads, kv_heads=kv_heads, scale=scale,
        causal=True, bias=biasf, window=5, softcap=2.0)
    fn = bind_registered(
        f"tessera_apple_gpu_flash_attn_bwd_variant_{dtype}_status")
    assert fn is not None
    input_pointer = ctypes.POINTER(ctypes.c_uint16)
    output_pointer = ctypes.POINTER(ctypes.c_float)
    dq = np.empty_like(qf)
    dk = np.empty_like(kf)
    dv = np.empty_like(vf)
    status = fn(
        q.ctypes.data_as(input_pointer), k.ctypes.data_as(input_pointer),
        v.ctypes.data_as(input_pointer), do.ctypes.data_as(input_pointer),
        bias.ctypes.data_as(input_pointer), dq.ctypes.data_as(output_pointer),
        dk.ctypes.data_as(output_pointer), dv.ctypes.data_as(output_pointer),
        q_heads, q_heads, kv_heads, sq, sk, dim, scale, 1, 5, 2.0, route)
    assert status == 1
    for actual, reference in zip((dq, dk, dv), expected):
        np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-4)
