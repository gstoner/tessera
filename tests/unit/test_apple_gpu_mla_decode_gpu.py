"""Apple GPU MLA decode — GPU promotion (2026-05-30).

`tessera_apple_gpu_mla_decode_f32` was promoted from a host reference to a real
on-GPU path: one cached MPSGraph fuses the latent down-projection
`c = X @ Wdkv`, the K/V up-projections `K = c @ Wuk` / `V = c @ Wuv`, and the
attention `O = softmax((Q @ Kᵀ)·scale) @ V` (B·S_q query rows folded to a
single matmul dim; K/V shared across batch). This test exercises the batched
(B>1) broadcast path and the graph-cache reuse, validating against a numpy
reference. See docs/audit/backend/apple/archive/apple_gpu_tier2_tier3_plan.md.
"""

from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

from tessera import runtime as R


def _ref_mla_decode(X, Wdkv, Wuk, Wuv, Q):
    X = X.astype(np.float64)
    c = X @ Wdkv.astype(np.float64)
    K = c @ Wuk.astype(np.float64)          # [S_kv, D_h]
    V = c @ Wuv.astype(np.float64)          # [S_kv, D_h]
    Q = Q.astype(np.float64)                # [B, S_q, D_h]
    B, S_q, D_h = Q.shape
    scale = 1.0 / math.sqrt(D_h)
    O = np.empty_like(Q)
    for b in range(B):
        s = (Q[b] @ K.T) * scale            # [S_q, S_kv]
        s = s - s.max(-1, keepdims=True)
        e = np.exp(s)
        O[b] = (e / e.sum(-1, keepdims=True)) @ V
    return O


def _mla_sym():
    rt = R._load_apple_gpu_runtime()
    sym = getattr(rt, "tessera_apple_gpu_mla_decode_f32", None)
    if sym is None:
        return None
    sym.argtypes = [ctypes.POINTER(ctypes.c_float)] * 6 + [ctypes.c_int32] * 6
    sym.restype = None
    return sym


def _call(sym, X, Wdkv, Wuk, Wuv, Q):
    B, S_q, D_h = Q.shape
    S_kv, D_x = X.shape
    D_lat = Wdkv.shape[1]
    fp = lambda a: np.ascontiguousarray(a, np.float32).ctypes.data_as(
        ctypes.POINTER(ctypes.c_float))
    O = np.zeros((B, S_q, D_h), np.float32)
    sym(fp(X), fp(Wdkv), fp(Wuk), fp(Wuv), fp(Q),
        O.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B, S_kv, D_x, D_lat, S_q, D_h)
    return O


_SHAPES = [
    pytest.param(1, 3, 8, 16, 4, 8, id="B1"),
    pytest.param(4, 5, 16, 32, 12, 16, id="B4_broadcast_kv"),
    pytest.param(2, 1, 12, 24, 32, 16, id="decode_step_Sq1"),
    pytest.param(3, 8, 32, 64, 20, 32, id="larger"),
]


@pytest.mark.parametrize("B,S_q,D_x,D_lat,S_kv,D_h", _SHAPES)
def test_mla_decode_matches_numpy(B, S_q, D_x, D_lat, S_kv, D_h):
    sym = _mla_sym()
    assert sym is not None
    rng = np.random.RandomState(0)
    X = (rng.randn(S_kv, D_x) * 0.3).astype(np.float32)
    Wdkv = (rng.randn(D_x, D_lat) * 0.3).astype(np.float32)
    Wuk = (rng.randn(D_lat, D_h) * 0.3).astype(np.float32)
    Wuv = (rng.randn(D_lat, D_h) * 0.3).astype(np.float32)
    Q = (rng.randn(B, S_q, D_h) * 0.3).astype(np.float32)
    out = _call(sym, X, Wdkv, Wuk, Wuv, Q)
    ref = _ref_mla_decode(X, Wdkv, Wuk, Wuv, Q)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_mla_decode_kv_shared_across_batch():
    """Each batch sees the same K/V (only Q varies) — distinct Q rows give
    distinct outputs, proving the fold-to-rows path keeps batches independent."""
    sym = _mla_sym()
    assert sym is not None
    rng = np.random.RandomState(1)
    S_kv, D_x, D_lat, D_h = 6, 12, 20, 16
    X = (rng.randn(S_kv, D_x) * 0.3).astype(np.float32)
    Wdkv = (rng.randn(D_x, D_lat) * 0.3).astype(np.float32)
    Wuk = (rng.randn(D_lat, D_h) * 0.3).astype(np.float32)
    Wuv = (rng.randn(D_lat, D_h) * 0.3).astype(np.float32)
    Q = (rng.randn(2, 2, D_h) * 0.3).astype(np.float32)
    out = _call(sym, X, Wdkv, Wuk, Wuv, Q)
    ref = _ref_mla_decode(X, Wdkv, Wuk, Wuv, Q)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
    # batch 0 and batch 1 use different queries -> different outputs
    assert not np.allclose(out[0], out[1])


def test_mla_decode_graph_cached():
    """Repeating the same shape must not grow the MPSGraph cache (graph reuse)."""
    rt = R._load_apple_gpu_runtime()
    cache_size = getattr(rt, "tessera_apple_gpu_mpsgraph_cache_size", None)
    if cache_size is None:
        pytest.skip("cache-size introspection symbol unavailable")
    cache_size.restype = ctypes.c_int32
    sym = _mla_sym()
    assert sym is not None
    rng = np.random.RandomState(2)
    S_kv, D_x, D_lat, D_h = 5, 8, 16, 8
    args = [
        (rng.randn(S_kv, D_x) * 0.3).astype(np.float32),
        (rng.randn(D_x, D_lat) * 0.3).astype(np.float32),
        (rng.randn(D_lat, D_h) * 0.3).astype(np.float32),
        (rng.randn(D_lat, D_h) * 0.3).astype(np.float32),
        (rng.randn(1, 3, D_h) * 0.3).astype(np.float32),
    ]
    _call(sym, *args)
    after_first = cache_size()
    for _ in range(3):
        _call(sym, *args)
    assert cache_size() == after_first  # no new graphs for the same signature
