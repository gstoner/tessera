"""Apple GPU fused batched matmul->softmax->matmul (2026-05-29).

`O = softmax((Q @ Kᵀ) * scale) @ V` per batch in a single dispatch — the
batched attention block fused into one MPSGraph (vs the bmm + softmax + bmm
3-call compose used by test_apple_gpu_batched_mha.py). Validated vs numpy.
See docs/apple_gpu_tier2_tier3_plan.md.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tessera import runtime as R


def _ref_attn(Q, K, V, scale):
    Q = Q.astype(np.float64)
    K = K.astype(np.float64)
    V = V.astype(np.float64)
    s = (Q @ np.swapaxes(K, -1, -2)) * scale
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return (e / e.sum(-1, keepdims=True)) @ V


@pytest.mark.parametrize("batch,T,D", [(6, 8, 16), (1, 12, 32), (4, 6, 8)])
def test_fused_attention_matches_numpy(batch, T, D):
    rng = np.random.RandomState(0)
    Q = (rng.randn(batch, T, D) * 0.5).astype(np.float32)
    K = (rng.randn(batch, T, D) * 0.5).astype(np.float32)
    V = (rng.randn(batch, T, D) * 0.5).astype(np.float32)
    scale = 1.0 / math.sqrt(D)
    out = R._apple_gpu_dispatch_batched_attention(Q, K, V, np, scale=scale)
    assert out is not None and out.shape == (batch, T, D)
    np.testing.assert_allclose(out, _ref_attn(Q, K, V, scale), rtol=1e-4, atol=1e-4)


def test_fused_attention_default_scale_and_rank4():
    # Rank-4 [B, H, T, D] folds to batch = B*H.
    rng = np.random.RandomState(1)
    B, H, T, D = 2, 4, 6, 16
    Q = (rng.randn(B, H, T, D) * 0.5).astype(np.float32)
    K = (rng.randn(B, H, T, D) * 0.5).astype(np.float32)
    V = (rng.randn(B, H, T, D) * 0.5).astype(np.float32)
    out = R._apple_gpu_dispatch_batched_attention(Q, K, V, np)  # default 1/sqrt(D)
    assert out.shape == (B, H, T, D)
    np.testing.assert_allclose(out, _ref_attn(Q, K, V, 1.0 / math.sqrt(D)),
                               rtol=1e-4, atol=1e-4)


def test_fused_attention_non_square_seq():
    # Sq != Sk (cross-attention).
    rng = np.random.RandomState(2)
    batch, Tq, Tk, D = 3, 5, 9, 16
    Q = (rng.randn(batch, Tq, D) * 0.5).astype(np.float32)
    K = (rng.randn(batch, Tk, D) * 0.5).astype(np.float32)
    V = (rng.randn(batch, Tk, D) * 0.5).astype(np.float32)
    scale = 1.0 / math.sqrt(D)
    out = R._apple_gpu_dispatch_batched_attention(Q, K, V, np, scale=scale)
    assert out.shape == (batch, Tq, D)
    np.testing.assert_allclose(out, _ref_attn(Q, K, V, scale), rtol=1e-4, atol=1e-4)


def test_fused_attention_f16():
    rng = np.random.RandomState(3)
    batch, T, D = 4, 8, 16
    Q = (rng.randn(batch, T, D) * 0.5).astype(np.float16)
    K = (rng.randn(batch, T, D) * 0.5).astype(np.float16)
    V = (rng.randn(batch, T, D) * 0.5).astype(np.float16)
    out = R._apple_gpu_dispatch_batched_attention(Q, K, V, np)
    assert out.dtype == np.float16
    np.testing.assert_allclose(out.astype(np.float32),
                               _ref_attn(Q, K, V, 1.0 / math.sqrt(D)),
                               rtol=5e-2, atol=5e-2)


def test_fused_attention_symbols_exported():
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_mpsgraph_bsmm_f32")
    assert hasattr(rt, "tessera_apple_gpu_mpsgraph_bsmm_f16")
    assert R._apple_gpu_bsmm_f32() is not None
    assert R._apple_gpu_bsmm_f16() is not None


def test_fused_attention_matches_bmm_compose():
    # The fused kernel and the bmm+softmax+bmm compose must agree.
    rng = np.random.RandomState(4)
    batch, T, D = 4, 10, 16
    Q = (rng.randn(batch, T, D) * 0.5).astype(np.float32)
    K = (rng.randn(batch, T, D) * 0.5).astype(np.float32)
    V = (rng.randn(batch, T, D) * 0.5).astype(np.float32)
    scale = 1.0 / math.sqrt(D)
    fused = R._apple_gpu_dispatch_batched_attention(Q, K, V, np, scale=scale)
    # compose path
    kt = np.ascontiguousarray(K.transpose(0, 2, 1))
    scores = np.asarray(R._apple_gpu_dispatch_bmm((Q * np.float32(scale)), kt, np))
    attn = np.asarray(R._apple_gpu_dispatch_mpsgraph_softmax(scores, np))
    ctx = np.asarray(R._apple_gpu_dispatch_bmm(attn, V, np))
    np.testing.assert_allclose(fused, ctx, rtol=1e-3, atol=1e-3)
