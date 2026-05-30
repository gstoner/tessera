"""Apple GPU native GQA / MQA KV-group indexing (2026-05-29).

`flash_attn_gqa_f32` lets query head `h` read KV group `h // (H/G)` directly,
so grouped/multi-query attention runs **without materializing the repeated KV**
(the Phase-2 bandwidth win). Validated against the repeat-KV + standard
attention reference. See docs/apple_gpu_tier2_tier3_plan.md.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from tessera import runtime as R

DARWIN = sys.platform == "darwin"


def _ref_gqa(Q, K, V, num_q_heads, num_kv_heads, scale, causal):
    """[B, H, S, D] Q; [B, G, S, D] K/V -> repeat KV G->H, then per-head SDPA."""
    Q = Q.astype(np.float64)
    B, H, Sq, D = Q.shape
    G = K.shape[1]
    Kr = np.repeat(K.astype(np.float64), H // G, axis=1)
    Vr = np.repeat(V.astype(np.float64), H // G, axis=1)
    out = np.empty_like(Q)
    for b in range(B):
        for h in range(H):
            s = (Q[b, h] @ Kr[b, h].T) * scale
            if causal:
                Sk = Kr.shape[2]
                s = np.where(np.triu(np.ones((Sq, Sk)), 1).astype(bool), -1e30, s)
            s = s - s.max(-1, keepdims=True)
            e = np.exp(s)
            out[b, h] = (e / e.sum(-1, keepdims=True)) @ Vr[b, h]
    return out


_CASES = [
    pytest.param(2, 8, 2, 6, 16, False, id="GQA_B2_H8_G2"),
    pytest.param(1, 8, 1, 5, 16, False, id="MQA_B1_H8_G1"),
    pytest.param(2, 4, 4, 7, 8, True, id="MHA_eq_H4_G4_causal"),
    pytest.param(1, 6, 3, 4, 32, True, id="GQA_H6_G3_causal"),
]


@pytest.mark.parametrize("B,H,G,S,D,causal", _CASES)
def test_gqa_matches_repeat_kv(B, H, G, S, D, causal):
    rng = np.random.RandomState(0)
    Q = rng.randn(B, H, S, D).astype(np.float32)
    K = rng.randn(B, G, S, D).astype(np.float32)
    V = rng.randn(B, G, S, D).astype(np.float32)
    scale = 1.0 / math.sqrt(D)
    out = R._apple_gpu_dispatch_gqa(Q, K, V, H, G, np, scale=scale, causal=causal)
    assert out is not None and out.shape == (B, H, S, D)
    ref = _ref_gqa(Q, K, V, H, G, scale, causal)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_gqa_default_scale():
    rng = np.random.RandomState(1)
    B, H, G, S, D = 2, 8, 2, 6, 16
    Q = rng.randn(B, H, S, D).astype(np.float32)
    K = rng.randn(B, G, S, D).astype(np.float32)
    V = rng.randn(B, G, S, D).astype(np.float32)
    out = R._apple_gpu_dispatch_gqa(Q, K, V, H, G, np)  # scale defaults to 1/sqrt(D)
    ref = _ref_gqa(Q, K, V, H, G, 1.0 / math.sqrt(D), False)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_gqa_kv_not_repeated_in_input():
    # The KV inputs carry only G heads (not H) — the kernel does the grouping.
    rng = np.random.RandomState(2)
    B, H, G, S, D = 1, 8, 2, 5, 16
    Q = rng.randn(B, H, S, D).astype(np.float32)
    K = rng.randn(B, G, S, D).astype(np.float32)  # G < H heads only
    V = rng.randn(B, G, S, D).astype(np.float32)
    assert K.shape[1] == G < H
    out = R._apple_gpu_dispatch_gqa(Q, K, V, H, G, np)
    assert out.shape == (B, H, S, D)


def test_gqa_symbol_exported():
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_flash_attn_gqa_f32")
    assert R._apple_gpu_flash_attn_gqa_f32() is not None


def test_gqa_f16_upcasts():
    rng = np.random.RandomState(3)
    B, H, G, S, D = 1, 4, 2, 6, 16
    Q = (rng.randn(B, H, S, D) * 0.5).astype(np.float16)
    K = (rng.randn(B, G, S, D) * 0.5).astype(np.float16)
    V = (rng.randn(B, G, S, D) * 0.5).astype(np.float16)
    out = R._apple_gpu_dispatch_gqa(Q, K, V, H, G, np)
    assert out.dtype == np.float16
    ref = _ref_gqa(Q.astype(np.float32), K.astype(np.float32), V.astype(np.float32),
                   H, G, 1.0 / math.sqrt(D), False)
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=3e-2, atol=3e-2)
