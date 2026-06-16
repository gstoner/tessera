"""Apple GPU codegen M2 — attention-region orientation fix.

The fused `matmul -> [scale] -> softmax -> matmul` (attention) path resolves the
score matmul's K orientation from its **transpose flag** (carried as a Phase-0a
dispatch role / on the AttentionRegion), not from value shapes — closing the
long-standing "ambiguous when D==Nk" blocker. `transpose_b=True` means the K
operand is natural (Nk,D) and the matmul does Q·Kᵀ; `transpose_b=False` means the
operand is already Kᵀ and is flipped. See
docs/audit/backend/apple/APPLE_GPU_CODEGEN_PLAN.md (M2).
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.compiler import fusion as F


def _attn_ref(Q, K, V, scale=1.0):
    s = (Q @ K.T) * np.float32(scale)
    e = np.exp(s - s.max(-1, keepdims=True))
    return (e / e.sum(-1, keepdims=True)) @ V


# ── discoverer reads the transpose flag ─────────────────────────────────────
def test_transpose_b_true_means_natural_k():
    ops = [F._Op("matmul", ("q", "k"), "s", {"transpose_b": True}),
           F._Op("softmax", ("s",), "p"),
           F._Op("matmul", ("p", "v"), "o")]
    region = F.discover_attention_regions(ops)[0][1]
    assert region.k_transposed is False   # K operand is natural (Nk,D)


def test_no_flag_means_pretransposed_k():
    ops = [F._Op("matmul", ("q", "kt"), "s"),   # operand is Kᵀ
           F._Op("softmax", ("s",), "p"),
           F._Op("matmul", ("p", "v"), "o")]
    region = F.discover_attention_regions(ops)[0][1]
    assert region.k_transposed is True    # needs flipping to (Nk,D)


def test_pv_matmul_with_transpose_not_fused():
    # P@V must be a plain contraction; a transpose on it isn't standard attn.
    ops = [F._Op("matmul", ("q", "k"), "s", {"transpose_b": True}),
           F._Op("softmax", ("s",), "p"),
           F._Op("matmul", ("p", "v"), "o", {"transpose_b": True})]
    assert F.discover_attention_regions(ops) == []


# ── region honors orientation (reference + natural normalization) ───────────
def test_region_reference_honors_k_transposed():
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((4, 8)).astype(np.float32)
    K = rng.standard_normal((6, 8)).astype(np.float32)        # natural (Nk,D)
    V = rng.standard_normal((6, 5)).astype(np.float32)
    natural = F.AttentionRegion(k_transposed=False).reference(Q, K, V)
    # Same K, pre-transposed operand + k_transposed=True → identical result.
    flipped = F.AttentionRegion(k_transposed=True).reference(
        Q, np.ascontiguousarray(K.T), V)
    np.testing.assert_allclose(natural, _attn_ref(Q, K, V), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(flipped, natural, rtol=1e-5, atol=1e-5)


# ── end-to-end @jit, including the ambiguous D==Nk case ─────────────────────
@pytest.mark.parametrize("M,D,Nk,Dv", [(6, 16, 6, 16), (8, 32, 12, 24),
                                       (4, 8, 4, 8)])
def test_jit_attention_transpose_b_matches_numpy(M, D, Nk, Dv):
    def attn(q, k, v):
        return ts.ops.matmul(ts.ops.softmax(ts.ops.matmul(q, k, transpose_b=True)), v)
    fn = ts.jit(target="apple_gpu")(attn)
    rng = np.random.default_rng(1)
    Q = rng.standard_normal((M, D)).astype(np.float32)
    K = rng.standard_normal((Nk, D)).astype(np.float32)       # D==Nk when equal
    V = rng.standard_normal((Nk, Dv)).astype(np.float32)
    np.testing.assert_allclose(np.asarray(fn(Q, K, V)), _attn_ref(Q, K, V),
                               rtol=1e-4, atol=1e-4)


def test_jit_generic_softmax_sandwich_still_works():
    # The non-transpose `softmax(A@B)@C` form must remain correct.
    def msm(a, b, c):
        return ts.ops.matmul(ts.ops.softmax(ts.ops.matmul(a, b)), c)
    fn = ts.jit(target="apple_gpu")(msm)
    rng = np.random.default_rng(2)
    a = rng.standard_normal((6, 8)).astype(np.float32)
    b = rng.standard_normal((8, 6)).astype(np.float32)
    c = rng.standard_normal((6, 16)).astype(np.float32)
    s = a @ b
    e = np.exp(s - s.max(-1, keepdims=True))
    ref = (e / e.sum(-1, keepdims=True)) @ c
    np.testing.assert_allclose(np.asarray(fn(a, b, c)), ref, rtol=1e-4, atol=1e-4)
