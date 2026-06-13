"""MSA — MiniMax Sparse Attention (arXiv:2606.13392), Phase 1 reference.

Forward correctness of the three MSA primitives (Index Branch scoring, Top-k
block selection, exact block-sparse Main Branch), the nn.functional wrapper,
autodiff registration, and the op_catalog / primitive_coverage contract. The
key anchor is dense-equivalence: when ``top_k == num_blocks`` MSA collapses to
ordinary (causal) GQA attention bit-for-bit. See docs/msa.md.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.autodiff.jvp import get_jvp
from tessera.autodiff.vjp import get_vjp
from tessera.compiler.op_catalog import get_op_spec
from tessera.compiler.primitive_coverage import coverage_for
from tessera.nn.functional import minimax_sparse_attention


# ── helpers ────────────────────────────────────────────────────────────────

def _qkv(B=1, Hq=4, Hkv=2, Sq=16, Sk=16, D=8, Dv=8, seed=0):
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(B, Hq, Sq, D))
    K = rng.normal(size=(B, Hkv, Sk, D))
    V = rng.normal(size=(B, Hkv, Sk, Dv))
    return Q, K, V


def _dense_gqa(Q, K, V, *, causal=True, scale=None):
    """Reference dense GQA attention (the MSA dense-equivalence target)."""
    B, Hq, Sq, D = Q.shape
    Hkv, Sk, Dv = K.shape[1], K.shape[2], V.shape[-1]
    g = Hq // Hkv
    scale = (1.0 / np.sqrt(D)) if scale is None else scale
    out = np.zeros((B, Hq, Sq, Dv))
    for b in range(B):
        for h in range(Hq):
            grp = h // g
            s = (Q[b, h] @ K[b, grp].T) * scale  # (Sq, Sk)
            if causal:
                fut = np.arange(Sk)[None, :] > np.arange(Sq)[:, None]
                s = np.where(fut, -np.inf, s)
            e = np.exp(s - s.max(axis=-1, keepdims=True))
            w = e / e.sum(axis=-1, keepdims=True)
            out[b, h] = w @ V[b, grp]
    return out


def _numeric_grad(fn, x, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        plus, minus = x.copy(), x.copy()
        plus[idx] += eps
        minus[idx] -= eps
        grad[idx] = (fn(plus) - fn(minus)) / (2.0 * eps)
        it.iternext()
    return grad


# ── shapes ───────────────────────────────────────────────────────────────────

def test_index_scores_shape_is_per_gqa_group():
    Q, K, V = _qkv(B=2, Hq=8, Hkv=2, Sq=16, Sk=16, D=8)
    scores = ts.ops.msa_index_scores(Q, K, block_size=4)
    # (B, Hkv, Sq, num_blocks) — one score per (group, query, KV block).
    assert scores.shape == (2, 2, 16, 4)


def test_select_blocks_shape_and_dtype():
    Q, K, V = _qkv()
    scores = ts.ops.msa_index_scores(Q, K, block_size=4)
    ids = ts.ops.msa_select_blocks(scores, top_k=2, block_size=4)
    assert ids.shape == (1, 2, 16, 2)
    assert ids.dtype == np.int64


def test_sparse_attention_output_shape_matches_q_heads():
    Q, K, V = _qkv(B=1, Hq=4, Hkv=2, Sq=16, Sk=16, D=8, Dv=5)
    out = ts.ops.msa_sparse_attention(Q, K, V, block_size=4, top_k=2)
    assert out.shape == (1, 4, 16, 5)  # (B, Hq, Sq, Dv)


# ── GQA grouping ─────────────────────────────────────────────────────────────

def test_gqa_divisibility_enforced():
    Q, K, V = _qkv(Hq=4, Hkv=3)  # 4 % 3 != 0
    with pytest.raises(ValueError):
        ts.ops.msa_index_scores(Q, K, block_size=4)
    with pytest.raises(ValueError):
        ts.ops.msa_sparse_attention(Q, K, V, block_size=4, top_k=2)


def test_selection_is_shared_across_a_gqa_group():
    Q, K, V = _qkv(B=1, Hq=6, Hkv=2, Sq=16, Sk=16, D=8)
    _, dbg = ts.ops.msa_sparse_attention(
        Q, K, V, block_size=4, top_k=2, return_debug=True
    )
    # Selection granularity is per group → the Hkv axis, not Hq.
    assert dbg["selected_block_ids"].shape[1] == 2


# ── causal masking ───────────────────────────────────────────────────────────

def test_causal_blocks_future_tokens():
    Q, K, V = _qkv(Sq=16, Sk=16, D=8)
    out = ts.ops.msa_sparse_attention(Q, K, V, block_size=4, top_k=4, causal=True)
    # Mutating strictly-future K/V must not change an early query's output.
    K2, V2 = K.copy(), V.copy()
    K2[:, :, 8:, :] += 5.0
    V2[:, :, 8:, :] += 5.0
    out2 = ts.ops.msa_sparse_attention(Q, K2, V2, block_size=4, top_k=4, causal=True)
    np.testing.assert_allclose(out[:, :, :8, :], out2[:, :, :8, :], rtol=1e-6, atol=1e-6)


# ── forced local block ───────────────────────────────────────────────────────

def test_force_local_block_always_selected():
    Q, K, V = _qkv(Sq=16, Sk=16, D=8)
    scores = ts.ops.msa_index_scores(Q, K, block_size=4)
    ids = ts.ops.msa_select_blocks(
        scores, top_k=2, block_size=4, force_local_block=True, causal=True
    )
    B, Hkv, Sq, _ = ids.shape
    for sq in range(Sq):
        local = min(sq // 4, 3)
        assert (ids[:, :, sq, :] == local).any(axis=-1).all(), sq


def test_local_block_hit_rate_is_one_when_forced():
    Q, K, V = _qkv()
    _, dbg = ts.ops.msa_sparse_attention(
        Q, K, V, block_size=4, top_k=2, force_local_block=True, return_debug=True
    )
    assert dbg["local_block_hit"] == 1.0


# ── deterministic top-k ──────────────────────────────────────────────────────

def test_top_k_selection_is_deterministic_and_sorted():
    Q, K, V = _qkv(seed=3)
    scores = ts.ops.msa_index_scores(Q, K, block_size=4)
    a = ts.ops.msa_select_blocks(scores, top_k=3, block_size=4)
    b = ts.ops.msa_select_blocks(scores, top_k=3, block_size=4)
    np.testing.assert_array_equal(a, b)
    assert (np.diff(a, axis=-1) >= 0).all()  # sorted ascending per row


def test_top_k_out_of_range_rejected():
    Q, K, V = _qkv()
    scores = ts.ops.msa_index_scores(Q, K, block_size=4)  # num_blocks = 4
    with pytest.raises(ValueError):
        ts.ops.msa_select_blocks(scores, top_k=5, block_size=4)
    with pytest.raises(ValueError):
        ts.ops.msa_select_blocks(scores, top_k=0, block_size=4)


# ── dense equivalence (the key anchor) ───────────────────────────────────────

@pytest.mark.parametrize("causal", [True, False])
def test_dense_equivalence_when_topk_equals_num_blocks(causal):
    Q, K, V = _qkv(B=2, Hq=4, Hkv=2, Sq=16, Sk=16, D=8, Dv=8, seed=11)
    num_blocks = 16 // 4
    out = ts.ops.msa_sparse_attention(
        Q, K, V, block_size=4, top_k=num_blocks, causal=causal
    )
    ref = _dense_gqa(Q, K, V, causal=causal)
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)


def test_debug_coverage_is_full_when_topk_equals_num_blocks():
    Q, K, V = _qkv()
    _, dbg = ts.ops.msa_sparse_attention(
        Q, K, V, block_size=4, top_k=4, causal=True, return_debug=True
    )
    assert dbg["num_blocks"] == 4
    np.testing.assert_allclose(dbg["coverage"], 1.0, rtol=1e-9)


# ── nn.functional wrapper ────────────────────────────────────────────────────

def test_functional_wrapper_matches_op():
    Q, K, V = _qkv(seed=5)
    a = minimax_sparse_attention(Q, K, V, block_size=4, top_k=2)
    b = ts.ops.msa_sparse_attention(Q, K, V, block_size=4, top_k=2)
    np.testing.assert_array_equal(a, b)


# ── autodiff (VJP finite-difference, smooth top_k==num_blocks regime) ─────────

def test_msa_sparse_attention_vjp_matches_finite_difference():
    Q, K, V = _qkv(B=1, Hq=2, Hkv=1, Sq=8, Sk=8, D=4, Dv=4, seed=2)
    kw = dict(block_size=4, top_k=2, causal=True)  # num_blocks=2 → all selected → smooth
    dout = np.ones((1, 2, 8, 4))
    dQ, dK, dV = get_vjp("msa_sparse_attention")(dout, Q, K, V, **kw)
    expected = _numeric_grad(
        lambda qq: float(ts.ops.msa_sparse_attention(qq, K, V, **kw).sum()), Q
    )
    np.testing.assert_allclose(dQ, expected, rtol=1e-3, atol=1e-4)


# ── registration / coverage contract ─────────────────────────────────────────

def test_ops_and_op_catalog_present():
    for name in ("msa_index_scores", "msa_select_blocks", "msa_sparse_attention"):
        assert hasattr(ts.ops, name), name
        assert get_op_spec(name) is not None, name


def test_autodiff_registration_matches_differentiability():
    # Smooth ops have VJP+JVP; the hard selector does not.
    assert get_vjp("msa_index_scores") is not None
    assert get_jvp("msa_index_scores") is not None
    assert get_vjp("msa_sparse_attention") is not None
    assert get_jvp("msa_sparse_attention") is not None
    assert get_vjp("msa_select_blocks") is None


def test_coverage_contract_axes():
    sa = coverage_for("msa_sparse_attention")
    assert sa.contract_status["vjp"] == "complete"
    assert sa.contract_status["jvp"] == "complete"
    assert sa.contract_status["math_semantics"] == "complete"
    assert sa.contract_status["masking_effect_rule"] == "complete"
    # Honest: no native sparse-block kernel yet (Phase 3) → backend stays partial.
    assert sa.contract_status["backend_kernel"] == "partial"

    sel = coverage_for("msa_select_blocks")
    assert sel.contract_status["vjp"] == "not_applicable"
    assert sel.contract_status["jvp"] == "not_applicable"

    idx = coverage_for("msa_index_scores")
    assert idx.contract_status["vjp"] == "complete"
    assert idx.contract_status["jvp"] == "complete"
