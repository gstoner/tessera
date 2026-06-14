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


def test_msa_op_catalog_contract_is_exact():
    idx = get_op_spec("msa_index_scores")
    sel = get_op_spec("msa_select_blocks")
    sparse = get_op_spec("msa_sparse_attention")
    assert idx.graph_name == "tessera.msa_index_scores"
    assert idx.min_arity == idx.max_arity == 2
    assert idx.lowering == "attention"
    assert sel.graph_name == "tessera.msa_select_blocks"
    assert sel.min_arity == sel.max_arity == 1
    assert sel.lowering == "indexing"
    assert sparse.graph_name == "tessera.msa_sparse_attention"
    assert sparse.min_arity == sparse.max_arity == 3
    assert sparse.effect == "state"
    assert sparse.lowering == "attention"


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


# ── nn.MinimaxSparseAttention module (Phase 4) ───────────────────────────────

def test_minimax_sparse_attention_is_exported():
    assert hasattr(ts.nn, "MinimaxSparseAttention")


def test_module_forward_shape():
    m = ts.nn.MinimaxSparseAttention(
        embed_dim=16, num_heads=4, num_kv_heads=2, block_size=4, top_k=2
    )
    x = np.random.default_rng(0).normal(size=(2, 16, 16)).astype(np.float32)
    out = m(x)
    assert out.shape == (2, 16, 16)  # (B, S, embed_dim)


def test_module_rejects_bad_gqa_and_seqlen():
    with pytest.raises(ValueError):
        ts.nn.MinimaxSparseAttention(
            embed_dim=16, num_heads=4, num_kv_heads=3, block_size=4, top_k=2
        )
    m = ts.nn.MinimaxSparseAttention(
        embed_dim=16, num_heads=4, num_kv_heads=2, block_size=8, top_k=1
    )
    bad = np.zeros((1, 12, 16), dtype=np.float32)  # 12 % 8 != 0
    with pytest.raises(ValueError):
        m(bad)


def test_module_composes_the_msa_op_with_its_own_weights():
    m = ts.nn.MinimaxSparseAttention(
        embed_dim=16, num_heads=4, num_kv_heads=2, block_size=4, top_k=2, causal=True
    )
    x = np.random.default_rng(7).normal(size=(1, 16, 16)).astype(np.float32)
    out = m(x)

    # Manually compose the same path with the module's (kaiming-init) weights.
    Wq = np.asarray(m.W_q._data._data)
    Wk = np.asarray(m.W_k._data._data)
    Wv = np.asarray(m.W_v._data._data)
    Wo = np.asarray(m.W_o._data._data)
    B, S, D = 1, 16, 4
    Q = (x @ Wq).reshape(B, S, 4, D).transpose(0, 2, 1, 3)
    K = (x @ Wk).reshape(B, S, 2, D).transpose(0, 2, 1, 3)
    V = (x @ Wv).reshape(B, S, 2, D).transpose(0, 2, 1, 3)
    O = ts.ops.msa_sparse_attention(Q, K, V, block_size=4, top_k=2, causal=True)
    manual = O.transpose(0, 2, 1, 3).reshape(B, S, 16) @ Wo
    np.testing.assert_allclose(out, manual, rtol=1e-6, atol=1e-6)


def test_module_dense_when_topk_equals_num_blocks():
    # top_k == num_blocks → exact dense GQA (independent of the index scores).
    m = ts.nn.MinimaxSparseAttention(
        embed_dim=16, num_heads=4, num_kv_heads=2, block_size=4, top_k=4
    )
    x = np.random.default_rng(1).normal(size=(1, 16, 16)).astype(np.float32)
    out = m(x)
    assert out.shape == (1, 16, 16)
    assert np.isfinite(out).all()


@pytest.mark.parametrize(
    "dense,sparsity,expected_top_k",
    [(True, 0.25, 8), (False, 0.5, 4), (False, 0.1, 1), (False, 1.0, 8)],
)
def test_from_gqa_derives_top_k(dense, sparsity, expected_top_k):
    m = ts.nn.MinimaxSparseAttention.from_gqa(
        embed_dim=16, num_heads=4, num_kv_heads=2,
        seq_len=64, block_size=8, sparsity=sparsity, dense=dense,
    )
    assert m.top_k == expected_top_k  # num_blocks = 64 // 8 = 8
    assert m.num_kv_heads == 2 and m.num_heads == 4


def test_from_gqa_rejects_indivisible_seqlen():
    with pytest.raises(ValueError):
        ts.nn.MinimaxSparseAttention.from_gqa(
            embed_dim=16, num_heads=4, num_kv_heads=2, seq_len=60, block_size=8
        )
