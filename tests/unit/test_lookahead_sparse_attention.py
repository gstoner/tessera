"""LSA composite policy — ``lookahead_sparse_attention``.

Covers the Phase-2 contract: the op matches an explicit composition of
local-window + selected-block attention; ``threshold=0`` collapses to dense
causal attention; ``tau`` is validated but pure-per-call (D2); and VJP/JVP match
finite difference. See ``docs/audit/domain/archive/lsa_scope.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tessera as ts
from tessera import lsa
from tessera.autodiff.jvp import get_jvp
from tessera.autodiff.vjp import get_vjp
from tessera.compiler.op_catalog import get_op_spec
from tessera.compiler.primitive_coverage import coverage_for

ROOT = Path(__file__).resolve().parents[2]


def _qkv(seed=0, B=2, H=2, S=16, D=8, dtype=np.float64):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((B, H, S, D)).astype(dtype)
    K = rng.standard_normal((B, H, S, D)).astype(dtype)
    V = rng.standard_normal((B, H, S, D)).astype(dtype)
    return Q, K, V


def _dense_causal(Q, K, V):
    B, H, S, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    out = np.zeros_like(V, dtype=np.float64)
    for b in range(B):
        for h in range(H):
            for i in range(S):
                s = (Q[b, h, i] @ K[b, h, : i + 1].T) * scale
                s -= s.max()
                w = np.exp(s)
                w /= w.sum()
                out[b, h, i] = w @ V[b, h, : i + 1]
    return out


def _explicit_footprint_attention(Q, K, V, mask, window_size, block_size, causal=True):
    """Independent reference: per-query softmax over (local window ∪ selected
    block tokens), computed from a boolean key-mask — a different code path than
    ``lsa.lookahead_sparse_attention``."""
    B, H, S, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    out = np.zeros_like(V, dtype=np.float64)
    for b in range(B):
        for h in range(H):
            for sq in range(S):
                keep = np.zeros(S, dtype=bool)
                lo = max(0, sq - window_size + 1)
                keep[lo : sq + 1] = True
                for blk in np.flatnonzero(mask[b, h, sq]):
                    keep[blk * block_size : blk * block_size + block_size] = True
                if causal:
                    keep[sq + 1 :] = False
                idx = np.flatnonzero(keep)
                s = (Q[b, h, sq] @ K[b, h, idx].T) * scale
                s -= s.max()
                w = np.exp(s)
                w /= w.sum()
                out[b, h, sq] = w @ V[b, h, idx]
    return out


def test_in_op_catalog_and_autodiff_complete():
    spec = get_op_spec("lookahead_sparse_attention")
    assert spec is not None and spec.graph_name == "tessera.lookahead_sparse_attention"
    assert get_vjp("lookahead_sparse_attention") is not None
    assert get_jvp("lookahead_sparse_attention") is not None
    cs = coverage_for("lookahead_sparse_attention").contract_status
    assert cs["vjp"] == "complete" and cs["jvp"] == "complete"
    assert cs["math_semantics"] == "complete"


def test_threshold_zero_equals_dense_causal_attention():
    # threshold=0 selects every causal block ⇒ footprint = full causal prefix.
    Q, K, V = _qkv(seed=1)
    out = ts.ops.lookahead_sparse_attention(
        Q, K, V, window_size=2, block_size=4, threshold=0.0, causal=True)
    np.testing.assert_allclose(out, _dense_causal(Q, K, V), atol=1e-10)


def test_matches_explicit_composition():
    Q, K, V = _qkv(seed=2)
    window_size, block_size, threshold = 3, 4, 0.5
    keys = lsa.compress_block_keys(K, block_size=block_size)
    mask = lsa.memory_index_select(
        keys, Q, block_size=block_size, threshold=threshold, causal=True, fallback_local=True).mask
    expected = _explicit_footprint_attention(Q, K, V, mask, window_size, block_size, causal=True)
    out = ts.ops.lookahead_sparse_attention(
        Q, K, V, window_size=window_size, block_size=block_size, threshold=threshold, causal=True)
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_tau_is_validated():
    Q, K, V = _qkv()
    with pytest.raises(ValueError):
        ts.ops.lookahead_sparse_attention(Q, K, V, window_size=2, block_size=4, tau=0)


def test_tau_is_pure_per_call_metadata_only():
    # D2: tau is the caller-owned re-selection cadence; a single forward call
    # does exactly one selection, so tau must not change the output.
    Q, K, V = _qkv(seed=3)
    kw = dict(window_size=3, block_size=4, threshold=0.5, causal=True)
    a = ts.ops.lookahead_sparse_attention(Q, K, V, tau=8, **kw)
    b = ts.ops.lookahead_sparse_attention(Q, K, V, tau=512, **kw)
    np.testing.assert_array_equal(a, b)


def test_vjp_matches_finite_difference():
    # threshold=0 ⇒ selection is perturbation-invariant (all causal blocks
    # always selected), so the composite is smooth and the numeric VJP is exact.
    Q, K, V = _qkv(seed=4, B=1, H=1, S=8, D=4)
    kw = dict(window_size=2, block_size=4, threshold=0.0, causal=True)
    dout = np.ones_like(V)
    d_q, d_k, d_v = get_vjp("lookahead_sparse_attention")(dout, Q, K, V, **kw)

    def _grad(fn, x, eps=1e-5):
        g = np.zeros_like(x)
        it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            plus, minus = x.copy(), x.copy()
            plus[idx] += eps
            minus[idx] -= eps
            g[idx] = (fn(plus) - fn(minus)) / (2.0 * eps)
            it.iternext()
        return g

    exp_q = _grad(lambda q: float(ts.ops.lookahead_sparse_attention(q, K, V, **kw).sum()), Q)
    np.testing.assert_allclose(d_q, exp_q, atol=1e-3)


def test_jvp_matches_finite_difference():
    Q, K, V = _qkv(seed=5, B=1, H=1, S=8, D=4)
    kw = dict(window_size=2, block_size=4, threshold=0.0, causal=True)
    dq = np.ones_like(Q) * 0.1
    _, tangent = get_jvp("lookahead_sparse_attention")(
        (Q, K, V), (dq, np.zeros_like(K), np.zeros_like(V)), **kw)
    eps = 1e-6
    plus = ts.ops.lookahead_sparse_attention(Q + eps * dq, K, V, **kw)
    minus = ts.ops.lookahead_sparse_attention(Q - eps * dq, K, V, **kw)
    np.testing.assert_allclose(tangent, (plus - minus) / (2.0 * eps), atol=1e-3)


def test_pass_is_declared_and_pipeline_wired():
    passes_h = (ROOT / "src/transforms/include/Tessera/Transforms/Passes.h").read_text()
    passes_cpp = (ROOT / "src/transforms/lib/Passes.cpp").read_text()
    pass_impl = (ROOT / "src/transforms/lib/AttentionFamilyPasses.cpp").read_text()
    assert "tessera-lookahead-sparse-attn-expand" in pass_impl
    assert "createLookaheadSparseAttnExpandPass" in passes_h
    assert "createLookaheadSparseAttnExpandPass" in passes_cpp
