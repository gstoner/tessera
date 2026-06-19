"""Workstream A — PagedKVState unifying ABI conformance + metamorphic oracle.

These guards lock the contract that joins Tessera's two KV substrates (contiguous
``KVCacheHandle`` and tiered ``TieredKVCache``) under one protocol an attention op
can consume. The metamorphic test is the seed of A's evaluator oracle: a paged
state and a contiguous state holding the same logical sequence must gather
bit-identical K/V.

See docs/audit/roadmap/CONTRACT_PASS_PLAN.md (Workstream A).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.cache import (KVCacheHandle, TieredKVCache, PagedKVState, PageTier,
                           KVKind, KVGeometry, PageTableEntry, as_paged_kv_state,
                           paged_attention)


def _rng(seed=0):
    return np.random.default_rng(seed)


def _fill_contiguous(num_heads, head_dim, n_tokens, *, seed=0, page_size=4):
    rng = _rng(seed)
    h = KVCacheHandle(num_heads=num_heads, head_dim=head_dim,
                      max_seq=max(n_tokens, 1), page_size=page_size)
    k = rng.standard_normal((n_tokens, num_heads, head_dim)).astype(np.float32)
    v = rng.standard_normal((n_tokens, num_heads, head_dim)).astype(np.float32)
    h.append(k, v)
    return h, k, v


def _fill_tiered(num_heads, head_dim, n_tokens, k, v, *, page_size=4,
                 resident_capacity=2):
    c = TieredKVCache(num_heads=num_heads, head_dim=head_dim,
                      max_seq=((n_tokens + page_size - 1) // page_size) * page_size,
                      page_size=page_size, resident_capacity=resident_capacity)
    c.write(k, v)
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Protocol conformance
# ─────────────────────────────────────────────────────────────────────────────


def test_contiguous_handle_satisfies_protocol():
    h, _, _ = _fill_contiguous(4, 8, 10)
    state = as_paged_kv_state(h)
    assert isinstance(state, PagedKVState)
    assert state.kind is KVKind.FULL
    geo = state.kv_geometry()
    assert isinstance(geo, KVGeometry)
    assert geo.num_heads == 4 and geo.head_dim == 8
    assert state.seq_len() == 10


def test_tiered_cache_satisfies_protocol():
    h, k, v = _fill_contiguous(4, 8, 10)
    c = _fill_tiered(4, 8, 10, k, v)
    state = as_paged_kv_state(c)
    assert isinstance(state, PagedKVState)
    assert state.seq_len() == 10
    assert state.kv_geometry().num_heads == 4


def test_unsupported_type_raises():
    with pytest.raises(TypeError):
        as_paged_kv_state(object())


# ─────────────────────────────────────────────────────────────────────────────
# Page table + tiers
# ─────────────────────────────────────────────────────────────────────────────


def test_contiguous_page_table_is_all_resident():
    h, _, _ = _fill_contiguous(2, 4, 10, page_size=4)
    state = as_paged_kv_state(h)
    table = state.page_table()
    assert [e.page_id for e in table] == [0, 1, 2]  # ceil(10/4)
    assert all(e.tier is PageTier.RESIDENT for e in table)
    assert all(isinstance(e, PageTableEntry) for e in table)


def test_tiered_page_table_reports_host_and_resident():
    h, k, v = _fill_contiguous(2, 4, 12, page_size=4)
    c = _fill_tiered(2, 4, 12, k, v, page_size=4, resident_capacity=1)
    # Stage exactly one page resident; the rest must report HOST.
    c.stage([2])
    state = as_paged_kv_state(c)
    tiers = {e.page_id: e.tier for e in state.page_table()}
    assert tiers[2] is PageTier.RESIDENT
    assert tiers[0] is PageTier.HOST and tiers[1] is PageTier.HOST
    assert state.tier(2) is PageTier.RESIDENT
    assert state.tier(0) is PageTier.HOST


# ─────────────────────────────────────────────────────────────────────────────
# gather — the consumer-facing read
# ─────────────────────────────────────────────────────────────────────────────


def test_contiguous_gather_matches_source():
    h, k, v = _fill_contiguous(3, 5, 9)
    state = as_paged_kv_state(h)
    idx = [0, 4, 8, 2]
    gk, gv = state.gather(idx)
    assert gk.shape == (4, 3, 5)
    np.testing.assert_array_equal(gk, k[idx])
    np.testing.assert_array_equal(gv, v[idx])


def test_tiered_gather_auto_stages_cold_pages():
    h, k, v = _fill_contiguous(2, 4, 12, page_size=4)
    c = _fill_tiered(2, 4, 12, k, v, page_size=4, resident_capacity=1)
    state = as_paged_kv_state(c)
    # Tokens spanning three different pages, none staged yet — gather must stage.
    idx = [1, 5, 9]
    gk, gv = state.gather(idx)
    np.testing.assert_allclose(gk, k[idx], rtol=0, atol=0)
    np.testing.assert_allclose(gv, v[idx], rtol=0, atol=0)
    assert c.stats.pages_staged >= 1  # the prefetch the adapter performed


# ─────────────────────────────────────────────────────────────────────────────
# Metamorphic oracle — paged ≡ contiguous (seed of evaluator.cross_path_equivalence)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_tokens,page_size,cap", [(8, 4, 1), (16, 4, 2), (10, 2, 3)])
def test_paged_equiv_contiguous_gather(n_tokens, page_size, cap):
    h, k, v = _fill_contiguous(4, 8, n_tokens, page_size=page_size)
    c = _fill_tiered(4, 8, n_tokens, k, v, page_size=page_size, resident_capacity=cap)
    cont = as_paged_kv_state(h)
    paged = as_paged_kv_state(c)

    idx = list(range(n_tokens))[::-1]  # arbitrary order, full coverage
    ck, cv = cont.gather(idx)
    pk, pv = paged.gather(idx)
    np.testing.assert_array_equal(ck, pk)
    np.testing.assert_array_equal(cv, pv)


# ─────────────────────────────────────────────────────────────────────────────
# paged_attention consumer — the contract is not an orphan
# ─────────────────────────────────────────────────────────────────────────────


def _ref_attention(Q, K, V, causal=False):
    """Per-head reference: Q,K,V are (H, S, d). Returns (H, q_len, d)."""
    d = Q.shape[-1]
    scores = np.matmul(Q, K.swapaxes(-1, -2)) / np.sqrt(d)
    if causal:
        q_len, k_len = scores.shape[-2], scores.shape[-1]
        mask = np.triu(np.ones((q_len, k_len), bool), k=1 + max(k_len - q_len, 0))
        scores = np.where(mask, -np.inf, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    return np.matmul(w, V)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("n_tokens,page_size,cap", [(8, 4, 1), (12, 4, 2)])
def test_paged_attention_equals_contiguous(causal, n_tokens, page_size, cap):
    num_heads, head_dim, q_len = 4, 8, 3
    h, k, v = _fill_contiguous(num_heads, head_dim, n_tokens, page_size=page_size)
    c = _fill_tiered(num_heads, head_dim, n_tokens, k, v,
                     page_size=page_size, resident_capacity=cap)
    Q = _rng(7).standard_normal((num_heads, q_len, head_dim)).astype(np.float32)

    # Reference: per-head attention over the full contiguous K/V.
    Kt = np.transpose(k, (1, 0, 2))
    Vt = np.transpose(v, (1, 0, 2))
    ref = _ref_attention(Q, Kt, Vt, causal=causal)

    o_contig = paged_attention(Q, h, causal=causal)
    o_tiered = paged_attention(Q, c, causal=causal)

    np.testing.assert_allclose(o_contig, ref, rtol=1e-5, atol=1e-5)
    # The metamorphic invariant: tiered/paged ≡ contiguous, bit-for-bit on gather.
    np.testing.assert_array_equal(o_contig, o_tiered)


def test_ops_namespace_exposes_paged_attention():
    import tessera
    h, k, v = _fill_contiguous(2, 4, 6)
    Q = _rng(3).standard_normal((2, 2, 4)).astype(np.float32)
    o_ops = tessera.ops.paged_attention(Q, h)
    o_flash = tessera.ops.flash_attn(Q, kv_state=h)  # kv_state alias
    np.testing.assert_array_equal(o_ops, o_flash)


def test_evaluator_paged_kv_oracle_equivalent():
    from tessera.compiler.evaluator import paged_kv_equivalence
    n_tokens = 12
    h, k, v = _fill_contiguous(4, 8, n_tokens, page_size=4)
    c1 = _fill_tiered(4, 8, n_tokens, k, v, page_size=4, resident_capacity=1)
    c_all = _fill_tiered(4, 8, n_tokens, k, v, page_size=4, resident_capacity=3)
    Q = _rng(5).standard_normal((4, 3, 8)).astype(np.float32)

    verdict = paged_kv_equivalence(
        [("contiguous", h), ("tiered_cap1", c1), ("tiered_capAll", c_all)], Q)
    assert verdict.relation == "equivalent", verdict.detail
    assert verdict.max_abs_err == 0.0  # bit-identical: residency schedule is inert


def test_quantized_contiguous_gather_dequantizes():
    rng = _rng(1)
    h = KVCacheHandle(num_heads=2, head_dim=4, max_seq=8, page_size=4,
                      quantize_bits=8)
    k = rng.standard_normal((6, 2, 4)).astype(np.float32)
    v = rng.standard_normal((6, 2, 4)).astype(np.float32)
    h.append(k, v)
    state = as_paged_kv_state(h)
    assert state.quant_bits() == 8
    gk, gv = state.gather([0, 3, 5])
    # int8 per-token symmetric quant round-trips within a coarse tolerance.
    np.testing.assert_allclose(gk, k[[0, 3, 5]], rtol=0.1, atol=0.1)
    np.testing.assert_allclose(gv, v[[0, 3, 5]], rtol=0.1, atol=0.1)
