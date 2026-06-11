"""Gap 1 — CPU cold-pool ↔ GPU-resident KV tiering for LSA.

`TieredKVCache` holds every KV page in a host cold pool and a bounded set of
pages in device-resident buffers; `stage`/`evict`/`gather` are the host↔device
staging ABI. `lookahead_attention_tiered` drives staging from the LSA selector
and must be numerically identical to the non-tiered oracle (tiering is a
residency optimization, not a math change). See
`docs/audit/domain/archive/lsa_scope.md` (Gap 1).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import cache, lsa


def _kqv(seed=0, H=2, D=8, S=16):
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((S, H, D)).astype(np.float32)
    V = rng.standard_normal((S, H, D)).astype(np.float32)
    Q = rng.standard_normal((S, H, D)).astype(np.float32)
    return K, V, Q


def _oracle(Q, K, V, *, window_size, block_size, threshold, causal=True):
    """Non-tiered oracle in (S,H,D) layout via lsa.lookahead_sparse_attention."""
    Kb = np.transpose(K, (1, 0, 2))[None]
    Vb = np.transpose(V, (1, 0, 2))[None]
    Qb = np.transpose(Q, (1, 0, 2))[None]
    ref = lsa.lookahead_sparse_attention(
        Qb, Kb, Vb, window_size=window_size, block_size=block_size,
        threshold=threshold, causal=causal)
    return np.transpose(ref[0], (1, 0, 2))


def test_exports():
    for name in ("TieredKVCache", "TieredKVCacheError", "StageStats",
                 "TieredStats", "lookahead_attention_tiered"):
        assert hasattr(cache, name)


def test_write_and_block_keys():
    K, V, _ = _kqv()
    c = cache.TieredKVCache(num_heads=2, head_dim=8, max_seq=16, page_size=4)
    c.write(K, V)
    assert c.current_seq == 16
    assert c.num_pages == 4
    keys = c.block_keys()
    assert keys.shape == (4, 2, 8)
    # page 0 summary == mean of its 4 tokens (the indexer-key contract).
    np.testing.assert_allclose(keys[0], K[0:4].mean(axis=0), atol=1e-6)


def test_stage_evict_residency_accounting():
    K, V, _ = _kqv()
    c = cache.TieredKVCache(num_heads=2, head_dim=8, max_seq=16, page_size=4,
                            resident_capacity=2)
    c.write(K, V)
    s = c.stage([0, 1])
    assert s.staged == 2 and s.evicted == 0 and s.already_resident == 0
    assert c.resident_pages() == {0, 1}
    # staging a 3rd page evicts the LRU (page 0).
    s2 = c.stage([2])
    assert s2.staged == 1 and s2.evicted == 1
    assert 0 not in c.resident_pages() and 2 in c.resident_pages()
    # re-staging an already-resident page copies nothing.
    s3 = c.stage([2])
    assert s3.staged == 0 and s3.already_resident == 1
    assert c.stats.bytes_staged == s.bytes_staged + s2.bytes_staged


def test_stage_more_than_capacity_raises():
    K, V, _ = _kqv()
    c = cache.TieredKVCache(num_heads=2, head_dim=8, max_seq=16, page_size=4,
                            resident_capacity=2)
    c.write(K, V)
    with pytest.raises(cache.TieredKVCacheError):
        c.stage([0, 1, 2])  # one call asks for 3 pages into a 2-page set


def test_gather_requires_resident_by_default():
    K, V, _ = _kqv()
    c = cache.TieredKVCache(num_heads=2, head_dim=8, max_seq=16, page_size=4,
                            resident_capacity=2)
    c.write(K, V)
    with pytest.raises(cache.TieredKVCacheError):
        c.gather([0])  # page 0 is cold
    c.stage([0])
    Kg, Vg = c.gather([0, 1, 2, 3])  # tokens of page 0
    np.testing.assert_allclose(Kg, K[0:4], atol=1e-6)
    np.testing.assert_allclose(Vg, V[0:4], atol=1e-6)


def test_gather_auto_stage_streams_cold_pages():
    K, V, _ = _kqv()
    c = cache.TieredKVCache(num_heads=2, head_dim=8, max_seq=16, page_size=4,
                            resident_capacity=1)
    c.write(K, V)
    # require_resident=False streams pages just-in-time even with capacity 1.
    Kg, Vg = c.gather(np.arange(16), require_resident=False)
    np.testing.assert_allclose(Kg, K, atol=1e-6)
    np.testing.assert_allclose(Vg, V, atol=1e-6)


@pytest.mark.parametrize("threshold,window_size", [(0.5, 3), (0.0, 2), (0.8, 4)])
def test_tiered_matches_oracle(threshold, window_size):
    K, V, Q = _kqv(seed=int(threshold * 100) + window_size)
    c = cache.TieredKVCache(num_heads=2, head_dim=8, max_seq=16, page_size=4,
                            resident_capacity=2)  # tight: 2 of 4 pages
    c.write(K, V)
    out, _ = cache.lookahead_attention_tiered(
        Q, c, window_size=window_size, threshold=threshold, causal=True)
    ref = _oracle(Q, K, V, window_size=window_size, block_size=4, threshold=threshold)
    np.testing.assert_allclose(out, ref, atol=1e-9)


def test_output_independent_of_resident_capacity():
    K, V, Q = _kqv(seed=11)
    tight = cache.TieredKVCache(num_heads=2, head_dim=8, max_seq=16, page_size=4,
                                resident_capacity=1)
    full = cache.TieredKVCache(num_heads=2, head_dim=8, max_seq=16, page_size=4)
    tight.write(K, V)
    full.write(K, V)
    o_tight, _ = cache.lookahead_attention_tiered(Q, tight, window_size=3, threshold=0.5)
    o_full, _ = cache.lookahead_attention_tiered(Q, full, window_size=3, threshold=0.5)
    np.testing.assert_allclose(o_tight, o_full, atol=1e-12)
    # Tight capacity forces real cold↔resident traffic; full residency evicts nothing.
    assert tight.stats.pages_evicted > 0
    assert full.stats.pages_evicted == 0


def test_lookahead_drives_staging():
    # threshold=1.0 → selector returns only the local/own block; threshold=0.0 →
    # every causal block selected. The selector therefore drives both the
    # footprint size (gather tokens) and, under tight capacity, the cold↔resident
    # staging traffic. window_size=1 isolates the selection contribution.
    K, V, Q = _kqv(seed=3)
    hi = cache.TieredKVCache(num_heads=2, head_dim=8, max_seq=16, page_size=4,
                             resident_capacity=1)
    lo = cache.TieredKVCache(num_heads=2, head_dim=8, max_seq=16, page_size=4,
                             resident_capacity=1)
    hi.write(K, V)
    lo.write(K, V)
    cache.lookahead_attention_tiered(Q, hi, window_size=1, threshold=1.0)
    cache.lookahead_attention_tiered(Q, lo, window_size=1, threshold=0.0)
    assert lo.stats.gather_tokens > hi.stats.gather_tokens
    assert lo.stats.bytes_staged > hi.stats.bytes_staged


def test_device_resident_flag_is_boolean():
    c = cache.TieredKVCache(num_heads=2, head_dim=8, max_seq=16, page_size=4)
    assert isinstance(c.device_resident, bool)
    c.free()  # idempotent cleanup
