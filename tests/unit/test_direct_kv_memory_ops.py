"""Direct numerical coverage for two thinly-tested ops (test_coverage closeout).

`kv_cache_prune` and `memory_index_score` were flagged `needs_direct_test`
(≤1 counted reference) in docs/audit/generated/test_coverage.md — existing tests
exercised them via the handle method / family namespace, which the reference
parser doesn't count. These are direct `tessera.ops.*` numerical tests.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera
from tessera.cache import KVCacheHandle


# ── kv_cache_prune ────────────────────────────────────────────────────────────


def test_kv_cache_prune_keeps_trailing_window():
    h = KVCacheHandle(num_heads=2, head_dim=4, max_seq=32, page_size=8)
    k = np.arange(10 * 2 * 4, dtype=np.float32).reshape(10, 2, 4)
    v = k + 100.0
    h.append(k, v)
    assert h.current_seq == 10
    pruned = tessera.ops.kv_cache_prune(h, max_entries=4)
    assert pruned.current_seq == 4
    # The trailing 4 tokens [6,7,8,9] are preserved at the front.
    ks, vs = pruned.read(0, 4)
    np.testing.assert_array_equal(ks, k[6:10])
    np.testing.assert_array_equal(vs, v[6:10])


def test_kv_cache_prune_max_seq_alias():
    h = KVCacheHandle(num_heads=1, head_dim=2, max_seq=16)
    h.append(np.ones((8, 1, 2), np.float32), np.ones((8, 1, 2), np.float32))
    out = tessera.ops.kv_cache_prune(h, max_seq=3)
    assert out.current_seq == 3


def test_kv_cache_prune_noop_when_under_limit():
    h = KVCacheHandle(num_heads=1, head_dim=2, max_seq=16)
    h.append(np.ones((5, 1, 2), np.float32), np.ones((5, 1, 2), np.float32))
    out = tessera.ops.kv_cache_prune(h, max_entries=10)
    assert out.current_seq == 5  # fewer entries than the limit → unchanged


# ── memory_index_score ────────────────────────────────────────────────────────


def test_memory_index_score_shape_and_range():
    rng = np.random.default_rng(0)
    B, H, nb, Sq, Dk = 2, 3, 5, 4, 8
    keys = rng.standard_normal((B, H, nb, Dk)).astype(np.float32)
    query = rng.standard_normal((B, H, Sq, Dk)).astype(np.float32)
    probs = tessera.ops.memory_index_score(keys, query)
    assert probs.shape == (B, H, Sq, nb)
    assert np.all((probs >= 0.0) & (probs <= 1.0))   # sigmoid range


def test_memory_index_score_matches_sigmoid_reference():
    B, H, nb, Sq, Dk = 1, 1, 2, 1, 4
    keys = np.ones((B, H, nb, Dk), np.float32)
    query = np.ones((B, H, Sq, Dk), np.float32)
    probs = tessera.ops.memory_index_score(keys, query, scale=1.0)
    # score = q·kᵀ·scale = Dk for every block → sigmoid(Dk).
    expected = 1.0 / (1.0 + np.exp(-float(Dk)))
    np.testing.assert_allclose(probs, expected, rtol=1e-6)


def test_memory_index_score_rejects_bad_rank():
    with pytest.raises(ValueError):
        tessera.ops.memory_index_score(np.zeros((2, 4)), np.zeros((2, 4)))
