"""Tests for `tessera.cache.KVCacheHandle` and the ops dispatch (Phase B2).

Coverage:
  * Construction validation
  * Append (rank-3 and packed rank-2)
  * Read (single-token + slice)
  * Prune
  * max_seq enforcement
  * `ops.kv_cache_*` dispatches handle vs. legacy `ReferenceKVCache`
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────


class TestConstruction:
    def test_basic(self):
        c = ts.cache.KVCacheHandle(num_heads=4, head_dim=8, max_seq=64)
        assert c.current_seq == 0
        assert c.num_heads == 4
        assert c.head_dim == 8
        assert c.max_seq == 64
        assert c.dtype == "fp32"
        assert c.page_size == 128
        assert c.keys.shape == (64, 4, 8)
        assert c.values.shape == (64, 4, 8)

    def test_dtype_options(self):
        c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=8, dtype="fp16")
        assert c.keys.dtype == np.float16

    def test_invalid_dtype_rejected(self):
        with pytest.raises(ValueError, match="Unknown KV-cache dtype"):
            ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=8, dtype="bogus")

    def test_zero_dimensions_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            ts.cache.KVCacheHandle(num_heads=0, head_dim=4, max_seq=8)
        with pytest.raises(ValueError, match="positive"):
            ts.cache.KVCacheHandle(num_heads=4, head_dim=0, max_seq=8)
        with pytest.raises(ValueError, match="positive"):
            ts.cache.KVCacheHandle(num_heads=4, head_dim=8, max_seq=0)


# ─────────────────────────────────────────────────────────────────────────────
# Append
# ─────────────────────────────────────────────────────────────────────────────


class TestAppend:
    def test_rank3_append(self):
        c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=16)
        k = np.random.randn(3, 2, 4).astype(np.float32)
        v = np.random.randn(3, 2, 4).astype(np.float32)
        c.append(k, v)
        assert c.current_seq == 3
        assert c.shape == (3, 2, 4)
        np.testing.assert_allclose(c.keys[:3], k)
        np.testing.assert_allclose(c.values[:3], v)

    def test_packed_rank2_append(self):
        # (seq, num_heads * head_dim) is reshaped to (seq, num_heads, head_dim)
        c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=16)
        k = np.random.randn(2, 8).astype(np.float32)  # 2 * 4 = 8
        v = np.random.randn(2, 8).astype(np.float32)
        c.append(k, v)
        assert c.current_seq == 2
        assert c.keys[:2].shape == (2, 2, 4)

    def test_append_advances_current_seq(self):
        c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=16)
        c.append(np.zeros((3, 2, 4), dtype=np.float32), np.zeros((3, 2, 4), dtype=np.float32))
        c.append(np.zeros((4, 2, 4), dtype=np.float32), np.zeros((4, 2, 4), dtype=np.float32))
        assert c.current_seq == 7

    def test_max_seq_overflow_rejected(self):
        c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=4)
        c.append(np.zeros((4, 2, 4), dtype=np.float32), np.zeros((4, 2, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="exceed max_seq"):
            c.append(np.zeros((1, 2, 4), dtype=np.float32), np.zeros((1, 2, 4), dtype=np.float32))

    def test_shape_mismatch_rejected(self):
        c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=16)
        with pytest.raises(ValueError, match="num_heads"):
            c.append(
                np.zeros((3, 4, 4), dtype=np.float32),  # wrong num_heads
                np.zeros((3, 4, 4), dtype=np.float32),
            )

    def test_kv_seq_length_mismatch_rejected(self):
        c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=16)
        with pytest.raises(ValueError, match="matching sequence"):
            c.append(
                np.zeros((3, 2, 4), dtype=np.float32),
                np.zeros((4, 2, 4), dtype=np.float32),
            )

    def test_dtype_coercion(self):
        # fp32 cache accepts fp16 input — gets promoted.
        c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=16, dtype="fp32")
        k = np.ones((1, 2, 4), dtype=np.float16)
        c.append(k, k)
        assert c.keys[:1].dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# Read
# ─────────────────────────────────────────────────────────────────────────────


class TestRead:
    def test_slice_read(self):
        c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=16)
        k = np.random.randn(8, 2, 4).astype(np.float32)
        v = np.random.randn(8, 2, 4).astype(np.float32)
        c.append(k, v)
        ks, vs = c.read(0, 5)
        np.testing.assert_allclose(ks, k[:5])
        np.testing.assert_allclose(vs, v[:5])

    def test_single_token_read(self):
        c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=16)
        c.append(np.arange(8 * 2 * 4).reshape(8, 2, 4).astype(np.float32),
                 np.zeros((8, 2, 4), dtype=np.float32))
        k1, v1 = c.read(4)  # default end = start+1
        assert k1.shape == (1, 2, 4)
        np.testing.assert_allclose(k1[0], np.arange(8 * 2 * 4).reshape(8, 2, 4)[4])

    def test_read_out_of_range(self):
        c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=8)
        c.append(np.zeros((2, 2, 4), dtype=np.float32), np.zeros((2, 2, 4), dtype=np.float32))
        with pytest.raises(IndexError, match="start"):
            c.read(-1)
        with pytest.raises(IndexError, match="end"):
            c.read(0, 5)


# ─────────────────────────────────────────────────────────────────────────────
# Prune
# ─────────────────────────────────────────────────────────────────────────────


class TestPrune:
    def test_prune_keeps_trailing(self):
        c = ts.cache.KVCacheHandle(num_heads=1, head_dim=2, max_seq=10)
        # Append 6 unique values
        for i in range(6):
            arr = np.full((1, 1, 2), float(i), dtype=np.float32)
            c.append(arr, arr)
        c.prune(max_entries=3)  # keep last 3 (3.0, 4.0, 5.0)
        assert c.current_seq == 3
        np.testing.assert_allclose(c.keys[:3, 0, 0], [3.0, 4.0, 5.0])

    def test_prune_zeros_freed_region(self):
        c = ts.cache.KVCacheHandle(num_heads=1, head_dim=2, max_seq=10)
        c.append(np.ones((4, 1, 2), dtype=np.float32), np.ones((4, 1, 2), dtype=np.float32))
        c.prune(max_entries=2)
        # Region [2, 4) should be zeroed for hygiene
        assert (c.keys[2:4] == 0).all()

    def test_prune_negative_rejected(self):
        c = ts.cache.KVCacheHandle(num_heads=1, head_dim=2, max_seq=4)
        with pytest.raises(ValueError, match="non-negative"):
            c.prune(-1)

    def test_prune_larger_than_current_is_noop(self):
        c = ts.cache.KVCacheHandle(num_heads=1, head_dim=2, max_seq=10)
        c.append(np.ones((3, 1, 2), dtype=np.float32), np.ones((3, 1, 2), dtype=np.float32))
        c.prune(max_entries=100)
        assert c.current_seq == 3


class TestTrim:
    def test_trim_rolls_back_newest_tokens_and_preserves_prefix(self):
        c = ts.cache.KVCacheHandle(num_heads=1, head_dim=2, max_seq=10)
        k = np.arange(6 * 1 * 2, dtype=np.float32).reshape(6, 1, 2)
        v = k + 100
        c.append(k, v)
        assert c.is_trimmable()
        c.trim(2)
        assert c.current_seq == 4
        ks, vs = c.read(0, 4)
        np.testing.assert_allclose(ks, k[:4])
        np.testing.assert_allclose(vs, v[:4])
        assert (c.keys[4:6] == 0).all()
        assert (c.values[4:6] == 0).all()

    def test_trim_oversized_clears_cache(self):
        c = ts.cache.KVCacheHandle(num_heads=1, head_dim=2, max_seq=4)
        c.append(np.ones((3, 1, 2), dtype=np.float32), np.ones((3, 1, 2), dtype=np.float32))
        c.trim(99)
        assert c.current_seq == 0
        assert (c.keys[:3] == 0).all()

    def test_trim_negative_rejected(self):
        c = ts.cache.KVCacheHandle(num_heads=1, head_dim=2, max_seq=4)
        with pytest.raises(ValueError, match="non-negative"):
            c.trim(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Ops dispatch — handle vs. legacy ReferenceKVCache
# ─────────────────────────────────────────────────────────────────────────────


class TestOpsDispatch:
    def test_ops_kv_cache_append_handle(self):
        c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=8)
        k = np.random.randn(2, 2, 4).astype(np.float32)
        v = np.random.randn(2, 2, 4).astype(np.float32)
        result = ts.ops.kv_cache_append(c, k, v)
        assert result is c
        assert c.current_seq == 2

    def test_ops_kv_cache_read_handle(self):
        c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=8)
        c.append(np.ones((4, 2, 4), dtype=np.float32),
                 np.ones((4, 2, 4), dtype=np.float32) * 2)
        k_slice, v_slice = ts.ops.kv_cache_read(c, 1, 3)
        assert k_slice.shape == (2, 2, 4)
        np.testing.assert_allclose(k_slice, 1.0)
        np.testing.assert_allclose(v_slice, 2.0)

    def test_ops_kv_cache_prune_handle(self):
        c = ts.cache.KVCacheHandle(num_heads=1, head_dim=2, max_seq=10)
        c.append(np.ones((5, 1, 2), dtype=np.float32), np.ones((5, 1, 2), dtype=np.float32))
        ts.ops.kv_cache_prune(c, max_entries=2)
        assert c.current_seq == 2

    def test_legacy_reference_kv_cache_still_works(self):
        # Backward compat — legacy ReferenceKVCache surface still functions.
        cache = ts.ops.kv_cache_append(None, np.ones((1, 8)), np.ones((1, 8)))
        cache = ts.ops.kv_cache_append(cache, np.ones((1, 8)) * 2, np.ones((1, 8)) * 2)
        assert len(cache.keys) == 2

    def test_kv_cache_read_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="unsupported cache type"):
            ts.ops.kv_cache_read("not a cache", 0)
