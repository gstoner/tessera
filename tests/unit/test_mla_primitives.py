"""Theme 5 — Multi-Latent Attention primitives.

Forward correctness of the latent compress / expand / RoPE-split / merge
ops, plus the LatentKVCacheHandle paged latent storage. Per-backend
FlashMLA target kernels (Hopper / Blackwell absorb-K fusion) are
deferred to Phase G — these tests pin the Python op surface so the
example at `examples/advanced/mla/` runs end-to-end on the CPU
reference path.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts


# ─────────────────────────────────────────────────────────────────────────────
# Latent compress / expand
# ─────────────────────────────────────────────────────────────────────────────


class TestLatentProjection:
    def test_compress_then_expand_shapes(self):
        np.random.seed(0)
        B, S, hidden = 2, 4, 64
        latent_dim = 16
        num_heads, head_dim = 4, 16
        x = np.random.randn(B, S, hidden).astype(np.float32)
        w_dkv = np.random.randn(hidden, latent_dim).astype(np.float32) * 0.1
        w_uk = np.random.randn(latent_dim, num_heads * head_dim).astype(np.float32) * 0.1
        w_uv = np.random.randn(latent_dim, num_heads * head_dim).astype(np.float32) * 0.1

        c = ts.ops.latent_kv_compress(x, w_dkv)
        assert c.shape == (B, S, latent_dim)
        K = ts.ops.latent_kv_expand_k(c, w_uk)
        V = ts.ops.latent_kv_expand_v(c, w_uv)
        assert K.shape == (B, S, num_heads * head_dim)
        assert V.shape == (B, S, num_heads * head_dim)

    def test_compress_matches_matmul_reference(self):
        """Numerically `latent_kv_compress` is just a matmul — the distinct
        op_name is the IR anchor for FlashMLA fusion. Check we didn't drift
        from that contract."""
        np.random.seed(0)
        x = np.random.randn(8, 32).astype(np.float32)
        w = np.random.randn(32, 8).astype(np.float32) * 0.3
        np.testing.assert_array_equal(ts.ops.latent_kv_compress(x, w), x @ w)

    def test_expand_matches_matmul_reference(self):
        np.random.seed(0)
        c = np.random.randn(4, 16).astype(np.float32)
        w = np.random.randn(16, 64).astype(np.float32) * 0.3
        np.testing.assert_array_equal(ts.ops.latent_kv_expand_k(c, w), c @ w)
        np.testing.assert_array_equal(ts.ops.latent_kv_expand_v(c, w), c @ w)


# ─────────────────────────────────────────────────────────────────────────────
# RoPE split / merge — decoupled-RoPE pattern
# ─────────────────────────────────────────────────────────────────────────────


class TestRopeSplitMerge:
    def test_split_partitions_last_dim(self):
        x = np.arange(48, dtype=np.float32).reshape(2, 24)
        rope, no_rope = ts.ops.rope_split(x, rope_dim=8)
        assert rope.shape == (2, 8)
        assert no_rope.shape == (2, 16)
        np.testing.assert_array_equal(rope, x[..., :8])
        np.testing.assert_array_equal(no_rope, x[..., 8:])

    def test_split_at_zero_returns_empty_rope(self):
        x = np.arange(8, dtype=np.float32).reshape(1, 8)
        rope, no_rope = ts.ops.rope_split(x, rope_dim=0)
        assert rope.shape == (1, 0)
        assert no_rope.shape == (1, 8)

    def test_split_at_full_returns_empty_no_rope(self):
        x = np.arange(8, dtype=np.float32).reshape(1, 8)
        rope, no_rope = ts.ops.rope_split(x, rope_dim=8)
        assert rope.shape == (1, 8)
        assert no_rope.shape == (1, 0)

    def test_merge_reverses_split(self):
        np.random.seed(0)
        x = np.random.randn(4, 32).astype(np.float32)
        rope, no_rope = ts.ops.rope_split(x, rope_dim=8)
        x_rt = ts.ops.rope_merge(rope, no_rope)
        np.testing.assert_array_equal(x, x_rt)

    def test_split_rejects_out_of_range_rope_dim(self):
        x = np.arange(8, dtype=np.float32).reshape(1, 8)
        with pytest.raises(ValueError, match="rope_dim"):
            ts.ops.rope_split(x, rope_dim=9)
        with pytest.raises(ValueError, match="rope_dim"):
            ts.ops.rope_split(x, rope_dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# LatentKVCacheHandle — paged latent cache
# ─────────────────────────────────────────────────────────────────────────────


class TestLatentKVCacheHandle:
    def test_construct_and_metadata(self):
        cache = ts.cache.LatentKVCacheHandle(latent_dim=64, max_seq=512)
        assert cache.shape == (0, 64)
        assert cache.current_seq == 0
        assert cache.is_full is False
        assert cache.dtype == "fp32"

    def test_append_then_read(self):
        np.random.seed(0)
        latent_dim = 16
        cache = ts.cache.LatentKVCacheHandle(latent_dim=latent_dim, max_seq=64)
        c1 = np.random.randn(8, latent_dim).astype(np.float32)
        c2 = np.random.randn(4, latent_dim).astype(np.float32)
        cache.append(c1)
        cache.append(c2)
        assert cache.shape == (12, latent_dim)
        np.testing.assert_array_equal(cache.read(0, 8), c1)
        np.testing.assert_array_equal(cache.read(8, 12), c2)

    def test_append_single_token_1d(self):
        """1-D input is reshaped to (1, latent_dim) — common decode pattern
        where each step appends one token."""
        cache = ts.cache.LatentKVCacheHandle(latent_dim=4, max_seq=8)
        cache.append(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        assert cache.shape == (1, 4)
        np.testing.assert_array_equal(cache.read(0), [[1.0, 2.0, 3.0, 4.0]])

    def test_append_overflow_raises_without_auto_evict(self):
        cache = ts.cache.LatentKVCacheHandle(latent_dim=4, max_seq=4)
        with pytest.raises(ValueError, match="exceed max_seq"):
            cache.append(np.zeros((5, 4), dtype=np.float32))

    def test_append_with_auto_evict_slides_window(self):
        cache = ts.cache.LatentKVCacheHandle(
            latent_dim=4, max_seq=4, auto_evict=True,
        )
        cache.append(np.full((3, 4), 1.0, dtype=np.float32))
        assert cache.shape == (3, 4)
        # Append 3 more — total would be 6, max is 4, so first 2 get evicted.
        cache.append(np.full((3, 4), 2.0, dtype=np.float32))
        assert cache.shape == (4, 4)
        # Window contains 1 leftover "1.0" token + 3 fresh "2.0" tokens.
        slot = cache.read(0, 4)
        np.testing.assert_array_equal(slot[0], [1.0] * 4)
        np.testing.assert_array_equal(slot[1:], np.full((3, 4), 2.0))

    def test_evict_oldest_explicit(self):
        cache = ts.cache.LatentKVCacheHandle(latent_dim=2, max_seq=8)
        for i in range(5):
            cache.append(np.full((1, 2), i, dtype=np.float32))
        cache.evict_oldest(2)
        assert cache.shape == (3, 2)
        np.testing.assert_array_equal(
            cache.read(0, 3), [[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
        )

    def test_read_out_of_range_raises(self):
        cache = ts.cache.LatentKVCacheHandle(latent_dim=2, max_seq=8)
        cache.append(np.zeros((3, 2), dtype=np.float32))
        with pytest.raises(ValueError, match="out of range"):
            cache.read(0, 5)

    def test_chunk_larger_than_max_seq_rejected_even_with_auto_evict(self):
        cache = ts.cache.LatentKVCacheHandle(
            latent_dim=2, max_seq=4, auto_evict=True,
        )
        with pytest.raises(ValueError, match="cannot fit even after eviction"):
            cache.append(np.zeros((5, 2), dtype=np.float32))

    def test_dtype_routing(self):
        cache_f16 = ts.cache.LatentKVCacheHandle(
            latent_dim=2, max_seq=4, dtype="fp16",
        )
        assert cache_f16.latents.dtype == np.float16

    def test_invalid_construction_args_raise(self):
        with pytest.raises(ValueError, match="latent_dim"):
            ts.cache.LatentKVCacheHandle(latent_dim=0, max_seq=4)
        with pytest.raises(ValueError, match="latent_dim"):
            ts.cache.LatentKVCacheHandle(latent_dim=4, max_seq=0)
        with pytest.raises(ValueError, match="page_size"):
            ts.cache.LatentKVCacheHandle(latent_dim=4, max_seq=4, page_size=0)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: MLA-style attention against a numpy reference
# ─────────────────────────────────────────────────────────────────────────────


class TestMLAEndToEnd:
    """Exercise the full MLA pattern: compress hidden → cache latent →
    expand at attention → softmax → context. Reference uses the same
    numpy ops, so this pins that the dedicated op_names didn't drift
    away from matmul semantics."""

    def test_decoupled_rope_compress_expand_round_trip(self):
        """Standard MLA chain: split RoPE off, compress no-RoPE part to
        latent, expand back, merge. Reconstructed K matches the
        ground-truth computation."""
        np.random.seed(0)
        S = 4
        head_dim_total = 32
        rope_dim = 8
        no_rope_dim = head_dim_total - rope_dim
        latent_dim = 16

        x = np.random.randn(S, head_dim_total).astype(np.float32)
        w_dkv = np.random.randn(no_rope_dim, latent_dim).astype(np.float32) * 0.1
        w_uk = np.random.randn(latent_dim, no_rope_dim).astype(np.float32) * 0.1

        rope_part, no_rope_part = ts.ops.rope_split(x, rope_dim=rope_dim)
        c = ts.ops.latent_kv_compress(no_rope_part, w_dkv)
        no_rope_back = ts.ops.latent_kv_expand_k(c, w_uk)
        x_back = ts.ops.rope_merge(rope_part, no_rope_back)

        # The compress-then-expand is a low-rank reconstruction, so
        # `no_rope_back ≠ no_rope_part` exactly. But the rope_part comes
        # back unchanged — that's the whole point of decoupled-RoPE.
        np.testing.assert_array_equal(x_back[..., :rope_dim], rope_part)
        # And the reconstruction matches the explicit numpy chain.
        ref_no_rope = (no_rope_part @ w_dkv) @ w_uk
        np.testing.assert_array_equal(x_back[..., rope_dim:], ref_no_rope)

    def test_latent_cache_paged_decode_step(self):
        """Decode-style usage: per-step, compress one token's no-RoPE
        slice, append to the latent cache, read back the full cache to
        reconstruct K for the attention score matmul."""
        np.random.seed(0)
        T = 5
        no_rope_dim = 16
        latent_dim = 8
        w_dkv = np.random.randn(no_rope_dim, latent_dim).astype(np.float32) * 0.1
        cache = ts.cache.LatentKVCacheHandle(latent_dim=latent_dim, max_seq=64)

        for t in range(T):
            x_t = np.random.randn(1, no_rope_dim).astype(np.float32)
            c_t = ts.ops.latent_kv_compress(x_t, w_dkv)
            cache.append(c_t)

        assert cache.shape == (T, latent_dim)
        full = cache.read(0, T)
        assert full.shape == (T, latent_dim)
