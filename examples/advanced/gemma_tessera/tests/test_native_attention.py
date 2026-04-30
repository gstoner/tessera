"""
Tests for native_flash_attention and paged attention kernels.

Runs on CPU when CUDA is unavailable; tolerances are widened accordingly.
"""
import pytest
import torch
from tessera_gemma.kernels.native_attention_tessera import native_flash_attention, _repeat_kv
from tessera_gemma.kernels.native_attention_paged_tessera import native_flash_attention_paged
from tessera_gemma.kernels.kv_cache_tessera import PagedKVCache


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATOL = 2e-2 if DEVICE == "cuda" else 4e-2
RTOL = 2e-2


# ---------------------------------------------------------------------------
# _repeat_kv
# ---------------------------------------------------------------------------
class TestRepeatKV:
    def test_no_repeat(self):
        k = torch.randn(2, 16, 8, 64)
        v = torch.randn_like(k)
        k2, v2 = _repeat_kv(k, v, 8, 8)
        assert k2 is k and v2 is v

    def test_repeat_x2(self):
        k = torch.randn(2, 16, 4, 64)
        v = torch.randn_like(k)
        k2, v2 = _repeat_kv(k, v, 8, 4)
        assert k2.shape == (2, 16, 8, 64)
        # First two Q-head groups should see the same KV
        assert torch.equal(k2[:, :, 0], k2[:, :, 1])
        assert torch.equal(k2[:, :, 2], k2[:, :, 3])

    def test_repeat_x4(self):
        k = torch.randn(1, 8, 2, 32)
        v = torch.randn_like(k)
        k2, v2 = _repeat_kv(k, v, 8, 2)
        assert k2.shape == (1, 8, 8, 32)


# ---------------------------------------------------------------------------
# native_flash_attention — causal, full attention
# ---------------------------------------------------------------------------
class TestNativeFlashAttention:
    @pytest.mark.parametrize("B,T,H,D", [
        (1, 64, 8, 64),
        (2, 128, 4, 64),
        (1, 32, 2, 128),
    ])
    def test_matches_sdpa(self, B, T, H, D):
        torch.manual_seed(42)
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        q = torch.randn(B, T, H, D, device=DEVICE, dtype=dtype)
        k = torch.randn(B, T, H, D, device=DEVICE, dtype=dtype)
        v = torch.randn(B, T, H, D, device=DEVICE, dtype=dtype)

        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True, dropout_p=0.0,
        ).transpose(1, 2)

        out = native_flash_attention(q, k, v, causal=True, dropout_p=0.0, block_size=32)
        assert torch.allclose(out.float(), ref.float(), atol=ATOL, rtol=RTOL), \
            f"max diff = {(out.float() - ref.float()).abs().max():.4f}"

    def test_output_shape(self):
        q = torch.randn(2, 64, 8, 64)
        k = torch.randn(2, 64, 8, 64)
        v = torch.randn_like(k)
        out = native_flash_attention(q, k, v, causal=True, block_size=64)
        assert out.shape == (2, 64, 8, 64)

    @pytest.mark.parametrize("block_size", [32, 64, 128])
    def test_block_size_invariance(self, block_size):
        torch.manual_seed(0)
        q = torch.randn(1, 64, 4, 32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        ref = native_flash_attention(q, k, v, causal=True, block_size=64)
        out = native_flash_attention(q, k, v, causal=True, block_size=block_size)
        assert torch.allclose(out, ref, atol=4e-2, rtol=4e-2)

    def test_gqa_with_repeat(self):
        torch.manual_seed(7)
        B, T, H, Hk, D = 1, 32, 8, 2, 64
        q = torch.randn(B, T, H, D)
        k = torch.randn(B, T, Hk, D)
        v = torch.randn(B, T, Hk, D)
        k_exp, v_exp = _repeat_kv(k, v, H, Hk)
        out = native_flash_attention(q, k_exp, v_exp, causal=True, block_size=32)
        assert out.shape == (B, T, H, D)


# ---------------------------------------------------------------------------
# Paged attention
# ---------------------------------------------------------------------------
class TestPagedAttention:
    def _make_paged(self, B, T, H, Hk, D, page_size=16):
        torch.manual_seed(99)
        k = torch.randn(B, T, Hk, D)
        v = torch.randn(B, T, Hk, D)
        cache = PagedKVCache(B, Hk, D, page_size=page_size)
        cache.append(k, v)
        return k, v, cache

    def test_paged_matches_full(self):
        B, T, H, Hk, D = 1, 48, 4, 2, 32
        k, v, cache = self._make_paged(B, T, H, Hk, D, page_size=16)
        q = torch.randn(B, T, H, D)
        k_exp, v_exp = _repeat_kv(k, v, H, Hk)

        ref  = native_flash_attention(q, k_exp, v_exp, causal=True, block_size=16)
        # paged: q against the cache pages (k/v already at full H in cache? No —
        # paged attn receives kv at Hk; we must expand first or pass raw)
        # Our paged kernel expects k_page/v_page at H (expanded). Re-expand:
        # Actually native_flash_attention_paged receives kv_pages as (k_page, v_page)
        # with shape (B, Tp, H, D) — expanded.  Build a page list from k_exp/v_exp.
        pages = [(k_exp[:, i*16:(i+1)*16], v_exp[:, i*16:(i+1)*16])
                 for i in range((T + 15) // 16)]
        out = native_flash_attention_paged(q, kv_pages=pages, causal=True, block_size=16)
        assert torch.allclose(out.float(), ref.float(), atol=5e-2, rtol=5e-2), \
            f"max diff = {(out - ref).abs().max():.4f}"

    def test_paged_output_shape(self):
        B, T, H, Hk, D = 2, 32, 4, 2, 32
        k, v, cache = self._make_paged(B, T, H, Hk, D, page_size=8)
        q = torch.randn(B, T, H, D)
        k_exp, v_exp = _repeat_kv(k, v, H, Hk)
        pages = [(k_exp, v_exp)]
        out = native_flash_attention_paged(q, kv_pages=pages, causal=True)
        assert out.shape == (B, T, H, D)

    def test_single_token_query(self):
        """Decode step: Tq = 1, full KV context."""
        B, Tc, H, D = 1, 64, 4, 32
        q = torch.randn(B, 1, H, D)
        k = torch.randn(B, Tc, H, D)
        v = torch.randn_like(k)
        pages = [(k, v)]
        out = native_flash_attention_paged(q, kv_pages=pages, causal=True)
        assert out.shape == (B, 1, H, D)


# ---------------------------------------------------------------------------
# PagedKVCache
# ---------------------------------------------------------------------------
class TestPagedKVCache:
    def test_append_and_len(self):
        cache = PagedKVCache(1, 4, 32, page_size=8)
        k = torch.randn(1, 10, 4, 32)
        v = torch.randn_like(k)
        cache.append(k, v)
        assert len(cache) == 10

    def test_multiple_appends(self):
        cache = PagedKVCache(1, 2, 16, page_size=4)
        for _ in range(3):
            k = torch.randn(1, 3, 2, 16)
            v = torch.randn_like(k)
            cache.append(k, v)
        assert len(cache) == 9

    def test_page_count(self):
        cache = PagedKVCache(1, 2, 16, page_size=4)
        cache.append(torch.randn(1, 7, 2, 16), torch.randn(1, 7, 2, 16))
        pages = list(cache.pages())
        assert len(pages) == 2  # ceil(7/4) = 2 pages

    def test_shape_mismatch_raises(self):
        cache = PagedKVCache(1, 2, 16, page_size=4)
        with pytest.raises(AssertionError):
            cache.append(torch.randn(1, 4, 3, 16), torch.randn(1, 4, 3, 16))

    def test_empty_cache_no_pages(self):
        cache = PagedKVCache(1, 2, 16, page_size=4)
        assert list(cache.pages()) == []
