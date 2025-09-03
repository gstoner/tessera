
import torch, torch.nn as nn
from einops import rearrange
from ..ops.rope import apply_rope
from .native_attention_tessera import native_flash_attention, _repeat_kv
from .native_attention_paged_tessera import native_flash_attention_paged
from .kv_cache_tessera import PagedKVCache
from ..utils.shapes import ShapeSpec, check_shape

class TesseraAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, rope_cache_fn=None, dropout_p: float=0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.dropout_p = dropout_p
        self.qkv = nn.Linear(hidden_size, hidden_size + 2 * (self.head_dim * num_kv_heads), bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rope_cache_fn = rope_cache_fn
        self._auto_blocks = {}

    def _choose_block(self, seq_len: int, head_dim: int, device: torch.device, dtype) -> int:
        key = (seq_len, head_dim, str(device), str(dtype))
        if key in self._auto_blocks:
            return self._auto_blocks[key]
        candidates = [64, 128, 256]
        x = torch.randn(1, seq_len, self.num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(1, seq_len, max(1,self.num_kv_heads), head_dim, device=device, dtype=dtype)
        v = torch.randn_like(k)
        k, v = _repeat_kv(k, v, self.num_heads, self.num_kv_heads)
        best = candidates[0]
        best_t = float("inf")
        for bs in candidates:
            if device.type == "cuda":
                torch.cuda.synchronize()
                t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                _ = native_flash_attention(x, k, v, causal=True, dropout_p=0.0, block_size=bs)
                t1.record(); torch.cuda.synchronize(); ms = t0.elapsed_time(t1)
            else:
                import time
                tstart = time.perf_counter()
                _ = native_flash_attention(x, k, v, causal=True, dropout_p=0.0, block_size=bs)
                ms = (time.perf_counter() - tstart) * 1000.0
            if ms < best_t:
                best_t, best = ms, bs
        self._auto_blocks[key] = best
        return best

    def forward(self, x, rope_cos=None, rope_sin=None, attn_mask=None, kv_cache: PagedKVCache = None, use_cache: bool = False, update_cache: bool = True):
        symbols = {}
        check_shape("x", tuple(x.shape), ShapeSpec(["B","T","C"]), symbols)

        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [self.hidden_size, self.head_dim * self.num_kv_heads, self.head_dim * self.num_kv_heads], dim=-1)
        q = rearrange(q, "b t (h d) -> b t h d", h=self.num_heads, d=self.head_dim)
        k = rearrange(k, "b t (h d) -> b t h d", h=self.num_kv_heads, d=self.head_dim)
        v = rearrange(v, "b t (h d) -> b t h d", h=self.num_kv_heads, d=self.head_dim)

        check_shape("q", tuple(q.shape), ShapeSpec(["B","T","H","D"]), symbols)
        check_shape("k", tuple(k.shape), ShapeSpec(["B","T","Hk","D"]), symbols)
        check_shape("v", tuple(v.shape), ShapeSpec(["B","T","Hk","D"]), symbols)

        if rope_cos is not None and rope_sin is not None:
            q = apply_rope(q, rope_cos[:, :T], rope_sin[:, :T])
            k = apply_rope(k, rope_cos[:, :T], rope_sin[:, :T])

        k, v = _repeat_kv(k, v, self.num_heads, self.num_kv_heads)
        block = self._choose_block(T, self.head_dim, x.device, x.dtype)

        if use_cache and kv_cache is not None:
            # Append new KV, then run paged attention over all pages
            if update_cache:
                kv_cache.append(k, v)
            # Query is only for the newly appended tokens by default; here we use all q for simplicity
            out = native_flash_attention_paged(q, kv_pages=list(kv_cache.pages()), causal=True, dropout_p=self.dropout_p, block_size=block)
        else:
            out = native_flash_attention(q, k, v, causal=True, dropout_p=self.dropout_p, block_size=block)
        out = rearrange(out, "b t h d -> b t (h d)")
        return self.proj(out)
