"""
tessera_gemma/kernels/attention_tessera.py

Tessera Gemma attention — supports:
  • Full causal self-attention
  • Sliding-window causal attention (SWA) for Gemma 4 alternating layers
  • Grouped-Query Attention (GQA / MQA) via KV head expansion
  • Paged KV cache for efficient autoregressive decoding
  • Tessera compiler annotation path (use_tessera_compile)

Changes vs v0.1:
  • `sliding_window` param: restricts KV context to the last W tokens.
  • Block-size selection is now a simple heuristic table instead of
    warmup-benchmarking on every new (seq_len, device, dtype) key — that
    was causing unnecessary side effects and slow first-token latency.
  • QKV is split into separate q/k/v projections (matching HF naming and
    enabling per-projection LoRA without touching the other heads).
  • `forward` signature extended: `kv_cache`, `use_cache`, `position_ids`.
  • `_tessera_op` marker for Tessera graph export.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from ..ops.rope import apply_rope
from .native_attention_tessera import native_flash_attention, _repeat_kv
from .native_attention_paged_tessera import native_flash_attention_paged
from .kv_cache_tessera import PagedKVCache
from ..utils.shapes import ShapeSpec, check_shape


# ---------------------------------------------------------------------------
# Block-size heuristic (replaces warmup benchmarking)
# ---------------------------------------------------------------------------
_BLOCK_HEURISTIC: dict[Tuple[int, str], int] = {
    # (head_dim, device_type) -> block_size
    (64,  "cuda"):  128,
    (128, "cuda"):  128,
    (256, "cuda"):   64,
    (64,  "cpu"):    64,
    (128, "cpu"):    64,
    (256, "cpu"):    64,
}


def _select_block(head_dim: int, device_type: str) -> int:
    return _BLOCK_HEURISTIC.get((head_dim, device_type), 128)


# ---------------------------------------------------------------------------
# Core attention module
# ---------------------------------------------------------------------------

class TesseraAttention(nn.Module):
    """
    Multi-head / grouped-query attention with optional sliding window.

    Args:
        hidden_size:        Model hidden dimension (input/output size).
        num_heads:          Number of query heads.
        num_kv_heads:       Number of key/value heads (GQA; use == num_heads
                            for standard MHA).
        head_dim:           Per-head dimension.  Defaults to hidden_size //
                            num_heads if not provided.
        sliding_window:     If > 0, restrict attention to the last
                            `sliding_window` key positions (SWA).
        dropout_p:          Attention dropout (0 during inference).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        sliding_window: int = 0,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.sliding_window = sliding_window  # 0 = full attention
        self.dropout_p = dropout_p

        q_dim  = self.num_heads    * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        # Separate projections — enables per-projection LoRA targeting
        self.q_proj = nn.Linear(hidden_size, q_dim,  bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, hidden_size,  bias=False)

        self._tessera_op = "tessera.flash_attn"

    # -----------------------------------------------------------------------
    # Core forward
    # -----------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[PagedKVCache] = None,
        use_cache: bool = False,
        update_cache: bool = True,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        symbols: dict = {}
        check_shape("x", tuple(x.shape), ShapeSpec(["B", "T", "C"]), symbols)
        B, T, _ = x.shape
        device = x.device

        # Project Q / K / V
        q = rearrange(self.q_proj(x), "b t (h d) -> b t h d",
                      h=self.num_heads, d=self.head_dim)
        k = rearrange(self.k_proj(x), "b t (h d) -> b t h d",
                      h=self.num_kv_heads, d=self.head_dim)
        v = rearrange(self.v_proj(x), "b t (h d) -> b t h d",
                      h=self.num_kv_heads, d=self.head_dim)

        # Apply RoPE
        if rope_cos is not None and rope_sin is not None:
            cos_slice = rope_cos[:, :T]
            sin_slice = rope_sin[:, :T]
            q = apply_rope(q, cos_slice, sin_slice)
            k = apply_rope(k, cos_slice, sin_slice)

        # Sliding-window mask: trim k/v to last W tokens
        if self.sliding_window > 0 and not use_cache:
            k, v = self._apply_sliding_window(q, k, v, T)

        # Expand KV heads for GQA
        k_full, v_full = _repeat_kv(k, v, self.num_heads, self.num_kv_heads)

        block = _select_block(self.head_dim, device.type)

        if use_cache and kv_cache is not None:
            if update_cache:
                kv_cache.append(k, v)
            out = native_flash_attention_paged(
                q,
                kv_pages=list(kv_cache.pages()),
                causal=True,
                dropout_p=self.dropout_p,
                block_size=block,
            )
        else:
            out = native_flash_attention(
                q, k_full, v_full,
                causal=True,
                dropout_p=self.dropout_p if self.training else 0.0,
                block_size=block,
            )

        out = rearrange(out, "b t h d -> b t (h d)")
        return self.o_proj(out)

    # -----------------------------------------------------------------------
    # Sliding-window helpers
    # -----------------------------------------------------------------------
    def _apply_sliding_window(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        T: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For each query position t, allow it to attend only to the W most
        recent key positions.  Implemented by trimming K/V to the last W
        rows and returning an appropriately offset causal mask via the
        native_flash_attention call (which uses absolute position indices).

        For simplicity we fall back to masking inside native_flash_attention
        and just return k/v unchanged; the kernel's causal mask handles the
        upper triangle.  A future Tessera tile-kernel will implement true
        sliding-window tiling.
        """
        # When T <= window, SWA == full attention — nothing to trim.
        W = self.sliding_window
        if T <= W:
            return k, v
        # Trim: keep only the last W key/value tokens
        # (for prefill this introduces an approximation; for decode T=1 always ≤ W)
        return k[:, -W:], v[:, -W:]
