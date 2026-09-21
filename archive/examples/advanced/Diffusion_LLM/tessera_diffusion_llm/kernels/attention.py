"""
tessera_diffusion_llm/kernels/attention.py

Bidirectional multi-head attention for diffusion transformers.

Key difference from autoregressive attention: causal=False — all token
positions attend to all other positions.  This is correct for diffusion
denoising where the model sees the entire (partially noised) sequence.

Supports:
  • GQA (num_kv_heads < num_heads)
  • Tessera compiler annotation path (use_tessera_compile flag)
  • Flash attention via torch.nn.functional.scaled_dot_product_attention
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _repeat_kv(k: torch.Tensor, v: torch.Tensor,
                num_heads: int, num_kv_heads: int):
    """Expand KV heads to match Q heads for GQA."""
    if num_kv_heads == num_heads:
        return k, v
    reps = num_heads // num_kv_heads
    return k.repeat_interleave(reps, dim=2), v.repeat_interleave(reps, dim=2)


class BidirectionalAttention(nn.Module):
    """
    Full (non-causal) multi-head attention with GQA support.

    Shapes throughout:
        x:  (B, T, hidden_size)
        q:  (B, T, num_heads,    head_dim)
        k:  (B, T, num_kv_heads, head_dim)
        v:  (B, T, num_kv_heads, head_dim)
        out:(B, T, hidden_size)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = head_dim
        self.dropout_p    = dropout_p

        q_dim  = num_heads    * head_dim
        kv_dim = num_kv_heads * head_dim

        self.q_proj = nn.Linear(hidden_size, q_dim,  bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, hidden_size,  bias=False)

        # Tessera compiler marker — bidirectional, causal=False
        self._tessera_op = "tessera.flash_attn"
        self._tessera_causal = False

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:         (B, T, hidden_size)
            attn_mask: Optional (B, T, T) or (T, T) boolean mask.
                       True = position to attend to (PyTorch SDPA convention).
        Returns:
            (B, T, hidden_size)
        """
        B, T, _ = x.shape

        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV for GQA
        if self.num_kv_heads != self.num_heads:
            reps = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)

        # Use PyTorch SDPA (routes to Flash Attention when available)
        # is_causal=False → bidirectional attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )

        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)
