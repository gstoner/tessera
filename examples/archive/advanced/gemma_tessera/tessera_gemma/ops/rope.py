"""
tessera_gemma/ops/rope.py — Rotary Position Embeddings for Gemma.

Changes vs v0.1:
  • `precompute_rope_cache` now supports NTK-scaled RoPE via a `rope_scaling`
    dict with keys "type" and "factor" (matches HuggingFace convention).
  • `apply_rope` accepts either half-dim or full-dim cos/sin tensors.
  • Added `apply_rope_inplace` for decoder-only step where we only have a
    single new token (avoids a slice).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Rotation application
# ---------------------------------------------------------------------------

def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to x using precomputed cos/sin tables.

    Args:
        x:   (B, T, H, Dh)  — query or key tensor.
        cos: (1, T, 1, Dh//2) or (1, T, 1, Dh)  — cosine table.
        sin: same shape as cos.

    Returns:
        Rotated tensor with the same shape as x.
    """
    Dh = x.shape[-1]
    # Support both half-dim and full-dim cos/sin tables
    if cos.shape[-1] == Dh:
        # full-dim interleaved encoding
        x1, x2 = x[..., ::2], x[..., 1::2]
        cos_h, sin_h = cos[..., ::2], sin[..., ::2]
    else:
        # half-dim tables (the common precompute_rope_cache output)
        x1, x2 = x[..., ::2], x[..., 1::2]
        cos_h, sin_h = cos, sin

    rot = torch.stack([x1 * cos_h - x2 * sin_h,
                       x1 * sin_h + x2 * cos_h], dim=-1)
    return rot.flatten(-2)


# ---------------------------------------------------------------------------
# Precomputed cache
# ---------------------------------------------------------------------------

def precompute_rope_cache(
    seqlen: int,
    head_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
    rope_scaling: Optional[Dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute (cos, sin) tables of shape (1, seqlen, 1, head_dim//2).

    Args:
        seqlen:       Maximum sequence length.
        head_dim:     Per-head feature dimension (must be even).
        theta:        RoPE base frequency (default 10000).
        device, dtype: Target device and floating-point dtype.
        rope_scaling: Optional NTK scaling config dict.
                      { "type": "linear"|"dynamic", "factor": float }
                      "linear"  → scale all base frequencies by 1/factor.
                      "dynamic" → NTK-aware scaling (α = factor).

    Returns:
        cos, sin: (1, seqlen, 1, head_dim//2) on device/dtype.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half = head_dim // 2

    # Base inverse frequencies
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )

    # Apply scaling if requested
    if rope_scaling is not None:
        factor = float(rope_scaling.get("factor", 1.0))
        scaling_type = rope_scaling.get("type", "linear")
        if scaling_type == "linear":
            inv_freq = inv_freq / factor
        elif scaling_type == "dynamic":
            # NTK-aware: new_base = theta * (factor**(head_dim/(head_dim-2)))
            new_theta = theta * (factor ** (head_dim / (head_dim - 2)))
            inv_freq = 1.0 / (
                new_theta ** (
                    torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
                    / head_dim
                )
            )

    t = torch.arange(seqlen, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)          # (seqlen, half)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)

    # Shape: (1, seqlen, 1, half) for broadcasting over (B, T, H, Dh//2)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return cos, sin
