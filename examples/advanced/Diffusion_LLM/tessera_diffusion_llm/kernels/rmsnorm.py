"""
tessera_diffusion_llm/kernels/rmsnorm.py

RMSNorm and Adaptive Layer Norm (adaLN) for diffusion transformers.

adaLN-Single (DiT-style)
-------------------------
Instead of a learned γ/β per layer, the time-step MLP outputs a single
(scale, shift, gate) triplet that is broadcast across all layers.  In
practice each block has a small projection head:

    (α, β, γ) = Linear(time_emb, 3 * hidden_size)
    h = RMSNorm(x)
    h = h * (1 + α.unsqueeze(1)) + β.unsqueeze(1)
    h = attn(h) * γ.unsqueeze(1)          ← gate on attn output

This file provides:
  • RMSNorm          — plain learnable RMS normalisation
  • AdaLNModulation  — projects time_emb → (scale, shift[, gate])
  • adaLN_forward    — applies scale+shift to normed input
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Norm (no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self._tessera_op = "tessera.elementwise.rmsnorm"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for numerical stability
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(x.dtype) * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps:.2e}"


class AdaLNModulation(nn.Module):
    """Projects a time (or condition) embedding into scale+shift [+gate] tensors.

    Args:
        hidden_size: Size of the transformer hidden dimension.
        cond_dim:    Dimension of the conditioning embedding.
        num_outputs: 2 = (scale, shift); 3 = (scale, shift, gate).
    """

    def __init__(
        self,
        hidden_size: int,
        cond_dim: int,
        num_outputs: int = 2,
    ) -> None:
        super().__init__()
        assert num_outputs in (2, 3)
        self.num_outputs = num_outputs
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, num_outputs * hidden_size, bias=True),
        )
        # Zero-init so the block starts as identity at t=0
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            cond: (B, cond_dim)
        Returns:
            Tuple of (scale, shift) or (scale, shift, gate), each (B, hidden_size).
        """
        out = self.proj(cond)
        return out.chunk(self.num_outputs, dim=-1)


def adaLN_forward(
    x: torch.Tensor,
    norm: RMSNorm,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """Apply adaLN: norm(x) * (1 + scale[:, None]) + shift[:, None].

    Args:
        x:     (B, T, D)
        norm:  RMSNorm module
        scale: (B, D)
        shift: (B, D)
    Returns:
        (B, T, D)
    """
    h = norm(x)
    return h * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
