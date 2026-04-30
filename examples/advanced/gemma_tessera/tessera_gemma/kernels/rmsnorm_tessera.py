"""
tessera_gemma/kernels/rmsnorm_tessera.py

RMSNorm for Gemma — used both as a pre-norm and a post-norm.

Changes vs v0.1:
  • Accepts an explicit `elementwise_affine` flag (True = learnable weight).
  • `forward` works for any trailing-dim layout: (..., dim).
  • When `use_tessera_compile` is set on the parent config, the forward is
    annotated so it can be JIT-compiled by the Tessera compiler as a fused
    elementwise kernel (tessera.elementwise graph op).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Norm, no mean subtraction.

    Gemma uses this with learnable per-channel scale (weight) and no bias.
    eps is added *inside* the sqrt to match the HuggingFace implementation.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(dim, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("weight", None)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps:.2e}, affine={self.elementwise_affine}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., dim)  — arbitrary batch / sequence prefix dimensions.
        Returns the same shape as input.
        """
        # Compute RMS in float32 for numerical stability even when x is bf16/f16
        x_fp32 = x.to(torch.float32)
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x_normed = (x_fp32 * rms).to(x.dtype)
        if self.weight is not None:
            return self.weight * x_normed
        return x_normed


# ---------------------------------------------------------------------------
# Tessera compiler annotation wrapper
# ---------------------------------------------------------------------------
# When `cfg.use_tessera_compile` is True the model builder calls
# `tessera_rmsnorm(cfg, dim)` instead of RMSNorm(dim) directly.  In the
# future this will route through the @tessera.jit decorator; for now it just
# returns a plain RMSNorm tagged with a `_tessera_op` attribute that the
# inspection utilities can detect.

def tessera_rmsnorm(dim: int, eps: float = 1e-6) -> RMSNorm:
    """Factory that creates an RMSNorm tagged for Tessera-compiler lowering."""
    m = RMSNorm(dim, eps)
    m._tessera_op = "tessera.elementwise.rmsnorm"  # marker for graph export
    return m
