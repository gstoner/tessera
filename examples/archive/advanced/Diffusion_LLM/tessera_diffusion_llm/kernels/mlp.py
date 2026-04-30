"""
tessera_diffusion_llm/kernels/mlp.py

Gated MLP (SwiGLU / GeGLU) for diffusion transformer blocks.
Identical design to GemmaMLP — separate gate/up/down projections.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionMLP(nn.Module):
    """
    Gated MLP: y = down(act(gate(x)) * up(x))

    SwiGLU: act = silu   (Gemma2/3 default)
    GeGLU:  act = gelu   (Gemma4, DiT default)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mlp_type: Literal["swiglu", "geglu"] = "swiglu",
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.mlp_type  = mlp_type
        self._tessera_op = "tessera.elementwise.mlp_gate"

    def _gate_act(self, x: torch.Tensor) -> torch.Tensor:
        if self.mlp_type == "geglu":
            return F.gelu(x, approximate="tanh")
        return F.silu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (*, hidden_size) — any batch / sequence prefix."""
        return self.down_proj(self._gate_act(self.gate_proj(x)) * self.up_proj(x))
