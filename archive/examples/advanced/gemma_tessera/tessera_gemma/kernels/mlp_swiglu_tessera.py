"""
tessera_gemma/kernels/mlp_swiglu_tessera.py

MLP block for Gemma with support for both SwiGLU and GeGLU activations.

Gemma 4 uses GeGLU (gate × GELU, not gate × SiLU).
Gemma 2/3 and the original Gemma use SwiGLU (gate × SiLU = Swish).

Architecture (all variants):
    gate(x) = Linear(hidden_size → intermediate_size, bias=False)
    up(x)   = Linear(hidden_size → intermediate_size, bias=False)
    down(y) = Linear(intermediate_size → hidden_size, bias=False)

    SwiGLU: y = down(silu(gate(x)) * up(x))
    GeGLU:  y = down(gelu(gate(x)) * up(x))

Note: the v0.1 implementation packed gate+up into a single `wi` (2× wide)
and called it `wo`.  That packing is still optionally supported for weight
compatibility but the canonical interface now uses separate projections.

Changes vs v0.1:
  • Separate `gate_proj` / `up_proj` / `down_proj` matching HF naming.
  • `mlp_type` param selects "swiglu" or "geglu".
  • `_tessera_op` marker for Tessera graph export.
  • `from_packed(wi, wo)` class-method for loading old checkpoint format.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

MlpKind = Literal["swiglu", "geglu"]


class GemmaMLP(nn.Module):
    """Gated MLP (SwiGLU or GeGLU) matching Gemma 2/3/4 architecture."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mlp_type: MlpKind = "geglu",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.mlp_type = mlp_type

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # Tessera compiler annotation
        self._tessera_op = "tessera.elementwise.mlp_gate"

    def _gate_act(self, x: torch.Tensor) -> torch.Tensor:
        if self.mlp_type == "geglu":
            return F.gelu(x, approximate="tanh")
        # default: swiglu = silu
        return F.silu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (*, hidden_size) — arbitrary batch/sequence prefix."""
        gate = self._gate_act(self.gate_proj(x))
        up   = self.up_proj(x)
        return self.down_proj(gate * up)

    # -----------------------------------------------------------------------
    # Checkpoint compatibility: load from old packed wi/wo format
    # -----------------------------------------------------------------------
    @classmethod
    def from_packed(
        cls,
        wi_weight: torch.Tensor,
        wo_weight: torch.Tensor,
        mlp_type: MlpKind = "swiglu",
    ) -> "GemmaMLP":
        """Construct from old v0.1 checkpoint where wi = [gate, up] packed."""
        intermediate_size = wi_weight.shape[0] // 2
        hidden_size = wo_weight.shape[0]
        m = cls(hidden_size, intermediate_size, mlp_type=mlp_type)
        with torch.no_grad():
            gate_w, up_w = wi_weight.chunk(2, dim=0)
            m.gate_proj.weight.copy_(gate_w)
            m.up_proj.weight.copy_(up_w)
            m.down_proj.weight.copy_(wo_weight)
        return m

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_size}, intermediate={self.intermediate_size}, "
            f"type={self.mlp_type}"
        )


# ---------------------------------------------------------------------------
# Back-compat alias — old code imports `SwiGLU`
# ---------------------------------------------------------------------------
class SwiGLU(GemmaMLP):
    """Backward-compatible alias. Prefer GemmaMLP(mlp_type='swiglu')."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__(hidden_size, intermediate_size, mlp_type="swiglu")
