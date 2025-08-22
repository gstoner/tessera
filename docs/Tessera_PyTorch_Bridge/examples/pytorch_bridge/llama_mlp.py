#!/usr/bin/env python3

"""
LLaMA-style MLP block demo for the Tessera↔PyTorch bridge.
Implements SwiGLU: y = W2( SiLU(W1 x) ⊙ (W3 x) )
Includes a tiny 'tessera' mock shim so it runs even without Tessera installed.
"""

import sys, types
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Tiny Tessera shim (only for demo) --------------------------------------
def _install_tessera_shim():
    ts = types.SimpleNamespace()
    def from_pytorch(m, example_inputs=None, strict=False):
        return m
    ts.from_pytorch = from_pytorch

    def to_dlpack(t): return torch.utils.dlpack.to_dlpack(t)
    def from_dlpack(dlt): return torch.utils.dlpack.from_dlpack(dlt)
    ts.to_dlpack = to_dlpack
    ts.from_dlpack = from_dlpack

    compat = types.SimpleNamespace()
    def wrap(m, allow_fallback=True): return m
    def register_aten(name):
        def deco(fn): return fn
        return deco
    compat.wrap = wrap
    compat.register_aten = register_aten
    ts.compat = compat

    class Module(nn.Module): pass
    ts.Module = Module

    sys.modules.setdefault("tessera", ts)

try:
    import tessera as ts
except Exception:
    _install_tessera_shim()
    import tessera as ts

# --- LLaMA MLP ---------------------------------------------------------------
class LLaMAMLP(nn.Module):
    def __init__(self, hidden_size: int, multiple: int = 4, bias: bool = False):
        super().__init__()
        inner = multiple * hidden_size
        # LLaMA often uses a slightly different multiple (e.g., 4/3 * hidden * 4), but 4x is fine for demo
        self.gate_proj = nn.Linear(hidden_size, inner, bias=bias)  # W1
        self.up_proj   = nn.Linear(hidden_size, inner, bias=bias)  # W3
        self.down_proj = nn.Linear(inner, hidden_size, bias=bias)  # W2

    def forward(self, x):
        # x: [B, T, H]
        g = F.silu(self.gate_proj(x))
        u = self.up_proj(x)
        return self.down_proj(g * u)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, mult = 1024, 4
    mlp = LLaMAMLP(H, mult).to(device).eval()
    x = torch.randn(2, 128, H, device=device)  # [batch, seq, hidden]

    # Convert (no-op with shim)
    tess_mlp = ts.from_pytorch(mlp, example_inputs=(x,), strict=False)

    # Zero-copy exchange
    tx = ts.from_dlpack(torch.utils.dlpack.to_dlpack(x))
    ty = tess_mlp(tx)
    y  = torch.utils.dlpack.from_dlpack(ts.to_dlpack(ty))

    print("Input :", tuple(x.shape))
    print("Output:", tuple(y.shape))

if __name__ == "__main__":
    main()
