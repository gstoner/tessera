from dataclasses import dataclass
from typing import Optional, Tuple

from tessera import Tensor, jit, nn
from tessera.stdlib import softmax_safe, dropout, rmsnorm_safe

@dataclass
class FullAttnConfig:
    d_model: int
    n_heads: int
    head_dim: int
    dtype: str = "bf16"
    accum: str = "fp32"

class FullAttention(nn.Module):
    def __init__(self, cfg: FullAttnConfig):
        super().__init__()
        self.cfg = cfg
        D, H, Dh = cfg.d_model, cfg.n_heads, cfg.head_dim
        self.Wq = nn.Linear(D, H*Dh, dtype=cfg.dtype, accum=cfg.accum)
        self.Wk = nn.Linear(D, H*Dh, dtype=cfg.dtype, accum=cfg.accum)
        self.Wv = nn.Linear(D, H*Dh, dtype=cfg.dtype, accum=cfg.accum)
        self.Wo = nn.Linear(H*Dh, D, dtype=cfg.dtype, accum=cfg.accum)

    @jit
    def __call__(self, x: Tensor["B","S","D"], *, state=None, causal: bool=True, streaming: bool=False):
        B,S,D = x.shape
        H, Dh = self.cfg.n_heads, self.cfg.head_dim
        xn = rmsnorm_safe(x)
        q = self.Wq(xn).reshape(B,S,H,Dh)
        k = self.Wk(xn).reshape(B,S,H,Dh)
        v = self.Wv(xn).reshape(B,S,H,Dh)
        scale = 1.0 / (Dh ** 0.5)
        attn = softmax_safe((q @ k.transpose(-2,-1)) * scale, causal=causal)
        y = (attn @ v).reshape(B,S,H*Dh)
        y = self.Wo(y)
        return y, None
