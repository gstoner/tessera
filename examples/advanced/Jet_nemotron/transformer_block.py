from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple

# Tessera surface placeholders
from tessera import Tensor, jit, nn
from tessera.stdlib import rmsnorm_safe, dropout
from .jetblock import JetBlock, JetBlockConfig
from .full_attention import FullAttention, FullAttnConfig

AttnType = Literal["full", "jet"]

@dataclass
class TransformerConfig:
    d_model: int
    n_heads: int
    head_dim: int
    mlp_hidden: int
    n_layers: int
    attn_types: List[AttnType]  # length n_layers, "full" or "jet"
    dropout_p: float = 0.0
    dtype: str = "fp8_e4m3"
    accum: str = "fp32"

class MLP(nn.Module):
    def __init__(self, d_model: int, hidden: int, dtype: str, accum: str):
        super().__init__()
        self.up = nn.Linear(d_model, hidden, dtype=dtype, accum=accum)
        self.act = nn.SiLU()
        self.down = nn.Linear(hidden, d_model, dtype=dtype, accum=accum)

    def __call__(self, x: Tensor["B","S","D"]) -> Tensor["B","S","D"]:
        return self.down(self.act(self.up(x)))

class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        D, H, Dh = cfg.d_model, cfg.n_heads, cfg.head_dim
        attn_kind = cfg.attn_types[layer_idx]
        if attn_kind == "full":
            self.attn = FullAttention(FullAttnConfig(D, H, Dh, dtype=cfg.dtype, accum=cfg.accum))
        else:
            jcfg = JetBlockConfig(d_model=D, n_heads=H, head_dim=Dh,
                                  feature_map="elu1", conv_ks=7, gate="token",
                                  attn_dropout=cfg.dropout_p, dtype=cfg.dtype, accum=cfg.accum)
            self.attn = JetBlock(jcfg)
        self.mlp = MLP(D, cfg.mlp_hidden, cfg.dtype, cfg.accum)

    @jit
    def __call__(self, x: Tensor["B","S","D"], *, attn_state=None, causal: bool=True, streaming: bool=False):
        y, attn_state = self.attn(x, state=attn_state, causal=causal, streaming=streaming)
        x = x + dropout(y, p=self.cfg.dropout_p)
        z = self.mlp(rmsnorm_safe(x))
        x = x + dropout(z, p=self.cfg.dropout_p)
        return x, attn_state

class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = nn.RMSNorm(cfg.d_model)

    @jit
    def __call__(self, x: Tensor["B","S","D"], *, causal: bool=True, streaming: bool=False):
        states = [None] * self.cfg.n_layers
        for i,layer in enumerate(self.layers):
            x, states[i] = layer(x, attn_state=states[i], causal=causal, streaming=streaming)
        return self.final_norm(x), states

def swap_attention_kinds(model: Transformer, kinds: List[AttnType]):
    assert len(kinds) == len(model.layers)
    for i,k in enumerate(kinds):
        if isinstance(model.layers[i].attn, FullAttention) and k == "jet":
            cfg = model.layers[i].attn.cfg
            jcfg = JetBlockConfig(d_model=cfg.d_model, n_heads=cfg.n_heads,
                                  head_dim=cfg.head_dim, feature_map="elu1",
                                  conv_ks=7, gate="token", attn_dropout=0.0,
                                  dtype=cfg.dtype, accum=cfg.accum)
            model.layers[i].attn = JetBlock(jcfg)
        elif isinstance(model.layers[i].attn, JetBlock) and k == "full":
            jcfg = model.layers[i].attn.cfg
            fcfg = FullAttnConfig(d_model=jcfg.d_model, n_heads=jcfg.n_heads,
                                  head_dim=jcfg.head_dim, dtype=jcfg.dtype, accum=jcfg.accum)
            model.layers[i].attn = FullAttention(fcfg)
