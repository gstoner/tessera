
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

# Tessera surface imports (modeled)
from tessera import Tensor, jit, nn
from tessera.stdlib import rmsnorm_safe
from .dynamic_linear_attention import lin_attn_forward, StreamingState
from .advanced_feature_maps import make_gaussian_proj_per_head, generate_rf_proj_on_device, RFState, linear_attention_rf

GateType = Literal["token", "head"]

@dataclass
class JetBlockConfig:
    d_model: int
    n_heads: int
    head_dim: int
    feature_map: Literal["elu1", "relu2", "favor"] = "elu1"
    conv_ks: int = 7
    gate: GateType = "token"
    attn_dropout: float = 0.0
    dtype: str = "fp8_e4m3"
    accum: str = "fp32"
    gate_hidden: int = 0

\1        self.Wq_rf_heads = None
        self.Wk_rf_heads = None
        self.rf_on_device = False
        H, Dh = cfg.n_heads, cfg.head_dim
        D = cfg.d_model
        self.Wq = nn.Linear(D, H*Dh, dtype=cfg.dtype, accum=cfg.accum)
        self.Wk = nn.Linear(D, H*Dh, dtype=cfg.dtype, accum=cfg.accum)
        self.Wv = nn.Linear(D, H*Dh, dtype=cfg.dtype, accum=cfg.accum)
        self.Wo = nn.Linear(H*Dh, D, dtype=cfg.dtype, accum=cfg.accum)
        self.dwconv = nn.DynamicDepthwiseConv1d(
            channels=H*Dh, kernel_size=cfg.conv_ks, groups=H,
            dtype=cfg.dtype, accum=cfg.accum)
        if cfg.gate == "token":
            self.gate = nn.Sequential(
                nn.Linear(D, D//4), nn.SiLU(), nn.Linear(D//4, 1), nn.Sigmoid()
            )
        else:
            self.gate = nn.Sequential(
                nn.Linear(D, D//4), nn.SiLU(), nn.Linear(D//4, H), nn.Sigmoid()
            )

    @jit
    def __call__(self, x: Tensor["B","S","D"], *, state: Optional[StreamingState]=None,
                 causal: bool=True, streaming: bool=False) -> Tuple[Tensor["B","S","D"], Optional[StreamingState]]:
        B,S,D = x.shape
        H, Dh = self.cfg.n_heads, self.cfg.head_dim
        xn = rmsnorm_safe(x)
        q = self.Wq(xn).reshape(B,S,H,Dh)
        k = self.Wk(xn).reshape(B,S,H,Dh)
        v = self.Wv(xn).reshape(B,S,H,Dh)
        y_attn, state = lin_attn_forward(q, k, v, feature_map=self.cfg.feature_map,
                                         dropout_p=self.cfg.attn_dropout, causal=causal,
                                         state=state, streaming=streaming)
        v_channels = v.reshape(B,S,H*Dh).transpose(0,2,1)
        y_conv = self.dwconv(v_channels).transpose(0,2,1).reshape(B,S,H,Dh)
        if self.cfg.gate == "token":
            g = self.gate(xn).reshape(B,S,1,1)
        else:
            g = self.gate(xn).reshape(B,S,H,1)
        y_mix = g * y_attn + (1 - g) * y_conv
        y = y_mix.reshape(B,S,H*Dh)
        y = self.Wo(y)
        return y, state
