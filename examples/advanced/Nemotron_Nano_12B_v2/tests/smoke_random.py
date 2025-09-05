#!/usr/bin/env python3
import torch, numpy as np
from tessera.model.nemotron_h.hybrid_pattern import NemotronHConfig
from tessera.model.nemotron_h.model import NemotronH

def tiny_cfg():
    return NemotronHConfig(
        hidden_size=5120, intermediate_size=20480,
        num_attention_heads=40, num_key_value_heads=8, head_dim=128,
        num_hidden_layers=3, hybrid_override_pattern="M*-",
        mamba_num_heads=128, mamba_head_dim=80, ssm_state_size=128,
        conv_kernel=4, chunk_size=128, rms_norm_eps=1e-5,
        attention_bias=False, mlp_bias=False
    )

if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = NemotronH(cfg)
    x = torch.randint(0, 32000, (2, 16))  # small seq for smoke
    y = m(x)
    print("OK shapes:", tuple(y.shape))
