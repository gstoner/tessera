"""
High-level Nemotron‑H model skeleton (reference execution).
Swap modules for Tessera Graph/Tile backed kernels in your repo.
"""
from .hybrid_pattern import NemotronHConfig, build_hybrid_stack
from .ops import RMSNorm, MLPReLU2, AttentionGQA, Mamba2MixerStub
import torch, torch.nn as nn

class NemotronH(nn.Module):
    def __init__(self, cfg: NemotronHConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size if hasattr(cfg, "vocab_size") else 131072, cfg.hidden_size)
        self.blocks = nn.ModuleList([])
        for ch in cfg.hybrid_override_pattern:
            if ch == "M":
                self.blocks.append(Mamba2MixerStub(cfg.hidden_size, cfg.mamba_num_heads, cfg.mamba_head_dim, cfg.ssm_state_size, cfg.conv_kernel, cfg.chunk_size))
            elif ch == "*":
                self.blocks.append(AttentionGQA(cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, bias=cfg.attention_bias))
            else:
                self.blocks.append(MLPReLU2(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, getattr(cfg, "vocab_size", 131072), bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = x + blk(x) if hasattr(blk, "forward") else x
        x = self.norm(x)
        return self.lm_head(x)
