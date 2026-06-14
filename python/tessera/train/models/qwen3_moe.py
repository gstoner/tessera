"""Qwen3-MoE — a self-contained, directly-instantiated MoE decoder.

This is the canonical example of principle #3 (no implicit indirection): the
entire model — config, decoder block, and stack — is in THIS file and built by
plain ``nn.Module`` instantiation. There is no ``ModuleSpec``, no submodule
registry, no string-keyed resolution. What runs at any call site is identifiable
by reading top-to-bottom.

Architecture (Qwen3-MoE-style, pre-norm):
    block(x) = x + attn(rmsnorm(x));  block += moe(rmsnorm(block))
    model(ids) = lm_head(rmsnorm(stack(embed(ids))))

Runs today on numpy (and ``@jit(target="apple_gpu")`` for the matmul-heavy
sublayers). To port a *different* architecture (MoBA, DynMoE, ...), copy this
file and edit it in place — see the ``add-moe-model`` skill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tessera import nn
from tessera.train.engine.moe import MoEFeedForward


@dataclass(frozen=True)
class Qwen3MoEConfig:
    vocab_size: int = 1024
    hidden_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    num_experts: int = 8
    top_k: int = 2
    expert_intermediate: int = 512
    shared_intermediate: int = 256
    rms_eps: float = 1e-6
    dtype: str = "fp32"


class Qwen3MoEBlock(nn.Module):
    """One pre-norm decoder block: causal self-attention + MoE FFN."""

    def __init__(self, cfg: Qwen3MoEConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_eps, dtype=cfg.dtype)
        self.self_attn = nn.MultiHeadAttention(
            cfg.hidden_size, cfg.num_heads, bias=False, dtype=cfg.dtype
        )
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_eps, dtype=cfg.dtype)
        self.mlp = MoEFeedForward(
            cfg.hidden_size, cfg.num_experts, cfg.top_k,
            cfg.expert_intermediate, cfg.shared_intermediate,
            dtype=cfg.dtype, seed=layer_idx,
        )

    def forward(self, x):
        # x: (B, S, H)
        h = self.self_attn(self.input_layernorm(x), causal=True)
        x = np.asarray(x) + np.asarray(h)
        moe_out, aux = self.mlp(self.post_attention_layernorm(x))
        x = x + np.asarray(moe_out)
        return x, aux


class Qwen3MoEModel(nn.Module):
    """Embedding → L decoder blocks → final norm → LM head.

    ``forward(ids)`` returns ``(logits (B,S,V), aux_losses)`` where
    ``aux_losses`` is the summed load-balancing + router-z loss across layers,
    ready to add (scaled) to the next-token loss in the training loop.
    """

    def __init__(self, cfg: Qwen3MoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=cfg.dtype)
        self.layers = nn.ModuleList([Qwen3MoEBlock(cfg, i) for i in range(cfg.num_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_eps, dtype=cfg.dtype)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, dtype=cfg.dtype)

    def forward(self, ids):
        x = self.embed_tokens(np.asarray(ids, dtype=np.int64))   # (B, S, H)
        lb_loss = 0.0
        z_loss = 0.0
        for layer in self.layers:
            x, aux = layer(x)
            lb_loss += aux["load_balancing_loss"]
            z_loss += aux["router_z_loss"]
        logits = self.lm_head(self.norm(x))
        return logits, {"load_balancing_loss": lb_loss, "router_z_loss": z_loss}
