"""MoBA — Mixture of Block Attention (Lu et al., 2025; arXiv:2502.13189).

Ported into tessera.train by the ``add-moe-model`` skill: copied from
``qwen3_moe.py`` and edited in place. The *only* structural change from the
template is the attention sublayer — full causal attention becomes
block-sparse MoBA attention; the MoE FFN, norms, embedding, and head are reused
unchanged. The whole model is still ONE self-contained file, built by direct
instantiation (no registry / spec indirection).

MoBA mechanism (numpy reference, fp64 internally):
  * The key/value sequence is partitioned into blocks of ``block_size``.
  * For each query, a gate scores every block by ``q · mean(keys-in-block)``.
  * The query attends to the top-``top_k_blocks`` *past* blocks (by gate score)
    plus its own current block (mandatory), with causal masking throughout.
This is the "mixture of attention experts (= blocks)" that gives MoBA its name —
a structural analogue of MoE routing, here over KV blocks instead of FFN experts.

The forward is a numpy reference (like the Qwen3 template's attention path); it
is tape-invisible in v1 and used for shape/inference + the skill's verify check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tessera import nn
from tessera.train.engine.moe import MoEFeedForward


def _arr(x) -> np.ndarray:
    while hasattr(x, "_data"):
        x = x._data
    return np.asarray(x)


@dataclass(frozen=True)
class MoBAConfig:
    vocab_size: int = 1024
    hidden_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    block_size: int = 4        # KV block granularity for MoBA routing
    top_k_blocks: int = 2      # past blocks each query attends to (+ its own)
    num_experts: int = 8
    top_k: int = 2
    expert_intermediate: int = 512
    shared_intermediate: int = 256
    rms_eps: float = 1e-6
    dtype: str = "fp32"


def _softmax_lastdim(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def moba_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                   block_size: int, top_k_blocks: int) -> np.ndarray:
    """Single-head MoBA attention. Q/K/V: ``(S, head_dim)`` → ``(S, head_dim)``."""
    S, hd = Q.shape
    Q = Q.astype(np.float64); K = K.astype(np.float64); V = V.astype(np.float64)
    scale = 1.0 / math.sqrt(hd)
    nb = (S + block_size - 1) // block_size
    block_of = np.arange(S) // block_size            # (S,) block id per position

    # Gate score: query vs. mean-pooled key of each block.
    block_mean = np.stack([K[block_of == b].mean(axis=0) for b in range(nb)])  # (nb, hd)
    gate = Q @ block_mean.T                           # (S, nb)

    out = np.zeros((S, hd), dtype=np.float64)
    for i in range(S):
        cb = int(block_of[i])
        past = np.arange(cb)
        if past.size > top_k_blocks:
            sel = past[np.argsort(-gate[i, past])[:top_k_blocks]]
        else:
            sel = past
        allowed = set(sel.tolist()) | {cb}
        # Causal key mask restricted to the selected blocks.
        mask = np.array([(block_of[j] in allowed) and (j <= i) for j in range(S)])
        scores = (Q[i] @ K.T) * scale
        scores = np.where(mask, scores, -1e30)
        out[i] = _softmax_lastdim(scores) @ V
    return out


class MoBABlock(nn.Module):
    """Pre-norm decoder block: MoBA block-sparse attention + MoE FFN."""

    def __init__(self, cfg: MoBAConfig, layer_idx: int) -> None:
        super().__init__()
        assert cfg.hidden_size % cfg.num_heads == 0
        self.cfg = cfg
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_eps, dtype=cfg.dtype)
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False, dtype=cfg.dtype)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False, dtype=cfg.dtype)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False, dtype=cfg.dtype)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False, dtype=cfg.dtype)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_eps, dtype=cfg.dtype)
        self.mlp = MoEFeedForward(
            cfg.hidden_size, cfg.num_experts, cfg.top_k,
            cfg.expert_intermediate, cfg.shared_intermediate,
            dtype=cfg.dtype, seed=layer_idx,
        )

    def _attn(self, x: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        B, S, H = x.shape
        nh, hd = cfg.num_heads, self.head_dim
        q = _arr(self.q_proj(x)).reshape(B, S, nh, hd)
        k = _arr(self.k_proj(x)).reshape(B, S, nh, hd)
        v = _arr(self.v_proj(x)).reshape(B, S, nh, hd)
        out = np.zeros((B, S, nh, hd), dtype=np.float64)
        for b in range(B):
            for h in range(nh):
                out[b, :, h, :] = moba_attention(
                    q[b, :, h, :], k[b, :, h, :], v[b, :, h, :],
                    cfg.block_size, cfg.top_k_blocks,
                )
        return _arr(self.o_proj(out.reshape(B, S, H).astype(np.float32)))

    def forward(self, x):
        x = np.asarray(x)
        x = x + self._attn(self.input_layernorm(x))
        moe_out, aux = self.mlp(self.post_attention_layernorm(x))
        x = x + np.asarray(moe_out)
        return x, aux


class MoBAModel(nn.Module):
    """Embedding → L MoBA blocks → final norm → LM head.

    ``forward(ids)`` → ``(logits (B,S,V), aux_losses)`` (summed MoE aux terms),
    matching the Qwen3MoEModel interface so the same training loop and the
    ``add-moe-model`` verify check apply unchanged.
    """

    def __init__(self, cfg: MoBAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=cfg.dtype)
        self.layers = nn.ModuleList([MoBABlock(cfg, i) for i in range(cfg.num_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_eps, dtype=cfg.dtype)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, dtype=cfg.dtype)

    def forward(self, ids):
        x = self.embed_tokens(np.asarray(ids, dtype=np.int64))
        lb_loss = 0.0
        z_loss = 0.0
        for layer in self.layers:
            x, aux = layer(x)
            lb_loss += aux["load_balancing_loss"]
            z_loss += aux["router_z_loss"]
        logits = self.lm_head(self.norm(x))
        return logits, {"load_balancing_loss": lb_loss, "router_z_loss": z_loss}
