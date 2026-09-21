"""
tessera_diffusion_llm/models/transformer.py

Shared bidirectional transformer backbone used by all three model variants.

Architecture (DiT-style with adaLN-Single time conditioning):
  • Input: token embeddings + sinusoidal positional encoding
  • Each block: adaLN → BidirectionalAttention → residual
                adaLN → GatedMLP → residual
  • Time conditioning: single adaLN modulation projected to scale+shift
    applied before attention and before MLP in each block

Tessera compiler integration:
  • Every attention module carries _tessera_op = "tessera.flash_attn"
    with causal=False
  • Every MLP carries _tessera_op = "tessera.elementwise.mlp_gate"
  • The sharding annotation for 2-way tensor-parallel maps attention
    heads and MLP intermediate dim to the "tp" mesh axis
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ..configs import TransformerConfig
from ..kernels.attention import BidirectionalAttention
from ..kernels.mlp import DiffusionMLP
from ..kernels.rmsnorm import RMSNorm, AdaLNModulation, adaLN_forward
from ..kernels.time_embed import CombinedTimeEmbed


# ---------------------------------------------------------------------------
# Single transformer block
# ---------------------------------------------------------------------------

class DiffusionTransformerBlock(nn.Module):
    """
    One transformer block with adaLN-Single conditioning.

    Args:
        cfg:       TransformerConfig
        cond_dim:  Dimension of conditioning vector (time embedding).
    """

    def __init__(self, cfg: TransformerConfig, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn  = BidirectionalAttention(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            dropout_p=cfg.dropout_p,
        )
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp   = DiffusionMLP(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            mlp_type=cfg.mlp_type,
        )

        # adaLN: each block gets its own (scale_attn, shift_attn, gate_attn,
        #                                 scale_mlp,  shift_mlp)
        # Using 2 modulations × 2 outputs each (scale + shift)
        self.adaLN_attn = AdaLNModulation(cfg.hidden_size, cond_dim, num_outputs=2)
        self.adaLN_mlp  = AdaLNModulation(cfg.hidden_size, cond_dim, num_outputs=2)

        self.time_cond_style = cfg.time_cond_style

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, T, hidden_size)
            cond: (B, cond_dim) — time or noise-level embedding
        Returns:
            (B, T, hidden_size)
        """
        # Attention sub-layer
        scale_a, shift_a = self.adaLN_attn(cond)
        h = adaLN_forward(x, self.norm1, scale_a, shift_a)
        h = self.attn(h, attn_mask)
        x = x + h

        # MLP sub-layer
        scale_m, shift_m = self.adaLN_mlp(cond)
        h = adaLN_forward(x, self.norm2, scale_m, shift_m)
        h = self.mlp(h)
        x = x + h
        return x


# ---------------------------------------------------------------------------
# Full backbone
# ---------------------------------------------------------------------------

class DiffusionTransformer(nn.Module):
    """
    Bidirectional transformer backbone for diffusion models.

    Wraps the embedding layer, positional encoding, N transformer blocks,
    and the final layer norm.  The specific output head (logits for MDLM,
    ε-prediction for continuous, velocity for flow matching) lives in the
    subclass models.

    Args:
        cfg: TransformerConfig
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size

        # Token embedding (also used as tie-target for output projection)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, H)

        # Learned positional embedding (absolute; simple and effective)
        self.pos_embed = nn.Embedding(cfg.max_position_embeddings, H)

        # Time / noise-level conditioning
        self.time_embed = CombinedTimeEmbed(H, embed_type="sinusoidal")

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [DiffusionTransformerBlock(cfg, cond_dim=H)
             for _ in range(cfg.num_hidden_layers)]
        )
        self.norm_out = RMSNorm(H, cfg.rms_norm_eps)

        self.dropout = nn.Dropout(cfg.dropout_p)
        self._init_weights()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------
    def _init_weights(self) -> None:
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------
    def forward(
        self,
        token_ids: Optional[torch.LongTensor],
        t: torch.Tensor,
        x_emb: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: (B, T) integer token ids OR None if x_emb provided.
            t:         (B,) timestep tensor (integer or float ∈ [0,1]).
            x_emb:     (B, T, H) pre-computed embedding (e.g. for continuous
                       diffusion where input is a noisy embedding vector).
                       If both token_ids and x_emb are given, x_emb is used.
            attn_mask: Optional (B, T, T) attention mask.

        Returns:
            (B, T, hidden_size) — contextualised hidden states before output head.
        """
        if x_emb is not None:
            x = x_emb
        else:
            x = self.embed_tokens(token_ids)

        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device)
        x = self.dropout(x + self.pos_embed(positions))

        # Time conditioning vector
        cond = self.time_embed(t)   # (B, H)

        for block in self.blocks:
            x = block(x, cond, attn_mask)

        return self.norm_out(x)

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------
    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters() if not trainable_only else (
            p for p in self.parameters() if p.requires_grad
        )
        return sum(p.numel() for p in params)
