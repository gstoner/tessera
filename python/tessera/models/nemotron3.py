"""Nemotron-3 Super config factory — hybrid Mamba-SSM + attention anchors + MoE.

Layout (verified, arXiv:2604.12374): 88 layers, predominantly Mamba-2 paired with
MoE, a limited number of GQA attention "anchor" layers; 1M context; 2 shared-weight
MTP heads.

Built on the Track L `stdlib.hybrid` reference: `linear_mixer="ssm"` + sparse
attention anchors + `ffn="moe"`.  **Reference approximations:** (1) the real model
uses **LatentMoE** (route in a down-projected ℓ-dim latent, 512 experts/top-22);
this reference uses a standard top-k MoE — LatentMoE is a distinct contract tracked
separately.  (2) exact anchor placement (Figure 2) is approximated by an
`attn_period` schedule.  Layer count, Mamba-state mixer, anchor pattern, and MoE
family are faithful at the shape level.
"""

from __future__ import annotations

from ..stdlib.hybrid import HybridConfig, nemotron_schedule


def config() -> HybridConfig:
    """Full-scale Nemotron-3 Super (artifact target; not Mac-executable)."""
    return HybridConfig(
        d_model=4096, num_heads=32, head_dim=128, ssm_state=128,
        schedule=nemotron_schedule(88, attn_period=8), linear_mixer="ssm",
        ffn="moe", num_experts=64, top_k=8, shared_expert=True)


def scaled_config() -> HybridConfig:
    """Structurally-faithful shrunk instance (Mamba + anchors + MoE)."""
    return HybridConfig(
        d_model=32, num_heads=4, head_dim=8, ssm_state=4,
        schedule=nemotron_schedule(8, attn_period=4), linear_mixer="ssm",
        ffn="moe", num_experts=8, top_k=2, shared_expert=True)


__all__ = ["config", "scaled_config"]
