"""Qwen3.6-35B-A3B config factory — gated-DeltaNet hybrid + MoE (Track L).

Layout (verified, HF config): 40 layers `[GatedDeltaNet ×3, GatedAttention] ×10`
(period 4), 256 experts / top-8 + 1 shared, MTP 1 layer.

Built on the Track L `stdlib.hybrid` reference stack: `linear_mixer="delta"` +
`ffn="moe"`.  **Reference approximation:** the real model uses different head
dims for the linear vs full layers (linear 32V/16K head_dim 128; full 16Q/2KV
head_dim 256) and a pre-mixer short conv; this reference uses one uniform head
config and no conv.  Schedule, layer count, mixer family, and MoE shape are
faithful; the config is a shape-level artifact, not a weight-faithful port.
"""

from __future__ import annotations

from ..stdlib.hybrid import HybridConfig, qwen3_6_schedule


def config() -> HybridConfig:
    """Full-scale Qwen3.6-35B-A3B (artifact target; not Mac-executable)."""
    return HybridConfig(
        d_model=2048, num_heads=16, head_dim=128,
        schedule=qwen3_6_schedule(40), linear_mixer="delta",
        ffn="moe", num_experts=256, top_k=8, shared_expert=True)


def scaled_config() -> HybridConfig:
    """Structurally-faithful shrunk instance — runs end-to-end on Apple GPU /
    numpy, gated against the recompute reference."""
    return HybridConfig(
        d_model=32, num_heads=4, head_dim=8,
        schedule=qwen3_6_schedule(8), linear_mixer="delta",
        ffn="moe", num_experts=8, top_k=2, shared_expert=True)


__all__ = ["config", "scaled_config"]
