"""LFM2.5-8B-A1B config factory — LIV gated short-conv hybrid + MoE (Track L).

Layout (verified, Liquid blog + LFM2 report): 24 layers = 18 double-gated LIV
short-conv (kernel 3) + 6 GQA; 32 experts / top-4; first 2 layers dense.

Built on the Track L `stdlib.hybrid` reference: `linear_mixer="liv"` (conv_kernel
3) + GQA anchors (period 4 → 6 full of 24) + `ffn="moe"`.  **Reference
approximation:** the real model keeps the first 2 layers dense and has no shared
expert; this reference applies one FFN type globally (all-MoE).  Layer count, the
LIV gated-conv mixer, the 18:6 conv:attention split, and the 32/top-4 MoE shape
are faithful at the shape level.
"""

from __future__ import annotations

from ..stdlib.hybrid import HybridConfig, HybridSchedule


def config() -> HybridConfig:
    """Full-scale LFM2.5-8B-A1B (artifact target; not Mac-executable)."""
    return HybridConfig(
        d_model=2048, num_heads=8, head_dim=128, conv_kernel=3,
        schedule=HybridSchedule(num_layers=24, period=4, full_offset=1),
        linear_mixer="liv",
        ffn="moe", num_experts=32, top_k=4, shared_expert=False)


def scaled_config() -> HybridConfig:
    """Structurally-faithful shrunk instance (LIV conv + GQA anchors + MoE)."""
    return HybridConfig(
        d_model=32, num_heads=4, head_dim=8, conv_kernel=3,
        schedule=HybridSchedule(num_layers=8, period=4, full_offset=1),
        linear_mixer="liv",
        ffn="moe", num_experts=8, top_k=2, shared_expert=False)


__all__ = ["config", "scaled_config"]
