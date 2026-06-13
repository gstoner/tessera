"""Tessera production model graphs (experimental).

Compiler-visible, dimension-checked model graphs built from Tessera primitives.
Today this hosts the DiffusionGemma text block — a block-diffusion MoE text
model — as a shape-only graph + config-aware verifier (the contract layer that
runtime/kernel lowering phases build on).
"""

from __future__ import annotations

from .diffusion_gemma import (
    DiffusionGemmaConfig,
    DiffusionGemmaDimError,
    GraphNode,
    TextBlockGraph,
    build_text_block,
    build_lm_head,
    verify_config,
    verify_text_block,
    verify_lm_head,
    estimated_param_counts,
    verify_param_budget,
)
from .moe_routing import (
    RoutingPlan,
    route_top_k,
    plan_packing,
    pack_tokens,
    unpack_combine,
    moe_forward,
    moe_forward_naive,
    synthetic_moe_weights,
)
from .sampler import (
    SamplerConfig,
    SamplerResult,
    temperature_schedule,
    entropy_bound_sample,
)
from .block_diffusion import (
    BlockDiffusionStepGraph,
    BlockDiffusionStepResult,
    build_block_diffusion_step,
    verify_block_diffusion_step,
    run_block_diffusion_step,
)
from .decode import BlockDiffusionDecoder, BlockDecodeResult
from .block_diffusion_runtime import (
    synthetic_layer_weights,
    synthetic_encoder_kv,
    denoise_layer,
    run_denoise,
)
from .staging import (
    StagingError,
    QuantizationPlan,
    VisionMetadata,
    ModelManifest,
    plan_quantization,
    default_vision_metadata,
    validate_vision_metadata,
    vision_execution_supported,
    import_model_metadata,
)

__all__ = [
    "StagingError",
    "QuantizationPlan",
    "VisionMetadata",
    "ModelManifest",
    "plan_quantization",
    "default_vision_metadata",
    "validate_vision_metadata",
    "vision_execution_supported",
    "import_model_metadata",
    "BlockDiffusionDecoder",
    "BlockDecodeResult",
    "BlockDiffusionStepGraph",
    "BlockDiffusionStepResult",
    "synthetic_layer_weights",
    "synthetic_encoder_kv",
    "denoise_layer",
    "run_denoise",
    "build_block_diffusion_step",
    "verify_block_diffusion_step",
    "run_block_diffusion_step",
    "SamplerConfig",
    "SamplerResult",
    "temperature_schedule",
    "entropy_bound_sample",
    "RoutingPlan",
    "route_top_k",
    "plan_packing",
    "pack_tokens",
    "unpack_combine",
    "moe_forward",
    "moe_forward_naive",
    "synthetic_moe_weights",
    "DiffusionGemmaConfig",
    "DiffusionGemmaDimError",
    "GraphNode",
    "TextBlockGraph",
    "build_text_block",
    "build_lm_head",
    "verify_config",
    "verify_text_block",
    "verify_lm_head",
    "estimated_param_counts",
    "verify_param_budget",
]
