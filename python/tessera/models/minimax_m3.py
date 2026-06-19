"""MiniMax-M3 config factory — GQA + MSA + MoE + multimodal metadata.

MiniMax-M3's released Hugging Face config identifies a native multimodal
``minimax_m3_vl`` model with a 60-layer text tower: GQA attention, MiniMax
Sparse Attention (MSA) after the first three dense layers, 128 routed experts
with top-4 routing, one shared expert, and BF16 weights.  This module keeps the
text tower as a compiler-visible contract and records the vision/video metadata
as importer-side staging only.
"""

from __future__ import annotations

from dataclasses import dataclass

from .moe_transformer import MoETransformerConfig


MINIMAX_M3_SPARSE_LAYER_FREQ: tuple[int, ...] = (0, 0, 0) + (1,) * 57


@dataclass(frozen=True)
class MiniMaxM3VisionMetadata:
    """Importer-visible multimodal metadata; execution is intentionally staged."""

    image_token_index: int = 200025
    video_token_index: int = 200026
    image_seq_length: int = 576
    patch_size: int = 14
    image_size: int = 2016
    projector_hidden_size: int = 6144
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    vision_segment_max_frames: int = 4
    vision_execution_supported: bool = False


VISION_METADATA = MiniMaxM3VisionMetadata()


def config() -> MoETransformerConfig:
    """Full-scale MiniMax-M3 text-tower contract from the released HF config."""
    return MoETransformerConfig(
        name="minimax_m3",
        hidden_size=6144,
        num_layers=60,
        vocab_size=200064,
        context_length=1_048_576,
        attn_kind="gqa",
        num_attention_heads=64,
        num_kv_heads=4,
        head_dim=128,
        rope_head_dim=64,
        rope_theta=5_000_000.0,
        sparse="msa",
        msa_top_k_blocks=16,
        msa_block_size=128,
        msa_index_dim=128,
        msa_num_index_heads=4,
        msa_score_type="max",
        msa_sparse_layer_freq=MINIMAX_M3_SPARSE_LAYER_FREQ,
        num_experts=128,
        num_experts_per_tok=4,
        num_shared_experts=1,
        moe_intermediate_size=3072,
        shared_expert_intermediate_size=3072,
        first_k_dense=3,
        dense_intermediate_size=12288,
        dtype="bf16",
        total_params_b=427.04,
        active_params_b=23.0,
        hf_model_size_b=427.04,
    )


def scaled_config() -> MoETransformerConfig:
    """Mac-executable shrink preserving dense warmup, GQA, MSA, and MoE shape.

    The small ``msa_block_size`` intentionally creates multiple KV blocks in
    unit tests, so runtime decode proves real MSA selection instead of a
    one-block dense fallback.
    """
    return MoETransformerConfig(
        name="minimax_m3_scaled",
        hidden_size=256,
        num_layers=4,
        vocab_size=2048,
        context_length=512,
        attn_kind="gqa",
        num_attention_heads=8,
        num_kv_heads=2,
        head_dim=32,
        rope_head_dim=16,
        rope_theta=5_000_000.0,
        sparse="msa",
        msa_top_k_blocks=2,
        msa_block_size=4,
        msa_index_dim=32,
        msa_num_index_heads=2,
        msa_score_type="max",
        msa_sparse_layer_freq=(0, 1, 1, 1),
        num_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
        first_k_dense=1,
        dense_intermediate_size=512,
        dtype="bf16",
    )


__all__ = [
    "MiniMaxM3VisionMetadata",
    "MINIMAX_M3_SPARSE_LAYER_FREQ",
    "VISION_METADATA",
    "config",
    "scaled_config",
]
