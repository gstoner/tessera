"""GLM-5.2 config factory — 744B-class MoE + MLA + DSA IndexShare + MTP.

The released Hugging Face config identifies the model as ``glm_moe_dsa`` with
78 layers, 6144 hidden width, 1M context, MLA ranks, 256 routed experts, DSA
index sharing, and one next-token MTP layer.  This module keeps those facts as a
compiler-visible contract; the scaled variant preserves the same structural
features while staying small enough for local numpy tests.
"""

from __future__ import annotations

from .moe_transformer import MoETransformerConfig


GLM52_INDEXER_TYPES: tuple[str, ...] = (
    "full", "full", "full",
    "shared", "shared", "shared", "full", "shared", "shared", "shared",
    "full", "shared", "shared", "shared", "full", "shared", "shared", "shared",
    "full", "shared", "shared", "shared", "full", "shared", "shared", "shared",
    "full", "shared", "shared", "shared", "full", "shared", "shared", "shared",
    "full", "shared", "shared", "shared", "full", "shared", "shared", "shared",
    "full", "shared", "shared", "shared", "full", "shared", "shared", "shared",
    "full", "shared", "shared", "shared", "full", "shared", "shared", "shared",
    "full", "shared", "shared", "shared", "full", "shared", "shared", "shared",
    "full", "shared", "shared", "shared", "full", "shared", "shared", "shared",
    "full", "shared", "shared", "shared",
)


def glm52_config() -> MoETransformerConfig:
    """Full-scale GLM-5.2 shape contract from the released HF config."""
    return MoETransformerConfig(
        name="glm5_2",
        hidden_size=6144,
        num_layers=78,
        vocab_size=154880,
        context_length=1_048_576,
        attn_kind="mla",
        num_attention_heads=64,
        num_kv_heads=64,
        head_dim=192,
        qk_head_dim=256,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        q_lora_rank=2048,
        kv_lora_rank=512,
        rope_head_dim=64,
        rope_variant="rope",
        rope_theta=8_000_000.0,
        sparse="dsa",
        dsa_top_k_blocks=32,
        dsa_block_size=64,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_types=GLM52_INDEXER_TYPES,
        indexer_rope_interleave=True,
        sliding_window=0,
        layer_types=("full",),
        num_experts=256,
        num_experts_per_tok=8,
        num_shared_experts=1,
        moe_intermediate_size=2048,
        shared_expert_intermediate_size=2048,
        first_k_dense=3,
        dense_intermediate_size=12288,
        weight_dtype=None,
        quant_group_size=0,
        rms_norm_eps=1e-5,
        dtype="bf16",
        total_params_b=744.0,
        active_params_b=40.0,
        hf_model_size_b=744.0,
        rollout_kv_dtype="fp8",
        mtp_num_steps=4,
        mtp_num_layers=1,
        mtp_share_parameters=True,
        mtp_index_share=True,
        mtp_kv_share=True,
    )


def config() -> MoETransformerConfig:
    """Canonical full-scale GLM config; now GLM-5.2, not a placeholder."""
    return glm52_config()


def scaled_config() -> MoETransformerConfig:
    """Mac-executable GLM-5.2 shrink preserving MLA, DSA, IndexShare, and MTP."""
    return MoETransformerConfig(
        name="glm5_2_scaled",
        hidden_size=256,
        num_layers=8,
        vocab_size=1024,
        context_length=4096,
        attn_kind="mla",
        num_attention_heads=8,
        num_kv_heads=8,
        head_dim=32,
        qk_head_dim=40,
        qk_nope_head_dim=32,
        qk_rope_head_dim=8,
        v_head_dim=32,
        q_lora_rank=64,
        kv_lora_rank=32,
        rope_head_dim=8,
        rope_variant="rope",
        rope_theta=8_000_000.0,
        sparse="dsa",
        dsa_top_k_blocks=2,
        dsa_block_size=4,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=8,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_types=(
            "full", "full", "full", "shared",
            "shared", "shared", "full", "shared",
        ),
        indexer_rope_interleave=True,
        num_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
        first_k_dense=3,
        dense_intermediate_size=512,
        weight_dtype=None,
        quant_group_size=0,
        rms_norm_eps=1e-5,
        dtype="bf16",
        total_params_b=0.0,
        active_params_b=0.0,
        hf_model_size_b=744.0,
        rollout_kv_dtype="fp8",
        mtp_num_steps=4,
        mtp_num_layers=1,
        mtp_share_parameters=True,
        mtp_index_share=True,
        mtp_kv_share=True,
    )


__all__ = ["GLM52_INDEXER_TYPES", "glm52_config", "config", "scaled_config"]
