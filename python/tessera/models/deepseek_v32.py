"""DeepSeek-V3.2 config factory — the north-star vertical (MoE + MLA + DSA + FP8).

DeepSeek-V3.2-Exp is the richest target: latent attention (MLA), DeepSeek Sparse
Attention (DSA), a 256-expert MoE with a shared expert and leading dense layers,
and FP8 weights.  Finishing it exercises every M1–M4 pillar.

Dimensions follow the public DeepSeek-V3 architecture (671B total / ~37B active,
61 layers, hidden 7168, MLA kv_lora_rank 512 / q_lora_rank 1536, 128 heads ×
128, 256 routed + 1 shared experts, 8 active, moe_intermediate 2048, 3 leading
dense layers, vocab 129280) plus V3.2's sparse-attention top-k block selection
(block size / top-k are approximate — the published indexer is token-level; the
block contract here is the shape the M4 kernel implements).  Treat the exact DSA
numbers as approximate.
"""

from __future__ import annotations

from .moe_transformer import MoETransformerConfig


def config() -> MoETransformerConfig:
    """Full-scale DeepSeek-V3.2 config (artifact target; not Mac-executable)."""
    return MoETransformerConfig(
        name="deepseek_v3.2",
        hidden_size=7168, num_layers=61, vocab_size=129280, context_length=163840,
        attn_kind="mla", num_attention_heads=128, num_kv_heads=128, head_dim=128,
        q_lora_rank=1536, kv_lora_rank=512, rope_head_dim=64, rope_variant="rope",
        sparse="dsa", dsa_top_k_blocks=32, dsa_block_size=64,
        num_experts=256, num_experts_per_tok=8, num_shared_experts=1,
        moe_intermediate_size=2048, shared_expert_intermediate_size=2048,
        first_k_dense=3, dense_intermediate_size=18432,
        weight_dtype="fp8_e4m3", quant_group_size=128,
        total_params_b=671.0, active_params_b=37.0,
    )


def scaled_config() -> MoETransformerConfig:
    """A structurally-faithful, Mac-executable shrink of DeepSeek-V3.2.

    Same architecture (MLA + DSA + MoE + FP8 contract, leading dense layer) at
    toy dimensions so the whole forward runs on Apple GPU / numpy and can be
    gated against a reference.  This is the M5 integration target's config.
    """
    return MoETransformerConfig(
        name="deepseek_v3.2_scaled",
        hidden_size=256, num_layers=4, vocab_size=1024, context_length=512,
        attn_kind="mla", num_attention_heads=4, num_kv_heads=4, head_dim=64,
        q_lora_rank=0, kv_lora_rank=64, rope_head_dim=32, rope_variant="rope",
        sparse="dsa", dsa_top_k_blocks=2, dsa_block_size=4,
        num_experts=8, num_experts_per_tok=2, num_shared_experts=1,
        moe_intermediate_size=256, shared_expert_intermediate_size=256,
        first_k_dense=1, dense_intermediate_size=512,
        weight_dtype="fp8_e4m3", quant_group_size=64,
    )


__all__ = ["config", "scaled_config"]
