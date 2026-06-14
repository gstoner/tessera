"""GLM-5 config factory — MoE + DSA + FP8 (dimensions UNCONFIRMED placeholder).

GLM-5 is not yet publicly specified.  This config is a *placeholder* built from
the GLM-4.5/4.6 MoE lineage and the roadmap assumption that GLM-5 adopts sparse
attention (DSA) and an FP8 release — it exists so the frontend surface and the
shared pillars are exercised for a GQA+DSA+FP8 model, not to assert real GLM-5
dimensions.  No parameter budget is declared (``verify_param_budget`` is a
no-op) precisely because the dims are unconfirmed.  Replace with real
hyperparameters when GLM-5 is published.
"""

from __future__ import annotations

from .moe_transformer import MoETransformerConfig


def config() -> MoETransformerConfig:
    """Placeholder full-scale GLM-5 config (GQA + DSA + FP8; dims unconfirmed)."""
    return MoETransformerConfig(
        name="glm5_placeholder",
        hidden_size=5120, num_layers=47, vocab_size=151552, context_length=131072,
        attn_kind="gqa", num_attention_heads=96, num_kv_heads=8, head_dim=128,
        rope_head_dim=128, rope_variant="rope",
        sparse="dsa", dsa_top_k_blocks=32, dsa_block_size=64,
        sliding_window=0, layer_types=("full",),
        num_experts=160, num_experts_per_tok=8, num_shared_experts=1,
        moe_intermediate_size=1536, shared_expert_intermediate_size=1536,
        first_k_dense=1, dense_intermediate_size=12288,
        weight_dtype="fp8_e4m3", quant_group_size=128,
        total_params_b=0.0, active_params_b=0.0,   # unconfirmed → budget unchecked
    )


def scaled_config() -> MoETransformerConfig:
    """Structurally-faithful, Mac-executable GLM-5 shrink (GQA + DSA + MoE + FP8)."""
    return MoETransformerConfig(
        name="glm5_scaled",
        hidden_size=256, num_layers=4, vocab_size=1024, context_length=512,
        attn_kind="gqa", num_attention_heads=8, num_kv_heads=2, head_dim=32,
        rope_head_dim=32, rope_variant="rope",
        sparse="dsa", dsa_top_k_blocks=2, dsa_block_size=4,
        num_experts=8, num_experts_per_tok=2, num_shared_experts=1,
        moe_intermediate_size=256, shared_expert_intermediate_size=256,
        first_k_dense=1, dense_intermediate_size=512,
        weight_dtype="fp8_e4m3", quant_group_size=64,
    )


__all__ = ["config", "scaled_config"]
