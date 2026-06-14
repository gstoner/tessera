"""Kimi-K2 config factory — MoE + MLA + native INT4 weights.

Kimi-K2 shares the DeepSeek-V3 architecture lineage (MLA latent attention, a
large MoE with a shared expert) and adds the deepest INT4 quant path: ~1T total
/ ~32B active, 61 layers, hidden 7168, 384 routed + 1 shared experts, 8 active,
moe_intermediate 2048, MLA kv_lora_rank 512 / q_lora_rank 1536, 64 heads × 128,
vocab 163840.  No DSA in the base model (dense MLA attention).  Native INT4 is
the differentiator — it drives the ``stdlib.quant`` group-wise INT4 path.
"""

from __future__ import annotations

from .moe_transformer import MoETransformerConfig


def config() -> MoETransformerConfig:
    """Full-scale Kimi-K2 config (artifact target; not Mac-executable)."""
    return MoETransformerConfig(
        name="kimi_k2",
        hidden_size=7168, num_layers=61, vocab_size=163840, context_length=131072,
        attn_kind="mla", num_attention_heads=64, num_kv_heads=64, head_dim=128,
        q_lora_rank=1536, kv_lora_rank=512, rope_head_dim=64, rope_variant="rope",
        sparse=None,
        num_experts=384, num_experts_per_tok=8, num_shared_experts=1,
        moe_intermediate_size=2048, shared_expert_intermediate_size=2048,
        first_k_dense=1, dense_intermediate_size=18432,
        weight_dtype="int4", quant_group_size=128,
        total_params_b=1026.0, active_params_b=32.0,
    )


def scaled_config() -> MoETransformerConfig:
    """Structurally-faithful, Mac-executable Kimi-K2 shrink (MLA + MoE + INT4)."""
    return MoETransformerConfig(
        name="kimi_k2_scaled",
        hidden_size=256, num_layers=4, vocab_size=1024, context_length=512,
        attn_kind="mla", num_attention_heads=4, num_kv_heads=4, head_dim=64,
        q_lora_rank=0, kv_lora_rank=64, rope_head_dim=32, rope_variant="rope",
        sparse=None,
        num_experts=8, num_experts_per_tok=2, num_shared_experts=1,
        moe_intermediate_size=256, shared_expert_intermediate_size=256,
        first_k_dense=1, dense_intermediate_size=512,
        weight_dtype="int4", quant_group_size=64,
    )


__all__ = ["config", "scaled_config"]
