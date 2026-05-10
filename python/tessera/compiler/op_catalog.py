"""Canonical Tessera frontend operator catalog.

This module is intentionally dependency-light so it can be shared by the
Python AST frontend, textual frontend, effect inference, and reference CPU
lowering without creating import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OpSpec:
    public_name: str
    graph_name: str
    min_arity: int
    max_arity: int
    effect: str = "pure"
    lowering: str = "elementwise"

    def valid_arity(self, arity: int) -> bool:
        return self.min_arity <= arity <= self.max_arity


_SPECS = [
    OpSpec("gemm", "tessera.matmul", 2, 2, lowering="loop_nest"),
    OpSpec("matmul", "tessera.matmul", 2, 2, lowering="loop_nest"),
    OpSpec("batched_gemm", "tessera.batched_gemm", 2, 2, lowering="loop_nest"),
    OpSpec("einsum", "tessera.einsum", 1, 99, lowering="contraction"),
    OpSpec("factorized_matmul", "tessera.factorized_matmul", 2, 2, lowering="loop_nest"),
    OpSpec("tri_solve", "tessera.tri_solve", 2, 2, lowering="linalg_solver"),
    OpSpec("cholesky", "tessera.cholesky", 1, 1, lowering="linalg_decomposition"),
    OpSpec("qr", "tessera.qr", 1, 1, lowering="linalg_decomposition"),
    OpSpec("svd", "tessera.svd", 1, 1, lowering="linalg_decomposition"),
    OpSpec("conv2d", "tessera.conv2d_nhwc", 2, 4, lowering="stencil"),
    OpSpec("conv3d", "tessera.conv3d_ndhwc", 2, 4, lowering="stencil"),
    OpSpec("layer_norm", "tessera.layer_norm", 1, 1, lowering="normalization"),
    OpSpec("softmax", "tessera.softmax", 1, 1, lowering="stable_reduction"),
    OpSpec("softmax_safe", "tessera.softmax_safe", 1, 1, lowering="stable_reduction"),
    OpSpec("reduce", "tessera.reduce", 1, 1, lowering="stable_reduction"),
    OpSpec("sum", "tessera.reduce", 1, 1, lowering="stable_reduction"),
    OpSpec("gelu", "tessera.gelu", 1, 1),
    OpSpec("tanh", "tessera.tanh", 1, 1),
    OpSpec("add", "tessera.add", 1, 2),
    OpSpec("mul", "tessera.mul", 1, 2),
    OpSpec("relu", "tessera.relu", 1, 1),
    OpSpec("silu", "tessera.silu", 1, 1),
    OpSpec("silu_mul", "tessera.silu_mul", 2, 2),
    OpSpec("sigmoid", "tessera.sigmoid", 1, 1),
    OpSpec("sin", "tessera.sin", 1, 1),
    # Theme 9 — utility tensor ops. `arange` has no differentiable inputs;
    # the rest follow the standard elementwise / shape pattern.
    OpSpec("arange", "tessera.arange", 0, 3, lowering="layout_transform"),
    OpSpec("gather", "tessera.gather", 2, 2, lowering="layout_transform"),
    OpSpec("clip", "tessera.clip", 1, 1),
    OpSpec("masked_fill", "tessera.masked_fill", 2, 2, lowering="layout_transform"),
    OpSpec("adam", "tessera.adam", 4, 4, lowering="functional_optimizer_step"),
    OpSpec("transpose", "tessera.transpose", 1, 1, lowering="layout_transform"),
    OpSpec("cast", "tessera.cast", 1, 1, lowering="layout_transform"),
    OpSpec("dropout", "tessera.dropout", 1, 1, effect="random", lowering="random_mask"),
    OpSpec("qkv_projection", "tessera.qkv_projection", 2, 2, lowering="projection"),
    OpSpec("flash_attn", "tessera.flash_attn", 3, 3, effect="state", lowering="attention"),
    # attention_variants_plan, LA-1 — linear / kernel-feature attention.
    # Returns (O, state) tuple; the runtime dispatcher unpacks both.
    OpSpec("linear_attn", "tessera.linear_attn", 3, 3, effect="state", lowering="attention"),
    OpSpec("linear_attn_state", "tessera.linear_attn_state", 3, 3, effect="state", lowering="attention"),
    # attention_variants_plan, LA-4 — Power attention + Retention promoted
    # from `examples/advanced/power_retention/`. Same recurrence backbone
    # as linear_attn with deg + window / log_g + chunk attrs.
    OpSpec("power_attn", "tessera.power_attn", 3, 3, effect="state", lowering="attention"),
    OpSpec("retention", "tessera.retention", 3, 3, effect="state", lowering="attention"),
    # attention_variants_plan, NSA — Native Sparse Attention branches.
    # Each is a single-output op (no tuple returns) so the tape can
    # record + back-propagate cleanly. compress_blocks is a tuple-returning
    # helper that's intentionally NOT in op_catalog (matches the
    # qkv_projection pattern).
    OpSpec("attn_sliding_window", "tessera.attn_sliding_window", 3, 3, effect="state", lowering="attention"),
    OpSpec("attn_compressed_blocks", "tessera.attn_compressed_blocks", 3, 3, effect="state", lowering="attention"),
    OpSpec("attn_top_k_blocks", "tessera.attn_top_k_blocks", 3, 3, effect="state", lowering="attention"),
    # Phase F-MoR — Mixture of Recursions primitives. mor_router maps
    # (x, w_router) → per-token depth assignment. mor_partition takes a
    # depth tensor + step int and returns a bool mask. mor_scatter writes
    # active-token updates back into the full hidden state buffer.
    OpSpec("mor_router", "tessera.mor_router", 2, 2, lowering="layout_transform"),
    OpSpec("mor_partition", "tessera.mor_partition", 2, 2, lowering="layout_transform"),
    OpSpec("mor_scatter", "tessera.mor_scatter", 3, 3, lowering="layout_transform"),
    OpSpec("moe", "tessera.moe", 2, 2, effect="collective", lowering="moe"),
    OpSpec("moe_dispatch", "tessera.moe_dispatch", 2, 2, effect="collective", lowering="moe_transport"),
    OpSpec("moe_combine", "tessera.moe_combine", 2, 2, effect="collective", lowering="moe_transport"),
    OpSpec("all_reduce", "tessera.all_reduce", 1, 1, effect="collective", lowering="collective"),
    OpSpec("reduce_scatter", "tessera.reduce_scatter", 1, 1, effect="collective", lowering="collective"),
    OpSpec("all_gather", "tessera.all_gather", 1, 1, effect="collective", lowering="collective"),
    OpSpec("all_to_all", "tessera.all_to_all", 1, 1, effect="collective", lowering="collective"),
    OpSpec("rng_uniform", "tessera.rng_uniform", 0, 0, effect="random", lowering="random_source"),
    OpSpec("rng_normal", "tessera.rng_normal", 0, 0, effect="random", lowering="random_source"),
    OpSpec("fused_epilogue", "tessera.fused_epilogue", 1, 3, lowering="fused_epilogue"),
    OpSpec("fft", "tessera.fft", 1, 1, lowering="spectral"),
    OpSpec("ifft", "tessera.ifft", 1, 1, lowering="spectral"),
    OpSpec("rfft", "tessera.rfft", 1, 1, lowering="spectral"),
    OpSpec("irfft", "tessera.irfft", 1, 1, lowering="spectral"),
    OpSpec("stft", "tessera.stft", 2, 2, lowering="spectral"),
    OpSpec("istft", "tessera.istft", 2, 2, lowering="spectral"),
    OpSpec("spectral_filter", "tessera.spectral_filter", 2, 2, lowering="spectral"),
    OpSpec("dct", "tessera.dct", 1, 1, lowering="spectral"),
    OpSpec("spectral_conv", "tessera.spectral_conv", 2, 2, lowering="spectral"),
    OpSpec("spmm_coo", "tessera.spmm_coo", 2, 2, lowering="sparse"),
    OpSpec("spmm_csr", "tessera.spmm_csr", 2, 2, lowering="sparse"),
    OpSpec("sddmm", "tessera.sddmm", 3, 3, lowering="sparse"),
    OpSpec("bsmm", "tessera.bsmm", 2, 2, lowering="sparse"),
    OpSpec("segment_reduce", "tessera.segment_reduce", 2, 2, lowering="segment_reduce"),
    OpSpec("rearrange", "tessera.rearrange", 1, 1, lowering="layout_transform"),
    OpSpec("pack", "tessera.pack", 1, 1, effect="movement", lowering="layout_transform"),
    OpSpec("unpack", "tessera.unpack", 1, 1, effect="movement", lowering="layout_transform"),
    OpSpec("tile_view", "tessera.tile_view", 1, 1, lowering="layout_transform"),
    OpSpec("rmsnorm", "tessera.rmsnorm", 1, 1, lowering="normalization"),
    OpSpec("rmsnorm_safe", "tessera.rmsnorm_safe", 1, 1, lowering="normalization"),
    OpSpec("rope", "tessera.rope", 2, 2, lowering="rotary_embedding"),
    OpSpec("kv_cache_append", "tessera.kv_cache.append", 3, 3, effect="state", lowering="state_update"),
    OpSpec("kv_cache_prune", "tessera.kv_cache.prune", 1, 1, effect="state", lowering="state_update"),
    # Theme 10 — fp8 quantize/dequantize ops. Per-tensor symmetric.
    OpSpec("quantize_fp8", "tessera.quantize_fp8", 1, 1, lowering="quantize"),
    OpSpec("dequantize_fp8", "tessera.dequantize_fp8", 2, 2, lowering="quantize"),
    # Deferred-items plan, Item 2 — fp6 / fp4 / nvfp4. Same shape as fp8.
    OpSpec("quantize_fp6", "tessera.quantize_fp6", 1, 1, lowering="quantize"),
    OpSpec("dequantize_fp6", "tessera.dequantize_fp6", 2, 2, lowering="quantize"),
    OpSpec("quantize_fp4", "tessera.quantize_fp4", 1, 1, lowering="quantize"),
    OpSpec("dequantize_fp4", "tessera.dequantize_fp4", 2, 2, lowering="quantize"),
    OpSpec("quantize_nvfp4", "tessera.quantize_nvfp4", 1, 1, lowering="quantize"),
    OpSpec("dequantize_nvfp4", "tessera.dequantize_nvfp4", 2, 2, lowering="quantize"),
    # Theme 5 — Multi-Latent Attention primitives. The three projection ops
    # are matmul-shaped but distinct names so a future FlashMLA target pass
    # can match the chain (compress → cache → expand) and emit a fused
    # absorbed-K kernel on Hopper/Blackwell.
    OpSpec("latent_kv_compress", "tessera.latent_kv_compress", 2, 2, lowering="loop_nest"),
    OpSpec("latent_kv_expand_k", "tessera.latent_kv_expand_k", 2, 2, lowering="loop_nest"),
    OpSpec("latent_kv_expand_v", "tessera.latent_kv_expand_v", 2, 2, lowering="loop_nest"),
    # MLA-1 fusion target — result of the MLAFusionPass collapse.
    OpSpec("mla_decode_fused", "tessera.mla_decode_fused", 5, 5, effect="state", lowering="attention"),
    OpSpec("rope_split", "tessera.rope_split", 1, 1, lowering="layout_transform"),
    OpSpec("rope_merge", "tessera.rope_merge", 2, 2, lowering="layout_transform"),
    # S-series sprint S2 — reductions. All accept (x, axis=, keepdims=).
    OpSpec("mean", "tessera.mean", 1, 1, lowering="reduction"),
    OpSpec("prod", "tessera.prod", 1, 1, lowering="reduction"),
    OpSpec("amax", "tessera.amax", 1, 1, lowering="reduction"),
    OpSpec("amin", "tessera.amin", 1, 1, lowering="reduction"),
    OpSpec("var", "tessera.var", 1, 1, lowering="reduction"),
    OpSpec("std", "tessera.std", 1, 1, lowering="reduction"),
    OpSpec("argmax", "tessera.argmax", 1, 1, lowering="reduction"),
    OpSpec("argmin", "tessera.argmin", 1, 1, lowering="reduction"),
    OpSpec("cumsum", "tessera.cumsum", 1, 1, lowering="reduction"),
    OpSpec("cumprod", "tessera.cumprod", 1, 1, lowering="reduction"),
    # S2 — numerical-stability primitives.
    OpSpec("logsumexp", "tessera.logsumexp", 1, 1, lowering="stable_reduction"),
    OpSpec("log_softmax", "tessera.log_softmax", 1, 1, lowering="stable_reduction"),
    OpSpec("log1p", "tessera.log1p", 1, 1),
    OpSpec("expm1", "tessera.expm1", 1, 1),
    OpSpec("softplus", "tessera.softplus", 1, 1),
    OpSpec("sigmoid_safe", "tessera.sigmoid_safe", 1, 1, lowering="stable_reduction"),
    # S2 — numeric helpers + comparisons.
    OpSpec("clamp", "tessera.clamp", 1, 1, lowering="numeric_helper"),
    OpSpec("where", "tessera.where", 3, 3, lowering="numeric_helper"),
    OpSpec("absolute", "tessera.absolute", 1, 1, lowering="numeric_helper"),
    OpSpec("sign", "tessera.sign", 1, 1, lowering="numeric_helper"),
    OpSpec("minimum", "tessera.minimum", 2, 2, lowering="numeric_helper"),
    OpSpec("maximum", "tessera.maximum", 2, 2, lowering="numeric_helper"),
    OpSpec("isnan", "tessera.isnan", 1, 1, lowering="numeric_helper"),
    OpSpec("isinf", "tessera.isinf", 1, 1, lowering="numeric_helper"),
    OpSpec("isfinite", "tessera.isfinite", 1, 1, lowering="numeric_helper"),
    OpSpec("eq", "tessera.eq", 2, 2, lowering="comparison"),
    OpSpec("ne", "tessera.ne", 2, 2, lowering="comparison"),
    OpSpec("lt", "tessera.lt", 2, 2, lowering="comparison"),
    OpSpec("le", "tessera.le", 2, 2, lowering="comparison"),
    OpSpec("gt", "tessera.gt", 2, 2, lowering="comparison"),
    OpSpec("ge", "tessera.ge", 2, 2, lowering="comparison"),
]

OP_SPECS: dict[str, OpSpec] = {spec.public_name: spec for spec in _SPECS}
GRAPH_OP_TO_SPEC: dict[str, OpSpec] = {spec.graph_name: spec for spec in _SPECS}
GRAPH_OP_MAP: dict[str, str] = {spec.public_name: spec.graph_name for spec in _SPECS}
SUPPORTED_CPU_OPS: frozenset[str] = frozenset(GRAPH_OP_TO_SPEC)
LEGACY_GRAPH_OP_ALIASES: dict[str, str] = {
    "tessera.gemm": "tessera.matmul",
    "tessera.conv2d": "tessera.conv2d_nhwc",
}


def normalize_op_name(name: str) -> str:
    """Return the public Tessera op name from a bare or qualified call name."""

    if name.startswith("tessera.ops."):
        return name.removeprefix("tessera.ops.")
    if name.startswith("ts.ops."):
        return name.removeprefix("ts.ops.")
    if name.startswith("ops."):
        return name.removeprefix("ops.")
    if name.startswith("op."):
        return name.removeprefix("op.")
    if name.startswith("tessera."):
        tail = name.removeprefix("tessera.")
        if tail.startswith("kv_cache."):
            return tail.replace(".", "_")
        return tail
    return name


def get_op_spec(name: str) -> Optional[OpSpec]:
    return OP_SPECS.get(normalize_op_name(name))


def graph_name_for(name: str) -> Optional[str]:
    spec = get_op_spec(name)
    return spec.graph_name if spec is not None else None


def canonical_graph_op_name(name: str) -> str:
    """Return the ODS-backed canonical Graph IR op name."""

    return LEGACY_GRAPH_OP_ALIASES.get(name, name)


__all__ = [
    "GRAPH_OP_MAP",
    "GRAPH_OP_TO_SPEC",
    "LEGACY_GRAPH_OP_ALIASES",
    "OP_SPECS",
    "SUPPORTED_CPU_OPS",
    "OpSpec",
    "canonical_graph_op_name",
    "get_op_spec",
    "graph_name_for",
    "normalize_op_name",
]
