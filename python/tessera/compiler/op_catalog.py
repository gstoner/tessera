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
    OpSpec("grouped_gemm", "tessera.grouped_gemm", 3, 3, lowering="loop_nest"),
    OpSpec("moe_swiglu_block", "tessera.moe_swiglu_block", 5, 5, lowering="loop_nest"),
    # Fused dequantize-into-GEMM (model-class roadmap M1): packed low-precision
    # weight codes + a separate per-group scale operand → fp32-accumulated GEMM.
    # operands: (x, w_codes, [w_scales]); grouped form adds group_sizes.
    OpSpec("dequant_matmul", "tessera.dequant_matmul", 2, 3, lowering="loop_nest"),
    OpSpec("dequant_grouped_gemm", "tessera.dequant_grouped_gemm", 3, 4, lowering="loop_nest"),
    # P3 (docs/audit/backend/apple/archive/apple_backend_capability_roadmap.md): PACKED int4 quantized matmul —
    # O = X @ dequant(W[N,K])^T with W stored as packed 4-bit codes (0.5 B/weight,
    # ~8× less weight traffic than dequant_matmul's f32 codes) + per-group affine
    # scale/bias. operands: (x, w_packed_codes, scales, biases); group_size attr.
    OpSpec("quantized_matmul", "tessera.quantized_matmul", 4, 4, lowering="loop_nest"),
    OpSpec("tri_solve", "tessera.tri_solve", 2, 2, lowering="linalg_solver"),
    OpSpec("cholesky_solve", "tessera.cholesky_solve", 2, 2, lowering="linalg_solver"),
    OpSpec("cholesky", "tessera.cholesky", 1, 1, lowering="linalg_decomposition"),
    OpSpec("qr", "tessera.qr", 1, 1, lowering="linalg_decomposition"),
    OpSpec("svd", "tessera.svd", 1, 1, lowering="linalg_decomposition"),
    OpSpec("lu", "tessera.lu", 1, 1, lowering="linalg_decomposition"),
    OpSpec("conv2d", "tessera.conv2d_nhwc", 2, 4, lowering="stencil"),
    OpSpec("conv3d", "tessera.conv3d_ndhwc", 2, 4, lowering="stencil"),
    OpSpec("layer_norm", "tessera.layer_norm", 1, 1, lowering="normalization"),
    OpSpec("softmax", "tessera.softmax", 1, 1, lowering="stable_reduction"),
    OpSpec("softmax_safe", "tessera.softmax_safe", 1, 1, lowering="stable_reduction"),
    OpSpec("reduce", "tessera.reduce", 1, 1, lowering="stable_reduction"),
    OpSpec("sum", "tessera.reduce", 1, 1, lowering="stable_reduction"),
    OpSpec("gelu", "tessera.gelu", 1, 1),
    OpSpec("tanh", "tessera.tanh", 1, 1),
    # Gemma-style logit soft-cap: cap * tanh(x / cap). Differentiable.
    OpSpec("softcap", "tessera.softcap", 1, 1),
    OpSpec("add", "tessera.add", 1, 2),
    OpSpec("mul", "tessera.mul", 1, 2),
    # Diffusion guidance score composition. Kept as a simple compiler-visible
    # numeric primitive: base + gamma * delta. CGG orchestration remains in the
    # library; this op is the IR bridge for the composition algebra.
    OpSpec("score_combine", "tessera.score_combine", 2, 2),
    OpSpec("relu", "tessera.relu", 1, 1),
    OpSpec("silu", "tessera.silu", 1, 1),
    OpSpec("silu_mul", "tessera.silu_mul", 2, 2),
    OpSpec("sigmoid", "tessera.sigmoid", 1, 1),
    OpSpec("sin", "tessera.sin", 1, 1),
    # Theme 9 — utility tensor ops. `arange` has no differentiable inputs;
    # the rest follow the standard elementwise / shape pattern.
    OpSpec("arange", "tessera.arange", 0, 3, lowering="layout_transform"),
    OpSpec("gather", "tessera.gather", 2, 2, lowering="indexing"),
    OpSpec("clip", "tessera.clip", 1, 1),
    OpSpec("masked_fill", "tessera.masked_fill", 2, 2, lowering="layout_transform"),
    OpSpec("adam", "tessera.adam", 4, 4, lowering="functional_optimizer_step"),
    OpSpec("adamw", "tessera.adamw", 2, 3, lowering="functional_optimizer_step"),
    OpSpec("momentum", "tessera.momentum", 2, 3, lowering="functional_optimizer_step"),
    OpSpec("adafactor", "tessera.adafactor", 2, 3, lowering="functional_optimizer_step"),
    OpSpec("lion", "tessera.lion", 2, 3, lowering="functional_optimizer_step"),
    # `ebm_energy_quadratic` is canonicalized to the flat-lane graph name
    # `tessera.ebm_energy_quadratic` below; the dotted Graph IR ODS spelling
    # `tessera.ebm.energy_quadratic` is a LEGACY_GRAPH_OP_ALIASES entry so it
    # does not collide on public_name with the canonical flat-lane OpSpec.
    OpSpec("ebm_langevin_step", "tessera.ebm.langevin_step", 3, 3,
           lowering="ebm"),
    OpSpec("transpose", "tessera.transpose", 1, 1, lowering="layout_transform"),
    OpSpec("cast", "tessera.cast", 1, 1, lowering="layout_transform"),
    OpSpec("dropout", "tessera.dropout", 1, 1, effect="random", lowering="random_mask"),
    OpSpec("qkv_projection", "tessera.qkv_projection", 2, 2, lowering="projection"),
    OpSpec("flash_attn", "tessera.flash_attn", 3, 4, effect="state", lowering="attention"),
    # Variable-length (packed-sequence) SDPA — Cosmos-3 "two-way flat attention"
    # IR contract. Operands: q, k, v, cu_seqlens_q, cu_seqlens_k (all required).
    OpSpec("varlen_sdpa", "tessera.varlen_sdpa", 5, 5, effect="pure", lowering="attention"),
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
    # Gap 4 (2026-05-20): 2D spatial-grid local-window attention.
    OpSpec("attn_local_window_2d", "tessera.attn_local_window_2d", 3, 3, effect="state", lowering="attention"),
    OpSpec("attn_compressed_blocks", "tessera.attn_compressed_blocks", 3, 3, effect="state", lowering="attention"),
    OpSpec("attn_top_k_blocks", "tessera.attn_top_k_blocks", 3, 3, effect="state", lowering="attention"),
    OpSpec("deepseek_sparse_attention", "tessera.deepseek_sparse_attention", 3, 4, effect="state", lowering="attention"),
    # MiniMax Sparse Attention (MSA, arXiv:2606.13392) — Index Branch (per-GQA-
    # group exp-free block scoring) + exact block-sparse Main Branch. The index
    # scorer is a smooth (differentiable) matmul; the block selector is a hard,
    # deterministic top-k (non-differentiable); the sparse attention is the
    # exact main branch. See docs/msa.md.
    OpSpec("msa_index_scores", "tessera.msa_index_scores", 2, 2, lowering="attention"),
    OpSpec("msa_select_blocks", "tessera.msa_select_blocks", 1, 1, lowering="indexing"),
    OpSpec("msa_sparse_attention", "tessera.msa_sparse_attention", 3, 3, effect="state", lowering="attention"),
    # Lookahead Sparse Attention (LSA) — experimental, inference-only. See
    # docs/audit/domain/archive/lsa_scope.md (D1-D5). `memory_index_select` is a
    # sigmoid-threshold block selector (non-differentiable, deterministic);
    # `lookahead_sparse_attention` is the composite policy op (local window ∪
    # selected historical blocks) that composes through the existing sparse
    # attention lane.
    OpSpec("memory_index_select", "tessera.memory_index_select", 2, 2, lowering="indexing"),
    # Differentiable indexer-training surface (the keys are learnable through
    # these even though the hard selector above is not). memory_index_score is
    # the smooth scoring head; memory_index_select_ste is hard-forward /
    # straight-through-backward.
    OpSpec("memory_index_score", "tessera.memory_index_score", 2, 2, lowering="attention"),
    OpSpec("memory_index_select_ste", "tessera.memory_index_select_ste", 2, 2, lowering="indexing"),
    OpSpec("lookahead_sparse_attention", "tessera.lookahead_sparse_attention", 3, 3, effect="state", lowering="attention"),
    OpSpec("gated_attention", "tessera.gated_attention", 4, 4, effect="state", lowering="attention"),
    OpSpec("hybrid_attention", "tessera.hybrid_attention", 3, 3, effect="state", lowering="attention"),
    OpSpec("lightning_attention", "tessera.lightning_attention", 3, 3, effect="state", lowering="attention"),
    OpSpec("gated_deltanet", "tessera.gated_deltanet", 3, 6, effect="state", lowering="attention"),
    OpSpec("kimi_delta_attention", "tessera.kimi_delta_attention", 3, 6, effect="state", lowering="attention"),
    OpSpec("modified_delta_attention", "tessera.modified_delta_attention", 3, 6, effect="state", lowering="attention"),
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
    # Group/instance/weight norm — reduce-then-normalize over a reshaped view, so
    # apple_gpu composes them from the rowop (layer_norm) + reduce opcode lanes.
    OpSpec("group_norm", "tessera.group_norm", 1, 3, lowering="normalization"),
    OpSpec("instance_norm", "tessera.instance_norm", 1, 3, lowering="normalization"),
    OpSpec("weight_norm", "tessera.weight_norm", 1, 1, lowering="normalization"),
    OpSpec("rope", "tessera.rope", 2, 2, lowering="rotary_embedding"),
    OpSpec("kv_cache_append", "tessera.kv_cache.append", 3, 3, effect="state", lowering="state_update"),
    OpSpec("kv_cache_prune", "tessera.kv_cache.prune", 1, 1, effect="state", lowering="state_update"),
    # ``end`` is optional at the Python surface. The explicit device_verified_jit form
    # carries (cache, start, end), while a single-token read carries two.
    OpSpec("kv_cache_read", "tessera.kv_cache.read", 2, 3, effect="state", lowering="state_update"),
    # SD1-3 — speculative-decode cache cursor ops (typed state effect, no device
    # kernel; ride KVCacheHandle.trim / SSMStateHandle.rollback).
    OpSpec("cache_commit", "tessera.cache.commit", 2, 2, effect="state", lowering="state_update"),
    OpSpec("cache_rollback", "tessera.cache.rollback", 2, 2, effect="state", lowering="state_update"),
    # SD1 — speculative-decode acceptance. spec_accept is a pure verifier
    # (draft/target → [path, length, bonus]); the cache commit/rollback live on
    # the state-effecting kv/ssm handles.
    OpSpec("spec_accept", "tessera.spec_accept", 2, 2, lowering="acceptance_verification"),
    # SD1-2 — distribution-preserving (Leviathan) rejection-sampling acceptance.
    # Pure given the explicit uniforms (accept_u, resid_u); CDF-inversion sampler.
    OpSpec("spec_accept_sample", "tessera.spec_accept_sample", 5, 5, lowering="acceptance_verification"),
    # Tree (multi-path) Leviathan rejection acceptance — device form of
    # speculative.batch_verify. (target_lp, draft_lp, accept_u) -> [path, length].
    OpSpec("spec_accept_tree_sample", "tessera.spec_accept_tree_sample", 3, 3, lowering="acceptance_verification"),
    # SD1-4 — speculative-decode target-verification I/O contract: (tokens, logits)
    # -> S×V log-probs. A composed-call marker (pure), reuses the verification
    # category (no fused kernel — that's a DK-track concern).
    OpSpec("target_verify", "tessera.target_verify", 2, 2, lowering="acceptance_verification"),
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
    OpSpec("alibi", "tessera.alibi", 0, 2, lowering="position_encoding"),
    OpSpec("ntk_rope", "tessera.ntk_rope", 2, 2, lowering="position_encoding"),
    OpSpec("multi_head_attention", "tessera.multi_head_attention", 3, 3, effect="state", lowering="attention"),
    OpSpec("gqa_attention", "tessera.gqa_attention", 3, 3, effect="state", lowering="attention"),
    OpSpec("mqa_attention", "tessera.mqa_attention", 3, 3, effect="state", lowering="attention"),
    OpSpec("mla_decode", "tessera.mla_decode", 3, 5, effect="state", lowering="attention"),
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
    OpSpec("cummax", "tessera.cummax", 1, 1, lowering="reduction"),
    OpSpec("cummin", "tessera.cummin", 1, 1, lowering="reduction"),
    OpSpec("max", "tessera.max", 1, 1, lowering="reduction"),
    OpSpec("min", "tessera.min", 1, 1, lowering="reduction"),
    # S2 — numerical-stability primitives.
    OpSpec("logsumexp", "tessera.logsumexp", 1, 1, lowering="stable_reduction"),
    OpSpec("log_softmax", "tessera.log_softmax", 1, 1, lowering="stable_reduction"),
    OpSpec("log1p", "tessera.log1p", 1, 1),
    OpSpec("expm1", "tessera.expm1", 1, 1),
    OpSpec("softplus", "tessera.softplus", 1, 1),
    OpSpec("sigmoid_safe", "tessera.sigmoid_safe", 1, 1, lowering="stable_reduction"),
    # S2 — scalar math breadth.
    OpSpec("sub", "tessera.sub", 2, 2),
    OpSpec("div", "tessera.div", 2, 2),
    OpSpec("floor_div", "tessera.floor_div", 2, 2),
    OpSpec("mod", "tessera.mod", 2, 2),
    OpSpec("exp", "tessera.exp", 1, 1),
    OpSpec("log", "tessera.log", 1, 1),
    OpSpec("sqrt", "tessera.sqrt", 1, 1),
    OpSpec("rsqrt", "tessera.rsqrt", 1, 1),
    OpSpec("pow", "tessera.pow", 2, 2),
    OpSpec("cos", "tessera.cos", 1, 1),
    OpSpec("tan", "tessera.tan", 1, 1),
    OpSpec("sinh", "tessera.sinh", 1, 1),
    OpSpec("cosh", "tessera.cosh", 1, 1),
    OpSpec("asin", "tessera.asin", 1, 1),
    OpSpec("acos", "tessera.acos", 1, 1),
    OpSpec("atan", "tessera.atan", 1, 1),
    OpSpec("atan2", "tessera.atan2", 2, 2),
    OpSpec("erf", "tessera.erf", 1, 1),
    OpSpec("erfc", "tessera.erfc", 1, 1),
    OpSpec("lgamma", "tessera.lgamma", 1, 1),
    OpSpec("digamma", "tessera.digamma", 1, 1),
    # S2 — numeric helpers + comparisons + logical/bitwise.
    OpSpec("clamp", "tessera.clamp", 1, 1, lowering="numeric_helper"),
    OpSpec("where", "tessera.where", 3, 3, lowering="numeric_helper"),
    OpSpec("absolute", "tessera.absolute", 1, 1, lowering="numeric_helper"),
    OpSpec("abs", "tessera.absolute", 1, 1, lowering="numeric_helper"),
    OpSpec("sign", "tessera.sign", 1, 1, lowering="numeric_helper"),
    OpSpec("reciprocal", "tessera.reciprocal", 1, 1, lowering="numeric_helper"),
    OpSpec("floor", "tessera.floor", 1, 1, lowering="numeric_helper"),
    OpSpec("ceil", "tessera.ceil", 1, 1, lowering="numeric_helper"),
    OpSpec("round", "tessera.round", 1, 1, lowering="numeric_helper"),
    OpSpec("trunc", "tessera.trunc", 1, 1, lowering="numeric_helper"),
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
    OpSpec("logical_and", "tessera.logical_and", 2, 2, lowering="logical"),
    OpSpec("logical_or", "tessera.logical_or", 2, 2, lowering="logical"),
    OpSpec("logical_not", "tessera.logical_not", 1, 1, lowering="logical"),
    OpSpec("logical_xor", "tessera.logical_xor", 2, 2, lowering="logical"),
    OpSpec("bitwise_and", "tessera.bitwise_and", 2, 2, lowering="logical"),
    OpSpec("bitwise_or", "tessera.bitwise_or", 2, 2, lowering="logical"),
    OpSpec("bitwise_xor", "tessera.bitwise_xor", 2, 2, lowering="logical"),
    OpSpec("bitwise_not", "tessera.bitwise_not", 1, 1, lowering="logical"),
    # S2 — tensor algebra and functional indexing. Most shape parameters are
    # kwargs in the Python surface, so arity only counts differentiable tensor
    # operands / sequence operands.
    OpSpec("reshape", "tessera.reshape", 1, 1, lowering="layout_transform"),
    OpSpec("view", "tessera.view", 1, 1, lowering="layout_transform"),
    OpSpec("flatten", "tessera.flatten", 1, 1, lowering="layout_transform"),
    OpSpec("squeeze", "tessera.squeeze", 1, 1, lowering="layout_transform"),
    OpSpec("unsqueeze", "tessera.unsqueeze", 1, 1, lowering="layout_transform"),
    OpSpec("permute", "tessera.permute", 1, 1, lowering="layout_transform"),
    OpSpec("broadcast", "tessera.broadcast", 1, 1, lowering="layout_transform"),
    OpSpec("expand", "tessera.expand", 1, 1, lowering="layout_transform"),
    # cat/stack are variadic: ``cat([a, b, …], axis)`` flattens its tensor list
    # into N≥1 operands (the AST/runtime frontends expand the list), so the spec
    # accepts a range rather than a fixed arity-1.
    OpSpec("cat", "tessera.cat", 1, 64, lowering="layout_transform"),
    OpSpec("stack", "tessera.stack", 1, 64, lowering="layout_transform"),
    OpSpec("split", "tessera.split", 1, 1, lowering="layout_transform"),
    OpSpec("chunk", "tessera.chunk", 1, 1, lowering="layout_transform"),
    OpSpec("pad", "tessera.pad", 1, 1, lowering="layout_transform"),
    OpSpec("tile", "tessera.tile", 1, 1, lowering="layout_transform"),
    OpSpec("repeat", "tessera.repeat", 1, 1, lowering="layout_transform"),
    OpSpec("roll", "tessera.roll", 1, 1, lowering="layout_transform"),
    OpSpec("flip", "tessera.flip", 1, 1, lowering="layout_transform"),
    OpSpec("slice", "tessera.slice", 1, 1, lowering="indexing"),
    OpSpec("select", "tessera.select", 1, 1, lowering="indexing"),
    OpSpec("dynamic_slice", "tessera.dynamic_slice", 1, 1, lowering="indexing"),
    OpSpec("dynamic_update_slice", "tessera.dynamic_update_slice", 2, 2, lowering="indexing"),
    OpSpec("take", "tessera.take", 2, 2, lowering="indexing"),
    OpSpec("index_select", "tessera.index_select", 2, 2, lowering="indexing"),
    OpSpec("scatter", "tessera.scatter", 3, 3, lowering="indexing"),
    OpSpec("scatter_add", "tessera.scatter_add", 3, 3, lowering="indexing"),
    OpSpec("scatter_reduce", "tessera.scatter_reduce", 3, 3, lowering="indexing"),
    OpSpec("index_update", "tessera.index_update", 3, 3, lowering="indexing"),
    OpSpec("nonzero", "tessera.nonzero", 1, 1, lowering="indexing"),
    # LDT / lattice reasoning primitives.
    OpSpec("count_nonzero", "tessera.count_nonzero", 1, 1, lowering="reduction"),
    OpSpec("popcount", "tessera.popcount", 1, 1, lowering="elementwise"),
    OpSpec("masked_categorical", "tessera.masked_categorical", 2, 2,
           effect="random", lowering="indexing"),
    # Geometric-algebra (Clifford Cl(3,0)) flat-coefficient lane. These are the
    # canonical tessera.ops projection of the tessera.ga.* Multivector surface;
    # the apple_gpu runtime routes them to the cl30 MSL kernels (see runtime.py
    # _apple_gpu_dispatch_clifford). Bilinear products = loop_nest; the diagonal
    # ±1 involutions/projection = elementwise; the scalar norms = reduction.
    OpSpec("clifford_geometric_product", "tessera.clifford_geometric_product", 2, 2, lowering="loop_nest"),
    OpSpec("clifford_wedge", "tessera.clifford_wedge", 2, 2, lowering="loop_nest"),
    OpSpec("clifford_left_contraction", "tessera.clifford_left_contraction", 2, 2, lowering="loop_nest"),
    OpSpec("clifford_inner", "tessera.clifford_inner", 2, 2, lowering="loop_nest"),
    OpSpec("clifford_rotor_sandwich", "tessera.clifford_rotor_sandwich", 2, 2, lowering="loop_nest"),
    OpSpec("clifford_reverse", "tessera.clifford_reverse", 1, 1, lowering="elementwise"),
    OpSpec("clifford_grade_involution", "tessera.clifford_grade_involution", 1, 1, lowering="elementwise"),
    OpSpec("clifford_conjugate", "tessera.clifford_conjugate", 1, 1, lowering="elementwise"),
    OpSpec("clifford_grade_projection", "tessera.clifford_grade_projection", 1, 1, lowering="elementwise"),
    OpSpec("clifford_hodge_star", "tessera.clifford_hodge_star", 1, 1, lowering="elementwise"),
    OpSpec("clifford_ext_deriv", "tessera.clifford_ext_deriv", 1, 1, lowering="stencil"),
    OpSpec("clifford_vec_deriv", "tessera.clifford_vec_deriv", 1, 1, lowering="stencil"),
    OpSpec("clifford_codiff", "tessera.clifford_codiff", 1, 1, lowering="stencil"),
    OpSpec("clifford_exp", "tessera.clifford_exp", 1, 1, lowering="elementwise"),
    OpSpec("clifford_log", "tessera.clifford_log", 1, 1, lowering="elementwise"),
    OpSpec("clifford_norm", "tessera.clifford_norm", 1, 1, lowering="reduction"),
    OpSpec("clifford_norm_squared", "tessera.clifford_norm_squared", 1, 1, lowering="reduction"),
    # Energy-based-model (EBM) flat-array lane — canonical tessera.ops projection
    # of the tensor-clean tessera.ebm.* subset; apple_gpu routes them to the EBM
    # MSL kernels (see runtime.py _apple_gpu_dispatch_ebm).
    OpSpec("ebm_energy_quadratic", "tessera.ebm_energy_quadratic", 2, 2, lowering="reduction"),
    OpSpec("ebm_self_verify", "tessera.ebm_self_verify", 2, 2, lowering="indexing"),
    OpSpec("ebm_refinement", "tessera.ebm_refinement", 2, 2, lowering="elementwise"),
    OpSpec("ebm_inner_step", "tessera.ebm_inner_step", 2, 2, lowering="elementwise"),
    OpSpec("top_k", "tessera.top_k", 1, 1, lowering="sort"),
    OpSpec("sort", "tessera.sort", 1, 1, lowering="sort"),
    OpSpec("argsort", "tessera.argsort", 1, 1, lowering="sort"),
    # S7/S10/S11 focused Graph IR entrypoints. These are Python-reference
    # primitives promoted into the frontend catalog so the Graph IR builder can
    # emit stable op names instead of treating them as opaque calls.
    OpSpec("linear_general", "tessera.linear_general", 2, 3, lowering="model_layer"),
    OpSpec("sgd", "tessera.sgd", 2, 2, lowering="functional_optimizer_step"),
    OpSpec("mse_loss", "tessera.loss.mse", 2, 2, lowering="loss"),
    OpSpec("mae_loss", "tessera.loss.mae", 2, 2, lowering="loss"),
    OpSpec("huber_loss", "tessera.loss.huber", 2, 2, lowering="loss"),
    OpSpec("smooth_l1_loss", "tessera.loss.smooth_l1", 2, 2, lowering="loss"),
    OpSpec("log_cosh_loss", "tessera.loss.log_cosh", 2, 2, lowering="loss"),
    OpSpec("cross_entropy_loss", "tessera.loss.cross_entropy", 2, 2, lowering="loss"),
    OpSpec("binary_cross_entropy_loss", "tessera.loss.binary_cross_entropy", 2, 2, lowering="loss"),
    OpSpec("asymmetric_bce", "tessera.loss.asymmetric_bce", 2, 2, lowering="loss"),
    OpSpec("z_loss", "tessera.loss.z_loss", 1, 1, lowering="loss"),
    OpSpec("load_balance_loss", "tessera.loss.load_balance_loss", 1, 1, lowering="loss"),
    OpSpec("ddpm_noise_pred_loss", "tessera.loss.ddpm_noise_pred", 2, 2, lowering="loss"),
    OpSpec("score_matching_loss", "tessera.loss.score_matching", 2, 2, lowering="loss"),
    # Distribution-matching losses — pure exp/log/sub/mul/sum-last-axis chains,
    # so apple_gpu composes them from the batch-1/2 opcode lanes (runtime.py).
    OpSpec("kl_divergence", "tessera.loss.kl_divergence", 2, 2, lowering="loss"),
    OpSpec("js_divergence", "tessera.loss.js_divergence", 2, 2, lowering="loss"),
    # EBM training losses (#5) — reductions over energy/score tensors; apple_gpu
    # routes reduction="mean" to the EBM-loss MPSGraph kernels (runtime.py).
    OpSpec("contrastive_divergence_loss", "tessera.loss.contrastive_divergence", 2, 2, lowering="loss"),
    OpSpec("persistent_cd_loss", "tessera.loss.persistent_cd", 2, 2, lowering="loss"),
    OpSpec("implicit_score_matching_loss", "tessera.loss.implicit_score_matching", 2, 2, lowering="loss"),
    OpSpec("denoising_score_matching_loss", "tessera.loss.denoising_score_matching", 3, 3, lowering="loss"),
    OpSpec("vlb_loss", "tessera.loss.vlb", 1, 1, lowering="loss"),
    OpSpec("ppo_policy_loss", "tessera.rl.ppo_policy_loss", 3, 6, lowering="rl_loss"),
    OpSpec("grpo_policy_loss", "tessera.rl.grpo_policy_loss", 2, 3, lowering="rl_loss"),
    OpSpec("cispo_policy_loss", "tessera.rl.cispo_policy_loss", 2, 3, lowering="rl_loss"),
    OpSpec("normalize_group_advantages", "tessera.rl.normalize_group_advantages", 1, 1, lowering="rl_loss"),
    # State-space / Mamba2 selective scan.  Inputs: x, A, B, C, [D, initial_state].
    # Lowered as a stateful sequence-axis scan (`state_space` lowering kind).
    OpSpec("selective_ssm", "tessera.selective_ssm", 4, 6, effect="state", lowering="state_space"),

    # M7 Visual Complex Analysis (E3, 2026-05-20).  These ops give the
    # M7 long-tail a real Graph IR identity so the frontend can emit
    # stable op names instead of treating ``tessera.complex.*`` calls
    # as opaque host code.  Lowering kinds:
    #   - ``elementwise``: pointwise over packed (re, im) tensors.
    #     Same lowering family as ``gelu`` / ``silu`` / ``sigmoid``.
    #   - ``stencil``: Wirtinger derivatives ∂/∂z + ∂/∂z̄ + Laplacian
    #     are 3×3 stencils on the (re, im) field.  Halo width = 1.
    # The first 4 (complex_mul/exp + mobius/stereographic) are already
    # E2-promoted via manifest dispatch — we list them here too so the
    # Graph IR builder can emit canonical tessera.* op names instead
    # of falling through to the opaque-call path.
    # — Pointwise complex math (7) —
    OpSpec("complex_mul",        "tessera.complex_mul",        2, 2),
    OpSpec("complex_div",        "tessera.complex_div",        2, 2),
    OpSpec("complex_exp",        "tessera.complex_exp",        1, 1),
    OpSpec("complex_log",        "tessera.complex_log",        1, 1),
    OpSpec("complex_sqrt",       "tessera.complex_sqrt",       1, 1),
    OpSpec("complex_pow",        "tessera.complex_pow",        2, 2),
    OpSpec("complex_conjugate",  "tessera.complex_conjugate",  1, 1),
    OpSpec("complex_abs",        "tessera.complex_abs",        1, 1),
    OpSpec("complex_arg",        "tessera.complex_arg",        1, 1),
    # — Möbius / projective family (3) —
    OpSpec("mobius",                   "tessera.mobius",                   2, 2),
    OpSpec("mobius_from_three_points", "tessera.mobius_from_three_points", 2, 2),
    OpSpec("stereographic",            "tessera.stereographic",            1, 1),
    # — Cross-ratio / cocircularity / Cauchy-Riemann certificate (3) —
    OpSpec("cross_ratio",          "tessera.cross_ratio",          4, 4),
    OpSpec("is_concyclic",         "tessera.is_concyclic",         4, 4),
    OpSpec("check_cauchy_riemann", "tessera.check_cauchy_riemann", 1, 1, lowering="stencil"),
    # — Wirtinger derivatives + Laplacian (3 stencils) —
    OpSpec("dz",           "tessera.dz",           1, 1, lowering="stencil"),
    OpSpec("dbar",         "tessera.dbar",         1, 1, lowering="stencil"),
    OpSpec("laplacian_2d", "tessera.laplacian_2d", 1, 1, lowering="stencil"),
    # — Conformal Jacobian + energy on sphere (2) —
    OpSpec("conformal_jacobian",         "tessera.conformal_jacobian",         1, 1, lowering="stencil"),
    OpSpec("conformal_energy_on_sphere", "tessera.conformal_energy_on_sphere", 1, 1, lowering="stable_reduction"),
]

OP_SPECS: dict[str, OpSpec] = {spec.public_name: spec for spec in _SPECS}
GRAPH_OP_TO_SPEC: dict[str, OpSpec] = {spec.graph_name: spec for spec in _SPECS}
GRAPH_OP_MAP: dict[str, str] = {spec.public_name: spec.graph_name for spec in _SPECS}
SUPPORTED_CPU_OPS: frozenset[str] = frozenset(GRAPH_OP_TO_SPEC)
LEGACY_GRAPH_OP_ALIASES: dict[str, str] = {
    "tessera.gemm": "tessera.matmul",
    "tessera.conv2d": "tessera.conv2d_nhwc",
    # Dotted Graph IR ODS spelling → canonical flat EBM lane op (see op spec note).
    "tessera.ebm.energy_quadratic": "tessera.ebm_energy_quadratic",
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
    for prefix in (
        "tessera.losses.",
        "ts.losses.",
        "losses.",
        "tessera.optim.",
        "ts.optim.",
        "optim.",
        "tessera.rl.",
        "ts.rl.",
        "rl.",
        "tessera.nn.",
        "ts.nn.",
        "nn.",
        "tessera.memory.",
        "ts.memory.",
        "memory.",
    ):
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    if name.startswith("tessera."):
        tail = name.removeprefix("tessera.")
        # Dotted stateful-cache graph names map to their underscore public specs
        # (kv_cache.append → kv_cache_append; cache.commit → cache_commit), so
        # get_op_spec resolves them and downstream effect inference sees the state
        # write rather than defaulting a lowered graph name to pure.
        if tail.startswith("kv_cache.") or tail.startswith("cache."):
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
