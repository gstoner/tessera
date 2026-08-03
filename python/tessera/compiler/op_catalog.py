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
    # W1.2 — the op's shape rule, by NAME. Empty means "take the declared
    # default for my `lowering` kind" (see LOWERING_SHAPE_RULE); it does not
    # mean "no rule". An op whose lowering kind has no declared default is
    # `unclassified`, which is a real, counted status -- not a silent fallback.
    shape_rule: str = ""

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
    # Optional affine operands are gamma and beta, in that order. RMSNorm has
    # gamma only. Their Graph ABI remains backward-compatible with unary calls.
    OpSpec("layer_norm", "tessera.layer_norm", 1, 3, lowering="normalization"),
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
    OpSpec("nesterov", "tessera.nesterov", 2, 3, lowering="functional_optimizer_step"),
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
    # 4 operands, not 3: `scores` is a keyword-only TENSOR (B, H, S_q,
    # num_blocks), unwrapped and asarray'd by the reference exactly like Q/K/V.
    # The positional-only arity gate could not see it, and the frontend was
    # emitting it as a string attribute rather than an operand (W1.3).
    # Contrast `varlen_sdpa`, whose keyword-only cu_seqlens WERE already counted.
    OpSpec("attn_top_k_blocks", "tessera.attn_top_k_blocks", 4, 4, effect="state", lowering="attention"),
    OpSpec("deepseek_sparse_attention", "tessera.deepseek_sparse_attention", 3, 4, effect="state", lowering="attention"),
    # MiniMax Sparse Attention (MSA, arXiv:2606.13392) — Index Branch (per-GQA-
    # group exp-free block scoring) + exact block-sparse Main Branch. The index
    # scorer is a smooth (differentiable) matmul; the block selector is a hard,
    # deterministic top-k (non-differentiable); the sparse attention is the
    # exact main branch. See docs/architecture/workloads/msa.md.
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
    # 2 required (x, experts) + 2 OPTIONAL keyword tensor operands (`scores`,
    # `route`). max_arity was 2, so a routed MoE -- the normal case for a real
    # model -- exceeded the declared arity once `route` correctly emitted as an
    # operand, and the op was dropped from the body entirely. That failure was
    # SILENT at the Graph IR level and only surfaced two stages later as
    # "schedule-ir stage was claimed but schedule_ir is empty".
    OpSpec("moe", "tessera.moe", 2, 4, effect="collective", lowering="moe"),
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
    OpSpec("rmsnorm", "tessera.rmsnorm", 1, 2, lowering="normalization"),
    OpSpec("rmsnorm_safe", "tessera.rmsnorm_safe", 1, 1, lowering="normalization"),
    # Group/instance/weight norm — reduce-then-normalize over a reshaped view, so
    # apple_gpu composes them from the rowop (layer_norm) + reduce opcode lanes.
    OpSpec("group_norm", "tessera.group_norm", 1, 3, lowering="normalization"),
    OpSpec("instance_norm", "tessera.instance_norm", 1, 3, lowering="normalization"),
    OpSpec("weight_norm", "tessera.weight_norm", 1, 1, lowering="normalization"),
    OpSpec("rope", "tessera.rope", 2, 2, lowering="rotary_embedding"),
    OpSpec("kv_cache_append", "tessera.kv_cache.append", 3, 3, effect="state", lowering="state_update"),
    OpSpec("kv_cache_prune", "tessera.kv_cache.prune", 1, 1, effect="state", lowering="state_update"),
    # ``end`` is optional at the Python surface. The explicit compiled form
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
    # Compiler-generated internal carrier produced only after autodiff. It has
    # four tensor operands (prediction, target, cotangent, parameter) and two
    # results (updated parameter, target gradient).
    OpSpec(
        "training.loss_sgd", "tessera.training.loss_sgd", 4, 4,
        lowering="optimizer",
    ),
    # Six tensor operands (loss triple plus parameter and two moments); the
    # prediction gradient is internal to the fused AdamW state transition.
    OpSpec(
        "training.loss_adamw", "tessera.training.loss_adamw", 6, 6,
        lowering="optimizer",
    ),
    OpSpec("mse_loss", "tessera.loss.mse", 2, 2, lowering="loss"),
    OpSpec("mae_loss", "tessera.loss.mae", 2, 2, lowering="loss"),
    OpSpec("huber_loss", "tessera.loss.huber", 2, 2, lowering="loss"),
    OpSpec("smooth_l1_loss", "tessera.loss.smooth_l1", 2, 2, lowering="loss"),
    OpSpec("log_cosh_loss", "tessera.loss.log_cosh", 2, 2, lowering="loss"),
    OpSpec("cross_entropy_loss", "tessera.loss.cross_entropy", 2, 2, lowering="loss"),
    OpSpec("label_smoothed_cross_entropy", "tessera.loss.cross_entropy", 2, 2, lowering="loss"),
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
    OpSpec("selective_ssm", "tessera.selective_ssm", 5, 6, effect="state", lowering="state_space"),

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
# Multiple public convenience functions may intentionally share one Graph op
# (for example label_smoothed_cross_entropy is cross_entropy with a non-zero
# attribute). Preserve the first, canonical public spelling for reverse lookup.
GRAPH_OP_TO_SPEC: dict[str, OpSpec] = {}
for _spec in _SPECS:
    GRAPH_OP_TO_SPEC.setdefault(_spec.graph_name, _spec)
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
    # ODS Graph spellings are canonical identities too.  Check that index
    # before public-name normalization so dotted names such as
    # ``tessera.loss.smooth_l1`` resolve to ``smooth_l1_loss`` instead of the
    # non-existent public key ``loss.smooth_l1``.
    graph_name = canonical_graph_op_name(name)
    graph_spec = GRAPH_OP_TO_SPEC.get(graph_name)
    if graph_spec is not None:
        return graph_spec
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


# ─────────────────────────────────────────────────────────────────────────────
# W1.2 — the one shape-rule registry
#
# `_infer_result_type` was a five-case if-chain ending in
# `return operand_types[0]`: correct for the 60 elementwise ops and silently
# wrong for anything else whose result shape differs from its first operand.
# Worse, `primitive_coverage` reported the `shape_rule` axis as CLOSED across
# 480 primitives while that if-chain was the whole implementation -- Decision
# #29's "declared but not consumed" in its purest form.
#
# The fix is not to hand-write 313 rules. It is to make the rule NAMED for
# every op, so that:
#   * the common case is declared once per lowering kind rather than implied,
#   * an op with no rule is a *counted* `unclassified`, not a silent default,
#   * `primitive_coverage.shape_rule` can auto-flip from real declarations the
#     same way it already does from `_VJPS` / `_JVPS`.
#
# Behavior is deliberately unchanged in this slice: `unclassified` still
# resolves to same-as-first-operand. What changes is that it is now visible and
# counted, so it can be driven down instead of read as closed.
# ─────────────────────────────────────────────────────────────────────────────

#: Shape rule per lowering kind. Only kinds whose shape behavior is genuinely
#: uniform are declared here; the rest resolve to `unclassified` on purpose.
LOWERING_SHAPE_RULE: dict = {
    # Result has the shape and dtype of the first operand.
    "elementwise": "same_as_first",
    "normalization": "same_as_first",
    "random_mask": "same_as_first",
    "position_encoding": "same_as_first",
    "rotary_embedding": "same_as_first",
    "numeric_helper": "same_as_first",
    # NOT declared: `functional_optimizer_step` / `optimizer`. An optimizer
    # legitimately keeps f32 master state while the parameters are bf16 --
    # that is standard mixed precision, not a dtype bug. Declaring them
    # storage-preserving would force the enforcement wrapper to *destroy* that
    # by rounding the state back to bf16. Measured: `adam` returns f32 for bf16
    # params, and it is right to.
    # A comparison yields a predicate, not a value in the operand dtype.
    "comparison": "same_shape_bool",
    # Contractions and reductions get per-op rules; naming the kind here would
    # be a guess. Left unclassified until each is declared.
}

#: Per-op rules that override the lowering-kind default.
OP_SHAPE_RULE: dict = {
    "tessera.matmul": "matmul_2d",
    "tessera.batched_gemm": "batched_gemm_3d",
    "tessera.transpose": "transpose",
    "tessera.ebm_energy_quadratic": "reduce_trailing",
    "tessera.ebm.langevin_step": "same_as_first",
    # The `logical` kind is NOT uniform: the connectives yield a predicate
    # while the bitwise ops preserve the operand's integer dtype. Declaring one
    # default for the kind would be wrong for half of it.
    "tessera.logical_and": "same_shape_bool",
    "tessera.logical_or": "same_shape_bool",
    "tessera.logical_not": "same_shape_bool",
    "tessera.logical_xor": "same_shape_bool",
    "tessera.bitwise_and": "same_as_first",
    "tessera.bitwise_or": "same_as_first",
    "tessera.bitwise_xor": "same_as_first",
    "tessera.bitwise_not": "same_as_first",
    # Predicates that happen to sit under the `numeric_helper` kind. Caught by
    # differentially probing predicted vs actual dtype -- the shape agreed, so
    # a shape-only check would have missed them exactly as it missed `eq`.
    "tessera.isnan": "same_shape_bool",
    "tessera.isinf": "same_shape_bool",
    "tessera.isfinite": "same_shape_bool",

    # ── Verified against actual op behavior (f32 AND bf16) ────────────────
    # Full reductions to a scalar; storage dtype preserved.
    **{f"tessera.{n}": "reduce_all" for n in
       ("amax", "amin", "max", "mean", "min", "prod", "std", "var",
        "logsumexp", "reduce")},
    # Full reductions yielding an index.
    **{f"tessera.{n}": "reduce_all_index" for n in
       ("argmax", "argmin", "count_nonzero")},
    # Shape-preserving: scans, softmax family, sort, and layout no-ops.
    **{f"tessera.{n}": "same_as_first" for n in
       ("cummax", "cummin", "cumprod", "cumsum", "softmax", "softmax_safe",
        "log_softmax", "sort", "flip", "squeeze", "stack", "unpack",
        "fused_epilogue", "all_reduce", "all_to_all")},
    "tessera.argsort": "same_shape_index",
    # Flattening layout transforms.
    "tessera.flatten": "flatten",
    "tessera.cat": "flatten",
    # Grade-reducing Clifford norms: drop the trailing axis.
    "tessera.clifford_norm": "reduce_trailing",
    "tessera.clifford_norm_squared": "reduce_trailing",
    # Spectral — WITHDRAWN, and the reason is worth keeping.
    #
    # `fft`/`ifft`/`rfft` genuinely return complex64; the probe confirms it.
    # But declaring that rule propagates complex64 into Graph IR, and the dtype
    # capability contracts mark complex64 `unsupported` / `planned_gated` on
    # x86 and ROCm. Decision #15a is explicit that planned/gated dtypes are not
    # first-class, so the verifier rejecting a complex-typed `dct` operand is
    # the POLICY WORKING, not a bug.
    #
    # These ops previously "passed" only because the fallback mistyped their
    # result as f32 — a wrong dtype that happened to satisfy the capability
    # check. Declaring the true rule surfaced the real conflict: the spectral
    # lane needs a first-class complex dtype, which is a dtype-policy decision
    # (Decision #15a), not a shape-rule one. Left unclassified until that is
    # taken; promoting complex64 in the capability tables here would silently
    # make a planned_gated dtype first-class.

    # ── Attribute-driven (needed the widened rule signature) ─────────────
    # Result shape lives in an attribute, not in any operand's type.
    **{f"tessera.{n}": "from_shape_attr" for n in
       ("reshape", "view", "broadcast", "expand", "tile_view")},
    "tessera.cast": "cast",

    # ── n-ary, verified by probing at f32 and bf16 ───────────────────────
    # Attention: the output carries the QUERY's shape and storage dtype. The
    # reference returned float64 for BOTH f32 and bf16 inputs -- on the single
    # hottest accelerator path -- so declaring this rule also makes the
    # storage-dtype enforcement apply to it.
    **{f"tessera.{n}": "same_as_first" for n in
       ("flash_attn", "gated_attention", "mla_decode")},
    # Elementwise-shaped indexing and transport: result keeps the data
    # operand's shape and dtype.
    **{f"tessera.{n}": "same_as_first" for n in
       ("index_update", "scatter", "take", "moe_dispatch", "spectral_filter")},
    # Clifford binary products keep the multivector shape; `inner` contracts
    # the trailing (blade) axis to a scalar per row.
    **{f"tessera.{n}": "same_as_first" for n in
       ("clifford_geometric_product", "clifford_wedge",
        "clifford_left_contraction", "clifford_rotor_sandwich")},
    "tessera.clifford_inner": "reduce_trailing",
    # Losses that reduce to a scalar in the operand's storage dtype. Several
    # returned float64 regardless of input.
    # NOTE the `loss.` prefix: the graph names are `tessera.loss.mse`, not
    # `tessera.mse_loss`. The first version of this block used the PUBLIC names
    # and therefore matched nothing -- 16 silently phantom declarations. A
    # declaration that names no real op is the same "declared but not consumed"
    # failure this registry exists to remove, so it is now drift-gated.
    **{f"tessera.loss.{n}": "reduce_all" for n in
       ("mse", "mae", "huber", "smooth_l1", "log_cosh", "cross_entropy",
        "binary_cross_entropy", "asymmetric_bce", "kl_divergence",
        "js_divergence", "contrastive_divergence", "persistent_cd",
        "ddpm_noise_pred", "score_matching", "vlb")},

    # ── Probed at f32 / bf16 / fp16; several ignored the input dtype ─────
    # These returned float64 for EVERY input dtype -- f32, bf16 and fp16 alike.
    # They are unclassified only in the sense that nobody had looked; declaring
    # the rule also FIXES them, because the storage-dtype enforcement then
    # computes at f32 and stores back at the operand's dtype.
    **{f"tessera.loss.{n}": "reduce_all" for n in ("z_loss", "load_balance_loss")},
    "tessera.rl.ppo_policy_loss": "reduce_all",
    "tessera.rl.normalize_group_advantages": "same_as_first",
    # sigmoid_safe was inconsistent: f32 for a bf16 input but f16 for an f16
    # input. Declaring it makes the behavior uniform instead of accidental.
    "tessera.sigmoid_safe": "same_as_first",
    "tessera.dct": "same_as_first",
    # Shape-reducing, dtype-preserving.
    # Derives from operand 1 (candidates), NOT operand 0 (energies): energies
    # only score, candidates carry the data. `reduce_trailing` on operand 0
    # predicted (B,) with the energies' dtype -- wrong shape AND wrong dtype.
    "tessera.ebm_self_verify": "select_from_second",
    # The quantize family is MULTI-RESULT: (codes, scale). `same_as_first` was
    # a false contract for it. nvfp4 is separated because its scale is
    # per-BLOCK (Blackwell's micro-scaled format), not per-tensor.
    **{f"tessera.quantize_{n}": "quantize_per_tensor" for n in ("fp8", "fp6", "fp4")},
    "tessera.quantize_nvfp4": "quantize_per_block",
    # dequantize is single-result: (codes, scale) -> tensor shaped like codes.
    **{f"tessera.dequantize_{n}": "same_as_first"
       for n in ("fp8", "fp6", "fp4", "nvfp4")},
    # Optimizers: (param, moment1, moment2). The param keeps its own storage
    # dtype while the moments follow the `state_dtype` attribute -- f32 master
    # state with bf16 params is standard mixed precision. Previously exempted
    # as "deliberately undeclared"; that was a vocabulary gap, not a genuine
    # exception, and the exemption also silently covered an optimizer wrongly
    # rounding its state DOWN to the param dtype.
    **{f"tessera.{n}": "optimizer_step" for n in
       ("adam", "adamw", "momentum", "nesterov", "adafactor", "lion", "sgd")},

    # Cache mutators thread the handle through -- the ODS says
    # `-> Tessera_KVCacheType:$updated` for each. `read` is the one member of
    # the family that returns TENSORS, not a handle.
    **{f"tessera.{n}": "state_handle" for n in
       ("kv_cache.append", "kv_cache.prune", "cache.commit", "cache.rollback")},
    "tessera.kv_cache.read": "kv_cache_read",

    # NOT declared on purpose: `all_gather` and `reduce_scatter` returned the
    # operand's shape only because the probe ran at world_size=1. Their real
    # shapes scale with the mesh, so declaring from that measurement would bake
    # in a degenerate case. They stay `unclassified` until probed multi-rank.
}

#: The declared vocabulary. `graph_ir` implements each name; a rule named here
#: with no implementation (or vice versa) is a drift-gated error.
SHAPE_RULE_NAMES = frozenset({
    "same_as_first",
    "matmul_2d",
    "batched_gemm_3d",
    "transpose",
    "same_shape_bool",
    "reduce_all",
    "reduce_all_index",
    "same_shape_index",
    "flatten",
    "complex_same",
    "rfft",
    "select_from_second",
    "quantize_per_tensor",
    "quantize_per_block",
    "optimizer_step",
    "from_shape_attr",
    "cast",
    "reduce_trailing",
    "state_handle",
    "kv_cache_read",
    "unclassified",
})


#: Ops EXAMINED and deliberately left without a rule, with the reason. This is
#: a third state, distinct from "not yet looked at": the shape or dtype is
#: genuinely not a function of the operands alone, or declaring a rule would
#: force wrong behavior. Answering "why is this unclassified?" is what makes the
#: remaining count meaningful.
DELIBERATELY_UNDECLARED: dict = {
    # The quantize family returns a TUPLE (codes, scale), not a single tensor,
    # so `same_as_first` was a false declaration -- it claims one result type
    # for a multi-result contract. The wrapper happened not to corrupt anything
    # (it passes non-arrays through), but a rule that misstates the contract is
    # exactly what this registry exists to remove. Declaring these needs a
    # tuple-aware rule vocabulary, which does not exist yet.
    #
    # Note the codes come back as f32, NOT as fp8/fp4 storage: this is
    # fake-quant. fp8_e4m3 / fp8_e5m2 / fp4_e2m1 / nvfp4 ARE canonical dtypes
    # in `tessera.dtype` and the per-backend contracts model them honestly
    # (gfx1151 `unsupported` -- RDNA 3.5 has no FP8 WMMA; x86 `emulated`), so
    # the type system can express the storage the reference never materializes.
    # Producing real sub-byte storage is a backend-path question, not a shape
    # rule one.
    "tessera.popcount": "returns an INTEGER bit count, never the operand's storage dtype, so it is not storage-preserving despite sitting in the `elementwise` kind. The exact integer width is NumPy-version dependent (uint8 under 2.x, int64 under 1.26), so pinning one would be wrong; declaring a shape rule needs an int-width-agnostic vocabulary first",
    "tessera.all_gather": "result shape scales with mesh size, not derivable from operand types",
    "tessera.reduce_scatter": "result shape shrinks with mesh size, not derivable from operand types",
    "tessera.fft": "returns complex64, which is planned_gated per Decision #15a; declaring it conflicts with the dtype capability contract",
    "tessera.ifft": "returns complex64, which is planned_gated per Decision #15a; declaring it conflicts with the dtype capability contract",
    "tessera.rfft": "returns complex64, which is planned_gated per Decision #15a; declaring it conflicts with the dtype capability contract",
    "tessera.irfft": "returns complex64, which is planned_gated per Decision #15a; declaring it conflicts with the dtype capability contract",
    # The five cache ops were here, under one shared sentence: "opaque cache
    # handle rather than a tensor type; its result is not describable by a
    # tensor shape rule". Both halves of that turned out to be wrong.
    #
    # The handle was never undescribable -- `Tessera_KVCacheType` has been in
    # `TesseraOps.td` the whole time, and the ODS states the signatures
    # exactly (`(!tessera.kv_cache, tensor, tensor) -> !tessera.kv_cache`).
    # What was missing was a way for the PYTHON emitter to name a non-tensor
    # type; it emitted `tensor<*x?>` at both ends and the handle silently
    # became an untyped tensor. They are now `state_handle`.
    #
    # And `kv_cache.read` does not return a handle at all -- it returns
    # `(K, V)` tensors. The shared sentence was true of four ops, so nothing
    # pointed at the fifth. It is now `kv_cache_read`.
}


def undeclared_reason(graph_name: str):
    """Why this op has no rule, when that was a decision rather than a gap."""
    return DELIBERATELY_UNDECLARED.get(graph_name)


#: Which OPERAND carries the result's storage dtype, per rule. Defaults to 0.
#: `ebm_self_verify` is the counterexample that forced this to be explicit:
#: operand 0 is a score vector and operand 1 is the data, so casting the result
#: to operand 0's dtype silently changed a bf16 candidate tensor to f32.
#: An "operand 0 is the tensor" assumption is a per-op question, not a global
#: one, and baking it in is how the wrapper produced a wrong dtype while
#: looking principled.
SHAPE_RULE_DTYPE_SOURCE: dict = {
    "select_from_second": 1,
}


def dtype_source_index(graph_name: str) -> int:
    """Index of the operand whose storage dtype the result should carry."""
    return SHAPE_RULE_DTYPE_SOURCE.get(shape_rule_for(graph_name), 0)


def shape_rule_for(graph_name: str) -> str:
    """The declared shape rule for `graph_name`.

    Returns `"unclassified"` -- an explicit, counted status -- when neither the
    op nor its lowering kind declares one. Never returns an empty string, so a
    caller cannot mistake "no rule" for "no answer".
    """
    # A deliberate non-declaration outranks any lowering-kind default. Without
    # this, an op could be listed as "examined, deliberately undeclared" while
    # `shape_rule_for` still handed back its kind's default -- and the gates
    # would enforce a rule the catalog had explicitly withdrawn. `popcount` hit
    # exactly that: excluded from the ratchet, yet still reported
    # `same_as_first` from the `elementwise` kind.
    if graph_name in DELIBERATELY_UNDECLARED:
        return "unclassified"
    explicit = OP_SHAPE_RULE.get(graph_name)
    if explicit:
        return explicit
    for spec in _SPECS:
        if spec.graph_name == graph_name:
            if spec.shape_rule:
                return spec.shape_rule
            return LOWERING_SHAPE_RULE.get(spec.lowering, "unclassified")
    return "unclassified"


def unclassified_shape_ops() -> list:
    """Graph op names whose shape rule is still `unclassified`.

    This list is a RATCHET: it may shrink, never grow. Driving it to zero is
    what actually closes W1's "no op reaches the `operand_types[0]` fallback".
    """
    seen = set()
    out = []
    for spec in _SPECS:
        if spec.graph_name in seen:
            continue
        seen.add(spec.graph_name)
        if spec.graph_name in DELIBERATELY_UNDECLARED:
            continue  # examined; the reason is recorded
        if shape_rule_for(spec.graph_name) == "unclassified":
            out.append(spec.graph_name)
    return sorted(out)


# ─────────────────────────────────────────────────────────────────────────────
# fp16 RANGE sensitivity — a separate axis from storage-dtype propagation
#
# bf16 and fp16 are the same WIDTH and fail in opposite ways, so neither
# substitutes for the other:
#
#   bf16  8 mantissa bits, f32's exponent range  -> loses RESOLUTION
#   fp16 10 mantissa bits, max 6.55e4            -> loses RANGE
#
# The shape-rule work above probes fp32 + bf16, which is right for *propagation*
# (does the op return the dtype it was given). It is blind to range: measured on
# this op set, 23 ops lose numerics at fp16 while fp32 AND bf16 are both fine.
# bf16 testing alone would never surface any of them.
#
# The failure is usually silent rather than loud. `rmsnorm_safe` at fp16 with
# 1e4 inputs returns 0.0 instead of ~1.0: sum(x**2) overflows to inf, then
# x/inf underflows to zero. No inf, no NaN, no error -- just wrong numbers from
# an op whose name asserts safety.
#
# These are DECLARED so the hazard is documented per op rather than rediscovered.
# Membership means "this op's numerics depend on fp16 range and must be probed
# there"; it does not by itself assert the op is broken.
# ─────────────────────────────────────────────────────────────────────────────

# Two DISTINCT classes, and conflating them is what made the original single
# list unactionable:
#
#   A. INTERMEDIATE overflow — the op's internal arithmetic leaves fp16 range
#      even though the ANSWER fits comfortably. `rmsnorm_safe` on 1e4 inputs
#      returns ~1.0, but computing sum(x**2) at fp16 overflows to inf and the
#      result collapses to 0.0. This is a REAL DEFECT and it is FIXED: the
#      storage-dtype enforcement now promotes reduced-precision operands to f32,
#      computes, and stores back, so the intermediate never happens at fp16.
#
#   B. RESULT unrepresentable — the answer itself exceeds fp16's 6.55e4 max.
#      A sum of 32 values of 1e4 is 3.2e5; a Clifford product of 1e4 magnitudes
#      is 4e8. No compute precision fixes that, because the number simply does
#      not fit. This is NOT a defect; it is fp16 being fp16, and the remedy
#      belongs to the caller (loss scaling, or bf16, whose max is 3.39e38).
#
# Keeping them in one bucket implied 22 things to fix. Six were fixable and are
# fixed; sixteen are a property of the format.

FP16_INTERMEDIATE_OVERFLOW: dict = {
    "tessera.rmsnorm": "sum(x**2) overflows fp16 while the normalized result is ~1.0",
    "tessera.rmsnorm_safe": "despite the name, returned 0.0 instead of ~1.0 at fp16 -- sum(x**2) -> inf, then x/inf -> 0. Fixed by computing at f32",
    "tessera.clifford_norm": "sum of squared blade coefficients overflows fp16; the norm itself fits",
    "tessera.clifford_log": "log of a large multivector norm overflowed fp16 intermediates; the log fits easily",
    "tessera.flash_attn": "QK^T contraction overflows fp16 before the softmax rescales; attention output is bounded",
    "tessera.mla_decode": "same contraction hazard as flash_attn, same bounded output",
}

FP16_RESULT_UNREPRESENTABLE: dict = {
    **{f"tessera.{n}": "a geometric product of large multivectors is ~1e8, far beyond fp16's 6.55e4 max"
       for n in ("clifford_geometric_product", "clifford_inner",
                 "clifford_left_contraction", "clifford_rotor_sandwich",
                 "clifford_wedge", "clifford_norm_squared")},
    **{f"tessera.{n}": "an accumulation over large values exceeds fp16 max; the answer does not fit regardless of compute width"
       for n in ("cumsum", "cumprod", "reduce", "segment_reduce")},
    **{f"tessera.loss.{n}": "loss over large logits exceeds fp16 max; caller-side loss scaling is the remedy"
       for n in ("cross_entropy", "binary_cross_entropy", "asymmetric_bce")},
    "tessera.lgamma": "log-gamma of moderate inputs already exceeds fp16 max",
    "tessera.silu_mul": "product of two large activations is ~1e8, beyond fp16 max",
    "tessera.spectral_filter": "accumulated spectral magnitude exceeds fp16 max",
}

#: Union, for callers that just want "does fp16 need care here".
FP16_RANGE_SENSITIVE: dict = {
    **FP16_INTERMEDIATE_OVERFLOW,
    **FP16_RESULT_UNREPRESENTABLE,
}


def fp16_range_hazard(graph_name: str):
    """Why this op needs an fp16 range probe, if it does."""
    return FP16_RANGE_SENSITIVE.get(graph_name)
