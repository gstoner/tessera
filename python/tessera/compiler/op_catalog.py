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
    # W2.2 scheduling semantics that are not represented by the scalar effect
    # lattice.  These travel with traced Graph IR so consumers never infer
    # aliasing or random-sample identity from Python or MLIR operation names.
    aliasing: str = "none"
    stochastic_identity: str = "none"

    def valid_arity(self, arity: int) -> bool:
        return self.min_arity <= arity <= self.max_arity


_SPECS = [
    OpSpec("gemm", "tessera.matmul", 2, 2, lowering="loop_nest"),
    OpSpec("matmul", "tessera.matmul", 2, 2, lowering="loop_nest"),
    OpSpec("batched_gemm", "tessera.batched_gemm", 2, 2, lowering="loop_nest"),
    OpSpec("es_low_rank_correction", "tessera.es_low_rank_correction", 3, 3,
           lowering="loop_nest", shape_rule="es_population_features"),
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
    # ── MC1: the matrix-function family (docs/audit/compiler/MATRIX_CALCULUS_REVIEW.md).
    # `linalg_function` is a NEW lowering kind and is deliberately absent from
    # LOWERING_SHAPE_RULE: these ops do not share one shape rule (det/logdet/
    # trace/norm reduce a matrix to a scalar, inv/matrix_power preserve shape,
    # vec/kron/solve each reshape differently), so naming a kind-wide default
    # would be the guess that file warns against. Per-op rules are declared in
    # OP_SHAPE_RULE below where they can be stated truthfully.
    OpSpec("det", "tessera.det", 1, 1, lowering="linalg_function"),
    OpSpec("logdet", "tessera.logdet", 1, 1, lowering="linalg_function"),
    OpSpec("inv", "tessera.inv", 1, 1, lowering="linalg_function"),
    OpSpec("solve", "tessera.solve", 2, 2, lowering="linalg_solver"),
    # trace and kron are (bi)linear, unlike the rest of the family: their VJP
    # IS their transpose, and they carry no condition number. Declaring them
    # `linalg_function` would have forced the category to claim
    # transpose="not_applicable", which is false for a linear map.
    OpSpec("trace", "tessera.trace", 1, 1, lowering="linalg_multilinear"),
    OpSpec("eigh", "tessera.eigh", 1, 1, lowering="linalg_decomposition"),
    OpSpec("kron", "tessera.kron", 2, 2, lowering="linalg_multilinear"),
    OpSpec("vec", "tessera.vec", 1, 1, lowering="layout_transform"),
    OpSpec("matrix_power", "tessera.matrix_power", 1, 2,
           lowering="linalg_function"),
    OpSpec("norm", "tessera.norm", 1, 1, lowering="linalg_function"),
    OpSpec("conv2d", "tessera.conv2d_nhwc", 2, 4, lowering="stencil"),
    OpSpec("conv3d", "tessera.conv3d_ndhwc", 2, 4, lowering="stencil"),
    # Optional affine operands are gamma and beta, in that order. RMSNorm has
    # gamma only. Their Graph ABI remains backward-compatible with unary calls.
    OpSpec("layer_norm", "tessera.layer_norm", 1, 3, lowering="normalization"),
    OpSpec("softmax", "tessera.softmax", 1, 1, lowering="stable_reduction"),
    OpSpec("softmax_safe", "tessera.softmax_safe", 1, 1, lowering="stable_reduction"),
    OpSpec("depth_attn", "tessera.depth_attn", 2, 2, lowering="attention",
           shape_rule="depth_attention"),
    # Differentiable discrete-choice relaxations. These are Graph-IR-visible
    # Python-reference operations; target execution remains explicitly
    # unsupported until a backend registers a physical lowering.
    OpSpec("sparsemax", "tessera.sparsemax", 1, 1, lowering="normalization", shape_rule="same_as_first"),
    OpSpec("entmax15", "tessera.entmax15", 1, 1, lowering="normalization", shape_rule="same_as_first"),
    OpSpec("soft_top_k", "tessera.soft_top_k", 1, 1, lowering="normalization", shape_rule="same_as_first"),
    OpSpec("gumbel_softmax", "tessera.gumbel_softmax", 1, 1, lowering="normalization", shape_rule="same_as_first"),
    OpSpec("perturbed_argmax", "tessera.perturbed_argmax", 1, 1, lowering="normalization", shape_rule="same_as_first"),
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
    OpSpec("adamw", "tessera.adamw", 2, 4, lowering="functional_optimizer_step"),
    OpSpec("momentum", "tessera.momentum", 2, 3, lowering="functional_optimizer_step"),
    OpSpec("nesterov", "tessera.nesterov", 2, 3, lowering="functional_optimizer_step"),
    OpSpec("adafactor", "tessera.adafactor", 2, 4, lowering="functional_optimizer_step"),
    OpSpec("lion", "tessera.lion", 2, 3, lowering="functional_optimizer_step"),
    # MSW-3 optimizer breadth. Same (params, grads, [state]) arity as the
    # single-slot optimizers above; each is transcribed from a numbered
    # definition in arXiv 2310.20360v3 and cites the label in its docstring.
    # Arities follow each method's FLAT ABI -- state as explicit tensor
    # operands, the convention adafactor records as "compiler-visible flat ABIs
    # keep optimizer state as explicit tensor operands". Adadelta and Shampoo
    # each carry TWO state tensors, so their maximum is 4, not 3 (review on
    # #695): declaring 3 made the only call that can actually execute exceed
    # the declared arity.
    OpSpec("adagrad", "tessera.adagrad", 2, 3, lowering="functional_optimizer_step"),
    OpSpec("rmsprop", "tessera.rmsprop", 2, 3, lowering="functional_optimizer_step"),
    OpSpec("adadelta", "tessera.adadelta", 2, 4, lowering="functional_optimizer_step"),
    OpSpec("shampoo", "tessera.shampoo", 2, 4, lowering="functional_optimizer_step"),
    #
    # `midpoint_sgd` is deliberately ABSENT from this catalog. Its second
    # operand is a gradient FUNCTION -- the method re-evaluates the gradient at
    # a probe point that does not exist until the first is known -- and
    # `TraceBuilder.record_op` requires every positional Graph operand to be a
    # Tracer, so every compiled use would fail by construction (review on
    # #695). Declaring it as an ordinary operand would advertise a Graph
    # boundary it cannot honour, which is worse than not declaring it (#29).
    # It remains a `tessera.optim` function with a `primitive_coverage` row.
    # Giving it a real Graph representation means a higher-order/region op, not
    # an operand slot, and that is a feature rather than a registration.
    # `ebm_energy_quadratic` is canonicalized to the flat-lane graph name
    # `tessera.ebm_energy_quadratic` below; the dotted Graph IR ODS spelling
    # `tessera.ebm.energy_quadratic` is a LEGACY_GRAPH_OP_ALIASES entry so it
    # does not collide on public_name with the canonical flat-lane OpSpec.
    OpSpec("ebm_langevin_step", "tessera.ebm.langevin_step", 3, 3,
           lowering="ebm"),
    OpSpec("transpose", "tessera.transpose", 1, 1, lowering="layout_transform"),
    OpSpec("stop_gradient", "tessera.stop_gradient", 1, 1,
           lowering="layout_transform", shape_rule="same_as_first"),
    OpSpec("cast", "tessera.cast", 1, 1, lowering="layout_transform"),
    OpSpec("dropout", "tessera.dropout", 1, 1, effect="random", lowering="random_mask",
           stochastic_identity="seed_counter"),
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
    OpSpec("rng_uniform", "tessera.rng_uniform", 0, 0, effect="random", lowering="random_source",
           stochastic_identity="implicit_stream"),
    OpSpec("rng_normal", "tessera.rng_normal", 0, 0, effect="random", lowering="random_source",
           stochastic_identity="implicit_stream"),
    OpSpec("rng_philox_uniform", "tessera.rng_philox_uniform", 2, 2,
           effect="pure", lowering="random_source", shape_rule="from_shape_attr",
           stochastic_identity="key_counter"),
    OpSpec("rng_philox_normal", "tessera.rng_philox_normal", 2, 2,
           effect="pure", lowering="random_source", shape_rule="from_shape_attr",
           stochastic_identity="key_counter"),
    OpSpec("fused_epilogue", "tessera.fused_epilogue", 1, 3, lowering="fused_epilogue"),
    # Coalition-lattice family (GAME_THEORY_PLAN.md G1, 2026-08-15). The
    # zeta/Möbius butterflies are structurally a radix-2 Stockham schedule with
    # a constant real 2x2 kernel, so they ride the spectral lowering lane —
    # the shared `tessera.butterfly_transform` consolidation (G1b) replaces
    # the per-name lowering, not this catalog surface. Python reference:
    # python/tessera/game/ (fp64 by the §6 numerics mandate).
    OpSpec("game_subset_zeta", "tessera.game_subset_zeta", 1, 1,
           lowering="spectral", shape_rule="same_as_first"),
    OpSpec("game_subset_mobius", "tessera.game_subset_mobius", 1, 1,
           lowering="spectral", shape_rule="same_as_first"),
    OpSpec("game_superset_zeta", "tessera.game_superset_zeta", 1, 1,
           lowering="spectral", shape_rule="same_as_first"),
    OpSpec("game_superset_mobius", "tessera.game_superset_mobius", 1, 1,
           lowering="spectral", shape_rule="same_as_first"),
    OpSpec("game_coalition_marginal", "tessera.game_coalition_marginal", 1, 1,
           lowering="spectral", shape_rule="coalition_marginal"),
    OpSpec("game_semivalue", "tessera.game_semivalue", 2, 2,
           lowering="contraction", shape_rule="coalition_players_axis"),
    # The §3.2 flagship: n-head softmax-weighted lattice reduction (streams
    # like flash-attention's online softmax on the kernel tier).
    OpSpec("game_boltzmann_value", "tessera.game_boltzmann_value", 2, 2,
           lowering="contraction", shape_rule="coalition_players_axis"),
    OpSpec("game_coalition_excess", "tessera.game_coalition_excess", 2, 2,
           lowering="contraction", shape_rule="same_as_first"),
    # Segmented minimum excludant (Grundy numbers); integer, non-diff, rides
    # the segment_reduce ragged encoding. Nim-sums use existing bitwise_xor.
    OpSpec("game_mex", "tessera.game_mex", 2, 2,
           lowering="segment_reduce", shape_rule="segment_mex"),
    # Thomas-algorithm tridiagonal solve (P2 tranche; PDE plan §III.1 —
    # required for Crank-Nicolson). Reference tier; fp64 accumulation with
    # the rhs storage dtype preserved; VJP = the transpose solve.
    OpSpec("tridiagonal_solve", "tessera.tridiagonal_solve", 4, 4,
           lowering="loop_nest", shape_rule="tridiagonal_rhs"),
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
    OpSpec("reshape", "tessera.reshape", 1, 1, lowering="layout_transform", aliasing="operand_0"),
    OpSpec("view", "tessera.view", 1, 1, lowering="layout_transform", aliasing="operand_0"),
    OpSpec("flatten", "tessera.flatten", 1, 1, lowering="layout_transform", aliasing="operand_0"),
    OpSpec("squeeze", "tessera.squeeze", 1, 1, lowering="layout_transform", aliasing="operand_0"),
    OpSpec("unsqueeze", "tessera.unsqueeze", 1, 1, lowering="layout_transform", aliasing="operand_0"),
    OpSpec("permute", "tessera.permute", 1, 1, lowering="layout_transform", aliasing="operand_0"),
    OpSpec("broadcast", "tessera.broadcast", 1, 1, lowering="layout_transform", aliasing="operand_0"),
    OpSpec("expand", "tessera.expand", 1, 1, lowering="layout_transform", aliasing="operand_0"),
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
    # 5 operands: `mobius(z, a, b, c, d)` -- the Mobius transform
    # (az + b) / (cz + d) takes its four coefficients as values, not
    # attributes. Catalog said 2; caught only once the op became reachable
    # from `tessera.ops` (W2.2), which is what let the arity gate see it.
    OpSpec("mobius",                   "tessera.mobius",                   5, 5),
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
    # 2 operands: `(p, p_target)` -- an energy BETWEEN two point sets.
    OpSpec("conformal_energy_on_sphere", "tessera.conformal_energy_on_sphere", 2, 2, lowering="stable_reduction"),
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
    # MC1: shape-preserving matrix functions. det/logdet/trace/norm reduce to a
    # scalar and vec/kron/solve/eigh each have their own rule, so only these two
    # can honestly claim the kind-wide default.
    "tessera.inv": "same_as_first",
    "tessera.matrix_power": "same_as_first",
    # A matrix consumed down to one number, keeping any batch axes. Not
    # `reduce_all` (which collapses everything) and not `reduce_trailing`
    # (which drops one axis).
    "tessera.det": "matrix_scalar",
    "tessera.logdet": "matrix_scalar",
    "tessera.trace": "matrix_scalar",
    "tessera.norm": "matrix_scalar",
    # x = A^-1 b takes the right-hand side's shape.
    "tessera.solve": "same_as_second",
    "tessera.vec": "vec",
    "tessera.kron": "kron",
    "tessera.eigh": "eigh",
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
    **{f"tessera.{n}": "optimizer_step" for n in ("adam", "adamw")},
    **{f"tessera.{n}": "optimizer_pair_step" for n in ("momentum", "nesterov")},
    # MSW-3. Same operand shapes as the momentum pair: one param-shaped moment
    # (`m`) alongside the param. `rmsprop`'s `step` slot is a scalar counter,
    # not a tensor operand, so it does not change the rule.
    **{f"tessera.{n}": "optimizer_pair_step" for n in ("adagrad", "rmsprop")},
    # Two param-shaped moments (`m` and `delta`), like adam/adamw.
    "tessera.adadelta": "optimizer_step",
    # (`tessera.midpoint_sgd` has no entry: it is not a catalog op -- see the
    # note beside the optimizer OpSpecs above.)
    "tessera.sgd": "same_as_first",
    # Lion's flat compiler ABI returns exactly (new_param, new_moment).  It is
    # not the three-result Adam-style contract used by optimizer_step.
    "tessera.lion": "optimizer_pair_step",
    "tessera.adafactor": "adafactor_step",

    # Cache mutators thread the handle through -- the ODS says
    # `-> Tessera_KVCacheType:$updated` for each. `read` is the one member of
    # the family that returns TENSORS, not a handle.
    **{f"tessera.{n}": "state_handle" for n in
       ("kv_cache.append", "kv_cache.prune", "cache.commit", "cache.rollback")},
    "tessera.kv_cache.read": "kv_cache_read",

    # A bit count is the operand's shape with the declared index width. It sits
    # in the `elementwise` kind, whose default rule is `same_as_first` -- which
    # would claim the operand's storage dtype and is wrong for a count.
    "tessera.popcount": "same_shape_index",

    # Mesh-scaling collectives. `all_reduce` and `all_to_all` are NOT here:
    # both preserve shape, so their `same_as_first` default is already right.
    "tessera.all_gather": "all_gather",
    "tessera.reduce_scatter": "reduce_scatter",

    # W1.4 wave 1 -- ops whose result is exactly operand 0, CONFIRMED with all
    # dims distinct (2,3,5,7 / 4x6). A square or equal-dim probe cannot tell
    # `same_as_first` from `transpose`, `matmul_2d` or `select_from_second`:
    # measured on a 4x4, `cholesky` matched six different rules. Every
    # assignment below survived a shape where only one of them can.
    #
    # The whole linear/sparse attention family lands here -- these are
    # attention VARIANTS, so the result is the query's shape whatever the
    # interior does.
    **{f"tessera.{n}": "same_as_first" for n in (
        "hybrid_attention", "kimi_delta_attention", "lightning_attention",
        "linear_attn", "modified_delta_attention", "power_attn", "retention",
        "attn_compressed_blocks", "attn_sliding_window", "gated_deltanet",
        "deepseek_sparse_attention", "lookahead_sparse_attention",
        "msa_sparse_attention", "attn_top_k_blocks", "mla_decode_fused",
        # Structural / in-place-shaped ops.
        "masked_fill", "mor_scatter", "roll", "dynamic_update_slice",
        "scatter_add", "scatter_reduce", "selective_ssm", "moe_swiglu_block",
        # cholesky of an (N, N) matrix is (N, N) -- verified on 6x6, since a
        # 4x4 probe would have matched five other rules equally well.
        "cholesky",
    )},

    # W1.4 wave 2 -- new rules, each verified on all-distinct dims.
    **{f"tessera.{n}": "matmul_trailing" for n in (
        "linear_general", "latent_kv_compress", "latent_kv_expand_k",
        "latent_kv_expand_v", "grouped_gemm", "moe")},
    # Operand 1 is the value; operand 0 is the operator or the key.
    **{f"tessera.{n}": "same_as_second" for n in (
        "cholesky_solve", "tri_solve", "target_verify")},
    "tessera.conv2d_nhwc": "conv_spatial",
    "tessera.conv3d_ndhwc": "conv_spatial",
    "tessera.gather": "index_along_axis",
    "tessera.index_select": "index_along_axis",
    "tessera.select": "drop_axis",
    "tessera.unsqueeze": "insert_axis",
    "tessera.slice": "from_slice_sizes",
    "tessera.dynamic_slice": "from_slice_sizes",
    "tessera.pad": "pad",
    # Query x key-BLOCK scores -- the trailing axis is a block count.
    **{f"tessera.{n}": "scores_per_block" for n in (
        "msa_index_scores", "memory_index_score", "memory_index_select_ste")},
    # Same block grid, but a MASK -- it selects blocks rather than scoring
    # them. Shape-identical to the rule above, so only a dtype comparison
    # separates them.
    "tessera.memory_index_select": "scores_per_block_mask",
    "tessera.masked_categorical": "reduce_trailing_index",
    "tessera.mor_router": "reduce_trailing_index",
    "tessera.mor_partition": "reduce_trailing_bool",
    "tessera.top_k": "top_k",
    "tessera.chunk": "split_equal",
    "tessera.split": "split_equal",
    "tessera.rope_split": "split_halves",
    "tessera.qkv_projection": "qkv_projection",
    "tessera.linear_attn_state": "state_matrix",
    "tessera.permute": "layout_permute",

    # W1.4 wave 3.
    # The remaining attention entry points are dispatch wrappers over the same
    # contract: the result is the query's shape. `gqa`/`mqa` differ only in how
    # many KV heads they broadcast over, which does not change the result.
    **{f"tessera.{n}": "same_as_first" for n in (
        "attn_local_window_2d", "gqa_attention", "mqa_attention",
        "multi_head_attention", "varlen_sdpa")},
    # Sparse and factorized products are still A @ B at the type level -- the
    # sparsity lives in the STORAGE of operand 0, not in the result shape.
    **{f"tessera.{n}": "matmul_2d" for n in (
        "spmm_csr", "spmm_coo", "bsmm", "sddmm", "factorized_matmul",
        "quantized_matmul")},
    "tessera.dequant_matmul": "matmul_trailing",
    "tessera.dequant_grouped_gemm": "matmul_trailing",
    # `moe_combine` sums the partials over the token axis.
    "tessera.moe_combine": "drop_axis",
    "tessera.rope_merge": "concat_trailing",
    "tessera.tile": "tile_trailing",
    "tessera.repeat": "flatten_repeat",
    "tessera.msa_select_blocks": "select_k_index",
    "tessera.segment_reduce": "segment_reduce",
    "tessera.lu": "lu",
    "tessera.qr": "qr",
    "tessera.svd": "svd",
    "tessera.nonzero": "nonzero",
    "tessera.spec_accept": "spec_accept",
    "tessera.spec_accept_sample": "spec_accept",
    "tessera.spec_accept_tree_sample": "spec_accept_tree",
    "tessera.stft": "stft",
    "tessera.spectral_conv": "conv_full",
    # Shape lives entirely in an attribute for these sources.
    **{f"tessera.{n}": "from_shape_attr" for n in ("rng_normal", "rng_uniform")},

    "tessera.arange": "arange",
    "tessera.einsum": "einsum",
    "tessera.istft": "istft",
    # Score-matching and RL policy losses reduce to a SCALAR. `reduce_all`
    # already states that, and being in the wrapper's preserving set it also
    # stops them widening f32 -> f64, which all four were measured doing.
    **{f"tessera.loss.{n}": "reduce_all" for n in (
        "denoising_score_matching", "implicit_score_matching")},
    **{f"tessera.rl.{n}": "reduce_all" for n in (
        "grpo_policy_loss", "cispo_policy_loss")},

    # W1.4 wave 3 tail. The Clifford field derivatives take a RAW coefficient
    # array `(spatial..., 2**n)` -- not a `MultivectorField` -- and return the
    # same layout. Three earlier probes failed only because they used the wrong
    # spatial rank for Cl(3,0); the ops were never the problem.
    **{f"tessera.{n}": "same_as_first" for n in (
        "clifford_ext_deriv", "clifford_codiff", "clifford_vec_deriv",
        "laplacian_2d")},
    # `(N, 3)` points on the sphere -> a per-point energy `(N,)`.
    "tessera.conformal_energy_on_sphere": "reduce_trailing",

    # W2.2 -- the complex family, now reachable from `tessera.ops`.
    # These were unreachable, so they had been sitting on the `elementwise`
    # default (`same_as_first`) unchecked. That default is actively wrong for
    # them and the storage-dtype wrapper enforces it: `complex_abs` returned a
    # float32 magnitude and got cast back to complex64.
    **{f"tessera.{n}": "complex_same" for n in (
        "complex_mul", "complex_div", "complex_exp", "complex_log",
        "complex_pow", "complex_sqrt", "complex_conjugate",
        "mobius", "cross_ratio")},
    # `stereographic` is NOT `complex_same`: it consumes `(..., 3)` real
    # coordinates and yields one complex value per point, so the trailing
    # coordinate axis is dropped.
    "tessera.stereographic": "complex_from_coords",
    # A magnitude and an angle are REAL.
    "tessera.complex_abs": "complex_to_real",
    "tessera.complex_arg": "complex_to_real",
    # Concyclicity is a predicate.
    "tessera.is_concyclic": "same_shape_bool",
    # `laplacian_2d` and `conformal_energy_on_sphere` are the real-valued
    # members of this family and are already declared with the wave-3 tail
    # above -- they were classifiable from `tessera.complex` before the ops
    # namespace reached them.

    # `pack` / `rearrange` mean two things depending on their `layout`
    # attribute: a tuple permutes, a named layout is identity.
    "tessera.pack": "layout_permute",
    "tessera.rearrange": "layout_permute",

    # Spectral family. `irfft` returns REAL values -- it is not `complex_same`.
    "tessera.fft": "complex_same",
    "tessera.ifft": "complex_same",
    "tessera.rfft": "rfft",
    "tessera.irfft": "irfft",

    # NOT declared on purpose: `all_gather` and `reduce_scatter` returned the
    # operand's shape only because the probe ran at world_size=1. Their real
    # shapes scale with the mesh, so declaring from that measurement would bake
    # in a degenerate case. They stay `unclassified` until probed multi-rank.
}

# ─────────────────────────────────────────────────────────────────────────────
# Declared result dtypes for non-storage-preserving results (W1.3)
#
# Two constants, because both were previously *derived* -- and a derived dtype
# is one the caller's storage choice can change out from under the compiler.
#
#   INDEX_DTYPE   an index or a count. Was hard-coded as "int64" in two shape
#                 rules and computed a third way by `popcount`, which returns
#                 `np.bitwise_count`'s width on numpy >= 2.0 (uint8 for int8
#                 input) and int64 from the masking fallback on 1.26. Same
#                 program, different result dtype, decided by which NumPy the
#                 reference happened to import.
#
#   COMPUTE_FLOAT the float an INTEGER input promotes to. NumPy picks this from
#                 the integer's width -- int8 -> f16, int16 -> f32,
#                 int32/int64 -> f64 -- so `cos` returned four different
#                 precisions for identical mathematics depending only on how
#                 the input was stored. Measured across the catalog: 29 ops.
#                 f32 is the compute width of the stack (Decision #15a); f64 is
#                 the oracle path and runs at 1/64 rate on the target GPUs,
#                 and f16-from-int8 silently caps a transcendental at 6.55e4.
#
# "Pinning one would be wrong" was the recorded reason `popcount` stayed
# undeclared. It is the opposite: a compiler must declare its result dtype
# precisely because it cannot be a function of the host library's promotion
# table.
INDEX_DTYPE = "int64"
COMPUTE_FLOAT_DTYPE = "fp32"


#: The declared vocabulary. `graph_ir` implements each name; a rule named here
#: with no implementation (or vice versa) is a drift-gated error.
SHAPE_RULE_NAMES = frozenset({
    # MC1 matrix-function family.
    "matrix_scalar",   # (..., m, n) -> (...)   det/logdet/trace/norm
    "vec",             # (..., m, n) -> (..., m*n), column-major
    "kron",            # (p, q) x (r, s) -> (p*r, q*s)
    "eigh",            # -> ((..., n), (..., n, n))
    "same_as_first",
    "depth_attention",
    "matmul_2d",
    "es_population_features",
    "coalition_marginal",
    "coalition_players_axis",
    "segment_mex",
    "tridiagonal_rhs",
    "batched_gemm_3d",
    "transpose",
    "same_shape_bool",
    "reduce_all",
    "reduce_all_index",
    "same_shape_index",
    "flatten",
    "complex_same",
    "rfft",
    "irfft",
    "select_from_second",
    "quantize_per_tensor",
    "quantize_per_block",
    "optimizer_step",
    "optimizer_pair_step",
    "adafactor_step",
    "from_shape_attr",
    "cast",
    "reduce_trailing",
    "state_handle",
    "layout_permute",
    "arange",
    "einsum",
    "istft",
    "complex_to_real",
    "complex_from_coords",
    "concat_trailing",
    "tile_trailing",
    "flatten_repeat",
    "select_k_index",
    "segment_reduce",
    "lu",
    "qr",
    "svd",
    "nonzero",
    "spec_accept",
    "spec_accept_tree",
    "stft",
    "conv_full",
    "matmul_trailing",
    "same_as_second",
    "conv_spatial",
    "index_along_axis",
    "drop_axis",
    "insert_axis",
    "from_slice_sizes",
    "pad",
    "scores_per_block",
    "scores_per_block_mask",
    "reduce_trailing_index",
    "reduce_trailing_bool",
    "top_k",
    "split_equal",
    "split_halves",
    "qkv_projection",
    "state_matrix",
    "kv_cache_read",
    "all_gather",
    "reduce_scatter",
    "unclassified",
})


#: Ops EXAMINED and deliberately left without a rule, with the reason. This is
#: a third state, distinct from "not yet looked at": the shape or dtype is
#: genuinely not a function of the operands alone, or declaring a rule would
#: force wrong behavior. Answering "why is this unclassified?" is what makes the
#: remaining count meaningful.
DELIBERATELY_UNDECLARED: dict = {
    # MSW-3. Shampoo's state is NOT param-shaped: its two preconditioners are
    # the Gram matrices `L` (d1 x d1) and `R` (d2 x d2) built from the
    # parameter's two axes, so a d1 x d2 parameter carries state of two
    # entirely different shapes. `optimizer_step` and `optimizer_pair_step`
    # both assert param-shaped moments and would be FALSE contracts here --
    # the same mistake the comment above records for `same_as_first` on the
    # quantize family. Adding an `optimizer_preconditioned_step` rule instead
    # would declare a vocabulary no pass consumes (#29), so this is recorded
    # as examined-and-undeclared until something needs to read it.
    "tessera.shampoo": (
        "two-sided full-matrix preconditioning: state is the Gram matrices "
        "L (d1 x d1) and R (d2 x d2), not param-shaped moments, so every "
        "existing optimizer shape rule would assert a false contract"),

    # W1.4 -- a genuinely new category, and the first exemption reason in this
    # registry that survives examination rather than dissolving under it.
    #
    # These four take a PYTHON CALLABLE as operand 0 and a complex scalar as
    # operand 1: they are higher-order numerical-differentiation operators
    # (`dz` / `dbar` are the Wirtinger derivatives, evaluated by finite
    # differences of `f` around `z0`). A shape rule is a function of
    # `operand_types`, and operand 0 here HAS no tensor type -- there is
    # nothing for the rule to read. This is not a vocabulary gap that a richer
    # signature would close, the way mesh context closed the collectives; the
    # operand is a function.
    #
    # They also live on `tessera.complex`, not `tessera.ops`, so the frontend
    # cannot emit them as Graph IR ops at all. Worth stating plainly: the
    # catalog names them, and no `@jit` body can reach them.
    **{f"tessera.{_n}": "takes a Python callable as operand 0 (higher-order "
                        "numerical differentiation of f around z0), so there "
                        "is no operand tensor type for a shape rule to read; "
                        "also reachable only via tessera.complex, not "
                        "tessera.ops"
       for _n in ("dz", "dbar", "conformal_jacobian", "check_cauchy_riemann")},

    # `training.loss_sgd` / `training.loss_adamw` are FUSED loss+optimizer
    # steps registered only in the runtime reference table, with no
    # `tessera.ops` entry point. Their results are (updated_param,
    # target_gradient) and (updated_param, m, v, target_gradient) -- describable
    # in principle, but the op cannot be called through the ops namespace, so
    # any rule declared here would be unverifiable against real behaviour. That
    # is the condition this registry exists to avoid.
    **{f"tessera.training.{_n}": "fused loss+optimizer step registered only in "
                                 "the runtime reference table; no tessera.ops "
                                 "entry point, so a declared rule could not be "
                                 "verified against the op"
       for _n in ("loss_sgd", "loss_adamw")},

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
    # `popcount` was here, on the grounds that its integer width is
    # NumPy-version dependent (uint8 under 2.x via `np.bitwise_count`, int64
    # under 1.26 via the masking fallback) "so pinning one would be wrong".
    # That has it backwards. A result dtype decided by which NumPy the host
    # imported is not a contract at all, and pinning one is precisely what a
    # compiler owes its users. It is `same_shape_index` over `INDEX_DTYPE` --
    # operand shape, declared index width -- and the reference now returns that
    # on every NumPy rather than inheriting the host's answer.
    # `all_gather` and `reduce_scatter` were here: "result shape scales with
    # mesh size, not derivable from operand types". The premise was right and
    # the conclusion did not follow -- it is not derivable from OPERAND TYPES,
    # which is an argument for giving the rule signature mesh context, not for
    # leaving the ops undeclared. They now take a `{axis: size}` mesh and FAIL
    # CLOSED to `?` on the scaled axis when it is unknown. Leaving them
    # unclassified meant falling back to the operand shape, which is not a
    # neutral answer but the positive claim `world_size == 1` -- and the
    # single-rank reference stubs made a probe agree with it.
    # The four spectral ops were here: "returns complex64, which is
    # planned_gated per Decision #15a; declaring it conflicts with the dtype
    # capability contract". It does not conflict -- the contract has an
    # explicit path for planned/gated dtypes (`allow_planned_gated=True` plus
    # `metadata.dtype_status`), and NAMING a dtype in a shape rule is not the
    # same as claiming a backend implements it. Promoting complex to CANONICAL
    # would be a capability change; declaring these rules is not, and complex
    # stays planned_gated.
    #
    # `complex_same` and `_shape_rfft` were already written and registered
    # while all four ops stayed exempt -- two rules with no consumer, which
    # Decision #29 exists to prevent, sitting next to the ops they were
    # written for.
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
    # `tri_solve(A, b)` / `target_verify(tokens, logits)`: operand 0 is the
    # operator or the key, operand 1 is the value that carries storage dtype.
    "same_as_second": 1,
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
