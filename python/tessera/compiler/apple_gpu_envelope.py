"""Apple GPU runtime envelope — the single source of dispatch truth (P1).

Until 2026-06-09 the envelope tables lived as two literal copies: one in
``driver.py`` (compile-time gating) and one in ``runtime.py`` (lane
dispatch), plus a third hand-maintained mirror in C++
(``TileToApple.cpp::kRuntimeOps``).  This module is now the only place
the tables are written down:

* ``driver.py`` and ``runtime.py`` import every ``_APPLE_GPU_*`` table
  from here (same names, drop-in).
* ``runtime.py`` dispatches per-op via :data:`APPLE_GPU_LANE_BY_OP`
  (op → lane), so adding a lane op means editing exactly one table.
* ``scripts/generate_apple_runtime_ops_table.py`` renders the C++
  X-macro ``apple_runtime_ops.inc`` from :data:`_APPLE_GPU_RUNTIME_OPS`
  so the Tile→Apple Target IR pass reads the same registry.
* ``apple_kernel_descriptor.py`` classifies descriptor families/lanes
  from these tables.

Drift gates: ``tests/unit/test_apple_gpu_envelope_dispatch.py`` (lane
oracle), ``tests/unit/test_apple_gpu_tile_pass_status_matches_envelope.py``
(real C++ pass vs envelope), and the generated-``.inc`` comparison gate.
"""

from __future__ import annotations

# MetalPerformanceShaders GEMM family — _apple_gpu_dispatch_matmul.
_APPLE_GPU_MPS_OPS = frozenset(
    {"tessera.matmul", "tessera.gemm", "tessera.batched_gemm"}
)
# Hand-written Metal Shading Language kernels.
_APPLE_GPU_MSL_OPS = frozenset({
    "tessera.rope",
    "tessera.flash_attn",
    "tessera.softmax",
    "tessera.softmax_safe",
    "tessera.gelu",
})

# 2026-05-29 — MetalPerformanceShadersGraph-backed Tier-1 / long-tail lane.
# op_name -> unary opcode (must match apple_gpu_runtime.mm mpsg_unary_node).
_APPLE_GPU_UNARY_OPCODES = {
    "tessera.relu": 0,
    "tessera.sigmoid": 1,
    "tessera.sigmoid_safe": 1,
    "tessera.tanh": 2,
    "tessera.softplus": 3,
    "tessera.silu": 4,
    "tessera.exp": 6,
    "tessera.log": 7,
    "tessera.sqrt": 8,
    "tessera.rsqrt": 9,
    "tessera.neg": 10,
    "tessera.negative": 10,
    "tessera.abs": 11,
    "tessera.absolute": 11,
    # Batch 1 (2026-06-08) — float-output elementwise math (MPSGraph nodes).
    "tessera.sin": 12, "tessera.cos": 13, "tessera.tan": 14,
    "tessera.asin": 15, "tessera.acos": 16, "tessera.atan": 17,
    "tessera.sinh": 18, "tessera.cosh": 19, "tessera.erf": 20, "tessera.erfc": 21,
    "tessera.expm1": 22, "tessera.log1p": 23, "tessera.reciprocal": 24,
    "tessera.sign": 25, "tessera.floor": 26, "tessera.ceil": 27,
    "tessera.round": 28, "tessera.trunc": 29,
    # Batch 2 (2026-06-08) — unary predicates / logical / bitwise → f32 mask.
    "tessera.isfinite": 30, "tessera.isinf": 31, "tessera.isnan": 32,
    "tessera.logical_not": 33, "tessera.bitwise_not": 34,
}
# Batch 1 (2026-06-08) — binary float math + comparison → f32 mask. op_name ->
# opcode in apple_gpu_runtime.mm mpsg_binary_node. add/sub/mul/div/maximum/minimum
# reuse the existing C nodes (0-5); 7-10 math; 11-16 comparison.
_APPLE_GPU_BINARY_OPCODES = {
    "tessera.add": 0, "tessera.sub": 1, "tessera.mul": 2, "tessera.div": 3,
    "tessera.maximum": 4, "tessera.minimum": 5,
    "tessera.pow": 7, "tessera.atan2": 8, "tessera.mod": 9, "tessera.floor_div": 10,
    "tessera.eq": 11, "tessera.ne": 12, "tessera.lt": 13, "tessera.le": 14,
    "tessera.gt": 15, "tessera.ge": 16,
    # Batch 2 (2026-06-08) — logical (→ f32 mask) + bitwise (int32) binary.
    "tessera.logical_and": 17, "tessera.logical_or": 18, "tessera.logical_xor": 19,
    "tessera.bitwise_and": 20, "tessera.bitwise_or": 21, "tessera.bitwise_xor": 22,
}
# op_name -> rowop kind (0 layer_norm, 1 rmsnorm, 3 log_softmax). Softmax stays
# on its dedicated MSL path for single-op; the MPSGraph softmax symbol is used
# by the f16/bf16 fused-chain completion.
_APPLE_GPU_ROWOP_KINDS = {
    "tessera.layer_norm": 0,
    "tessera.rmsnorm": 1,
    "tessera.rmsnorm_safe": 1,
    "tessera.log_softmax": 3,
}
# Batch 2 (2026-06-08) — composed on the GPU binary lane (no dedicated kernel):
# clamp/clip = max(min(x,hi),lo); where = c*a + (1-c)*b.
_APPLE_GPU_COMPOSE_OPS = frozenset({"tessera.clamp", "tessera.clip", "tessera.where"})
# Parameterized unary composed on the GPU lanes (2026-06-17): softcap = the Gemma
# logit soft-cap cap*tanh(x/cap) — div-by-scalar -> tanh -> mul-by-scalar. Unlike
# the param-free unary opcodes it carries a scalar `cap`, so it rides a compose
# handler rather than a pointwise-vocab entry. A first-class runtime op (in the
# master set below) so @jit(apple_gpu) routes a softcap program to the GPU.
_APPLE_GPU_SOFTCAP_OPS = frozenset({"tessera.softcap"})
# Structural transpose / N-D permute on a real MPSGraph kernel (2026-06-17,
# transposeTensor:permutation:). The first structural layout op displaced off the
# numpy lane — value-preserving data movement on Metal, so a transpose mid-program
# no longer demotes residency to the reference path. A first-class runtime op.
_APPLE_GPU_TRANSPOSE_OPS = frozenset({"tessera.transpose"})
# Batch 3 (2026-06-08) — regression / CE losses composed from the GPU opcode
# lanes (per-element recipe + reduce). One dispatcher, no dedicated kernels.
_APPLE_GPU_LOSS_COMPOSE_OPS = frozenset({
    "tessera.loss.mse", "tessera.loss.mae", "tessera.loss.huber",
    "tessera.loss.smooth_l1", "tessera.loss.log_cosh", "tessera.loss.vlb",
    "tessera.loss.ddpm_noise_pred", "tessera.loss.binary_cross_entropy",
    "tessera.loss.cross_entropy",
    "tessera.loss.kl_divergence", "tessera.loss.js_divergence",
})
# Group/instance/weight norm composed from the rowop (layer_norm) + reduce lanes.
_APPLE_GPU_NORM_COMPOSE_OPS = frozenset({
    "tessera.group_norm", "tessera.instance_norm", "tessera.weight_norm",
})
# Standard attention family (Sub-sprint A) — thin wrappers over the proven GQA
# flash-attention kernel (multi_head/gqa/mqa/mla_decode/gated_attention).
_APPLE_GPU_ATTN_WRAPPER_OPS = frozenset({
    "tessera.multi_head_attention", "tessera.gqa_attention",
    "tessera.mqa_attention", "tessera.mla_decode", "tessera.mla_decode_fused",
    "tessera.gated_attention",
})
# Linear / recurrent attention family (Sub-sprint B) via the quadratic-parallel
# form (φ(Q)φ(K)ᵀ ⊙ causal[⊙decay]) @ V — two GPU bmms + a mask multiply.
_APPLE_GPU_LINEAR_ATTN_OPS = frozenset({
    "tessera.linear_attn", "tessera.linear_attn_state",
    "tessera.lightning_attention", "tessera.power_attn", "tessera.retention",
})
# NSA masked-softmax attention (Sub-sprint C) — compressed-block (plain) +
# sliding-window (structured causal/window mask) via bmm→+mask→softmax→bmm.
_APPLE_GPU_MASKED_ATTN_OPS = frozenset({
    "tessera.attn_compressed_blocks", "tessera.attn_sliding_window",
})
# Delta-rule attention family (Sub-sprint D) — gated_deltanet / kimi_delta /
# modified_delta as the quadratic form with a per-token column-weight mask.
_APPLE_GPU_DELTA_ATTN_OPS = frozenset({
    "tessera.gated_deltanet", "tessera.kimi_delta_attention",
    "tessera.modified_delta_attention",
})
# hybrid_attention — policy wrapper routing to the now-proven delegates.
_APPLE_GPU_HYBRID_ATTN_OPS = frozenset({"tessera.hybrid_attention"})
# Data-dependent NSA attention (Sub-sprint E) — host select/gather + GPU attention.
# Lookahead Sparse Attention (LSA, experimental) joins this lane: its sigmoid-
# threshold block selection is host-mediated (data-dependent), the per-query
# footprint attention runs on the GPU. See docs/audit/domain/archive/lsa_scope.md.
_APPLE_GPU_SPARSE_ATTN_OPS = frozenset({
    "tessera.attn_top_k_blocks", "tessera.deepseek_sparse_attention",
    "tessera.attn_local_window_2d", "tessera.lookahead_sparse_attention",
    # MiniMax Sparse Attention (MSA, arXiv:2606.13392) — host-select + GPU exact
    # attention (Phase 3): the Index Branch top-k block selection runs on the
    # host (data-dependent, like attn_top_k_blocks); the exact Main Branch
    # attention over the gathered selected blocks runs on the GPU. See docs/msa.md.
    "tessera.msa_sparse_attention",
})
_APPLE_GPU_MPSGRAPH_OPS = (
    frozenset(_APPLE_GPU_UNARY_OPCODES)
    | frozenset(_APPLE_GPU_BINARY_OPCODES)
    | frozenset(_APPLE_GPU_ROWOP_KINDS)
    | _APPLE_GPU_COMPOSE_OPS
    | frozenset({"tessera.silu_mul"})
)
# 2026-05-29 — Tier-2 projections routed through the matmul / bmm lane.
_APPLE_GPU_PROJECTION_OPS = frozenset(
    {"tessera.linear_general", "tessera.qkv_projection"}
)
# 2026-05-29 — Tier-3 reductions / scans via the MPSGraph reduce lane.
# op_name -> (kind, op_code); kinds: "reduce" (scalar per row), "arg" (int
# index), "scan" (cumulative, same shape).
_APPLE_GPU_REDUCE_OPS = {
    "tessera.reduce": ("reduce", 0),   # sum
    "tessera.mean": ("reduce", 1),
    "tessera.amax": ("reduce", 2),
    "tessera.amin": ("reduce", 3),
    "tessera.prod": ("reduce", 4),
    "tessera.var": ("reduce", 5),
    "tessera.std": ("reduce", 6),
    "tessera.argmax": ("arg", 0),
    "tessera.argmin": ("arg", 1),
    "tessera.cumsum": ("scan", 0),
    "tessera.cumprod": ("scan", 1),
    # Batch 2 (2026-06-08) — reduce/scan opcode completions.
    "tessera.logsumexp": ("reduce", 7),
    "tessera.cummax": ("scan", 2),
    "tessera.cummin": ("scan", 3),
    "tessera.max": ("reduce", 2),   # reduce-max (alias of amax over an axis)
    "tessera.min": ("reduce", 3),   # reduce-min (alias of amin)
}
_APPLE_GPU_REDUCTION_OPS = frozenset(_APPLE_GPU_REDUCE_OPS)
# Hard top-k (k>1) via MPSGraph's native TopK op (values + indices) — the
# segmented_topk_gpu primitive. argmax/argmin (above) only give k==1.  The @jit
# AST frontend now emits `tessera.top_k(%x) {k = N}` (positional scalar k → attr;
# see graph_ir._POSITIONAL_ATTR_PARAMS), so it flows the Tile→Apple pipeline and
# this envelope row routes it to metal_runtime (multi-output dispatch returns
# (values, indices), like qkv_projection).
_APPLE_GPU_TOPK_OPS = frozenset({"tessera.top_k"})
# 2026-05-30 — Tier-3 convolutions: conv2d via the MPSGraph convolution2D node
# (NHWC/HWIO); conv3d via im2col + a GPU MPSGraph batched matmul (NDHWC/DHWIO).
_APPLE_GPU_CONV_OPS = frozenset({"tessera.conv2d", "tessera.conv3d"})
# GPU linear-algebra lane (MPSMatrix) — only the registered Graph IR ops:
# tessera.cholesky (1 operand) + tessera.tri_solve (2 operands, `lower` kwarg).
_APPLE_GPU_LINALG_OPS = frozenset({"tessera.cholesky", "tessera.tri_solve"})
# Mamba-2 selective state-space scan — chunked-parallel SSD with its batched
# contractions on the Metal bmm lane (scalar-state A; (D,N) A falls back).
# NOTE: the ReplaySSM *decode* routes (output_only / state_and_output / spec —
# see compiler/ssm_replay.py) run host-side on `tessera.cache.SSMStateHandle`
# today; their fused Metal decode kernels are `planned` (Track-R Phase 5) and
# join this runtime set only once they exist — do NOT add them here before then.
_APPLE_GPU_SSM_OPS = frozenset({"tessera.selective_ssm"})
# Ragged grouped matmul (MoE expert-FFN compute core) — per-group MPS matmul.
_APPLE_GPU_MOE_OPS = frozenset({"tessera.grouped_gemm", "tessera.moe_swiglu_block"})
# Spectral / FFT lane (2026-06-10) — the "special" kernel class. fft/ifft/rfft/
# irfft run on MPSGraph FourierTransform (macOS 14+); dct/stft/istft/
# spectral_conv compose over them; spectral_filter is elementwise.
_APPLE_GPU_SPECTRAL_OPS = frozenset({
    "tessera.fft", "tessera.ifft", "tessera.rfft", "tessera.irfft",
    "tessera.dct", "tessera.stft", "tessera.istft",
    "tessera.spectral_conv", "tessera.spectral_filter",
})
# LDT candidate-axis ops with dedicated Metal kernels (popcount intrinsic,
# innermost-axis nonzero count).
_APPLE_GPU_LDT_OPS = frozenset({
    "tessera.popcount", "tessera.count_nonzero", "tessera.loss.z_loss",
    "tessera.loss.asymmetric_bce", "tessera.loss.load_balance_loss",
    "tessera.masked_categorical",
})
# Geometric-algebra (Clifford Cl(3,0)) flat-coefficient lane — the canonical
# tessera.ops projection of the tessera.ga.* Multivector surface. The dispatcher
# calls the GA lane, which internally routes Cl(3,0) f32 to the cl30 MSL kernels.
_APPLE_GPU_CLIFFORD_OPS = frozenset({
    "tessera.clifford_geometric_product", "tessera.clifford_wedge",
    "tessera.clifford_left_contraction", "tessera.clifford_inner",
    "tessera.clifford_rotor_sandwich",
    "tessera.clifford_reverse", "tessera.clifford_grade_involution",
    "tessera.clifford_conjugate", "tessera.clifford_grade_projection",
    "tessera.clifford_hodge_star",
    "tessera.clifford_ext_deriv", "tessera.clifford_vec_deriv",
    "tessera.clifford_codiff",
    "tessera.clifford_exp", "tessera.clifford_log",
    "tessera.clifford_norm", "tessera.clifford_norm_squared",
})
# Energy-based-model flat-array lane — canonical tessera.ops projection of the
# tensor-clean tessera.ebm.* subset; the dispatcher calls the EBM lane, which
# internally routes f32 inputs to the dedicated EBM MSL kernels.
_APPLE_GPU_EBM_OPS = frozenset({
    "tessera.ebm_energy_quadratic", "tessera.ebm_self_verify",
    "tessera.ebm_refinement", "tessera.ebm_inner_step",
})
# EBM training losses (CD / PCD / score-matching / ISM / DSM) — MPSGraph
# reductions over energy/score tensors. reduction="mean" runs on GPU.
_APPLE_GPU_EBM_LOSS_OPS = frozenset({
    "tessera.loss.contrastive_divergence", "tessera.loss.persistent_cd",
    "tessera.loss.score_matching", "tessera.loss.implicit_score_matching",
    "tessera.loss.denoising_score_matching",
})
_APPLE_GPU_RUNTIME_OPS = (
    _APPLE_GPU_MPS_OPS | _APPLE_GPU_MSL_OPS | _APPLE_GPU_MPSGRAPH_OPS
    | _APPLE_GPU_PROJECTION_OPS | _APPLE_GPU_REDUCTION_OPS | _APPLE_GPU_TOPK_OPS
    | _APPLE_GPU_CONV_OPS
    | _APPLE_GPU_LINALG_OPS | _APPLE_GPU_SSM_OPS | _APPLE_GPU_MOE_OPS
    | _APPLE_GPU_SPECTRAL_OPS
    | _APPLE_GPU_LDT_OPS | _APPLE_GPU_CLIFFORD_OPS | _APPLE_GPU_EBM_OPS
    | _APPLE_GPU_EBM_LOSS_OPS | _APPLE_GPU_LOSS_COMPOSE_OPS
    | _APPLE_GPU_SOFTCAP_OPS | _APPLE_GPU_TRANSPOSE_OPS
    | _APPLE_GPU_NORM_COMPOSE_OPS | _APPLE_GPU_ATTN_WRAPPER_OPS
    | _APPLE_GPU_LINEAR_ATTN_OPS | _APPLE_GPU_MASKED_ATTN_OPS
    | _APPLE_GPU_DELTA_ATTN_OPS | _APPLE_GPU_HYBRID_ATTN_OPS
    | _APPLE_GPU_SPARSE_ATTN_OPS
)


def _build_lane_by_op() -> dict[str, str]:
    """op → fine-grained dispatch lane, first-match-wins in the historical
    runtime elif order. Lane names map 1:1 onto runtime handler adapters."""
    table: dict[str, str] = {}

    def put(ops, lane):
        for op in ops:
            table.setdefault(op, lane)

    put(_APPLE_GPU_MPS_OPS, "mps")
    put({"tessera.rope"}, "rope")
    put({"tessera.flash_attn"}, "flash_attn")
    put({"tessera.softmax", "tessera.softmax_safe"}, "softmax")
    put({"tessera.gelu"}, "gelu")
    put(_APPLE_GPU_UNARY_OPCODES, "unary")
    put(_APPLE_GPU_BINARY_OPCODES, "binary")
    put({"tessera.clamp", "tessera.clip"}, "clamp")
    put({"tessera.where"}, "where")
    put(_APPLE_GPU_SOFTCAP_OPS, "softcap")
    put(_APPLE_GPU_TRANSPOSE_OPS, "transpose")
    put(_APPLE_GPU_LOSS_COMPOSE_OPS, "loss_compose")
    put(_APPLE_GPU_NORM_COMPOSE_OPS, "norm_compose")
    put(_APPLE_GPU_ATTN_WRAPPER_OPS, "attn_wrapper")
    put(_APPLE_GPU_LINEAR_ATTN_OPS, "linear_attn")
    put(_APPLE_GPU_MASKED_ATTN_OPS, "masked_attn")
    put(_APPLE_GPU_DELTA_ATTN_OPS, "delta_attn")
    put(_APPLE_GPU_HYBRID_ATTN_OPS, "hybrid_attn")
    put(_APPLE_GPU_SPARSE_ATTN_OPS, "sparse_attn")
    put(_APPLE_GPU_ROWOP_KINDS, "rowop")
    put({"tessera.silu_mul"}, "silu_mul")
    put({"tessera.linear_general"}, "linear_general")
    put({"tessera.qkv_projection"}, "qkv_projection")
    put(_APPLE_GPU_REDUCE_OPS, "reduce")
    put(_APPLE_GPU_TOPK_OPS, "topk")
    put({"tessera.conv2d"}, "conv2d")
    put({"tessera.conv3d"}, "conv3d")
    put(_APPLE_GPU_LINALG_OPS, "linalg")
    put(_APPLE_GPU_SSM_OPS, "ssm")
    put({"tessera.moe_swiglu_block"}, "moe_swiglu_block")
    put({"tessera.grouped_gemm"}, "grouped_gemm")
    put(_APPLE_GPU_SPECTRAL_OPS, "spectral")
    put({"tessera.popcount"}, "popcount")
    put({"tessera.count_nonzero"}, "count_nonzero")
    put({"tessera.loss.z_loss"}, "z_loss")
    put({"tessera.loss.asymmetric_bce"}, "asymmetric_bce")
    put({"tessera.loss.load_balance_loss"}, "load_balance_loss")
    put({"tessera.masked_categorical"}, "masked_categorical")
    put(_APPLE_GPU_CLIFFORD_OPS, "clifford")
    put(_APPLE_GPU_EBM_OPS, "ebm")
    put(_APPLE_GPU_EBM_LOSS_OPS, "ebm_loss")
    missing = _APPLE_GPU_RUNTIME_OPS - set(table)
    if missing:  # structural invariant — every runtime op has a lane
        raise AssertionError(f"apple_gpu envelope ops without a lane: {sorted(missing)}")
    return table


#: op → dispatch lane for every op in ``_APPLE_GPU_RUNTIME_OPS``.
APPLE_GPU_LANE_BY_OP: dict[str, str] = _build_lane_by_op()

#: All lane names (runtime.py registers exactly one handler per lane).
APPLE_GPU_LANES: frozenset[str] = frozenset(APPLE_GPU_LANE_BY_OP.values())


def lane_for(op_name: str) -> str | None:
    """Dispatch lane for ``op_name`` (dotted or bare); None when the op is
    outside the runtime envelope."""
    if not op_name.startswith("tessera."):
        op_name = f"tessera.{op_name}"
    return APPLE_GPU_LANE_BY_OP.get(op_name)


def runtime_ops() -> frozenset[str]:
    """The full Apple GPU runtime envelope (dotted op names)."""
    return _APPLE_GPU_RUNTIME_OPS
