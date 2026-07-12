"""G4 — single-source runtime execution matrix.

Before this module, three places each had their own answer to "given an artifact,
what does the runtime actually do with it?":

- `capabilities.py` knew the per-target / per-op compile-time status
  (`ready` / `artifact_only` / `unimplemented`).
- `runtime.launch()` had a chain of hard-coded `target == "apple_cpu" and ...`,
  `target == "apple_gpu" and ...`, and `target != "cpu" -> unimplemented` branches.
- The docs / dashboards described it in prose.

They could drift. This module is the **one place** that maps a
``(target, compiler_path)`` pair to a structured `ExecutionRow`. The row tells
``launch()`` *which* executor to call (when any), what telemetry strings to use,
and what to return when no executor exists. ``capabilities.py`` consults the same
table to know which (target, compiler_path) pairs have a real runtime executor
backing the compile-time status. A generated dashboard renders the table for
humans; a drift test fails if anything diverges.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]

from .capabilities import TARGET_CAPABILITIES, normalize_target


# An executor takes (artifact, args) and returns the op output. Resolved lazily
# from runtime.py via name to avoid an import cycle (runtime imports this module).
EXECUTOR_ID = str
EXECUTOR_FN = Callable[..., object]


@dataclass(frozen=True)
class ExecutionRow:
    """One row of the execution matrix.

    A `(target, compiler_path)` pair resolves to **exactly one** row. The row is
    the runtime's contract: it names the executor (if any), the labels to use in
    telemetry + the result dict, and a precise reason when no executor exists.
    """

    target: str               # canonical target name (matches TARGET_CAPABILITIES)
    compiler_path: str        # e.g. "apple_cpu_accelerate", "apple_gpu_mps",
                              # "jit_cpu_numpy", "native_cpu", "artifact_only"
    execution_kind: str       # telemetry label: "native_cpu" / "native_gpu" /
                              # "reference_cpu" / "cpu_accelerate" / "artifact_only"
    executable: bool          # True iff there's a real executor function below
    executor_id: Optional[EXECUTOR_ID]   # symbolic name resolved at launch time
    runtime_status: str       # what to report when there's no executor:
                              # "unimplemented" / "missing_backend" / etc.
    reason: str = ""          # human-readable explanation for telemetry / errors
    execution_mode: str = ""  # telemetry-only: "metal_runtime" / "cpu_accelerate" / ""
    # Autodiff facet (AUTODIFF_UNIFICATION_PLAN §9a — Phase 4). `direction`
    # distinguishes a forward lane from a backward (VJP) launch; `op_family` names
    # the op whose backward this row launches (e.g. "flash_attn"), so the autodiff
    # ledger can source its native backward rungs from the matrix instead of
    # asserting them. Forward rows leave both at their defaults.
    direction: str = "forward"   # "forward" | "backward"
    op_family: str = ""          # backward rows: the op family this VJP launches
    # Exact-target backward proof.  An executable row alone is only
    # runtime-bound; device verification additionally requires one of the two
    # canonical proof statuses plus an architecture-aligned numerical fixture.
    device_proof: str = ""       # "device_verified_jit" | "device_verified_abi"
    evidence_target: str = ""    # e.g. "rocm_gfx1151" / "x86_avx512"
    numerical_fixture: str = ""  # repo-relative execute-and-compare fixture


# Catalog of every executor name → docstring describing what it runs. The actual
# functions live in `runtime.py`; this module deliberately does NOT import
# runtime.py (avoid the cycle — runtime.py imports `execution_matrix`).
KNOWN_EXECUTORS: dict[EXECUTOR_ID, str] = {
    "apple_cpu_accelerate": "Apple Silicon CPU via the Accelerate cblas_sgemm shim",
    "apple_gpu_mps":        "Apple Silicon GPU via MPS / MSL / MPSGraph (per envelope)",
    "apple_value_target_ir": "Apple CPU value-call dispatch — invokes the C ABI "
                             "symbol named in a tessera_apple.cpu.call value op "
                             "(Value Target IR sprint; CPU cholesky executable)",
    "apple_gpu_value_target_ir": "Apple GPU value-call dispatch — invokes the C "
                             "ABI symbol named in a tessera_apple.gpu.kernel_call "
                             "value op (rank-3 batched matmul f32/f16/bf16; "
                             "native sparse attention and PPO policy-loss variants "
                             "plus EBM quadratic energy/Langevin value kernels "
                             "when their Metal/MPSGraph executor probes are active)",
    "apple_gpu_structured_compute_compiled": "Apple GPU structured-compute tail — "
                             "the conv family (conv1d / conv_transpose / "
                             "depthwise_conv1d) reaches an executable apple_gpu "
                             "path via runtime.launch() and matches the reference "
                             "primitive (host-structured im2col/layout bookkeeping; "
                             "direct execute/compare evidence, not a bespoke fused "
                             "Metal kernel — parity with the x86/ROCm lanes)",
    "apple_gpu_loss_compiled": "Apple GPU pointwise-regression loss lane — mse / "
                             "mae / huber / smooth_l1 / log_cosh compose their "
                             "residual (pred-target) and none/mean/sum reduction "
                             "on the MPSGraph binary + reduce lanes (mse/mae also "
                             "use the GPU mul/abs opcodes; huber/smooth_l1/log_cosh "
                             "apply the piecewise/transcendental middle host-side). "
                             "Matches tessera.losses — parity with x86/ROCm "
                             "rocm_loss_compiled.",
    "apple_gpu_loss_family_compiled": "Apple GPU loss-family lane — binary-CE, "
                             "class-axis (cross_entropy / kl / js / z_loss), RL "
                             "policy (ppo / cispo / grpo), and EBM-diffusion "
                             "losses. The per-sample loss composes through the "
                             "standalone reference (host structure: gather/one-hot/"
                             "clip/softplus, some f64); the none/mean/sum reduction "
                             "runs on the MPSGraph reduce lane. Matches tessera."
                             "losses / tessera.rl — parity with the x86/ROCm "
                             "binary/class/rl/ebm loss lanes.",
    "apple_gpu_complex_compiled": "Apple GPU complex-arithmetic lane — the 9 "
                             "pointwise complex ops (mul / div / conjugate / abs / "
                             "arg / exp / log / sqrt / pow) compose interleaved-f32 "
                             "on the Apple GPU unary / binary / atan2 lanes; the "
                             "geometric/certificate ops (cross_ratio / dz / dbar / "
                             "laplacian_2d / conformal_* / is_concyclic / "
                             "check_cauchy_riemann / mobius_from_three_points) reuse "
                             "the tessera.complex reference (host structure, the "
                             "same path x86/ROCm take). Matches tessera.complex — "
                             "parity with x86/rocm_complex_compiled.",
    "apple_gpu_conformal_compiled": "Apple GPU conformal-geometry lane — mobius "
                             "f(z)=(az+b)/(cz+d) and stereographic projection "
                             "compose on the interleaved-f32 Apple GPU complex_mul "
                             "/ complex_div / binary-div lanes (no new kernel). "
                             "Matches tessera.complex — parity with "
                             "x86/rocm_conformal_compiled.",
    "apple_gpu_rng_compiled": "Apple GPU Philox RNG lane — rng_uniform / "
                             "rng_normal / dropout draw from the counter-based "
                             "Philox-4x32-10 reference (tessera.rng_device; Apple "
                             "ships no device Philox kernel), and the distribution "
                             "samplers (bernoulli / beta / categorical / dirichlet "
                             "/ gamma / poisson / randint / truncated_normal / "
                             "permutation / multinomial, RNGKey key/split/fold_in/"
                             "clone, and the MCMC samplers) run the public "
                             "tessera.rng RNGKey contract (the same path x86/ROCm "
                             "take for the distributions). Matches tessera.rng / "
                             "tessera.rng_device — parity with x86/rocm_rng_compiled.",
    "apple_gpu_linalg_compiled": "Apple GPU linalg lane — cholesky / tri_solve / "
                             "cholesky_solve / lu / qr / svd. Apple ships no MPS "
                             "lu/qr/svd primitive, so the decompositions resolve "
                             "on the numpy reference the x86/ROCm device kernels "
                             "match (qr/svd/cholesky via np.linalg; a standalone "
                             "partial-pivot LU; triangular solves via the "
                             "extracted triangle). Matches np.linalg — parity with "
                             "x86/rocm_linalg_compiled.",
    "apple_gpu_matmul_family_compiled": "Apple GPU matmul-family lane — einsum "
                             "(single-contraction spec) and factorized_matmul "
                             "(GEMM + rank-r SVD truncation) via the numpy "
                             "reference the x86/ROCm GEMM lanes match. Matches "
                             "numpy — parity with x86/rocm_matmul_family_compiled.",
    "apple_gpu_optimizer_compiled": "Apple GPU optimizer lane — sgd / momentum / "
                             "adam / adamw / lion per-parameter update. Apple "
                             "ships no device optimizer kernel, so the elementwise "
                             "update rules run on the numpy reference the x86/ROCm "
                             "device kernels are matched against (state m/v in/"
                             "out). Matches tessera.optim — parity with "
                             "x86/rocm_optimizer_compiled.",
    "apple_gpu_shape_compiled": "Apple GPU 0-move + sort lane — pad / roll / flip "
                             "/ tile / repeat / stack (host index-map + numpy "
                             "gather) and sort / argsort (numpy stable sort). "
                             "Apple ships no device gather/sort kernel, so this "
                             "runs on the CPU reference path. Matches "
                             "tessera.ops / numpy — parity with the x86/ROCm "
                             "strided + sort lanes.",
    "apple_gpu_reduce_compiled": "Apple GPU reduce lane — sum over an axis "
                             "(default: all) with keepdims, on the MPSGraph "
                             "reduce lane (numpy fallback when Metal is "
                             "unavailable). Matches numpy.sum — parity with "
                             "x86/rocm_reduce_compiled.",
    "apple_gpu_scatter_compiled": "Apple GPU scatter lane — scatter (set) / "
                             "scatter_add (sum) / scatter_reduce (min/max), "
                             "row-wise along an axis. Apple ships no device "
                             "scatter kernel, so the indexed store runs on the "
                             "numpy reference (np.add.at / np.minimum.at / "
                             "np.maximum.at). Matches numpy — parity with "
                             "x86/rocm_scatter_compiled.",
    "apple_gpu_sparse_compiled": "Apple GPU sparse + MoE lane — spmm_csr / "
                             "spmm_coo / sddmm / bsmm (numpy CSR SpMM / (a@b)*mask "
                             "/ a@b) and moe (routed per-token expert GEMVs, "
                             "top-1). Apple ships no device sparse/moe kernel, so "
                             "these run on the numpy reference. Matches numpy / "
                             "tessera — parity with the x86/ROCm sparse + moe "
                             "lanes.",
    "apple_gpu_tail_compiled": "Apple GPU reference tail lane — the heterogeneous "
                             "remainder: MLA latent-KV (compress / expand_k / "
                             "expand_v), alibi bias, lgamma / digamma, "
                             "fused_epilogue, asymmetric_bce, "
                             "normalize_group_advantages, and the "
                             "speculative-decode accept ops. Apple ships no "
                             "device kernel for these, so each reuses its public "
                             "tessera reference (tessera.ops / losses / rl). "
                             "Matches the reference — parity with the x86/ROCm "
                             "lanes.",
    "native_cpu":           "x86 AMX / native CPU runtime via the C runtime ABI",
    "jit_cpu_numpy":        "JIT CPU fallback via the numpy reference path",
    "rocm_wmma":            "AMD GPU RDNA WMMA matrix-core GEMM via the shipped "
                            "libtessera_rocm_gemm.so tessera_rocm_wmma_gemm_{f16,"
                            "bf16} C ABI symbol (HIPRTC-device_verified_jit for the device "
                            "arch; f16/bf16 storage, f32 accumulate)",
    "rocm_compiled":        "AMD GPU RDNA WMMA GEMM the Tessera compiler GENERATES "
                            "(Stage L): tessera-opt generates + serializes the "
                            "kernel to hsaco in-process (no mlir-opt), then HIP "
                            "loads + launches it. Opt-in; f16 storage, f32 accum; "
                            "the rocm_wmma lane stays the default + oracle",
    "rocm_flash_attn_compiled": "AMD GPU RDNA WMMA FA-2 forward the Tessera "
                            "compiler GENERATES (generate-wmma-flash-attn-kernel "
                            "-> ROCDL -> hsaco, in-process via tessera-opt), then "
                            "HIP loads + launches it. f16/bf16 storage, f32 "
                            "softmax + accumulate; the attention analog of "
                            "rocm_compiled",
    "rocm_flash_attn_bwd_compiled": "AMD GPU RDNA WMMA FA-2 BACKWARD the Tessera "
                            "compiler GENERATES (generate-wmma-flash-attn-bwd-"
                            "kernel -> three fa_pre/fa_dkdv/fa_dq WMMA kernels -> "
                            "hsaco), launched in sequence to produce dQ/dK/dV; O "
                            "is recomputed via the forward lane (nothing saved "
                            "from forward). MHA + GQA/MQA (grouped dkdv atomic-"
                            "accumulates dK/dV) + additive attn_bias + sliding-"
                            "window + logit-softcap; f16/bf16 storage, f32 "
                            "accumulate; the reverse-mode analog of "
                            "rocm_flash_attn_compiled",
    "rocm_selective_ssm_bwd_compiled": "AMD GPU RDNA Mamba2 selective_ssm "
                            "BACKWARD the Tessera compiler GENERATES "
                            "(generate-rocm-selective-ssm-bwd-kernel: one thread "
                            "per (b,d), atomic cross-channel reductions -> hsaco, "
                            "in-process via tessera-opt), HIP-launched to produce "
                            "(dx, dA, dB, dC, ddelta) from operands "
                            "(dout, x, A, B, C, delta[, gate[, state]]); the "
                            "reverse-mode analog of rocm_selective_ssm_compiled. "
                            "f32, matches autodiff.vjp.vjp_selective_ssm",
    "rocm_linear_attn_compiled": "AMD GPU RDNA WMMA linear-attention forward the "
                            "Tessera compiler GENERATES "
                            "(generate-wmma-linear-attn-kernel -> ROCDL -> hsaco, "
                            "in-process via tessera-opt), then HIP loads + "
                            "launches it. Quadratic-parallel form "
                            "O = (φ(Q)φ(K)ᵀ ⊙ causal [⊙ λ^(i-j)]) @ V, NO "
                            "softmax; f16/bf16 storage, f32 accumulate. Handles "
                            "tessera.linear_attn + the decay-masked siblings "
                            "tessera.lightning_attention (identity+decay) and "
                            "tessera.retention (x²+decay) by op name",
    "rocm_dspark_draft_block_compiled": "DS2 DSpark draft-block ROCm compiler "
                            "path — compiler-generated fused HIP/ROCDL "
                            "draft-block kernel (generate-rocm-dspark-draft-"
                            "block-kernel) with DS1 oracle fallback when ROCm "
                            "hardware is unavailable. f32/i64",
    "rocm_softmax_compiled": "AMD GPU RDNA row-reduction softmax the Tessera "
                            "compiler GENERATES (generate-rocm-softmax-kernel -> "
                            "ROCDL -> hsaco, in-process via tessera-opt), then HIP "
                            "loads + launches it. Stable softmax over the last "
                            "axis (one workgroup per row, LDS tree-reduce); the "
                            "first non-matmul/non-WMMA device_verified_jit ROCm kernel. "
                            "f32/f16/bf16 storage, f32 reduce",
    "rocm_norm_compiled":   "AMD GPU RDNA row-reduction rmsnorm / layer_norm the "
                            "Tessera compiler GENERATES (generate-rocm-norm-kernel "
                            "-> ROCDL -> hsaco, in-process via tessera-opt), then "
                            "HIP loads + launches it. Unweighted row normalize "
                            "over the last axis (one workgroup per row, LDS "
                            "tree-reduce of Σx and Σx²); handles "
                            "tessera.rmsnorm(_safe) + tessera.layer_norm by op "
                            "name. f32/f16/bf16 storage, f32 reduce",
    "rocm_reduce_compiled": "AMD GPU RDNA row reduction (sum/mean/max/min) the "
                            "Tessera compiler GENERATES (generate-rocm-reduce-"
                            "kernel -> ROCDL -> hsaco, in-process via "
                            "tessera-opt), then HIP loads + launches it — the "
                            "ROCm analog of the x86 AVX-512 reduction lane. An "
                            "arbitrary reduced axis folds to [outer,inner] "
                            "last-axis (one workgroup per row, LDS tree-reduce); "
                            "tessera.sum/mean/max/min (amax/amin) by op name. "
                            "f16/bf16/f32 storage, f32 reduce",
    "rocm_argreduce_compiled": "AMD GPU RDNA row arg-reduction (argmax/argmin) "
                            "the Tessera compiler GENERATES (generate-rocm-"
                            "argreduce-kernel -> ROCDL -> hsaco, in-process via "
                            "tessera-opt), then HIP launches it — a CUB ArgMax-"
                            "style warp-shuffle reduce carrying the (value,index) "
                            "pair, first-occurrence tie-break, along one axis. "
                            "f16/bf16/f32 input, i32 index output",
    "rocm_scan_compiled": "AMD GPU RDNA row inclusive prefix scan (cumsum/"
                            "cumprod/cummax/cummin) the Tessera compiler GENERATES "
                            "(generate-rocm-scan-kernel -> ROCDL -> hsaco), then "
                            "HIP launches it — the CUB BlockScan technique "
                            "(gpu.shuffle up Kogge-Stone warp-scan + subgroup "
                            "offset + cross-tile carry) along one axis; same-shape "
                            "output. f16/bf16/f32",
    "x86_reduce_compiled": "x86 CPU row reduction (sum/mean/max/min) — the "
                            "hand-written AVX-512 kernel (tessera_x86_avx512_"
                            "reduce_f32) the Python runtime ctypes-loads from "
                            "libtessera_x86_elementwise.so and calls directly; "
                            "an arbitrary reduced axis folds to [outer,inner] "
                            "last-axis (16 f32 lanes/__m512, NaN-propagating "
                            "max/min); tessera.sum/mean/max/min (amax/amin) by "
                            "op name. f32 only",
    "x86_scan_compiled": "x86 CPU inclusive prefix scan (cumsum/cumprod/cummax/"
                            "cummin) — tessera_x86_avx512_scan_f32 from "
                            "libtessera_x86_elementwise.so along one axis; "
                            "same-shape output (CPU analog of the ROCm block-scan "
                            "lane). f32 only",
    "x86_argreduce_compiled": "x86 CPU argmax/argmin — "
                            "tessera_x86_avx512_argreduce_f32 from "
                            "libtessera_x86_elementwise.so along one axis, "
                            "first-occurrence tie-break. f32 in, i32 index out",
    "x86_unary_compiled": "x86 CPU flat elementwise unary math — the hand-written "
                            "AVX-512 kernel (tessera_x86_avx512_unary_f32) the "
                            "Python runtime ctypes-loads from "
                            "libtessera_x86_elementwise.so and calls directly; the "
                            "direct-intrinsic algebraic + rounding subset "
                            "(sqrt/rsqrt/reciprocal/abs/sign/floor/ceil/trunc/"
                            "round), one __m512 of 16 f32 lanes at a time. "
                            "Transcendentals stay numpy-reference. f32 only",
    "x86_binary_compiled": "x86 CPU flat 2-operand elementwise arithmetic — the "
                            "hand-written AVX-512 kernel (tessera_x86_avx512_"
                            "binary_f32) the Python runtime ctypes-loads from "
                            "libtessera_x86_elementwise.so and calls directly; the "
                            "direct-intrinsic subset sub/div/maximum/minimum "
                            "(NaN-propagating max/min), 16 f32 lanes/__m512. `pow` "
                            "is transcendental → numpy-reference. f32 only",
    "x86_predicate_compiled": "x86 CPU unary predicate (isnan / isinf / "
                            "isfinite) — the hand-written AVX-512 kernel "
                            "(tessera_x86_avx512_predicate_f32; mask -> 0/1 "
                            "bytes), runtime-loaded. f32 in, bool out",
    "x86_complex_compiled": "x86 CPU complex arithmetic (9 pointwise ops) — "
                            "interleaved-f32 composed on AVX-512 transcendental/"
                            "unary/binary/atan2 kernels. f32",
    "x86_clamp_compiled": "x86 CPU clamp / clip — min(max(x, lo), hi) composed on "
                            "the AVX-512 binary max/min kernel (either bound "
                            "optional; scalar bounds broadcast on host). f32",
    "x86_strided_compiled": "x86 CPU 0-move lane — pad/cat/roll/flip/tile/"
                            "repeat/stack via the AVX-512 masked-gather kernel "
                            "(host index map). f32",
    "x86_conformal_compiled": "x86 CPU conformal lane — mobius (az+b)/(cz+d) "
                              "on the AVX-512 complex mul/div lane, "
                              "stereographic on the binary div lane. f32",
    "x86_sort_compiled": "x86 CPU sort lane — sort/argsort/top_k via the "
                         "AVX-512 bitonic sort network kernel (key+index; "
                         "host pads to a power of two + flips). f32",
    "x86_clifford_compiled": "x86 CPU geometric-algebra lane — Cl(3,0) "
                             "geometric_product/wedge/left_contraction/inner/"
                             "rotor_sandwich via the AVX-512 table-driven "
                             "bilinear kernel (compile-time Cayley table). f32",
    "x86_flash_attn_compiled": "x86 CPU attention lane — FA-style online-softmax "
                               "flash_attn forward via the AVX-512 kernel "
                               "(MHA scale+causal; the ROCm-WMMA partner). f32",
    "x86_mla_compiled": "x86 CPU MLA latent-KV lane — DeepSeek "
                        "latent_kv_compress/expand_k/expand_v/mla_decode_fused "
                        "composed on the AVX-512 GEMM + flash_attn lanes. f32",
    "x86_conv_compiled": "x86 CPU convolution lane — conv2d/conv3d via im2col "
                         "(host) + the AVX-512 f32 GEMM (device); bias/groups "
                         "on host. f32",
    "rocm_conv_compiled": "AMD GPU RDNA convolution lane — conv2d/conv3d via "
                          "im2col (host) + the COMPILER-GENERATED WMMA GEMM "
                          "(f16/bf16 storage, f32 accumulate). f16",
    "x86_nsa_compiled": "x86 CPU NSA lane — deepseek_sparse_attention: sliding "
                        "+ compressed-block + top-k-block branches on the "
                        "AVX-512 flash_attn kernels, gate blend on host. f32",
    "x86_msa_compiled": "x86 CPU MSA lane — msa_sparse_attention: exp-free "
                        "index-score + per-GQA-group top-k block select on host, "
                        "exact attend on the AVX-512 flash_attn kernel as dense "
                        "attention with a non-selected/causal additive -inf mask. "
                        "f32; dense-equivalence (top_k==num_blocks) → dense GQA",
    "x86_linear_attn_compiled": "x86 CPU linear-attention backbone — "
                                "linear_attn/power_attn/retention via the "
                                "quadratic-parallel form (φQ·φKᵀ ⊙ causal ⊙ "
                                "decay)@V on two AVX-512 batched GEMMs; feature "
                                "map / mask / decay on host (ROCm-linear_attn "
                                "partner). f32",
    "x86_scatter_compiled": "x86 CPU scatter lane — scatter/scatter_add/"
                            "scatter_reduce (0-reduce indexed store) via the "
                            "AVX-512 row-scatter kernel. f32",
    "x86_composite_helper_compiled": "x86 CPU composite helper lane for "
                            "memory_index_score / msa_index_scores / "
                            "varlen_sdpa / score_combine. Host shape/metadata "
                            "logic composes existing AVX-512-compatible "
                            "matmul/attention/binary runtime semantics and is "
                            "validated end-to-end through runtime.launch. f32",
    "rocm_scatter_compiled": "AMD GPU RDNA scatter lane — scatter/scatter_add/"
                             "scatter_reduce via the COMPILER-GENERATED gfx1151 "
                             "kernel (one thread per element; atomic_rmw). f32",
    "rocm_kv_cache_compiled": "AMD GPU RDNA KV-cache paged-movement lane — "
                             "kv_cache append/read/prune over a resident cache "
                             "buffer by composing the gfx1151 scatter (write) + "
                             "gather (read/prune) kernels; host page-index math. "
                             "f32, matches the KVCacheHandle reference",
    "x86_kv_cache_compiled": "x86 native KV-cache movement ABI — append/read/"
                             "prune over a contiguous resident f32 cache through "
                             "libtessera_x86_elementwise.so; matches the "
                             "KVCacheHandle reference",
    "x86_rng_compiled": "x86 CPU device RNG — counter-based Philox-4x32-10 "
                            "uniform kernel + host transform (uniform/normal/"
                            "dropout). f32",
    "x86_softcap_compiled": "x86 CPU softcap — cap*tanh(x/cap) composed on the "
                            "AVX-512 transcendental tanh kernel (scalar cap "
                            "broadcast on host). f32",
    "x86_atan2_compiled": "x86 CPU atan2 — quadrant-aware atan2(y, x) composed "
                            "on the AVX-512 transcendental atan kernel "
                            "(sign/quadrant on host). f32",
    "x86_compare_compiled": "x86 CPU flat 2-operand elementwise comparison — the "
                            "hand-written AVX-512 kernel (tessera_x86_avx512_"
                            "compare_f32) the Python runtime ctypes-loads from "
                            "libtessera_x86_elementwise.so and calls directly; "
                            "eq/ne/lt/le/gt/ge via _mm512_cmp_ps_mask + "
                            "_mm_maskz_set1_epi8, NaN semantics match numpy "
                            "(ordered except ne). f32 in, bool out",
    "x86_logical_compiled": "x86 CPU flat elementwise logical — the hand-written "
                            "AVX-512 kernel (tessera_x86_avx512_logical_i8) the "
                            "Python runtime ctypes-loads from "
                            "libtessera_x86_elementwise.so and calls directly; "
                            "and/or/xor (binary) + not (unary) over i8 booleans "
                            "(inputs normalized via != 0, _mm512_cmpneq_epi8 + "
                            "and/or/xor_si512), dispatched by op name; bool in/out",
    "x86_bitwise_compiled": "x86 CPU flat elementwise bitwise — the hand-written "
                            "AVX-512 kernel (tessera_x86_avx512_bitwise_i32) the "
                            "Python runtime ctypes-loads from "
                            "libtessera_x86_elementwise.so and calls directly; "
                            "and/or/xor (binary) + not (unary) over i32 integers "
                            "(full bit pattern, _mm512_{and,or,xor}_si512), "
                            "dispatched by op name; i32 in/out",
    "rocm_activation_compiled": "AMD GPU RDNA flat elementwise activation the "
                            "Tessera compiler GENERATES "
                            "(generate-rocm-activation-kernel -> ROCDL -> hsaco, "
                            "in-process via tessera-opt), then HIP loads + "
                            "launches it. Standalone gelu / silu / relu (one "
                            "thread per element), dispatched by op name; "
                            "f32/f16/bf16 storage, f32 compute",
    "rocm_unary_compiled": "AMD GPU RDNA flat elementwise unary-math kernel the "
                            "Tessera compiler GENERATES (generate-rocm-unary-"
                            "kernel -> ROCDL -> hsaco, in-process via tessera-"
                            "opt), then HIP loads + launches it — the S2 scalar-"
                            "math / stability family (exp/log/sqrt/rsqrt/"
                            "reciprocal/abs/sign/erf/tanh/sigmoid/log1p/expm1/"
                            "softplus), one thread per element, dispatched by op "
                            "name; f32/f16/bf16 storage, f32 compute",
    "rocm_binary_compiled": "AMD GPU RDNA flat 2-operand elementwise binary-"
                            "arithmetic kernel the Tessera compiler GENERATES "
                            "(generate-rocm-binary-kernel -> ROCDL -> hsaco, in-"
                            "process via tessera-opt), then HIP loads + launches "
                            "it — the S2 binary-arithmetic family (sub/div/pow/"
                            "maximum/minimum), one thread per element, dispatched "
                            "by op name; f32/f16/bf16 storage, f32 compute",
    "rocm_predicate_compiled": "AMD GPU RDNA unary predicate (isnan / isinf / "
                            "isfinite) kernel the Tessera compiler GENERATES "
                            "(generate-rocm-predicate-kernel, kind StrAttr → ROCDL "
                            "→ hsaco), then HIP launches — one thread per element. "
                            "f32 in, i8/bool out",
    "rocm_complex_compiled": "AMD GPU RDNA complex arithmetic (9 pointwise ops) "
                            "— interleaved-f32 composed on gfx1151 unary/binary/"
                            "atan2 kernels. f32",
    "rocm_clamp_compiled": "AMD GPU RDNA clamp / clip — min(max(x, lo), hi) "
                            "composed on the rocm_binary_compiled max/min kernel "
                            "(either bound optional; scalar bounds broadcast on "
                            "host). f32",
    "rocm_strided_compiled": "AMD GPU RDNA 0-move lane — pad/cat/roll/flip/tile/"
                            "repeat/stack via the gfx1151 masked-gather kernel "
                            "(host index map). f32",
    "rocm_conformal_compiled": "AMD GPU RDNA conformal lane — mobius "
                               "(az+b)/(cz+d) on the gfx1151 complex mul/div "
                               "lane, stereographic on the binary div lane. f32",
    "rocm_sort_compiled": "AMD GPU RDNA sort lane — sort/argsort/top_k via the "
                          "COMPILER-GENERATED cooperative bitonic kernel "
                          "(one block per row; host pads + flips). f32",
    "rocm_clifford_compiled": "AMD GPU RDNA geometric-algebra lane — Cl(3,0) "
                              "geometric_product/wedge/left_contraction/inner/"
                              "rotor_sandwich via the COMPILER-GENERATED "
                              "table-driven bilinear kernel "
                              "(triples unrolled at generation time). f32",
    "rocm_rng_compiled": "AMD GPU RDNA device RNG — COMPILER-GENERATED gfx1151 "
                            "Philox-4x32-10 uniform kernel + host transform "
                            "(uniform/normal/dropout). f32",
    "rocm_softcap_compiled": "AMD GPU RDNA softcap — cap*tanh(x/cap) composed on "
                            "the rocm_unary_compiled tanh kernel (scalar cap "
                            "broadcast on host). f32",
    "rocm_atan2_compiled": "AMD GPU RDNA atan2 — quadrant-aware atan2(y, x) "
                            "composed on the rocm_unary_compiled atan kernel "
                            "(sign/quadrant on host). f32",
    "rocm_compare_compiled": "AMD GPU RDNA flat 2-operand elementwise comparison "
                            "kernel the Tessera compiler GENERATES (generate-rocm-"
                            "compare-kernel -> ROCDL -> hsaco, in-process via "
                            "tessera-opt), then HIP loads + launches it — the S2 "
                            "comparison family (eq/ne/lt/le/gt/ge), one thread per "
                            "element, dispatched by op name; f32/f16/bf16 input "
                            "storage, f32 compare, i8/bool output",
    "rocm_logical_compiled": "AMD GPU RDNA flat elementwise logical kernel the "
                            "Tessera compiler GENERATES (generate-rocm-logical-"
                            "kernel -> ROCDL -> hsaco, in-process via tessera-"
                            "opt), then HIP loads + launches it — the S2 logical "
                            "family (and/or/xor binary, not unary) over i8 "
                            "booleans (inputs normalized via != 0), one thread "
                            "per element, dispatched by op name; bool in/out",
    "rocm_bitwise_compiled": "AMD GPU RDNA flat elementwise bitwise kernel the "
                            "Tessera compiler GENERATES (generate-rocm-bitwise-"
                            "kernel -> ROCDL -> hsaco, in-process via tessera-"
                            "opt), then HIP loads + launches it — the S2 bitwise "
                            "family (and/or/xor binary, not unary) over i32 "
                            "integers (full bit pattern, no normalization), one "
                            "thread per element, dispatched by op name; i32 in/out",
    "rocm_where_compiled": "AMD GPU RDNA flat 3-operand ternary select "
                            "where(cond,a,b)=cond?a:b the Tessera compiler "
                            "GENERATES (generate-rocm-where-kernel -> ROCDL -> "
                            "hsaco), then HIP launches it; cond i8 normalized != "
                            "0, a/b/out f16/bf16/f32",
    "x86_where_compiled": "x86 CPU ternary select where(cond,a,b) — "
                            "tessera_x86_avx512_where_f32 from "
                            "libtessera_x86_elementwise.so "
                            "(_mm512_cmpneq_epi8_mask + _mm512_mask_blend_ps); "
                            "cond i8 != 0, a/b/out f32",
    "x86_nvfp4_compiled": "x86 CPU NVFP4 — block-scaled fp4 (E2M1 codes + one "
                            "fp8-E4M3 scale per 16-elem block): per-block amax / "
                            "reassembly on the host, the fp8 scale and the e2m1 "
                            "codes on the AVX-512 fpquant kernel. Matches the "
                            "compiler/microscaling reference exactly. f32 "
                            "fake-quant",
    "x86_fpquant_compiled": "x86 CPU low-precision float quantize — quantize/"
                            "dequantize fp8 / fp6 / fp4: per-tensor symmetric "
                            "scale + grid-snap on the AVX-512 fpquant kernel "
                            "(tessera_x86_avx512_fpquant_f32: getexp/roundscale/"
                            "scalef mantissa-snap, RNE); dequantize is the fp32 "
                            "passthrough. f32, matches the reference exactly",
    "x86_intquant_compiled": "x86 CPU integer quantization — qparam selection and "
                            "int8 container conversion around AVX-512 round/"
                            "max/min/mul kernels; covers int8 and signed int4 "
                            "values stored in int8 containers",
    "x86_pooling_compiled": "x86 CPU pooling — host window matrix / adaptive "
                            "window partitioning with max/min/mean on the "
                            "AVX-512 reduce kernel",
    "x86_image_affine_compiled": "x86 CPU image affine preprocessing — "
                            "image_normalize as sub/div on AVX-512 binary "
                            "kernels with host layout and per-channel broadcast",
    "x86_metric_loss_compiled": "x86 CPU metric/contrastive loss tail — "
                            "reductions and exp/log on AVX-512 kernels with "
                            "host label/mask/sort/matrix structure",
    "x86_structured_compute_compiled": "x86 CPU structured compute tail — "
                            "CTC dynamic programming, VLM image/layout "
                            "transforms, conv/recurrent cells, and streaming "
                            "depthwise conv through runtime.launch(); host "
                            "shape/control structure with target-owned "
                            "single-GPU dispatch evidence",
    "x86_class_loss_compiled": "x86 CPU class-axis loss — cross_entropy / kl / "
                            "js / focal / label_smoothed_cross_entropy / z_loss: "
                            "exp/log on the AVX-512 transcendental kernel, "
                            "class-axis max/sum/gather/one-hot on the host, "
                            "leading-axis reduction on the reduce kernel. f32, "
                            "matches numpy 2e-4",
    "x86_ebm_loss_compiled": "x86 CPU EBM/diffusion loss — score_matching / "
                             "denoising / implicit / contrastive_divergence / "
                             "persistent_cd / ddpm_noise_pred / vlb / "
                             "load_balance: diff/square + reductions on the "
                             "AVX-512 binary + reduce kernels, host structure. "
                             "f32",
    "x86_ebm_compute_compiled": "x86 CPU EBM compute — energy_quadratic / "
                                "inner_step / refinement / self_verify: "
                                "diff/square/reduce on the AVX-512 binary + "
                                "reduce kernels, scalar scale / argmin gather on "
                                "the host. f32",
    "x86_ebm_langevin_compiled": "x86 CPU EBM Langevin sampling — y − η·grad + "
                                 "noise_scale·z with z drawn on-device from "
                                 "Philox-4x32-10 Box-Muller (AVX-512 kernel). "
                                 "f32",
    "x86_rl_loss_compiled": "x86 CPU RL policy loss — ppo / cispo / grpo core "
                            "surrogate on the AVX-512 policy-loss kernel "
                            "(tessera_x86_avx512_policy_loss_f32, ratio=exp(ln-"
                            "lo), clipped surrogate) + normalize_group_"
                            "advantages on the AVX-512 layer_norm kernel over the "
                            "group axis. KL/entropy/mask add-ons diagnose out. "
                            "f32, matches numpy 2e-5",
    "x86_binary_loss_compiled": "x86 CPU binary-cross-entropy loss — bce / "
                            "asymmetric_bce over (logits, targets): per-element "
                            "loss on the AVX-512 binary-loss kernel "
                            "(tessera_x86_avx512_binary_loss_f32, stable "
                            "softplus form) + none/mean/sum reduction on the "
                            "reduce kernel. f32, matches numpy 2e-5",
    "x86_fft_compiled": "x86 CPU spectral FFT (fft / ifft / rfft / irfft) over "
                            "ANY axis length — AVX-512 radix-2 Cooley-Tukey C2C "
                            "kernel (tessera_x86_fft_c2c_f32; deinterleave-permute "
                            "SIMD complex butterflies) for power-of-two; tiny "
                            "non-pow2 via the DFT-matrix on the AVX-512 GEMM; "
                            "other non-pow2 via Bluestein (chirp-z) over the "
                            "radix-2 kernel. SpectralPlan owns strategy + "
                            "normalization. complex64/f32",
    "x86_spectral_compiled": "x86 CPU spectral composites (dct / stft / istft / "
                            "spectral_conv / spectral_filter) — compose the "
                            "x86_fft_compiled FFT lane (frame / window / "
                            "overlap-add / pointwise complex-mul on host). f32",
    "x86_sparse_compiled": "x86 CPU sparse linear algebra (spmm_csr / spmm_coo / "
                            "sddmm / bsmm) — GENUINELY sparse AVX-512 kernels "
                            "(spmm = row-wise AXPY over CSR nonzeros, sddmm = "
                            "sampled dense-dense dot over masked entries), bsmm "
                            "via the AVX-512 GEMM microkernel. COO folds to CSR "
                            "on host. f32",
    "x86_moe_compiled": "x86 CPU mixture-of-experts compute (moe) — AVX-512 "
                            "routed per-token expert GEMV kernel (top-1; routing "
                            "resolved on host, out_dim vectorized). f32",
    "x86_optimizer_compiled": "x86 CPU optimizer steps (sgd / momentum / adam / "
                            "adamw / lion) — AVX-512 fused per-parameter update "
                            "kernel (kind-selected; m/v state in-place; host "
                            "computes the 1-β^t bias correction). f32",
    "x86_normcompose_compiled": "x86 CPU group/instance/weight norm — composed "
                            "on the AVX-512 layer_norm + reduce kernels (host "
                            "reshape/divide). f32",
    "x86_grad_clip_compiled": "x86 CPU grad_clip_norm — g*min(1,max_norm/||g||); "
                            "the global L2 sum-of-squares runs on the AVX-512 "
                            "reduce kernel, host sqrt + scale. f32",
    "rocm_grad_clip_compiled": "AMD GPU RDNA grad_clip_norm — g*min(1,max_norm/"
                            "||g||); the global L2 sum-of-squares runs on the "
                            "gfx1151 reduce kernel, host sqrt + scale. f32",
    "x86_lamb_compiled": "x86 CPU LAMB — AVX-512 adam kernel (lr=1/wd=0) + host "
                            "per-tensor trust ratio ‖p‖/‖update‖. f32",
    "x86_muon_compiled": "x86 CPU Muon — AVX-512 SVD U·Vh orthogonalization of "
                            "the momentum matrix + host U@Vh/sgd. f32",
    "x86_selective_ssm_compiled": "x86 CPU Mamba2 selective_ssm — AVX-512 fused "
                            "selective-scan kernel (single pass over S, "
                            "vectorized over the state dim N, exp via the Cephes "
                            "core; in-place (B,D,N) state). f32",
    "x86_selective_ssm_bwd_compiled": "x86 CPU Mamba2 selective_ssm BACKWARD — "
                            "AVX-512 fused backward kernel "
                            "(tessera_x86_selective_ssm_bwd_f32: forward-fill "
                            "h_traj then reverse scan accumulating dx/dA/dB/dC/"
                            "ddelta) behind the runtime.launch() ABI, operands "
                            "(dout, x, A, B, C, delta[, gate[, state]]); the "
                            "reverse-mode analog of x86_selective_ssm_compiled. "
                            "f32, matches autodiff.vjp.vjp_selective_ssm",
    "x86_linalg_compiled": "x86 CPU dense linear algebra (cholesky / tri_solve / "
                            "cholesky_solve / lu / qr / svd) — AVX-512 Cholesky–"
                            "Banachiewicz factorization, forward/back triangular "
                            "substitution (RHS columns vectorized), getrf partial-"
                            "pivot LU, Householder QR, one-sided Jacobi SVD "
                            "(vectorized rank-1/reflector/rotation updates); "
                            "cholesky_solve = two triangular solves. Batched. f32",
    "x86_stat_reduce_compiled": "x86 CPU statistical reduction (var / std / "
                            "count_nonzero) composed from the AVX-512 reduce "
                            "kernel: var=mean(x^2)-mean(x)^2, std=sqrt(var), "
                            "count_nonzero=sum(x!=0) over an axis. f32",
    "x86_stable_reduce_compiled": "x86 CPU stable reduction (logsumexp / "
                            "log_softmax / softmax_safe / sigmoid_safe) — "
                            "max-shifted, composed from the AVX-512 reduce "
                            "(max/sum) + the transcendental exp/log lane; "
                            "softmax_safe / sigmoid_safe alias the stable softmax "
                            "/ sigmoid lanes. f32",
    "x86_loss_compiled": "x86 CPU pointwise loss — mse / mae / huber / "
                            "smooth_l1 / log_cosh over (pred, target): per-element "
                            "loss on the AVX-512 loss kernel "
                            "(tessera_x86_avx512_pointwise_loss_f32) + "
                            "none/mean/sum reduction on the AVX-512 reduce "
                            "kernel. f32, matches numpy 2e-5",
    "x86_attention_compiled": "x86 CPU softmax-attention — multi_head / gqa / "
                            "mqa / mla_decode, all composed from the AVX-512 f32 "
                            "GEMM (QK^T and probs*V) + the AVX-512 row-softmax "
                            "kernel, with reshape/scale/causal-mask/KV-group in "
                            "Python; the CPU analog of the ROCm WMMA flash-"
                            "attention family. f32",
    "x86_rope_compiled": "x86 CPU interleaved-pair rotary position embedding "
                            "(rope) — tessera_x86_avx512_rope_f32 from "
                            "libtessera_x86_elementwise.so (AVX-512 deinterleave "
                            "+ Cephes sincos + re-interleave); operands x, theta "
                            "both [.., D] (D even). The CPU analog of the ROCm "
                            "rope lane. f32, matches numpy 2e-5",
    "x86_alibi_compiled": "x86 CPU ALiBi positional-bias generator "
                            "bias[h,i,j]=slope[h]*(j-i) over [H,S,S] — "
                            "tessera_x86_avx512_alibi_f32 from "
                            "libtessera_x86_elementwise.so; default slope ramp "
                            "2^(-8k/H), optional slopes operand. The CPU analog "
                            "of the ROCm alibi lane. f32",
    "x86_matmul_family_compiled": "x86 CPU matmul-family kernel — matmul / gemm "
                            "/ batched_gemm / linear_general / qkv_projection / "
                            "factorized_matmul / einsum, all built on the "
                            "AVX-512 f32 GEMM microkernel "
                            "(tessera_x86_avx512_gemm_f32) with the "
                            "reshape/batch/einsum logic in Python; the CPU "
                            "analog of the ROCm WMMA matmul-family lane. f32",
    "x86_norm_compiled": "x86 CPU row-reduction norm kernel — unweighted "
                            "rmsnorm / layer_norm over the last axis "
                            "(tessera_x86_avx512_{rmsnorm,layernorm}_f32 from "
                            "libtessera_x86_elementwise.so; AVX-512 horizontal "
                            "reduce); the CPU analog of the ROCm norm lane. f32, "
                            "matches numpy 2e-5",
    "x86_softmax_compiled": "x86 CPU row-reduction softmax kernel — "
                            "numerically-stable softmax over the last axis "
                            "(tessera_x86_avx512_softmax_f32 from "
                            "libtessera_x86_elementwise.so; AVX-512 max+exp+sum, "
                            "exp via the Cephes core); the CPU analog of the "
                            "ROCm softmax lane. f32, matches numpy 2e-5",
    "x86_binary_math_compiled": "x86 CPU transcendental-backed binary kernel — "
                            "pow(a,b) (positive base, a^b via exp(b*log(a))) + "
                            "silu_mul(a,b)=silu(a)*b (SwiGLU gate-multiply) from "
                            "libtessera_x86_elementwise.so, sharing the AVX-512 "
                            "exp/log/sigmoid cores; f32, matches numpy 2e-5",
    "x86_transcendental_compiled": "x86 CPU vectorized transcendental / "
                            "activation kernel (exp/log/tanh/sigmoid/silu/gelu/"
                            "erf/softplus/expm1/log1p/cos/tan/sinh/cosh/asin/"
                            "acos/atan/erfc) — "
                            "tessera_x86_avx512_transcendental_f32 from "
                            "libtessera_x86_elementwise.so (Cephes exp/log "
                            "minimax cores + Abramowitz-Stegun erf; activations "
                            "compose); reaches ROCm math->ROCDL parity. gelu is "
                            "the tanh approximation. f32, matches numpy 2e-5",
    "rocm_silu_mul_compiled": "AMD GPU RDNA SwiGLU gate-multiply the Tessera "
                            "compiler GENERATES (generate-rocm-silu-mul-kernel "
                            "-> ROCDL -> hsaco, in-process via tessera-opt), then "
                            "HIP loads + launches it. Flat 2-operand elementwise "
                            "silu(a)·b (one thread per element); the standalone "
                            "analog of the fused SwiGLU gate-multiply; "
                            "f32/f16/bf16 storage, f32 compute",
    "rocm_loss_compiled": "AMD GPU RDNA pointwise regression loss (mse / mae / "
                            "huber / smooth_l1 / log_cosh) the Tessera compiler "
                            "GENERATES (generate-rocm-pointwise-loss-kernel -> "
                            "ROCDL -> hsaco), then HIP launches it; per-element "
                            "loss on gfx1151 + none/mean/sum reduction. The CPU "
                            "analog is the x86_loss lane. f32/f16/bf16",
    "rocm_fft_compiled": "AMD GPU RDNA spectral FFT (fft / ifft / rfft / irfft) "
                            "the Tessera compiler GENERATES (generate-rocm-dft-"
                            "kernel -> ROCDL -> hsaco; one thread per output bin, "
                            "cos/sin twiddles), then HIP launches it. Direct DFT "
                            "(any length) on gfx1151 + r2c/c2r pack-unpack + plan "
                            "scale (radix-2/Bluestein perf is a follow-up). "
                            "complex64/f32",
    "rocm_spectral_compiled": "AMD GPU RDNA spectral composites (dct / stft / "
                            "istft / spectral_conv / spectral_filter) — compose "
                            "the rocm_fft_compiled DFT lane (frame / window / "
                            "overlap-add / pointwise complex-mul on host). f32",
    "rocm_sparse_compiled": "AMD GPU RDNA sparse linear algebra (spmm_csr / "
                            "spmm_coo / sddmm / bsmm) — COMPILER-GENERATED gfx1151 "
                            "sparse kernels (generate-rocm-spmm/sddmm-kernel: row-"
                            "wise CSR SpMM, sampled dense-dense SDDMM that skips "
                            "masked-zero entries) then HIP-launched; bsmm via the "
                            "WMMA matmul (bf16). f32",
    "rocm_sparse_attn_compiled": "AMD GPU RDNA DK2 selected-block sparse "
                            "attention (MSA and DSA/NSA block-id layouts) — "
                            "COMPILER-GENERATED scalar + row-tiled block-sparse "
                            "attention kernels plus GPU-resident top-k selection "
                            "over explicit B,Hkv,Sq,top_k block ids with GQA, "
                            "causal q-position masking, and f32 softmax",
    "rocm_composite_helper_compiled": "ROCm composite helper lane for "
                            "memory_index_score / msa_index_scores / "
                            "varlen_sdpa / score_combine. The Target IR lane "
                            "keeps the op compiler-visible and delegates to "
                            "existing matmul/flash-attn/binary helper semantics; "
                            "runtime.launch has an exact reference fallback "
                            "until a HIP-native helper is hardware-proven. f32",
    "rocm_optimizer_compiled": "AMD GPU RDNA optimizer steps (sgd / momentum / "
                            "adam / adamw / lion) — COMPILER-GENERATED gfx1151 "
                            "fused per-parameter update kernel (generate-rocm-"
                            "optimizer-kernel, kind StrAttr-selected; host "
                            "computes the 1-β^t bias correction). f32",
    "rocm_lamb_compiled": "AMD GPU RDNA LAMB — COMPILER-GENERATED gfx1151 adam "
                            "kernel (lr=1/wd=0) + host per-tensor trust ratio "
                            "‖p‖/‖update‖. f32",
    "rocm_muon_compiled": "AMD GPU RDNA Muon — gfx1151 SVD U·Vh "
                            "orthogonalization of the momentum matrix + host "
                            "U@Vh/sgd. f32",
    "rocm_moe_compiled": "AMD GPU RDNA mixture-of-experts compute (moe) — "
                            "COMPILER-GENERATED gfx1151 kernel (generate-rocm-moe-"
                            "kernel: routed per-token expert GEMV, one thread per "
                            "(token, out-col); routing resolved on host) HIP-"
                            "launched. f32",
    "rocm_moe_transport_compiled": "DK3 MoE transport ROCm compiler path — "
                            "moe_dispatch/moe_combine/grouped_swiglu over a "
                            "DispatchPlan through the runtime ABI; reference_cpu "
                            "until HIP gather/scatter transport kernels are promoted.",
    "rocm_normcompose_compiled": "AMD GPU RDNA group/instance/weight norm — "
                            "composed on the gfx1151 layer_norm + reduce kernels "
                            "(host reshape/divide). f32",
    "rocm_selective_ssm_compiled": "AMD GPU RDNA Mamba2 selective_ssm — COMPILER-"
                            "GENERATED gfx1151 selective-scan kernel (generate-"
                            "rocm-selective-ssm-kernel, one thread per (b,d) "
                            "channel, exp via math->rocdl) HIP-launched. f32",
    "rocm_linalg_compiled": "AMD GPU RDNA dense linear algebra (cholesky / "
                            "tri_solve / cholesky_solve / lu / qr / svd) — "
                            "COMPILER-GENERATED gfx1151 kernels (generate-rocm-"
                            "cholesky / tri-solve / lu / qr / svd-kernel, one "
                            "thread per matrix or matrix/RHS-column) HIP-launched; "
                            "cholesky_solve = two triangular solves. f32",
    "rocm_stat_reduce_compiled": "AMD GPU RDNA statistical reduction (var / std "
                            "/ count_nonzero) composed from the warp-shuffle "
                            "reduce kernel: var=mean(x^2)-mean(x)^2, "
                            "std=sqrt(var), count_nonzero=sum(x!=0) over an axis. "
                            "f32",
    "rocm_stable_reduce_compiled": "AMD GPU RDNA stable reduction (logsumexp / "
                            "log_softmax / softmax_safe / sigmoid_safe) — "
                            "max-shifted, composed from the warp-shuffle reduce "
                            "(max/sum) + the unary exp/log lane; softmax_safe / "
                            "sigmoid_safe alias the stable softmax / sigmoid "
                            "lanes. f32",
    "rocm_class_loss_compiled": "AMD GPU RDNA class-axis loss (cross_entropy / "
                            "kl / js / focal / label_smoothed / z_loss) — exp/log "
                            "on the rocm unary lane (gfx1151), class-axis "
                            "max/sum/gather/one-hot on the host. ROCm mirror of "
                            "x86_class_loss. f32",
    "rocm_ebm_loss_compiled": "AMD GPU RDNA EBM/diffusion loss (score_matching / "
                            "denoising / implicit / contrastive_divergence / "
                            "persistent_cd / ddpm_noise_pred / vlb / "
                            "load_balance) — diff/square + reductions on the "
                            "gfx1151 binary + reduce kernels, host structure. "
                            "ROCm mirror of x86_ebm_loss. f32",
    "rocm_ebm_compute_compiled": "AMD GPU RDNA EBM compute (energy_quadratic / "
                            "inner_step / refinement / self_verify) — "
                            "diff/square/reduce on the gfx1151 binary + reduce "
                            "kernels, host structure. ROCm mirror of "
                            "x86_ebm_compute. f32",
    "rocm_ebm_langevin_compiled": "AMD GPU RDNA EBM Langevin sampling — y − "
                            "η·grad + noise_scale·z with z drawn on-device from "
                            "Philox-4x32-10 Box-Muller via the COMPILER-"
                            "GENERATED gfx1151 kernel. f32",
    "rocm_fpquant_compiled": "AMD GPU RDNA low-precision float quantize "
                            "(quantize/dequantize fp8 / fp6 / fp4) — grid-snap on "
                            "generate-rocm-fpquant-kernel (log2/exp2/roundeven) + "
                            "per-tensor scale. ROCm mirror of x86_fpquant. f32",
    "rocm_intquant_compiled": "AMD GPU RDNA integer quantization — qparam "
                            "selection and int8 container conversion around "
                            "generated ROCm unary/binary kernels; covers int8 "
                            "and signed int4 values stored in int8 containers",
    "rocm_pooling_compiled": "AMD GPU RDNA pooling — host window matrix / "
                            "adaptive window partitioning with max/min/mean on "
                            "the generated ROCm reduce kernel",
    "rocm_image_affine_compiled": "AMD GPU RDNA image affine preprocessing — "
                            "image_normalize as sub/div on generated ROCm "
                            "binary kernels with host layout and per-channel "
                            "broadcast",
    "rocm_metric_loss_compiled": "AMD GPU RDNA metric/contrastive loss tail — "
                            "reductions and exp/log on generated ROCm kernels "
                            "with host label/mask/sort/matrix structure",
    "rocm_structured_compute_compiled": "AMD GPU RDNA structured compute tail — "
                            "CTC dynamic programming, VLM image/layout "
                            "transforms, conv/recurrent cells, and streaming "
                            "depthwise conv through runtime.launch(); host "
                            "shape/control structure with target-owned "
                            "single-GPU dispatch evidence",
    "rocm_dequant_gemm_compiled": "DK4 dequant-GEMM ROCm compiler path — "
                            "compiler-generated fused HIP/ROCDL packed-code "
                            "dequant-into-GEMM kernel (generate-rocm-dequant-"
                            "gemm-kernel) with packed int4/int8 codes + scales "
                            "through the runtime ABI. f32 accumulate",
    "rocm_nvfp4_compiled": "AMD GPU RDNA NVFP4 block-scaled fp4 — per-block "
                            "fp8-E4M3 scale + E2M1 codes on the fpquant kernel + "
                            "host block structure. ROCm mirror of x86_nvfp4. f32",
    "rocm_binary_loss_compiled": "AMD GPU RDNA binary-cross-entropy loss (bce / "
                            "asymmetric_bce) the Tessera compiler GENERATES "
                            "(generate-rocm-binary-loss-kernel -> ROCDL -> "
                            "hsaco), then HIP launches it; per-element on gfx1151 "
                            "+ reduction. ROCm mirror of x86_binary_loss. "
                            "f32/f16/bf16",
    "rocm_rl_loss_compiled": "AMD GPU RDNA RL policy loss (ppo / cispo / grpo "
                            "surrogate on generate-rocm-policy-loss-kernel + "
                            "normalize_group_advantages on the norm lane) the "
                            "Tessera compiler GENERATES, then HIP launches; "
                            "per-element on gfx1151 + reduction. ROCm mirror of "
                            "x86_rl_loss. f32/f16/bf16",
    "rocm_alibi_compiled":  "AMD GPU RDNA ALiBi positional-bias generator the "
                            "Tessera compiler GENERATES (generate-rocm-alibi-"
                            "kernel -> ROCDL -> hsaco, in-process via "
                            "tessera-opt), then HIP loads + launches it. "
                            "bias[h,i,j] = slope[h]·(j−i) over [H, S, S] (one "
                            "thread per element); slopes default to the "
                            "2^(-8(k+1)/H) ramp; f32/f16/bf16 output",
    "rocm_matmul_family_compiled": "AMD GPU RDNA matmul-family ops (batched_gemm "
                            "/ linear_general / qkv_projection / "
                            "factorized_matmul / einsum) built on the "
                            "COMPILER-GENERATED WMMA GEMM kernel (the "
                            "rocm_compiled spine), reshaped/batched/split in the "
                            "runtime; f16/bf16 storage, f32 accumulate. "
                            "factorized_matmul's rank-r SVD truncation is an "
                            "exact host epilogue; einsum handles single-"
                            "contraction matmul specs",
    "rocm_exotic_attn_compiled": "AMD GPU RDNA exotic-attention compositions "
                            "(gated_attention, mla_decode, mla_decode_fused, "
                            "mla_decode_step absorbed-latent decode) "
                            "built by COMPOSING the COMPILER-GENERATED WMMA "
                            "flash_attn kernel + the WMMA GEMM kernel (MLA latent "
                            "projections) + an elementwise gate, plus the DK1 "
                            "generated absorbed-latent ROCm decode kernel; f16/bf16 storage, "
                            "f32 softmax+accumulate. The block-sparse deepseek "
                            "variant stays artifact_only",
    "rocm_deltanet_compiled": "AMD GPU RDNA gated/delta linear-attention the "
                            "Tessera compiler GENERATES as a causal "
                            "SEQUENTIAL-SCAN kernel (generate-rocm-deltanet-"
                            "kernel -> ROCDL -> hsaco) — the first RECURRENT "
                            "device_verified_jit ROCm kernel: one workgroup per (b,h), one "
                            "thread per value-column, LDS state. Handles "
                            "gated_deltanet / kimi_delta_attention / "
                            "modified_delta_attention (erase/modified/gate/beta/"
                            "decay flags); f16/bf16/f32 storage, f32 compute",
    "x86_deltanet_compiled": "x86 CPU gated/delta linear-attention — the "
                            "hand-written AVX-512 causal delta-rule sequential "
                            "scan (avx512_deltanet_f32, runtime-loaded): per (b,h) "
                            "a Dqk x Dv state scanned over S with erase/decay/beta/"
                            "modified/gate variants. Handles gated_deltanet / "
                            "kimi_delta_attention / modified_delta_attention. f32, "
                            "matches numpy _delta_attention_impl. The x86 analog "
                            "of rocm_deltanet_compiled",
    "rocm_rope_compiled":   "AMD GPU RDNA rotary-position-embedding the Tessera "
                            "compiler GENERATES (generate-rocm-rope-kernel -> "
                            "ROCDL -> hsaco, in-process via tessera-opt), then HIP "
                            "loads + launches it. Interleaved-pair RoPE over "
                            "[M, D] (one workgroup per row); f32/f16/bf16",
    "nvidia_mma":           "NVIDIA GPU (consumer Blackwell sm_120) warp-level "
                            "mma.sync GEMM via the shipped libtessera_nvidia_gemm.so "
                            "tessera_nvidia_mma_gemm_{f16,bf16,tf32} C ABI symbol "
                            "(NVRTC-device_verified_jit for the device arch; f16/bf16/"
                            "fp32(tf32-math) storage, f32 accumulate)",
    # Note: pure-numpy `reference_cpu` is reached only as an internal *fallback*
    # inside `launch()`'s native_cpu branch (when `_execute_native_cpu_artifact`
    # raises and `_execute_jit_cpu_artifact` succeeds). It's not a directly
    # dispatched executor — no matrix row points at it — so it's intentionally
    # not in this catalog (the drift test would flag dead entries otherwise).
}


# The execution matrix itself: (target, compiler_path) -> ExecutionRow. Adding a
# new backend executor means (1) adding the function in runtime.py, (2) adding it
# to KNOWN_EXECUTORS, (3) adding an ExecutionRow here. `launch()` picks it up
# automatically; the dashboard regenerates; the drift test enforces it.
_MATRIX: dict[tuple[str, str], ExecutionRow] = {
    # --- Apple Silicon CPU (Accelerate) ---
    ("apple_cpu", "apple_cpu_accelerate"): ExecutionRow(
        target="apple_cpu", compiler_path="apple_cpu_accelerate",
        execution_kind="native_cpu", executable=True,
        executor_id="apple_cpu_accelerate", runtime_status="success",
        reason="Apple CPU artifact runs through Accelerate cblas_sgemm + multi-op chain.",
        execution_mode="cpu_accelerate"),
    # --- Apple Silicon GPU (MPS / MSL / MPSGraph) ---
    ("apple_gpu", "apple_gpu_mps"): ExecutionRow(
        target="apple_gpu", compiler_path="apple_gpu_mps",
        execution_kind="native_gpu", executable=True,
        executor_id="apple_gpu_mps", runtime_status="success",
        reason="Apple GPU artifact runs through MPS / MSL / MPSGraph per the runtime envelope.",
        execution_mode="metal_runtime"),
    # --- Apple Value Target IR (sprint 2) — CPU value-call execution ---
    # The value-preserving `-full` lane lowers to tessera_apple.cpu.call value
    # ops; this row executes them by invoking the C ABI `symbol` named in the IR
    # (read from metadata["apple_value_calls"]). CPU cholesky is executable now.
    ("apple_cpu", "apple_value_target_ir"): ExecutionRow(
        target="apple_cpu", compiler_path="apple_value_target_ir",
        execution_kind="native_cpu", executable=True,
        executor_id="apple_value_target_ir", runtime_status="success",
        reason="Apple CPU value-call (tessera_apple.cpu.call) dispatches to the "
               "named Accelerate/LAPACK C ABI symbol.",
        execution_mode="cpu_accelerate"),
    # Apple GPU value-call execution for narrow, explicitly allowlisted lanes:
    # rank-3 batched matmul (Sprint 8), native sparse attention (Sprint 11),
    # PPO policy loss (Stages 13/14), and the first EBM value kernels. The
    # executor rejects cpu.call, package_call, multi-op programs, inactive
    # stubs, and off-allowlist symbols.
    ("apple_gpu", "apple_value_target_ir"): ExecutionRow(
        target="apple_gpu", compiler_path="apple_value_target_ir",
        execution_kind="native_gpu", executable=True,
        executor_id="apple_gpu_value_target_ir", runtime_status="success",
        reason="Apple GPU value-call (tessera_apple.gpu.kernel_call) dispatches "
               "named C ABI symbols for strict rank-3 batched matmul, native "
               "sparse attention, PPO policy-loss, and EBM value envelopes.",
        execution_mode="metal_runtime"),
    # Apple GPU structured-compute lane (2026-07-09) — parity with the x86/ROCm
    # structured-compute tails (conv family, vision/layout, recurrent, MoR, VLM,
    # RoPE, …). These reuse the tessera reference primitives (ops.* / F.* /
    # memory.*) with no Metal dispatch, so the lane runs entirely on the CPU
    # reference: an executable apple_gpu artifact path with a compare fixture,
    # but it does NOT run on the GPU — execution_kind=reference_cpu so
    # telemetry/audit never miscount it as Apple GPU execution.
    ("apple_gpu", "apple_gpu_structured_compute_compiled"): ExecutionRow(
        target="apple_gpu", compiler_path="apple_gpu_structured_compute_compiled",
        execution_kind="reference_cpu", executable=True,
        executor_id="apple_gpu_structured_compute_compiled",
        runtime_status="success",
        reason="Apple GPU structured-compute artifact covers the conv family "
               "(conv1d / conv_transpose / depthwise_conv1d) + the structured "
               "tail through runtime.launch(). It reuses the tessera reference "
               "primitives with no Metal dispatch, so it runs on the CPU "
               "reference path — execution_kind=reference_cpu; direct "
               "execute/compare evidence, not a bespoke fused Metal kernel — "
               "parity with the x86/ROCm structured-compute lanes."),
    # Apple GPU pointwise-regression loss lane (2026-07-09) — parity with the
    # x86/ROCm loss lanes. mse / mae / huber / smooth_l1 / log_cosh compose the
    # residual + none/mean/sum reduction on the MPSGraph binary + reduce lanes
    # (host structure for the piecewise/transcendental middle); matches
    # tessera.losses.
    ("apple_gpu", "apple_gpu_loss_compiled"): ExecutionRow(
        target="apple_gpu", compiler_path="apple_gpu_loss_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="apple_gpu_loss_compiled", runtime_status="success",
        reason="Apple GPU loss artifact runs mse / mae / huber / smooth_l1 / "
               "log_cosh: residual (pred-target) on the MPSGraph binary lane, "
               "mse/mae per-element on the GPU mul/abs opcodes, the piecewise/"
               "transcendental middle (huber/smooth_l1/log_cosh) host-side, and "
               "the none/mean/sum reduction on the MPSGraph reduce lane. f32, "
               "matches tessera.losses — parity with rocm_loss_compiled.",
        execution_mode="metal_runtime"),
    # Apple GPU loss-family lane (2026-07-09) — parity with the x86/ROCm binary/
    # class/rl/ebm loss lanes. binary-CE, class-axis (cross_entropy/kl/js/z_loss),
    # RL policy (ppo/cispo/grpo), and EBM-diffusion losses: per-sample loss via
    # the standalone reference (host structure), none/mean/sum reduction on the
    # MPSGraph reduce lane.
    ("apple_gpu", "apple_gpu_loss_family_compiled"): ExecutionRow(
        target="apple_gpu", compiler_path="apple_gpu_loss_family_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="apple_gpu_loss_family_compiled", runtime_status="success",
        reason="Apple GPU loss-family artifact runs binary-CE, class-axis "
               "(cross_entropy / kl / js / z_loss), RL policy (ppo / cispo / "
               "grpo), and EBM-diffusion losses: the per-sample loss composes "
               "through the standalone reference (host structure — gather/one-hot/"
               "clip/softplus, some f64) and the none/mean/sum reduction runs on "
               "the MPSGraph reduce lane. f32, matches tessera.losses / tessera.rl "
               "— parity with the x86/ROCm loss lanes.",
        execution_mode="metal_runtime"),
    # Apple GPU complex-arithmetic lane (2026-07-09) — parity with the x86/ROCm
    # complex lanes. 9 pointwise ops compose interleaved-f32 on the Apple GPU
    # unary/binary/atan2 lanes; the geometric/certificate ops reuse the
    # tessera.complex reference (host structure, as x86/ROCm do).
    ("apple_gpu", "apple_gpu_complex_compiled"): ExecutionRow(
        target="apple_gpu", compiler_path="apple_gpu_complex_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="apple_gpu_complex_compiled", runtime_status="success",
        reason="Apple GPU complex artifact runs the 9 pointwise complex ops "
               "(mul / div / conjugate / abs / arg / exp / log / sqrt / pow) as "
               "interleaved-f32 compositions on the Apple GPU unary / binary / "
               "atan2 lanes, and the geometric/certificate ops (cross_ratio / dz "
               "/ dbar / laplacian_2d / conformal_* / is_concyclic / "
               "check_cauchy_riemann / mobius_from_three_points) via the "
               "tessera.complex reference (host structure). f32, matches "
               "tessera.complex — parity with x86/rocm_complex_compiled.",
        execution_mode="metal_runtime"),
    # Apple GPU conformal-geometry lane (2026-07-09) — mobius / stereographic
    # composed on the interleaved-f32 Apple GPU complex_mul/complex_div/binary-div
    # lanes. Parity with x86/rocm_conformal_compiled.
    ("apple_gpu", "apple_gpu_conformal_compiled"): ExecutionRow(
        target="apple_gpu", compiler_path="apple_gpu_conformal_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="apple_gpu_conformal_compiled", runtime_status="success",
        reason="Apple GPU conformal artifact runs mobius f(z)=(az+b)/(cz+d) and "
               "stereographic projection composed on the interleaved-f32 Apple "
               "GPU complex_mul / complex_div / binary-div lanes (no new kernel). "
               "f32, matches tessera.complex — parity with "
               "x86/rocm_conformal_compiled.",
        execution_mode="metal_runtime"),
    # Apple GPU Philox RNG lane (2026-07-10) — parity with the x86/ROCm rng
    # lanes. Apple ships no device Philox kernel, so this lane runs entirely on
    # the CPU reference (tessera.rng_device Philox core + tessera.rng RNGKey
    # contract) — no Metal dispatch. It is an executable apple_gpu artifact path
    # with a compare fixture, but it does NOT run on the GPU: execution_kind is
    # reference_cpu so telemetry/audit never miscount it as Apple GPU execution.
    ("apple_gpu", "apple_gpu_rng_compiled"): ExecutionRow(
        target="apple_gpu", compiler_path="apple_gpu_rng_compiled",
        execution_kind="reference_cpu", executable=True,
        executor_id="apple_gpu_rng_compiled", runtime_status="success",
        reason="Apple GPU RNG artifact runs rng_uniform / rng_normal / dropout "
               "from the counter-based Philox-4x32-10 reference "
               "(tessera.rng_device; Apple ships no device Philox), and the "
               "distribution samplers (bernoulli/beta/categorical/dirichlet/"
               "gamma/poisson/randint/truncated_normal/permutation/multinomial, "
               "RNGKey key/split/fold_in/clone, MCMC samplers) via the public "
               "tessera.rng RNGKey contract. Runs on the CPU reference path (no "
               "Metal dispatch) — execution_kind=reference_cpu; matches "
               "tessera.rng / tessera.rng_device, parity with x86/rocm_rng_compiled."),
    # Apple GPU linalg + matmul-family lanes (2026-07-10) — parity with the
    # x86/ROCm linalg / matmul-family lanes. Apple has no MPS lu/qr/svd primitive,
    # so these lanes run entirely on the CPU numpy reference (no Metal dispatch):
    # executable apple_gpu artifact paths with compare fixtures, but they do NOT
    # run on the GPU, so execution_kind is reference_cpu (telemetry/audit must not
    # miscount them as Apple GPU execution).
    ("apple_gpu", "apple_gpu_linalg_compiled"): ExecutionRow(
        target="apple_gpu", compiler_path="apple_gpu_linalg_compiled",
        execution_kind="reference_cpu", executable=True,
        executor_id="apple_gpu_linalg_compiled", runtime_status="success",
        reason="Apple GPU linalg artifact runs cholesky / tri_solve / "
               "cholesky_solve / lu / qr / svd via the numpy reference (Apple "
               "ships no MPS lu/qr/svd primitive): qr/svd/cholesky on np.linalg, "
               "a standalone partial-pivot LU, triangular solves via the "
               "extracted triangle. Runs on the CPU reference path (no Metal "
               "dispatch) — execution_kind=reference_cpu; matches np.linalg, "
               "parity with x86/rocm_linalg_compiled."),
    ("apple_gpu", "apple_gpu_matmul_family_compiled"): ExecutionRow(
        target="apple_gpu", compiler_path="apple_gpu_matmul_family_compiled",
        execution_kind="reference_cpu", executable=True,
        executor_id="apple_gpu_matmul_family_compiled", runtime_status="success",
        reason="Apple GPU matmul-family artifact runs einsum (single-contraction "
               "spec) and factorized_matmul (GEMM + rank-r SVD truncation) via "
               "the numpy reference the x86/ROCm GEMM lanes match. Runs on the "
               "CPU reference path (no Metal dispatch) — "
               "execution_kind=reference_cpu; matches numpy, parity with "
               "x86/rocm_matmul_family_compiled."),
    # Apple GPU optimizer lane (2026-07-10) — sgd/momentum/adam/adamw/lion. Apple
    # ships no device optimizer kernel, so the elementwise update rules run on
    # the numpy reference (no Metal dispatch) — execution_kind=reference_cpu.
    ("apple_gpu", "apple_gpu_optimizer_compiled"): ExecutionRow(
        target="apple_gpu", compiler_path="apple_gpu_optimizer_compiled",
        execution_kind="reference_cpu", executable=True,
        executor_id="apple_gpu_optimizer_compiled", runtime_status="success",
        reason="Apple GPU optimizer artifact runs sgd / momentum / adam / adamw "
               "/ lion per-parameter updates (state m/v in/out) via the numpy "
               "reference the x86/ROCm device kernels are matched against. Runs "
               "on the CPU reference path (no Metal dispatch) — "
               "execution_kind=reference_cpu; matches tessera.optim, parity with "
               "x86/rocm_optimizer_compiled."),
    # Apple GPU 0-move + sort lane (2026-07-10) — pad/roll/flip/tile/repeat/stack
    # (host index-map + numpy gather) + sort/argsort (numpy stable sort). Apple
    # ships no device gather/sort kernel, so this runs on the CPU reference path
    # (no Metal dispatch) — execution_kind=reference_cpu.
    ("apple_gpu", "apple_gpu_shape_compiled"): ExecutionRow(
        target="apple_gpu", compiler_path="apple_gpu_shape_compiled",
        execution_kind="reference_cpu", executable=True,
        executor_id="apple_gpu_shape_compiled", runtime_status="success",
        reason="Apple GPU 0-move/sort artifact runs pad / roll / flip / tile / "
               "repeat / stack (host index-map + numpy gather) and sort / "
               "argsort (numpy stable sort). Apple ships no device gather/sort "
               "kernel, so it runs on the CPU reference path (no Metal dispatch) "
               "— execution_kind=reference_cpu; matches tessera.ops / numpy, "
               "parity with the x86/ROCm strided + sort lanes."),
    # Apple GPU reduce lane (2026-07-10) — sum genuinely on the MPSGraph reduce
    # lane (numpy fallback when Metal is unavailable) -> native_gpu.
    ("apple_gpu", "apple_gpu_reduce_compiled"): ExecutionRow(
        target="apple_gpu", compiler_path="apple_gpu_reduce_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="apple_gpu_reduce_compiled", runtime_status="success",
        reason="Apple GPU reduce artifact runs sum over an axis (default: all) "
               "with keepdims on the MPSGraph reduce lane (numpy fallback when "
               "Metal is unavailable). f32, matches numpy.sum — parity with "
               "x86/rocm_reduce_compiled.",
        execution_mode="metal_runtime"),
    # Apple GPU scatter + sparse/MoE lanes (2026-07-10) — Apple ships no device
    # scatter/spmm/sddmm/moe kernel, so these run the numpy reference (no Metal
    # dispatch) — execution_kind=reference_cpu.
    ("apple_gpu", "apple_gpu_scatter_compiled"): ExecutionRow(
        target="apple_gpu", compiler_path="apple_gpu_scatter_compiled",
        execution_kind="reference_cpu", executable=True,
        executor_id="apple_gpu_scatter_compiled", runtime_status="success",
        reason="Apple GPU scatter artifact runs scatter / scatter_add / "
               "scatter_reduce (row-wise indexed store, set/add/min/max) via the "
               "numpy reference (np.add.at / np.minimum.at / np.maximum.at). Runs "
               "on the CPU reference path (no Metal dispatch) — "
               "execution_kind=reference_cpu; matches numpy, parity with "
               "x86/rocm_scatter_compiled."),
    ("apple_gpu", "apple_gpu_sparse_compiled"): ExecutionRow(
        target="apple_gpu", compiler_path="apple_gpu_sparse_compiled",
        execution_kind="reference_cpu", executable=True,
        executor_id="apple_gpu_sparse_compiled", runtime_status="success",
        reason="Apple GPU sparse/MoE artifact runs spmm_csr / spmm_coo / sddmm / "
               "bsmm (numpy CSR SpMM / (a@b)*mask / a@b) and moe (routed "
               "per-token expert GEMVs, top-1) via the numpy reference. Runs on "
               "the CPU reference path (no Metal dispatch) — "
               "execution_kind=reference_cpu; matches numpy / tessera, parity "
               "with the x86/ROCm sparse + moe lanes."),
    # Apple GPU reference tail lane (2026-07-10) — the heterogeneous remainder
    # (MLA latent-KV, alibi, lgamma/digamma, fused_epilogue, asymmetric_bce,
    # normalize_group_advantages, speculative-decode accept). Apple ships no
    # device kernel, so each reuses its public tessera reference (no Metal
    # dispatch) — execution_kind=reference_cpu.
    ("apple_gpu", "apple_gpu_tail_compiled"): ExecutionRow(
        target="apple_gpu", compiler_path="apple_gpu_tail_compiled",
        execution_kind="reference_cpu", executable=True,
        executor_id="apple_gpu_tail_compiled", runtime_status="success",
        reason="Apple GPU reference tail artifact runs the heterogeneous "
               "remainder — MLA latent-KV (compress / expand_k / expand_v), "
               "alibi, lgamma / digamma, fused_epilogue, asymmetric_bce, "
               "normalize_group_advantages, and speculative-decode accept — each "
               "via its public tessera reference (tessera.ops / losses / rl). "
               "Runs on the CPU reference path (no Metal dispatch) — "
               "execution_kind=reference_cpu; matches the reference, parity with "
               "the x86/ROCm lanes."),
    # --- x86 / native CPU (AMX path) ---
    ("cpu", "native_cpu"): ExecutionRow(
        target="cpu", compiler_path="native_cpu",
        execution_kind="native_cpu", executable=True,
        executor_id="native_cpu", runtime_status="success",
        reason="CPU artifact runs through the x86 AMX / native CPU runtime."),
    # --- x86 AVX-512 elementwise device_verified_jit lane (runtime-loaded C-ABI kernels) ---
    ("x86", "x86_reduce_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_reduce_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_reduce_compiled", runtime_status="success",
        reason="x86 reduce artifact runs the hand-written AVX-512 row-reduction "
               "kernel (sum/mean/max/min over the last axis, 16 f32 lanes/__m512, "
               "NaN-propagating max/min): the Python runtime ctypes-loads "
               "libtessera_x86_elementwise.so and calls "
               "tessera_x86_avx512_reduce_f32. An arbitrary reduced axis folds to "
               "[outer,inner]; handles tessera.sum/mean/max/min (amax/amin) by op "
               "name. f32 only.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_scan_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_scan_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_scan_compiled", runtime_status="success",
        reason="x86 scan artifact runs the inclusive prefix scan (cumsum/cumprod/"
               "cummax/cummin) from libtessera_x86_elementwise.so "
               "(tessera_x86_avx512_scan_f32) along one axis; same-shape output. "
               "The CPU analog of the ROCm block-scan lane (scalar row "
               "recurrence, rows parallel; SIMD prefix is a perf follow-up). "
               "f32 only.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_argreduce_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_argreduce_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_argreduce_compiled", runtime_status="success",
        reason="x86 argreduce artifact runs argmax/argmin from "
               "libtessera_x86_elementwise.so (tessera_x86_avx512_argreduce_f32) "
               "along one axis, numpy first-occurrence tie-break (strict compare). "
               "f32 input, i32 index output.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_where_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_where_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_where_compiled", runtime_status="success",
        reason="x86 where artifact runs the ternary select where(cond,a,b) from "
               "libtessera_x86_elementwise.so (tessera_x86_avx512_where_f32, "
               "_mm512_cmpneq_epi8_mask + _mm512_mask_blend_ps); cond i8 "
               "normalized != 0, a/b/out f32.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_nvfp4_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_nvfp4_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_nvfp4_compiled", runtime_status="success",
        reason="x86 nvfp4 artifact runs block-scaled fp4: per-block amax / "
               "reassembly on the host, the per-block fp8-E4M3 scale and the "
               "E2M1 codes on the AVX-512 fpquant kernel; dequantize is the fp32 "
               "passthrough. Matches the compiler/microscaling reference exactly "
               "(0 abs err). f32 fake-quant.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_fpquant_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_fpquant_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_fpquant_compiled", runtime_status="success",
        reason="x86 fpquant artifact runs quantize/dequantize fp8 / fp6 / fp4: "
               "per-tensor symmetric scale (amax/max_normal) + grid-snap on the "
               "AVX-512 fpquant kernel (getexp/roundscale/scalef mantissa-snap, "
               "RNE), rescale; dequantize is the fp32 passthrough. Matches the "
               "tessera.ops reference exactly (0 abs err).",
        execution_mode="cpu_avx512"),
    ("x86", "x86_intquant_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_intquant_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_intquant_compiled", runtime_status="success",
        reason="x86 intquant artifact runs quantize/dequantize int8/int4 and "
               "fake_quantize: qparam selection and int8 container conversion on "
               "host, round/max/min/mul on AVX-512 elementwise kernels. int4 is "
               "signed int4 values in int8 containers. Matches "
               "tessera.quantization reference exactly.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_pooling_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_pooling_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_pooling_compiled", runtime_status="success",
        reason="x86 pooling artifact runs max/avg/min/adaptive_pool by forming "
               "the pooling window matrix on host and reducing each row on "
               "tessera_x86_avx512_reduce_f32. f32, matches nn.functional.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_image_affine_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_image_affine_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_image_affine_compiled", runtime_status="success",
        reason="x86 image affine artifact runs image_normalize as "
               "(x-mean)/std: layout and per-channel broadcast on host, sub/div "
               "on tessera_x86_avx512_binary_f32. f32, matches tessera.ops.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_metric_loss_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_metric_loss_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_metric_loss_compiled", runtime_status="success",
        reason="x86 metric-loss artifact runs wasserstein / cosine_embedding / "
               "contrastive / triplet / InfoNCE / NT-Xent / seq2seq losses with "
               "AVX-512 reductions and exp/log; label, mask, sort, and compact "
               "matrix structure remain on host. f32, matches tessera.losses.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_structured_compute_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_structured_compute_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_structured_compute_compiled", runtime_status="success",
        reason="x86 structured-compute artifact covers CTC loss, VLM image/"
               "layout transforms, conv1d/conv_transpose/LoRA, GRU/simple-RNN, "
               "and depthwise_conv1d through runtime.launch(). Shape/control "
               "bookkeeping remains host-structured; the row is direct "
               "single-GPU executable evidence, not a bespoke fused kernel.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_class_loss_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_class_loss_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_class_loss_compiled", runtime_status="success",
        reason="x86 class-loss artifact runs cross_entropy / kl / js / focal / "
               "label_smoothed_cross_entropy / z_loss: exp/log on the AVX-512 "
               "transcendental kernel, class-axis max/sum/gather/one-hot on the "
               "host, leading-axis reduction on the reduce kernel. f32, "
               "numpy 2e-4.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_ebm_loss_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_ebm_loss_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_ebm_loss_compiled", runtime_status="success",
        reason="x86 EBM/diffusion loss artifact runs score_matching / "
               "denoising / implicit / contrastive_divergence / persistent_cd / "
               "ddpm_noise_pred / vlb / load_balance: the diff/square and "
               "reductions run on the AVX-512 binary + reduce kernels, the "
               "cheap structure (argmax/one-hot/scalar scale) on the host. f32, "
               "matches the numpy loss reference.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_ebm_compute_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_ebm_compute_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_ebm_compute_compiled", runtime_status="success",
        reason="x86 EBM compute artifact runs energy_quadratic / inner_step / "
               "refinement / self_verify: the diff/square and reductions run on "
               "the AVX-512 binary + reduce kernels, the cheap structure (scalar "
               "scale, argmin gather) on the host. f32, matches the numpy "
               "tessera.ebm reference.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_ebm_langevin_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_ebm_langevin_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_ebm_langevin_compiled", runtime_status="success",
        reason="x86 EBM Langevin sampling artifact runs y − η·grad + "
               "noise_scale·z where z is Box-Muller Gaussian noise drawn "
               "ON-DEVICE from counter-based Philox-4x32-10 (the AVX-512 "
               "langevin kernel; counter (c0+i,…); the noise never round-trips "
               "the host). f32, matches tessera.ebm.langevin_step_philox.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_rl_loss_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_rl_loss_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_rl_loss_compiled", runtime_status="success",
        reason="x86 rl-loss artifact runs the ppo / cispo / grpo core surrogate "
               "(ratio=exp(logp_new−logp_old), clipped) on the AVX-512 policy-"
               "loss kernel + normalize_group_advantages on the AVX-512 "
               "layer_norm kernel over the group axis; reduction on the reduce "
               "kernel. KL/entropy/mask add-ons get a stable diagnostic. f32, "
               "numpy 2e-5.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_binary_loss_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_binary_loss_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_binary_loss_compiled", runtime_status="success",
        reason="x86 binary-loss artifact runs bce / asymmetric_bce over "
               "(logits, targets): per-element loss on the AVX-512 binary-loss "
               "kernel (tessera_x86_avx512_binary_loss_f32, stable softplus "
               "form) + none/mean/sum reduction on the reduce kernel. f32, "
               "numpy 2e-5.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_fft_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_fft_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_fft_compiled", runtime_status="success",
        reason="x86 FFT artifact runs fft / ifft / rfft / irfft over any axis "
               "length on the AVX-512 radix-2 C2C kernel (power-of-two; SIMD "
               "complex butterflies) + r2c/c2r pack-unpack; tiny non-pow2 via "
               "the DFT-matrix on the AVX-512 GEMM, other non-pow2 via Bluestein "
               "(chirp-z) over the radix-2 kernel. The SpectralPlan owns strategy "
               "+ normalization. complex64/f32.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_spectral_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_spectral_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_spectral_compiled", runtime_status="success",
        reason="x86 spectral composites (dct / stft / istft / spectral_conv / "
               "spectral_filter) compose the x86_fft_compiled FFT lane — "
               "framing / windowing / overlap-add / pointwise complex-mul on "
               "host, the transform on the AVX-512 radix-2 kernel. f32, "
               "matches np.fft.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_sparse_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_sparse_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_sparse_compiled", runtime_status="success",
        reason="x86 sparse lane (spmm_csr / spmm_coo / sddmm / bsmm) runs the "
               "AVX-512 sparse kernels — spmm = row-wise AXPY iterating the CSR "
               "nonzeros (COO folds to CSR on host), sddmm = sampled dense-dense "
               "dot over masked entries, bsmm via the GEMM microkernel. f32, "
               "matches numpy.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_moe_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_moe_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_moe_compiled", runtime_status="success",
        reason="x86 moe-compute lane runs the routed per-token expert GEMVs "
               "(top-1) on the AVX-512 kernel — routing (argmax/round-robin) "
               "resolved on host, the expert matmuls vectorized over out_dim. "
               "f32, matches numpy.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_optimizer_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_optimizer_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_optimizer_compiled", runtime_status="success",
        reason="x86 optimizer lane runs sgd / momentum / adam / adamw / lion as a "
               "fused per-parameter update on the AVX-512 kernel (kind-selected; "
               "m/v state updated in place; the 1-β^t bias correction computed on "
               "host from the step). f32, matches the optim.py reference.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_normcompose_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_normcompose_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_normcompose_compiled", runtime_status="success",
        reason="x86 group/instance/weight-norm lane composed on the AVX-512 "
               "layer_norm (row mean/var) + reduce (sum-of-squares) kernels; "
               "host does the reshape / per-axis divide. f32, matches "
               "nn.functional.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_grad_clip_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_grad_clip_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_grad_clip_compiled", runtime_status="success",
        reason="x86 grad_clip_norm lane — global gradient-norm clipping "
               "g*min(1, max_norm/||g||): the L2 norm's global sum-of-squares "
               "runs on the AVX-512 reduce kernel, host does sqrt + the clip "
               "scale + the elementwise scale; norm_type=inf uses max|g|. f32, "
               "matches optim.clip_grad_norm within f32 tolerance.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_lamb_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_lamb_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_lamb_compiled", runtime_status="success",
        reason="x86 LAMB lane runs the AVX-512 adam kernel (lr=1/wd=0) then "
               "applies the per-tensor trust ratio ‖p‖/‖update‖ on host (the "
               "reduction the elementwise lane can't do). f32, matches "
               "optim.lamb.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_muon_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_muon_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_muon_compiled", runtime_status="success",
        reason="x86 Muon lane orthogonalizes the momentum matrix via the "
               "AVX-512 one-sided-Jacobi SVD kernel (U·Vh polar factor); the "
               "small U@Vh + momentum/sgd run on host. <2-D params normalize. "
               "f32, matches optim.muon.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_selective_ssm_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_selective_ssm_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_selective_ssm_compiled", runtime_status="success",
        reason="x86 state-space lane runs selective_ssm (Mamba2) on the AVX-512 "
               "fused selective-scan kernel — a single pass over S vectorized "
               "over the state dim N, exp via the Cephes core, (B,D,N) state "
               "updated in place. f32, matches the numpy reference.",
        execution_mode="cpu_avx512"),
    # Mamba2 selective_ssm BACKWARD on x86 (tessera_x86_selective_ssm_bwd_f32):
    # operands (dout, x, A, B, C, delta[, gate[, state]]) -> (dx, dA, dB, dC,
    # ddelta). The second native backward TARGET after ROCm (AUTODIFF §9a) — same
    # paired contract, AVX-512 kernel instead of the gfx1151 one.
    ("x86", "x86_selective_ssm_bwd_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_selective_ssm_bwd_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_selective_ssm_bwd_compiled", runtime_status="success",
        reason="x86 selective_ssm (Mamba2) backward runs the AVX-512 fused "
               "backward kernel (tessera_x86_selective_ssm_bwd_f32: forward-fill "
               "h_traj then reverse scan accumulating dx/dA/dB/dC/ddelta) behind "
               "the runtime.launch() ABI, from operands (dout, x, A, B, C, delta"
               "[, gate[, state]]). f32, matches autodiff.vjp.vjp_selective_ssm; "
               "the reverse-mode analog of x86_selective_ssm_compiled.",
        execution_mode="cpu_avx512",
        direction="backward", op_family="selective_ssm",
        device_proof="device_verified_abi", evidence_target="x86_avx512",
        numerical_fixture="tests/unit/test_x86_ssm_bwd_launch_execute.py"),
    ("x86", "x86_linalg_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_linalg_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_linalg_compiled", runtime_status="success",
        reason="x86 linalg lane (cholesky / tri_solve / cholesky_solve / lu / qr "
               "/ svd) runs the AVX-512 kernels — Cholesky–Banachiewicz "
               "factorization, forward/back triangular substitution (RHS columns "
               "vectorized), getrf partial-pivot LU, Householder QR, one-sided "
               "Jacobi SVD; cholesky_solve composes two triangular solves. "
               "Batched. f32, matches numpy.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_stat_reduce_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_stat_reduce_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_stat_reduce_compiled", runtime_status="success",
        reason="x86 stat-reduce artifact runs var / std / count_nonzero over an "
               "axis, composed from the AVX-512 reduce kernel "
               "(var=mean(x^2)-mean(x)^2, std=sqrt(var), count_nonzero=sum(x!=0))"
               ". f32.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_stable_reduce_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_stable_reduce_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_stable_reduce_compiled", runtime_status="success",
        reason="x86 stable-reduce artifact runs logsumexp / log_softmax / "
               "softmax_safe / sigmoid_safe — max-shifted, composed from the "
               "AVX-512 reduce (max/sum) + the transcendental exp/log lane; "
               "softmax_safe / sigmoid_safe alias the stable softmax / sigmoid "
               "lanes. f32.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_loss_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_loss_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_loss_compiled", runtime_status="success",
        reason="x86 loss artifact runs the pointwise regression losses (mse / "
               "mae / huber / smooth_l1 / log_cosh) over (pred, target): the "
               "per-element loss on the AVX-512 loss kernel "
               "(tessera_x86_avx512_pointwise_loss_f32) + none/mean/sum "
               "reduction on the AVX-512 reduce kernel. f32, numpy 2e-5.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_attention_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_attention_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_attention_compiled", runtime_status="success",
        reason="x86 attention artifact runs multi_head / gqa / mqa / mla_decode "
               "as O = softmax(Q·Kᵀ·scale [+causal])·V, composed from the AVX-512 "
               "f32 GEMM (QKᵀ and probs·V) + the AVX-512 row-softmax kernel; "
               "reshape/scale/mask/KV-group expansion in Python. The CPU analog "
               "of the ROCm WMMA flash-attention family. f32.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_rope_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_rope_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_rope_compiled", runtime_status="success",
        reason="x86 rope artifact runs the interleaved-pair rotary position "
               "embedding from libtessera_x86_elementwise.so "
               "(tessera_x86_avx512_rope_f32; AVX-512 deinterleave + Cephes "
               "sincos + re-interleave). Operands x, theta both [.., D] (D "
               "even). The CPU analog of the ROCm rope lane. f32, numpy 2e-5.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_alibi_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_alibi_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_alibi_compiled", runtime_status="success",
        reason="x86 alibi artifact runs the ALiBi positional-bias generator "
               "bias[h,i,j]=slope[h]*(j-i) over [H,S,S] from "
               "libtessera_x86_elementwise.so (tessera_x86_avx512_alibi_f32; "
               "default slope ramp 2^(-8k/H), optional slopes operand). The CPU "
               "analog of the ROCm alibi lane. f32.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_matmul_family_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_matmul_family_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_matmul_family_compiled", runtime_status="success",
        reason="x86 matmul-family artifact runs matmul / gemm / batched_gemm / "
               "linear_general / qkv_projection / factorized_matmul / einsum "
               "on the AVX-512 f32 "
               "GEMM microkernel (tessera_x86_avx512_gemm_f32), with the "
               "reshape/batch/single-contraction-einsum logic in Python. The CPU "
               "analog of the ROCm WMMA matmul-family lane. f32, K-scaled tol.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_kv_cache_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_kv_cache_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_kv_cache_compiled", runtime_status="success",
        reason="x86 KV-cache artifact invokes the native f32 append/read/prune "
               "C ABI in libtessera_x86_elementwise.so. The read proof copies "
               "the exact [start,end) cache rows and compares them with the "
               "KVCacheHandle reference.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_norm_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_norm_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_norm_compiled", runtime_status="success",
        reason="x86 norm artifact runs the unweighted rmsnorm / layer_norm "
               "row-reduction over the last axis from "
               "libtessera_x86_elementwise.so (tessera_x86_avx512_rmsnorm_f32 / "
               "_layernorm_f32; AVX-512 horizontal reduce, eps from kwargs). The "
               "CPU analog of the ROCm warp-shuffle norm lane. f32, numpy 2e-5.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_softmax_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_softmax_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_softmax_compiled", runtime_status="success",
        reason="x86 softmax artifact runs the numerically-stable softmax "
               "row-reduction over the last axis from "
               "libtessera_x86_elementwise.so (tessera_x86_avx512_softmax_f32; "
               "AVX-512 max + exp via the Cephes core + sum). The CPU analog of "
               "the ROCm softmax lane. f32, numpy 2e-5.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_binary_math_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_binary_math_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_binary_math_compiled", runtime_status="success",
        reason="x86 binary-math artifact runs pow(a,b) (positive base, a^b via "
               "exp(b*log(a))) and silu_mul(a,b)=silu(a)*b from "
               "libtessera_x86_elementwise.so (tessera_x86_avx512_pow_f32 / "
               "_silu_mul_f32), sharing the AVX-512 exp/log/sigmoid cores. f32, "
               "matches numpy 2e-5.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_transcendental_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_transcendental_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_transcendental_compiled", runtime_status="success",
        reason="x86 transcendental artifact runs the vectorized AVX-512 "
               "transcendental / activation kernel (exp/log/tanh/sigmoid/silu/"
               "gelu/erf/softplus/expm1/log1p) from "
               "libtessera_x86_elementwise.so (tessera_x86_avx512_transcendental"
               "_f32): Cephes exp/log minimax cores (~1 ulp) + Abramowitz-Stegun "
               "erf, activations compose. The CPU analog reaching ROCm math->"
               "ROCDL parity; gelu = tanh approximation. f32, matches numpy "
               "2e-5.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_unary_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_unary_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_unary_compiled", runtime_status="success",
        reason="x86 unary artifact runs the hand-written AVX-512 elementwise "
               "kernel (direct-intrinsic subset sqrt/rsqrt/reciprocal/abs/sign/"
               "floor/ceil/trunc/round, 16 f32 lanes/__m512): the Python runtime "
               "ctypes-loads libtessera_x86_elementwise.so and calls "
               "tessera_x86_avx512_unary_f32. Transcendentals stay numpy-"
               "reference. f32 only.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_binary_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_binary_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_binary_compiled", runtime_status="success",
        reason="x86 binary artifact runs the hand-written AVX-512 2-operand "
               "elementwise kernel (sub/div/maximum/minimum, NaN-propagating "
               "max/min, 16 f32 lanes/__m512): the Python runtime ctypes-loads "
               "libtessera_x86_elementwise.so and calls "
               "tessera_x86_avx512_binary_f32. `pow` stays numpy-reference. "
               "f32 only.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_clamp_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_clamp_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_clamp_compiled", runtime_status="success",
        reason="x86 clamp lane runs clamp / clip as min(max(x, lo), hi) composed "
               "on the AVX-512 binary max/min kernel (either bound optional; the "
               "scalar bounds are broadcast on host). f32, matches np.clip.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_complex_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_complex_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_complex_compiled", runtime_status="success",
        reason="x86 complex-arithmetic lane (9 pointwise ops) over "
               "interleaved-f32 [...,2] composed on the AVX-512 transcendental / "
               "unary / binary / atan2 kernels; host packs the interleave. f32, "
               "matches tessera.complex.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_softcap_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_softcap_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_softcap_compiled", runtime_status="success",
        reason="x86 softcap lane runs cap*tanh(x/cap) composed on the AVX-512 "
               "transcendental tanh kernel (scalar cap broadcast on host). "
               "f32, matches cap*tanh(x/cap).",
        execution_mode="cpu_avx512"),
    ("x86", "x86_rng_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_rng_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_rng_compiled", runtime_status="success",
        reason="x86 RNG lane runs counter-based Philox-4x32-10 on the AVX-512 "
               "kernel for the uniform bits; host applies the distribution "
               "transform (uniform scale / Box-Muller normal / dropout mask). "
               "f32, bit-exact vs tessera.rng_device.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_strided_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_strided_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_strided_compiled", runtime_status="success",
        reason="x86 0-move lane realizes pad/cat/roll/flip/tile/repeat/stack via "
               "the AVX-512 masked-gather kernel (host computes the index map "
               "from numpy shape arithmetic; the device moves the f32 data). "
               "f32, matches numpy.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_scatter_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_scatter_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_scatter_compiled", runtime_status="success",
        reason="x86 scatter lane realizes scatter/scatter_add/scatter_reduce "
               "(0-reduce indexed store) via the AVX-512 row-scatter kernel "
               "(host moves the scatter axis to 0 + flattens rows; the device "
               "reduces duplicate targets set/add/min/max). f32, matches the "
               "numpy scatter reference.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_composite_helper_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_composite_helper_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_composite_helper_compiled", runtime_status="success",
        reason="x86 composite-helper lane keeps memory_index_score, "
               "msa_index_scores, varlen_sdpa, and score_combine on "
               "compiler-visible Target IR while composing existing "
               "matmul/attention/binary runtime semantics plus host metadata "
               "logic. f32, matches the public ops references.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_conformal_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_conformal_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_conformal_compiled", runtime_status="success",
        reason="x86 conformal lane runs mobius (az+b)/(cz+d) on the AVX-512 "
               "complex mul/div lane and stereographic (x+iy)/(1-z) on the "
               "binary div lane (host orchestration). f32, matches "
               "tessera.complex.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_sort_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_sort_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_sort_compiled", runtime_status="success",
        reason="x86 sort lane runs sort/argsort/top_k via the AVX-512 bitonic "
               "sort network kernel (data-independent key+index "
               "compare-exchange; host pads each row to a power of two and "
               "flips for descending). f32, matches numpy.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_clifford_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_clifford_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_clifford_compiled", runtime_status="success",
        reason="x86 GA lane runs the Cl(3,0) table-driven bilinear products "
               "(geometric_product/wedge/left_contraction, + inner/"
               "rotor_sandwich by composition) on the AVX-512 kernel "
               "(blade-major [8,n]; compile-time Cayley table). f32, matches "
               "the numpy GA reference.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_flash_attn_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_flash_attn_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_flash_attn_compiled", runtime_status="success",
        reason="x86 attention lane runs the flash_attn forward as an FA-style "
               "online softmax on the AVX-512 kernel (running max/denominator + "
               "rescaled accumulator; the S×S scores never materialize). MHA, "
               "scale + causal, f32; the AVX-512 partner to the ROCm WMMA "
               "flash_attn. Matches the dense attention reference.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_mla_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_mla_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_mla_compiled", runtime_status="success",
        reason="x86 MLA latent-KV lane composes the DeepSeek building blocks on "
               "the AVX-512 GEMM (latent_kv_compress/expand_k/expand_v = batched "
               "matmul) + the flash_attn lane (mla_decode_fused chains "
               "compress→expand→flash_attn). f32, matches the numpy MLA "
               "reference.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_conv_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_conv_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_conv_compiled", runtime_status="success",
        reason="x86 convolution lane runs conv2d/conv3d as im2col + a GEMM: the "
               "host lays out the NHWC/NDHWC patch matrix (shape arithmetic "
               "only), the FLOP-heavy GEMM runs on the AVX-512 f32 kernel, and "
               "bias / groups / activation finish on the host. f32, matches the "
               "conv reference.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_nsa_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_nsa_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_nsa_compiled", runtime_status="success",
        reason="x86 NSA (deepseek_sparse_attention) blends three branches — "
               "sliding-window (the AVX-512 windowed flash_attn), "
               "compressed-block (dense flash_attn over per-block mean "
               "summaries), and top-k-block (host top-k block select + gather + "
               "dense flash_attn) — through the learned gate. The attention "
               "FLOPs run on the device kernels; compression / selection / "
               "blend on the host. f32, matches the dense-masked reference.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_msa_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_msa_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_msa_compiled", runtime_status="success",
        reason="x86 MSA (msa_sparse_attention) — the exp-free index scoring + "
               "per-GQA-group top-k block selection run on the host (bit-"
               "identical to the reference ops); the exact attend over the "
               "selected blocks runs on the AVX-512 flash_attn kernel as dense "
               "attention with the non-selected / causal-invalid keys folded "
               "into an additive -inf bias. The attention FLOPs run on the "
               "device kernel; selection on host. f32, matches the reference; "
               "dense-equivalence (top_k==num_blocks) reduces to dense GQA.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_linear_attn_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_linear_attn_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_linear_attn_compiled", runtime_status="success",
        reason="x86 linear-attention backbone (linear_attn / power_attn / "
               "retention) runs the quadratic-parallel form O = (φ(Q)·φ(K)ᵀ ⊙ "
               "causal ⊙ decay) @ V on two AVX-512 batched GEMMs; the feature "
               "map (elu/relu/identity/polynomial_2 or x^deg), the causal mask, "
               "and the decay matrix (c[t]/c[r] from cumprod) on the host. The "
               "AVX-512 partner to the ROCm linear_attn lane. f32, matches the "
               "numpy recurrence reference.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_atan2_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_atan2_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_atan2_compiled", runtime_status="success",
        reason="x86 atan2 lane runs quadrant-aware atan2(y, x) composed on the "
               "AVX-512 transcendental atan kernel (sign/quadrant logic on "
               "host). f32, matches np.arctan2.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_predicate_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_predicate_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_predicate_compiled", runtime_status="success",
        reason="x86 predicate artifact runs the hand-written AVX-512 unary "
               "predicate kernel (isnan = x!=x, isinf = |x|==inf, isfinite = "
               "ordered & |x|<inf; mask -> 0/1 bytes): the Python runtime "
               "ctypes-loads libtessera_x86_elementwise.so and calls "
               "tessera_x86_avx512_predicate_f32. f32 in, bool out.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_compare_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_compare_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_compare_compiled", runtime_status="success",
        reason="x86 compare artifact runs the hand-written AVX-512 2-operand "
               "comparison kernel (eq/ne/lt/le/gt/ge via _mm512_cmp_ps_mask + "
               "_mm_maskz_set1_epi8, NaN semantics matching numpy): the Python "
               "runtime ctypes-loads libtessera_x86_elementwise.so and calls "
               "tessera_x86_avx512_compare_f32. f32 in, bool out.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_logical_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_logical_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_logical_compiled", runtime_status="success",
        reason="x86 logical artifact runs the hand-written AVX-512 elementwise "
               "logical kernel (and/or/xor binary, not unary, over i8 booleans, "
               "inputs normalized via != 0): the Python runtime ctypes-loads "
               "libtessera_x86_elementwise.so and calls "
               "tessera_x86_avx512_logical_i8. Dispatched by op name. bool in/out.",
        execution_mode="cpu_avx512"),
    ("x86", "x86_bitwise_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_bitwise_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_bitwise_compiled", runtime_status="success",
        reason="x86 bitwise artifact runs the hand-written AVX-512 elementwise "
               "bitwise kernel (and/or/xor binary, not unary, over i32 integers, "
               "full bit pattern): the Python runtime ctypes-loads "
               "libtessera_x86_elementwise.so and calls "
               "tessera_x86_avx512_bitwise_i32. Dispatched by op name. i32 in/out.",
        execution_mode="cpu_avx512"),
    # --- CPU JIT (numpy reference for non-AMX ops) ---
    ("cpu", "jit_cpu_numpy"): ExecutionRow(
        target="cpu", compiler_path="jit_cpu_numpy",
        execution_kind="reference_cpu", executable=True,
        executor_id="jit_cpu_numpy", runtime_status="success",
        reason="CPU JIT artifact runs through the numpy reference path."),
    # --- AMD ROCm GPU (RDNA WMMA matrix-core GEMM) ---
    # Strix Halo bring-up (2026-06-22): the shipped libtessera_rocm_gemm.so runs
    # a real WMMA GEMM on the AMD GPU. The artifact is only stamped
    # executable=True by the jit path on a host that passes the runtime probe
    # (lib loads + a live HIP device); elsewhere launch() reports unimplemented.
    # This row is host-independent — the dashboard renders it everywhere; only
    # `metadata["executable"] is True` (a ROCm box) actually dispatches here.
    ("rocm", "rocm_wmma"): ExecutionRow(
        target="rocm", compiler_path="rocm_wmma",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_wmma", runtime_status="success",
        reason="ROCm matmul via the hand-written RDNA WMMA GEMM "
               "(tessera_rocm_wmma_gemm_{f16,bf16} C ABI symbol, HIPRTC-device_verified_jit "
               "for the device arch). Now the reference ORACLE + availability "
               "fallback for the device_verified_jit lane (rocm_compiled) — still directly "
               "selectable by stamping compiler_path=\"rocm_wmma\".",
        execution_mode="hip_runtime"),
    # --- AMD ROCm GPU (COMPILED lane — Stage L, the DEFAULT rocm matmul lane) ---
    # The kernel the Tessera compiler GENERATES: the in-process Stage L pipeline
    # (generate-wmma-gemm-kernel -> ROCDL -> gpu-module-to-binary, all in
    # tessera-opt, no mlir-opt shell-out) emits an hsaco that runs the RDNA WMMA
    # GEMM. This is now the DEFAULT for `@jit(target="rocm")` matmul on a capable
    # host (jit.py stamps compiler_path="rocm_compiled"); it reaches
    # parity-or-better vs the hand-written kernel across aligned/ragged/f16/bf16
    # (ROCM_AUDIT L4). The hand-written rocm_wmma lane is the oracle + the
    # availability fallback.
    ("rocm", "rocm_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_compiled", runtime_status="success",
        reason="ROCm matmul artifact runs the COMPILER-GENERATED RDNA WMMA GEMM "
               "(Stage L): tessera-opt generates + serializes the kernel to hsaco "
               "in-process (no mlir-opt), then HIP loads + launches it. The "
               "DEFAULT rocm matmul lane; degrades to the hand-written rocm_wmma "
               "oracle when the device_verified_jit lane is unavailable on the host.",
        execution_mode="hip_runtime"),
    # --- AMD ROCm GPU (COMPILED flash_attn lane — the matmul-L4 analog) ---
    # The compiler-GENERATED FA-2 forward (generate-wmma-flash-attn-kernel ->
    # ROCDL -> hsaco, in-process via tessera-opt) loaded + launched through HIP.
    # Reaches runtime.launch() exactly like the device_verified_jit GEMM; f16/bf16 storage,
    # f32 softmax + accumulate; validated vs a numpy attention reference.
    ("rocm", "rocm_flash_attn_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_flash_attn_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_flash_attn_compiled", runtime_status="success",
        reason="ROCm flash_attn artifact runs the COMPILER-GENERATED RDNA WMMA "
               "FA-2 forward: tessera-opt generates + serializes the kernel to "
               "hsaco in-process, then HIP loads + launches it. The attention "
               "analog of the device_verified_jit GEMM lane (rocm_compiled).",
        execution_mode="hip_runtime"),
    # The compiler-GENERATED FA-2 backward (generate-wmma-flash-attn-bwd-kernel
    # -> three fa_pre/fa_dkdv/fa_dq WMMA kernels -> hsaco) launched in sequence.
    # Self-contained VJP over (dO, Q, K, V): O is recomputed via the forward
    # lane, so nothing is saved from forward. Validated vs the numpy attention
    # backward / autodiff vjp_flash_attn. Core MHA (scale + causal).
    ("rocm", "rocm_flash_attn_bwd_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_flash_attn_bwd_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_flash_attn_bwd_compiled", runtime_status="success",
        reason="ROCm flash_attn backward artifact runs the COMPILER-GENERATED "
               "RDNA WMMA FA-2 backward: tessera-opt expands one "
               "tessera_rocm.flash_attn_bwd directive into three fa_pre/fa_dkdv/"
               "fa_dq WMMA kernels serialized to hsaco in-process, then HIP "
               "launches them in sequence to produce dQ/dK/dV. O is recomputed "
               "via the forward lane (nothing saved from forward). The "
               "reverse-mode analog of rocm_flash_attn_compiled; MHA + GQA/MQA "
               "(gqa dkdv atomic-accumulates dK/dV across the group) + additive "
               "attn_bias + sliding-window (implicitly causal, masks keys older "
               "than W) + Gemma-2 logit-softcap (dS scaled by 1-tanh^2), scale + "
               "causal, f16/bf16 storage, f32 accumulate.",
        execution_mode="hip_runtime",
        direction="backward", op_family="flash_attn",
        device_proof="device_verified_jit", evidence_target="rocm_gfx1151",
        numerical_fixture="tests/unit/test_rocm_flash_attn_bwd_compiled.py"),
    # Mamba2 selective_ssm BACKWARD (generate-rocm-selective-ssm-bwd-kernel):
    # operands (dout, x, A, B, C, delta[, gate[, state]]) -> (dx, dA, dB, dC,
    # ddelta). The reverse-mode analog of rocm_selective_ssm_compiled; the second
    # native backward launch lane after flash_attn (AUTODIFF_UNIFICATION §9a).
    ("rocm", "rocm_selective_ssm_bwd_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_selective_ssm_bwd_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_selective_ssm_bwd_compiled", runtime_status="success",
        reason="ROCm selective_ssm (Mamba2) backward artifact runs the "
               "COMPILER-GENERATED gfx1151 backward kernel "
               "(generate-rocm-selective-ssm-bwd-kernel: one thread per (b,d), "
               "atomic cross-channel reductions) HIP-launched to produce "
               "(dx, dA, dB, dC, ddelta) from operands (dout, x, A, B, C, delta"
               "[, gate[, state]]). f32, matches autodiff.vjp.vjp_selective_ssm; "
               "the reverse-mode analog of rocm_selective_ssm_compiled.",
        execution_mode="hip_runtime",
        direction="backward", op_family="selective_ssm",
        device_proof="device_verified_jit", evidence_target="rocm_gfx1151",
        numerical_fixture="tests/unit/test_rocm_ssm_bwd_launch_execute.py"),
    # Linear-attention family (quadratic-parallel form, no softmax; a distinct
    # algorithm from flash_attn): tessera.linear_attn + the decay-masked siblings
    # tessera.lightning_attention / tessera.retention, dispatched by op name.
    # f16/bf16, f32 accumulate; validated vs the numpy linear-attention reference.
    ("rocm", "rocm_linear_attn_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_linear_attn_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_linear_attn_compiled", runtime_status="success",
        reason="ROCm linear-attention-family artifact runs the COMPILER-GENERATED "
               "RDNA WMMA forward (quadratic-parallel form, no softmax): "
               "tessera-opt generates + serializes the kernel to hsaco "
               "in-process, then HIP loads + launches it. Handles linear_attn + "
               "lightning_attention (identity+decay) + retention (x²+decay) by "
               "op name.",
        execution_mode="hip_runtime"),
    # DS2 native row: the compiler/runtime ABI for the fused draft-block path
    # launches a generated ROCm kernel when hardware is present, with an
    # internal DS1-oracle fallback for hardware-free CI.
    ("rocm", "rocm_dspark_draft_block_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_dspark_draft_block_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_dspark_draft_block_compiled", runtime_status="success",
        reason="ROCm DSpark draft-block artifact launches the compiler-"
               "generated fused HIP/ROCDL draft-block kernel "
               "(generate-rocm-dspark-draft-block-kernel) for logits, "
               "confidence logits, greedy tokens, and hidden states. The "
               "executor falls back to the DS1 oracle only when ROCm hardware "
               "or tessera-opt is unavailable.",
        execution_mode="hip_runtime"),
    # Row-reduction softmax — the first non-matmul/non-WMMA device_verified_jit ROCm kernel.
    # Stable softmax over the last axis; f32/f16/bf16; validated vs numpy.
    ("rocm", "rocm_softmax_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_softmax_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_softmax_compiled", runtime_status="success",
        reason="ROCm softmax artifact runs the COMPILER-GENERATED RDNA row-"
               "reduction kernel (stable softmax over the last axis, one "
               "workgroup per row, LDS tree-reduce): tessera-opt generates + "
               "serializes the kernel to hsaco in-process, then HIP loads + "
               "launches it. The first non-matmul/non-WMMA device_verified_jit ROCm kernel.",
        execution_mode="hip_runtime"),
    # Row-reduction rmsnorm / layer_norm — siblings of the softmax kernel.
    # Unweighted row normalize over the last axis; f32/f16/bf16; vs numpy.
    ("rocm", "rocm_norm_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_norm_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_norm_compiled", runtime_status="success",
        reason="ROCm norm artifact runs the COMPILER-GENERATED RDNA row-reduction "
               "kernel (unweighted rmsnorm / layer_norm over the last axis, one "
               "workgroup per row, LDS tree-reduce of Σx and Σx²): tessera-opt "
               "generates + serializes the kernel to hsaco in-process, then HIP "
               "loads + launches it. Handles tessera.rmsnorm(_safe) + "
               "tessera.layer_norm by op name.",
        execution_mode="hip_runtime"),
    # Row reduction (sum/mean/max/min) over the last axis — the ROCm analog of
    # the x86 AVX-512 reduction lane. vs numpy.
    ("rocm", "rocm_reduce_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_reduce_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_reduce_compiled", runtime_status="success",
        reason="ROCm reduce artifact runs the COMPILER-GENERATED RDNA row-"
               "reduction kernel (sum/mean/max/min over the last axis, one "
               "workgroup per row, LDS tree-reduce): tessera-opt generates + "
               "serializes the kernel to hsaco in-process, then HIP loads + "
               "launches it. An arbitrary reduced axis folds to [outer,inner]; "
               "handles tessera.sum/mean/max/min (amax/amin) by op name.",
        execution_mode="hip_runtime"),
    # argmax/argmin — CUB ArgMax-style warp-shuffle arg-reduce, i32 index output.
    ("rocm", "rocm_argreduce_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_argreduce_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_argreduce_compiled", runtime_status="success",
        reason="ROCm argreduce artifact runs the COMPILER-GENERATED RDNA row "
               "arg-reduction kernel (argmax/argmin along one axis): each thread "
               "carries the best (value,index) pair, a gpu.shuffle xor butterfly "
               "reduces the pair within a 32-lane subgroup (CUB ArgMax pattern), "
               "first-occurrence tie-break. tessera-opt → hsaco in-process, HIP "
               "launches it. f16/bf16/f32 input, i32 index output.",
        execution_mode="hip_runtime"),
    # cumsum/cumprod/cummax/cummin — CUB BlockScan, inclusive prefix, same shape.
    ("rocm", "rocm_scan_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_scan_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_scan_compiled", runtime_status="success",
        reason="ROCm scan artifact runs the COMPILER-GENERATED RDNA row inclusive "
               "prefix scan (cumsum/cumprod/cummax/cummin along one axis): the CUB "
               "BlockScan technique — gpu.shuffle up (Kogge-Stone) warp-scan + "
               "per-subgroup exclusive offset + cross-tile carry. tessera-opt → "
               "hsaco in-process, HIP launches it. Same-shape output. f16/bf16/f32.",
        execution_mode="hip_runtime"),
    # Standalone elementwise activations (gelu/silu/relu) — flat per-element
    # kernel; the standalone analog of the GEMM fused epilogue. vs numpy.
    ("rocm", "rocm_activation_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_activation_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_activation_compiled", runtime_status="success",
        reason="ROCm activation artifact runs the COMPILER-GENERATED flat "
               "elementwise kernel (standalone gelu/silu/relu, one thread per "
               "element): tessera-opt generates + serializes the kernel to hsaco "
               "in-process, then HIP loads + launches it. Dispatched by op name.",
        execution_mode="hip_runtime"),
    # Standalone elementwise unary math (exp/log/sqrt/erf/…) — the S2 scalar-math
    # / stability family, flat per-element; the unary sibling of activation.
    ("rocm", "rocm_unary_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_unary_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_unary_compiled", runtime_status="success",
        reason="ROCm unary artifact runs the COMPILER-GENERATED flat elementwise "
               "unary-math kernel (S2 scalar-math/stability: exp/log/sqrt/rsqrt/"
               "reciprocal/abs/sign/erf/tanh/sigmoid/log1p/expm1/softplus, one "
               "thread per element): tessera-opt generates + serializes the "
               "kernel to hsaco in-process, then HIP loads + launches it. "
               "Dispatched by op name.",
        execution_mode="hip_runtime"),
    # Binary arithmetic sub/div/pow/maximum/minimum — flat 2-operand elementwise.
    ("rocm", "rocm_binary_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_binary_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_binary_compiled", runtime_status="success",
        reason="ROCm binary artifact runs the COMPILER-GENERATED flat 2-operand "
               "elementwise binary-arithmetic kernel (sub/div/pow/maximum/minimum, "
               "one thread per element): tessera-opt generates + serializes the "
               "kernel to hsaco in-process, then HIP loads + launches it. "
               "Dispatched by op name.",
        execution_mode="hip_runtime"),
    # Comparison eq/ne/lt/le/gt/ge — flat 2-operand elementwise, bool output.
    ("rocm", "rocm_clamp_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_clamp_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_clamp_compiled", runtime_status="success",
        reason="ROCm clamp lane runs clamp / clip as min(max(x, lo), hi) composed "
               "on the COMPILER-GENERATED gfx1151 binary max/min kernel (either "
               "bound optional; scalar bounds broadcast on host). f32, matches "
               "np.clip.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_complex_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_complex_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_complex_compiled", runtime_status="success",
        reason="ROCm complex-arithmetic lane (9 pointwise ops) over "
               "interleaved-f32 [...,2] composed on the gfx1151 unary / binary / "
               "atan2 kernels; host packs the interleave. f32, matches "
               "tessera.complex.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_softcap_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_softcap_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_softcap_compiled", runtime_status="success",
        reason="ROCm softcap lane runs cap*tanh(x/cap) composed on the "
               "COMPILER-GENERATED gfx1151 unary tanh kernel (scalar cap "
               "broadcast on host). f32, matches cap*tanh(x/cap).",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_rng_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_rng_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_rng_compiled", runtime_status="success",
        reason="ROCm RNG lane runs counter-based Philox-4x32-10 on the "
               "COMPILER-GENERATED gfx1151 kernel (generate-rocm-philox-kernel) "
               "for the uniform bits; host applies the distribution transform. "
               "f32, bit-exact vs tessera.rng_device.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_strided_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_strided_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_strided_compiled", runtime_status="success",
        reason="ROCm 0-move lane realizes pad/cat/roll/flip/tile/repeat/stack "
               "via the COMPILER-GENERATED gfx1151 masked-gather kernel "
               "(generate-rocm-gather-kernel; host index map). f32, matches "
               "numpy.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_scatter_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_scatter_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_scatter_compiled", runtime_status="success",
        reason="ROCm scatter lane realizes scatter/scatter_add/scatter_reduce "
               "(0-reduce indexed store) via the COMPILER-GENERATED gfx1151 "
               "kernel (generate-rocm-scatter-kernel; one thread per element; "
               "atomic_rmw for add/min/max). f32, matches the numpy scatter "
               "reference.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_kv_cache_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_kv_cache_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_kv_cache_compiled", runtime_status="success",
        reason="ROCm KV-cache paged-movement lane realizes kv_cache "
               "append/read/prune over a resident cache buffer (max_seq, H, D) "
               "by COMPOSING the COMPILER-GENERATED gfx1151 scatter (append row "
               "write) + masked-gather (read/prune) kernels with host page-index "
               "math. quantize_kv rides the intquant lane. f32, matches the "
               "KVCacheHandle append/read/prune reference.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_conformal_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_conformal_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_conformal_compiled", runtime_status="success",
        reason="ROCm conformal lane runs mobius (az+b)/(cz+d) on the gfx1151 "
               "complex mul/div lane and stereographic (x+iy)/(1-z) on the "
               "binary div lane (host orchestration). f32, matches "
               "tessera.complex.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_sort_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_sort_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_sort_compiled", runtime_status="success",
        reason="ROCm sort lane runs sort/argsort/top_k via the "
               "COMPILER-GENERATED cooperative bitonic kernel "
               "(generate-rocm-sort-kernel; one block per row; host pads each "
               "row to a power of two and flips for descending). f32, matches "
               "numpy.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_clifford_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_clifford_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_clifford_compiled", runtime_status="success",
        reason="ROCm GA lane runs the Cl(3,0) table-driven bilinear products "
               "(geometric_product/wedge/left_contraction, + inner/"
               "rotor_sandwich by composition) on the COMPILER-GENERATED "
               "gfx1151 kernel (generate-rocm-clifford-kernel; one thread per "
               "batch element; triples unrolled at generation time). f32, "
               "matches the numpy GA reference.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_atan2_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_atan2_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_atan2_compiled", runtime_status="success",
        reason="ROCm atan2 lane runs quadrant-aware atan2(y, x) composed on the "
               "COMPILER-GENERATED gfx1151 unary atan kernel (sign/quadrant "
               "logic on host). f32, matches np.arctan2.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_predicate_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_predicate_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_predicate_compiled", runtime_status="success",
        reason="ROCm predicate artifact runs the COMPILER-GENERATED unary "
               "predicate kernel (isnan/isinf/isfinite, kind StrAttr-selected, one "
               "thread per element): tessera-opt generates generate-rocm-predicate-"
               "kernel -> ROCDL -> hsaco, then HIP launches it. f32 in, bool out.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_compare_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_compare_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_compare_compiled", runtime_status="success",
        reason="ROCm compare artifact runs the COMPILER-GENERATED flat 2-operand "
               "elementwise comparison kernel (eq/ne/lt/le/gt/ge, one thread per "
               "element, i8/bool output): tessera-opt generates + serializes the "
               "kernel to hsaco in-process, then HIP loads + launches it. "
               "Dispatched by op name.",
        execution_mode="hip_runtime"),
    # Logical and/or/xor/not — flat elementwise over i8 booleans, bool output.
    ("rocm", "rocm_logical_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_logical_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_logical_compiled", runtime_status="success",
        reason="ROCm logical artifact runs the COMPILER-GENERATED flat "
               "elementwise logical kernel (and/or/xor binary, not unary, over "
               "i8 booleans, one thread per element): tessera-opt generates + "
               "serializes the kernel to hsaco in-process, then HIP loads + "
               "launches it. Dispatched by op name.",
        execution_mode="hip_runtime"),
    # Bitwise and/or/xor/not — flat elementwise over i32 integers, i32 output.
    ("rocm", "rocm_bitwise_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_bitwise_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_bitwise_compiled", runtime_status="success",
        reason="ROCm bitwise artifact runs the COMPILER-GENERATED flat "
               "elementwise bitwise kernel (and/or/xor binary, not unary, over "
               "i32 integers, one thread per element): tessera-opt generates + "
               "serializes the kernel to hsaco in-process, then HIP loads + "
               "launches it. Dispatched by op name.",
        execution_mode="hip_runtime"),
    # Ternary select where(cond,a,b) — flat 3-operand elementwise. vs numpy.
    ("rocm", "rocm_where_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_where_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_where_compiled", runtime_status="success",
        reason="ROCm where artifact runs the COMPILER-GENERATED flat 3-operand "
               "ternary select where(cond,a,b)=cond?a:b (one thread per element): "
               "tessera-opt generates + serializes the kernel to hsaco in-process "
               "(generate-rocm-where-kernel -> ROCDL), then HIP loads + launches "
               "it. cond i8 normalized != 0, a/b/out f16/bf16/f32.",
        execution_mode="hip_runtime"),
    # SwiGLU gate-multiply silu(a)·b — flat 2-operand elementwise. vs numpy.
    ("rocm", "rocm_fft_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_fft_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_fft_compiled", runtime_status="success",
        reason="ROCm FFT artifact runs fft / ifft / rfft / irfft over any axis "
               "length on the COMPILER-GENERATED one-thread-per-bin DFT kernel "
               "(generate-rocm-dft-kernel -> ROCDL, cos/sin twiddles) on gfx1151 "
               "+ r2c/c2r pack-unpack + plan scale. Direct DFT (radix-2/Bluestein "
               "perf is a follow-up). complex64/f32.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_spectral_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_spectral_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_spectral_compiled", runtime_status="success",
        reason="ROCm spectral composites (dct / stft / istft / spectral_conv / "
               "spectral_filter) compose the rocm_fft_compiled DFT lane — "
               "framing / windowing / overlap-add / pointwise complex-mul on "
               "host, the transform on the gfx1151 DFT kernel. f32, matches "
               "np.fft.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_sparse_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_sparse_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_sparse_compiled", runtime_status="success",
        reason="ROCm sparse lane (spmm_csr / spmm_coo / sddmm / bsmm) runs the "
               "COMPILER-GENERATED gfx1151 sparse kernels (generate-rocm-spmm/"
               "sddmm-kernel: row-wise CSR SpMM, sampled SDDMM skipping masked-"
               "zero entries) HIP-launched; COO folds to CSR on host; bsmm via "
               "the WMMA matmul (bf16). f32, matches numpy.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_sparse_attn_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_sparse_attn_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_sparse_attn_compiled", runtime_status="success",
        reason="ROCm DK2 sparse-attention lane lowers selected-block layouts "
               "(MSA explicit selected_block_ids plus DSA/NSA-compatible block "
               "worklists) to COMPILER-GENERATED scalar/row-tiled block-sparse "
               "attention kernels, with a GPU-resident top-k selector for MSA/"
               "NSA score rows. It preserves GQA grouping, q-position causal "
               "masking, and dense-equivalence when selected blocks cover the "
               "full KV range. f32 softmax/accumulate.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_composite_helper_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_composite_helper_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_composite_helper_compiled", runtime_status="success",
        reason="ROCm composite-helper lane keeps memory_index_score, "
               "msa_index_scores, varlen_sdpa, and score_combine visible to "
               "Target IR while composing the existing matmul/flash-attn/binary "
               "helper semantics. runtime.launch reports reference_cpu until "
               "the HIP-native helper path is hardware-proven. f32.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_optimizer_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_optimizer_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_optimizer_compiled", runtime_status="success",
        reason="ROCm optimizer lane runs sgd / momentum / adam / adamw / lion as "
               "a fused per-parameter update on the COMPILER-GENERATED gfx1151 "
               "kernel (generate-rocm-optimizer-kernel, kind StrAttr-selected, "
               "one thread per element; the 1-β^t bias correction computed on "
               "host) HIP-launched. f32, matches the optim.py reference.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_lamb_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_lamb_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_lamb_compiled", runtime_status="success",
        reason="ROCm LAMB lane runs the COMPILER-GENERATED gfx1151 adam kernel "
               "(lr=1/wd=0) then applies the per-tensor trust ratio "
               "‖p‖/‖update‖ on host. f32, matches optim.lamb.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_muon_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_muon_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_muon_compiled", runtime_status="success",
        reason="ROCm Muon lane orthogonalizes the momentum matrix via the "
               "gfx1151 SVD kernel (U·Vh polar factor); the small U@Vh + "
               "momentum/sgd run on host. <2-D params normalize. f32, matches "
               "optim.muon.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_moe_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_moe_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_moe_compiled", runtime_status="success",
        reason="ROCm moe-compute lane runs the routed per-token expert GEMVs "
               "(top-1) on the COMPILER-GENERATED gfx1151 kernel (generate-rocm-"
               "moe-kernel: one thread per (token, out-col); routing resolved on "
               "host) HIP-launched. f32, matches numpy.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_moe_transport_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_moe_transport_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_moe_transport_compiled", runtime_status="success",
        reason="ROCm DK3 MoE transport + expert GEMM run NATIVELY on gfx1151: "
               "moe_dispatch on the device gather kernel (token_of_slot = "
               "sort_perm//top_k row gather), moe_combine on the device scatter "
               "(add) kernel (host pre-scales each packed row by its route "
               "weight, then atomic scatter-add to token order), and "
               "grouped_swiglu's three expert GEMMs on the f32 GEMM device "
               "kernel (generate-rocm-gemm-f32-kernel; silu*mul host-side). All "
               "three report native_gpu vs the stdlib DispatchPlan oracle (f32 "
               "vs the f64 oracle for combine/swiglu). Off-box they fall back to "
               "the oracle + reference_cpu.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_normcompose_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_normcompose_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_normcompose_compiled", runtime_status="success",
        reason="ROCm group/instance/weight-norm lane composed on the gfx1151 "
               "layer_norm (row mean/var) + reduce (sum-of-squares) kernels; "
               "host does the reshape / per-axis divide. f32, matches "
               "nn.functional.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_grad_clip_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_grad_clip_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_grad_clip_compiled", runtime_status="success",
        reason="ROCm grad_clip_norm lane — global gradient-norm clipping "
               "g*min(1, max_norm/||g||): the L2 norm's global sum-of-squares "
               "runs on the gfx1151 reduce kernel (the FLOP-heavy O(n) part), "
               "host does sqrt + the clip scale + the elementwise scale; "
               "norm_type=inf uses max|g|. f32, matches optim.clip_grad_norm "
               "within f32 tolerance (the reference accumulates in f64).",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_selective_ssm_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_selective_ssm_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_selective_ssm_compiled", runtime_status="success",
        reason="ROCm state-space lane runs selective_ssm (Mamba2) on the "
               "COMPILER-GENERATED gfx1151 selective-scan kernel (generate-rocm-"
               "selective-ssm-kernel: one thread per (b,d) channel, sequential "
               "over time, exp via math->rocdl) HIP-launched. f32, matches the "
               "numpy reference.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_linalg_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_linalg_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_linalg_compiled", runtime_status="success",
        reason="ROCm linalg lane (cholesky / tri_solve / cholesky_solve / lu / qr "
               "/ svd) runs the COMPILER-GENERATED gfx1151 kernels (generate-rocm-"
               "cholesky / tri-solve / lu / qr / svd-kernel, one thread per matrix "
               "or matrix/RHS-column) HIP-launched; cholesky_solve composes two "
               "triangular solves. f32, matches numpy.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_stat_reduce_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_stat_reduce_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_stat_reduce_compiled", runtime_status="success",
        reason="ROCm stat-reduce artifact runs var / std / count_nonzero over an "
               "axis, composed from the warp-shuffle reduce kernel "
               "(var=mean(x^2)-mean(x)^2, std=sqrt(var), count_nonzero=sum(x!=0))"
               ". f32.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_stable_reduce_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_stable_reduce_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_stable_reduce_compiled", runtime_status="success",
        reason="ROCm stable-reduce artifact runs logsumexp / log_softmax / "
               "softmax_safe / sigmoid_safe — max-shifted, composed from the "
               "warp-shuffle reduce (max/sum) + the unary exp/log lane; "
               "softmax_safe / sigmoid_safe alias the stable softmax / sigmoid "
               "lanes. f32.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_class_loss_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_class_loss_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_class_loss_compiled", runtime_status="success",
        reason="ROCm class-loss artifact runs cross_entropy / kl / js / focal / "
               "label_smoothed_cross_entropy / z_loss: exp/log on the rocm unary "
               "lane (gfx1151), class-axis max/sum/gather/one-hot on the host. "
               "ROCm mirror of x86_class_loss. f32.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_ebm_loss_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_ebm_loss_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_ebm_loss_compiled", runtime_status="success",
        reason="ROCm EBM/diffusion loss artifact runs score_matching / "
               "denoising / implicit / contrastive_divergence / persistent_cd / "
               "ddpm_noise_pred / vlb / load_balance: the diff/square and "
               "reductions run on the gfx1151 binary + reduce kernels, the "
               "structure on the host. ROCm mirror of x86_ebm_loss. f32.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_ebm_compute_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_ebm_compute_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_ebm_compute_compiled", runtime_status="success",
        reason="ROCm EBM compute artifact runs energy_quadratic / inner_step / "
               "refinement / self_verify: the diff/square and reductions run on "
               "the gfx1151 binary + reduce kernels, the structure on the host. "
               "ROCm mirror of x86_ebm_compute. f32.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_ebm_langevin_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_ebm_langevin_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_ebm_langevin_compiled", runtime_status="success",
        reason="ROCm EBM Langevin sampling artifact runs y − η·grad + "
               "noise_scale·z where z is Box-Muller Gaussian noise drawn "
               "ON-DEVICE from counter-based Philox-4x32-10 via the "
               "COMPILER-GENERATED gfx1151 kernel "
               "(generate-rocm-ebm-langevin-kernel). f32, matches "
               "tessera.ebm.langevin_step_philox.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_fpquant_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_fpquant_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_fpquant_compiled", runtime_status="success",
        reason="ROCm fpquant artifact runs quantize/dequantize fp8 / fp6 / fp4: "
               "per-tensor scale + grid-snap on the COMPILER-GENERATED fpquant "
               "kernel (generate-rocm-fpquant-kernel: log2/exp2/roundeven -> "
               "ROCDL). ROCm mirror of x86_fpquant. f32.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_intquant_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_intquant_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_intquant_compiled", runtime_status="success",
        reason="ROCm intquant artifact runs quantize/dequantize int8/int4 and "
               "fake_quantize: qparam selection and int8 container conversion on "
               "host, round/max/min/mul on generated ROCm unary/binary kernels. "
               "int4 is signed int4 values in int8 containers.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_pooling_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_pooling_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_pooling_compiled", runtime_status="success",
        reason="ROCm pooling artifact runs max/avg/min/adaptive_pool by forming "
               "the pooling window matrix on host and reducing each row on the "
               "generated ROCm reduce kernel. f32, matches nn.functional.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_image_affine_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_image_affine_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_image_affine_compiled", runtime_status="success",
        reason="ROCm image affine artifact runs image_normalize as "
               "(x-mean)/std: layout and per-channel broadcast on host, sub/div "
               "on generated ROCm binary kernels. f32, matches tessera.ops.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_metric_loss_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_metric_loss_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_metric_loss_compiled", runtime_status="success",
        reason="ROCm metric-loss artifact runs wasserstein / cosine_embedding / "
               "contrastive / triplet / InfoNCE / NT-Xent / seq2seq losses with "
               "generated ROCm reductions and exp/log; label, mask, sort, and "
               "compact matrix structure remain on host. f32, matches "
               "tessera.losses.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_structured_compute_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_structured_compute_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_structured_compute_compiled", runtime_status="success",
        reason="ROCm structured-compute artifact covers CTC loss, VLM image/"
               "layout transforms, conv1d/conv_transpose/LoRA, GRU/simple-RNN, "
               "and depthwise_conv1d through runtime.launch(). Shape/control "
               "bookkeeping remains host-structured; the row is direct "
               "single-GPU executable evidence, not a bespoke fused kernel.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_dequant_gemm_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_dequant_gemm_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_dequant_gemm_compiled", runtime_status="success",
        reason="ROCm DK4 dequant-GEMM artifact launches the compiler-generated "
               "fused HIP/ROCDL packed-code dequant-into-GEMM kernel "
               "(generate-rocm-dequant-gemm-kernel) for int4/int8 codes + "
               "per-group scales, f32 accumulate. The executor falls back to "
               "the packed-weight oracle only when ROCm hardware or tessera-opt "
               "is unavailable.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_nvfp4_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_nvfp4_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_nvfp4_compiled", runtime_status="success",
        reason="ROCm nvfp4 artifact runs block-scaled fp4: per-block fp8-E4M3 "
               "scale + E2M1 codes on the fpquant kernel + host block structure. "
               "ROCm mirror of x86_nvfp4. f32.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_binary_loss_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_binary_loss_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_binary_loss_compiled", runtime_status="success",
        reason="ROCm binary-loss artifact runs the COMPILER-GENERATED bce / "
               "asymmetric_bce per-element loss over (logits, targets) on "
               "gfx1151 (generate-rocm-binary-loss-kernel -> ROCDL, stable "
               "softplus) + reduction. ROCm mirror of x86_binary_loss. "
               "f32/f16/bf16.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_rl_loss_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_rl_loss_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_rl_loss_compiled", runtime_status="success",
        reason="ROCm rl-loss artifact runs the ppo / cispo / grpo core surrogate "
               "on the COMPILER-GENERATED policy-loss kernel (generate-rocm-"
               "policy-loss-kernel -> ROCDL) + normalize_group_advantages on the "
               "norm lane, then reduction. ROCm mirror of x86_rl_loss. "
               "f32/f16/bf16.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_loss_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_loss_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_loss_compiled", runtime_status="success",
        reason="ROCm loss artifact runs the COMPILER-GENERATED per-element "
               "regression loss (mse/mae/huber/smooth_l1/log_cosh) over "
               "(pred, target) on gfx1151 (generate-rocm-pointwise-loss-kernel "
               "-> ROCDL, exp/log1p via math->rocdl), then the none/mean/sum "
               "reduction. The CPU analog of the x86_loss lane. f32/f16/bf16.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_silu_mul_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_silu_mul_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_silu_mul_compiled", runtime_status="success",
        reason="ROCm silu_mul artifact runs the COMPILER-GENERATED flat 2-operand "
               "elementwise SwiGLU gate-multiply silu(a)·b (one thread per "
               "element): tessera-opt generates + serializes the kernel to hsaco "
               "in-process, then HIP loads + launches it.",
        execution_mode="hip_runtime"),
    # ALiBi positional-bias generator — bias[h,i,j]=slope[h]·(j−i). vs numpy.
    ("rocm", "rocm_alibi_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_alibi_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_alibi_compiled", runtime_status="success",
        reason="ROCm alibi artifact runs the COMPILER-GENERATED ALiBi positional-"
               "bias generator (bias[h,i,j]=slope[h]·(j−i) over [H,S,S], one "
               "thread per element): tessera-opt generates + serializes the "
               "kernel to hsaco in-process, then HIP loads + launches it. Slopes "
               "default to the 2^(-8(k+1)/H) ramp.",
        execution_mode="hip_runtime"),
    # matmul-family — batched_gemm / linear_general / qkv_projection /
    # factorized_matmul / einsum on the WMMA GEMM kernel. vs numpy.
    ("rocm", "rocm_matmul_family_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_matmul_family_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_matmul_family_compiled", runtime_status="success",
        reason="ROCm matmul-family artifact runs the COMPILER-GENERATED WMMA GEMM "
               "kernel (the rocm_compiled spine) reshaped/batched/split in the "
               "runtime — batched_gemm, linear_general, qkv_projection, "
               "factorized_matmul (GPU matmul + exact host SVD-truncate), and "
               "single-contraction einsum. f16/bf16, f32 accumulate.",
        execution_mode="hip_runtime"),
    ("rocm", "rocm_conv_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_conv_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_conv_compiled", runtime_status="success",
        reason="ROCm convolution lane runs conv2d/conv3d as im2col + the "
               "COMPILER-GENERATED WMMA GEMM: the host lays out the NHWC/NDHWC "
               "patch matrix, the GEMM runs on the gfx1151 WMMA kernel (f16/bf16 "
               "storage, f32 accumulate), bias/groups on the host. Matches the "
               "conv reference to WMMA f16 tolerance.",
        execution_mode="hip_runtime"),
    # Exotic-attention compositions — gated_attention / mla_decode /
    # mla_decode_fused on WMMA flash/GEMM plus mla_decode_step absorbed-latent
    # decode on the DK1 generated ROCm kernel. vs stdlib/numpy.
    ("rocm", "rocm_exotic_attn_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_exotic_attn_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_exotic_attn_compiled", runtime_status="success",
        reason="ROCm exotic-attention artifact composes the COMPILER-GENERATED "
               "WMMA flash_attn kernel with the WMMA GEMM kernel (MLA latent "
               "projections) + an elementwise gate — gated_attention, mla_decode, "
               "mla_decode_fused — and routes mla_decode_step through the DK1 "
               "absorbed-latent decode kernel against stdlib.attention."
               "mla_decode_step. f16/bf16, f32 softmax+accumulate.",
        execution_mode="hip_runtime"),
    # DeltaNet — causal sequential-scan gated/delta linear attention. vs numpy.
    ("rocm", "rocm_deltanet_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_deltanet_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_deltanet_compiled", runtime_status="success",
        reason="ROCm deltanet artifact runs the COMPILER-GENERATED causal "
               "sequential-scan kernel (one workgroup per (b,h), one thread per "
               "value-column, LDS state) — the gated/delta linear-attention "
               "recurrence for gated_deltanet / kimi_delta_attention / "
               "modified_delta_attention. f16/bf16/f32 storage, f32 compute.",
        execution_mode="hip_runtime"),
    # x86 analog of the ROCm deltanet lane — the AVX-512 causal delta-rule scan
    # (avx512_deltanet_f32) for gated_deltanet / kimi_delta_attention /
    # modified_delta_attention. f32; matches numpy _delta_attention_impl.
    ("x86", "x86_deltanet_compiled"): ExecutionRow(
        target="x86", compiler_path="x86_deltanet_compiled",
        execution_kind="native_cpu", executable=True,
        executor_id="x86_deltanet_compiled", runtime_status="success",
        reason="x86 deltanet artifact runs the hand-written AVX-512 causal "
               "delta-rule sequential scan (avx512_deltanet_f32, runtime-loaded "
               "from libtessera_x86_elementwise.so): per (b,h) a Dqk x Dv state "
               "scanned over S with erase/decay/beta/modified/gate variants — the "
               "gated/delta linear-attention recurrence for gated_deltanet / "
               "kimi_delta_attention / modified_delta_attention. f32, matches "
               "numpy _delta_attention_impl.",
        execution_mode="cpu_avx512"),
    # Rotary position embedding — interleaved-pair RoPE over [M, D]. vs numpy.
    ("rocm", "rocm_rope_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_rope_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_rope_compiled", runtime_status="success",
        reason="ROCm rope artifact runs the COMPILER-GENERATED interleaved-pair "
               "rotary-position-embedding kernel (one workgroup per row): "
               "tessera-opt generates + serializes the kernel to hsaco "
               "in-process, then HIP loads + launches it.",
        execution_mode="hip_runtime"),
    # --- NVIDIA GPU (consumer Blackwell, sm_120 warp-level mma.sync GEMM) ---
    # sm_120 bring-up (2026-06-25): the shipped libtessera_nvidia_gemm.so runs a
    # real mma.sync GEMM on the RTX 5070 Ti. Like the rocm_wmma row, this is
    # host-independent in the dashboard; the jit path stamps executable=True only
    # on a host passing the runtime probe (lib loads + a live CUDA device). The
    # analog of rocm_wmma (shipped symbol); a compiler-generated nvidia lane (the
    # rocm_compiled analog) is a later follow-up. The row targets the proven arch
    # nvidia_sm120 — the NVRTC symbol auto-detects compute_XX, but only sm_120 is
    # hardware-proven, so the other arches stay unimplemented.
    ("nvidia_sm120", "nvidia_mma"): ExecutionRow(
        target="nvidia_sm120", compiler_path="nvidia_mma",
        execution_kind="native_gpu", executable=True,
        executor_id="nvidia_mma", runtime_status="success",
        reason="NVIDIA sm_120 matmul via the shipped warp-level mma.sync GEMM "
               "(tessera_nvidia_mma_gemm_{f16,bf16,tf32} C ABI symbol in "
               "libtessera_nvidia_gemm.so, NVRTC-device_verified_jit for the device arch; "
               "f16/bf16/fp32(tf32-math) storage, f32 accumulate). Directly "
               "selectable by stamping compiler_path=\"nvidia_mma\".",
        execution_mode="cuda_runtime"),
}


# Targets recognized by the capability registry but with NO executable runtime
# row (yet). `launch()` reports `unimplemented` (target capability present) or
# `missing_backend` (target capability absent). Listed explicitly so the drift
# test catches accidental status drift.
#
# Note: ``rocm`` is NO LONGER here — it has an executable ``rocm_wmma`` row
# above (RDNA WMMA GEMM). The named ROCm sub-arches — INCLUDING ``rocm_gfx1151``,
# the Strix Halo box's own arch — stay listed here as "no per-arch executor row":
# the shipped GEMM symbol HIPRTC-compiles for whatever arch the device
# enumerates, so the generic ``rocm`` lane is what actually executes on gfx1151;
# the sub-arch aliases earn distinct rows only if a sub-arch needs distinct
# dispatch. Listing every registered ROCm sub-arch here (not just some) keeps the
# classification total — every capability is either executable or explicitly
# unimplemented, no silent ``lookup() -> None`` gaps.
#
# Note: ``nvidia_sm120`` is NO LONGER here — it has an executable ``nvidia_mma``
# row above (consumer-Blackwell mma.sync GEMM, proven on the RTX 5070 Ti). The
# other NVIDIA arches (sm_80/90/100) stay listed: the shipped NVRTC symbol
# auto-detects the device arch, but only sm_120 is hardware-proven today.
_UNIMPLEMENTED_TARGETS: tuple[str, ...] = (
    "nvidia_sm80", "nvidia_sm90", "nvidia_sm100",
    "rocm_gfx90a", "rocm_gfx940", "rocm_gfx942", "rocm_gfx950",
    "rocm_gfx1100", "rocm_gfx1151", "rocm_gfx1200",
)


def lookup(target: str, compiler_path: str) -> Optional[ExecutionRow]:
    """The exact matrix lookup. Returns None when (target, compiler_path) isn't a
    runtime-executable pair — `launch()` then falls back to the
    target-default-status path (unimplemented / missing_backend)."""
    return _MATRIX.get((target, compiler_path))


def executor_for_metadata(metadata: Mapping[str, object]) -> Optional[ExecutionRow]:
    """The interpretation `launch()` uses: read `target` + `compiler_path` from
    an artifact's metadata and resolve the row. None if there is no executor."""
    target = str(metadata.get("target", "cpu") or "cpu")
    compiler_path = str(metadata.get("compiler_path", "") or "")
    if not compiler_path:
        # Legacy artifacts without compiler_path: fall through to the historical
        # `executable + execution_kind == native_cpu` logic in launch().
        return None
    return lookup(target, compiler_path)


def all_rows() -> list[ExecutionRow]:
    """Stable order: by (target, compiler_path) — what the dashboard renders."""
    return [_MATRIX[k] for k in sorted(_MATRIX)]


def unimplemented_targets() -> tuple[str, ...]:
    """The targets the capability registry knows about but for which no
    executable row exists; `launch()` reports unimplemented / missing_backend."""
    return _UNIMPLEMENTED_TARGETS


# --- autodiff facet: native backward launches (AUTODIFF_UNIFICATION_PLAN §9a) ---

_DEVICE_PROOFS: frozenset[str] = frozenset(
    {"device_verified_jit", "device_verified_abi"}
)


def backward_rows() -> list[ExecutionRow]:
    """Every backward (VJP) launch row in the matrix (``direction == "backward"``)."""
    return [r for r in all_rows() if r.direction == "backward"]


def native_backward_targets() -> dict[str, dict[str, tuple[str, ...]]]:
    """Per op-family, the targets whose **backward** has a native launch.

    Runtime binding and device proof are deliberately separate.  A verified
    row must name an exact evidence target and a checked-in numerical fixture;
    ``execution_kind`` alone is never device-verification evidence.
    """
    bound: dict[str, set[str]] = {}
    oracle: dict[str, set[str]] = {}
    jit: dict[str, set[str]] = {}
    abi: dict[str, set[str]] = {}
    for r in backward_rows():
        if not r.executable or not r.op_family:
            continue
        target = r.evidence_target or r.target
        bound.setdefault(r.op_family, set()).add(target)
        if r.device_proof:
            if r.device_proof not in _DEVICE_PROOFS:
                raise ValueError(f"unknown backward device proof {r.device_proof!r}")
            fixture = _REPO_ROOT / r.numerical_fixture if r.numerical_fixture else None
            if not r.evidence_target or fixture is None or not fixture.is_file():
                raise ValueError(
                    f"{r.compiler_path} claims {r.device_proof} without exact "
                    "evidence_target and checked-in numerical_fixture")
            oracle.setdefault(r.op_family, set()).add(target)
            (jit if r.device_proof == "device_verified_jit" else abi).setdefault(
                r.op_family, set()).add(target)
    return {
        fam: {
            "runtime_bound": tuple(sorted(bound.get(fam, ()))),
            "oracle_proven": tuple(sorted(oracle.get(fam, ()))),
            "device_verified_jit": tuple(sorted(jit.get(fam, ()))),
            "device_verified_abi": tuple(sorted(abi.get(fam, ()))),
        }
        for fam in (set(bound) | set(oracle) | set(jit) | set(abi))
    }


def has_native_backward(op_family: str, target: str) -> bool:
    """True iff ``target`` runs a native, device-executable **backward** launch
    for ``op_family``. The hook Phase 4 (A3) flips so a
    ``@jit(autodiff="reverse", native_required=True)`` request is honored instead
    of rejected. Sourced from the matrix — never a hand-maintained table."""
    info = native_backward_targets().get(op_family)
    if not info:
        return False
    verified = set(info["device_verified_jit"]) | set(info["device_verified_abi"])
    # Public requests still use family targets; exact proof labels are retained
    # in the ledger while this compatibility match resolves the runtime route.
    return target in verified or any(t.startswith(f"{target}_") for t in verified)


#: Stable CSV column order for the execution matrix — append-only.
EXECUTION_MATRIX_CSV_COLUMNS: tuple[str, ...] = (
    "target", "compiler_path", "execution_kind", "executable",
    "executor_id", "runtime_status", "execution_mode", "reason",
)


def render_csv() -> str:
    """Render the canonical machine-readable execution matrix.

    One row per (target, compiler_path) in `all_rows()` order.  This is
    the drift-gated artifact; the Markdown is the human companion.
    """
    import csv as _csv
    import io as _io

    buf = _io.StringIO()
    writer = _csv.writer(buf, lineterminator="\n")
    writer.writerow(EXECUTION_MATRIX_CSV_COLUMNS)
    for r in all_rows():
        writer.writerow([
            r.target, r.compiler_path, r.execution_kind,
            "1" if r.executable else "0",
            r.executor_id or "", r.runtime_status, r.execution_mode, r.reason,
        ])
    return buf.getvalue()


def render_dashboard() -> str:
    """Render the matrix as a Markdown table for `docs/audit/generated/runtime_execution_matrix.md`.
    Pure function so the drift test can compare bytes."""
    lines = [
        "# Runtime execution matrix",
        "",
        "**Generated from `tessera.compiler.execution_matrix._MATRIX` — do not hand-edit.**",
        "Regenerate with:",
        "",
        "```",
        "python3 -c 'from tessera.compiler.execution_matrix import write_dashboard; write_dashboard()'",
        "```",
        "",
        "Single source of truth for what `runtime.launch()` does with each "
        "`(target, compiler_path)` pair. `capabilities.py`, `runtime.launch()`, "
        "and this dashboard all derive from the same `_MATRIX`. The drift test "
        "`test_runtime_execution_matrix` fails if they diverge.",
        "",
        "## Executable rows",
        "",
        "| Target | Compiler path | Executor | Execution kind | Telemetry mode | Reason |",
        "|--------|---------------|----------|----------------|----------------|--------|",
    ]
    for row in all_rows():
        lines.append(
            f"| `{row.target}` | `{row.compiler_path}` | "
            f"`{row.executor_id or '-'}` | `{row.execution_kind}` | "
            f"{'`' + row.execution_mode + '`' if row.execution_mode else '-'} | "
            f"{row.reason} |"
        )
    lines += [
        "",
        "## Targets with no executable row",
        "",
        "These targets are recognized by the capability registry (so an artifact "
        "can carry them and lower correctly) but have no executable runtime row. "
        "`launch()` returns `runtime_status = \"unimplemented\"` when the target "
        "capability is present, or `\"missing_backend\"` otherwise — never silent "
        "success, never a fabricated output.",
        "",
        "```",
        ", ".join(unimplemented_targets()),
        "```",
        "",
        "## Known executor IDs",
        "",
        "| Executor ID | What it runs |",
        "|-------------|--------------|",
    ]
    for eid in sorted(KNOWN_EXECUTORS):
        lines.append(f"| `{eid}` | {KNOWN_EXECUTORS[eid]} |")
    lines.append("")
    return "\n".join(lines)


def write_dashboard() -> str:
    """Render and write the dashboard; returns the path."""
    from pathlib import Path
    p = (Path(__file__).resolve().parents[2].parent / "docs" / "audit"
         / "generated" / "runtime_execution_matrix.md")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_dashboard())
    return str(p)


def validate_against_capabilities() -> list[str]:
    """Cross-check: every executable row's target must exist in the capability
    registry, and every `_UNIMPLEMENTED_TARGETS` entry too. Returns a list of
    error strings (empty = OK). Used by the drift test."""
    errors: list[str] = []
    for row in all_rows():
        try:
            normalize_target(row.target)
        except ValueError:
            errors.append(f"matrix row target {row.target!r} is not in TARGET_CAPABILITIES")
        if row.executor_id is not None and row.executor_id not in KNOWN_EXECUTORS:
            errors.append(f"matrix row uses executor_id {row.executor_id!r} not in KNOWN_EXECUTORS")
    for t in unimplemented_targets():
        if t not in TARGET_CAPABILITIES:
            errors.append(f"_UNIMPLEMENTED_TARGETS entry {t!r} is not in TARGET_CAPABILITIES")
    return errors
