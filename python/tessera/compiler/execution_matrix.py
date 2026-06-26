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
from typing import Callable, Mapping, Optional

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
    "native_cpu":           "x86 AMX / native CPU runtime via the C runtime ABI",
    "jit_cpu_numpy":        "JIT CPU fallback via the numpy reference path",
    "rocm_wmma":            "AMD GPU RDNA WMMA matrix-core GEMM via the shipped "
                            "libtessera_rocm_gemm.so tessera_rocm_wmma_gemm_{f16,"
                            "bf16} C ABI symbol (HIPRTC-compiled for the device "
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
    "rocm_softmax_compiled": "AMD GPU RDNA row-reduction softmax the Tessera "
                            "compiler GENERATES (generate-rocm-softmax-kernel -> "
                            "ROCDL -> hsaco, in-process via tessera-opt), then HIP "
                            "loads + launches it. Stable softmax over the last "
                            "axis (one workgroup per row, LDS tree-reduce); the "
                            "first non-matmul/non-WMMA compiled ROCm kernel. "
                            "f32/f16/bf16 storage, f32 reduce",
    "rocm_norm_compiled":   "AMD GPU RDNA row-reduction rmsnorm / layer_norm the "
                            "Tessera compiler GENERATES (generate-rocm-norm-kernel "
                            "-> ROCDL -> hsaco, in-process via tessera-opt), then "
                            "HIP loads + launches it. Unweighted row normalize "
                            "over the last axis (one workgroup per row, LDS "
                            "tree-reduce of Σx and Σx²); handles "
                            "tessera.rmsnorm(_safe) + tessera.layer_norm by op "
                            "name. f32/f16/bf16 storage, f32 reduce",
    "rocm_activation_compiled": "AMD GPU RDNA flat elementwise activation the "
                            "Tessera compiler GENERATES "
                            "(generate-rocm-activation-kernel -> ROCDL -> hsaco, "
                            "in-process via tessera-opt), then HIP loads + "
                            "launches it. Standalone gelu / silu / relu (one "
                            "thread per element), dispatched by op name; "
                            "f32/f16/bf16 storage, f32 compute",
    "rocm_silu_mul_compiled": "AMD GPU RDNA SwiGLU gate-multiply the Tessera "
                            "compiler GENERATES (generate-rocm-silu-mul-kernel "
                            "-> ROCDL -> hsaco, in-process via tessera-opt), then "
                            "HIP loads + launches it. Flat 2-operand elementwise "
                            "silu(a)·b (one thread per element); the standalone "
                            "analog of the fused SwiGLU gate-multiply; "
                            "f32/f16/bf16 storage, f32 compute",
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
    "rocm_rope_compiled":   "AMD GPU RDNA rotary-position-embedding the Tessera "
                            "compiler GENERATES (generate-rocm-rope-kernel -> "
                            "ROCDL -> hsaco, in-process via tessera-opt), then HIP "
                            "loads + launches it. Interleaved-pair RoPE over "
                            "[M, D] (one workgroup per row); f32/f16/bf16",
    "nvidia_mma":           "NVIDIA GPU (consumer Blackwell sm_120) warp-level "
                            "mma.sync GEMM via the shipped libtessera_nvidia_gemm.so "
                            "tessera_nvidia_mma_gemm_{f16,bf16,tf32} C ABI symbol "
                            "(NVRTC-compiled for the device arch; f16/bf16/"
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
    # --- x86 / native CPU (AMX path) ---
    ("cpu", "native_cpu"): ExecutionRow(
        target="cpu", compiler_path="native_cpu",
        execution_kind="native_cpu", executable=True,
        executor_id="native_cpu", runtime_status="success",
        reason="CPU artifact runs through the x86 AMX / native CPU runtime."),
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
               "(tessera_rocm_wmma_gemm_{f16,bf16} C ABI symbol, HIPRTC-compiled "
               "for the device arch). Now the reference ORACLE + availability "
               "fallback for the compiled lane (rocm_compiled) — still directly "
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
               "oracle when the compiled lane is unavailable on the host.",
        execution_mode="hip_runtime"),
    # --- AMD ROCm GPU (COMPILED flash_attn lane — the matmul-L4 analog) ---
    # The compiler-GENERATED FA-2 forward (generate-wmma-flash-attn-kernel ->
    # ROCDL -> hsaco, in-process via tessera-opt) loaded + launched through HIP.
    # Reaches runtime.launch() exactly like the compiled GEMM; f16/bf16 storage,
    # f32 softmax + accumulate; validated vs a numpy attention reference.
    ("rocm", "rocm_flash_attn_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_flash_attn_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_flash_attn_compiled", runtime_status="success",
        reason="ROCm flash_attn artifact runs the COMPILER-GENERATED RDNA WMMA "
               "FA-2 forward: tessera-opt generates + serializes the kernel to "
               "hsaco in-process, then HIP loads + launches it. The attention "
               "analog of the compiled GEMM lane (rocm_compiled).",
        execution_mode="hip_runtime"),
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
    # Row-reduction softmax — the first non-matmul/non-WMMA compiled ROCm kernel.
    # Stable softmax over the last axis; f32/f16/bf16; validated vs numpy.
    ("rocm", "rocm_softmax_compiled"): ExecutionRow(
        target="rocm", compiler_path="rocm_softmax_compiled",
        execution_kind="native_gpu", executable=True,
        executor_id="rocm_softmax_compiled", runtime_status="success",
        reason="ROCm softmax artifact runs the COMPILER-GENERATED RDNA row-"
               "reduction kernel (stable softmax over the last axis, one "
               "workgroup per row, LDS tree-reduce): tessera-opt generates + "
               "serializes the kernel to hsaco in-process, then HIP loads + "
               "launches it. The first non-matmul/non-WMMA compiled ROCm kernel.",
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
    # SwiGLU gate-multiply silu(a)·b — flat 2-operand elementwise. vs numpy.
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
               "libtessera_nvidia_gemm.so, NVRTC-compiled for the device arch; "
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
