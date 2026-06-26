---
last_updated: 2026-06-25
audit_role: sub_audit
---

# ROCm Backend Audit

This document consolidates ROCm-specific audit material.

> **Real-hardware bring-up:** see [`STRIX_HALO_EXECUTION_PLAN.md`](STRIX_HALO_EXECUTION_PLAN.md)
> — the gfx1151 (RDNA 3.5 / Ryzen AI Max+ 395) target model is now grounded in the
> RDNA3.5 ISA (WMMA 16×16×16, no FP8), and the doc lays out the rung ladder to the
> first real non-Apple `backend_kernel` execution proof (emit → assemble → HIP-launch →
> execute-and-compare). This is the unblock for the "Still Open" / "Next Work" items below.
>
> **Design patterns from the AMD ROCm ecosystem:** see
> [`ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md`](ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md) — a
> source-grounded survey of AITER, ATOM, hipBLASLt, rocWMMA, Mori, Iris, XIO, and the
> AMD Gluon GEMM tutorial, with ranked, Tessera-mapped patterns (hardware-free IR/dispatch
> wins to adopt now, the GEMM perf ladder for Strix Halo bring-up, and the GPU-initiated
> comm track).
>
> **RDNA ISA data archive:** see [`docs/reference/isa/rdna/`](../../../reference/isa/rdna/README.md)
> — a structured, regenerable extraction of AMD's RDNA3 / RDNA3.5 / RDNA4 ISA guides:
> per-version instruction DB (opcodes, pseudocode), microcode encoding bit-fields, and a
> cross-version opcode matrix. The fast "does this op exist on my target" check before
> emitting — e.g. gfx1151 is RDNA3.5 (WMMA F16/BF16/IU8/IU4, **no** FP8/BF8 WMMA, which
> are RDNA4-only along with sparse SWMMAC). Regenerate via its `tools/build_archive.py`.

## Finished

- ROCm target-map generation exists at `../../generated/rocm_target_map.md`.
- ROCm/gfx target handling and HIP toolchain gates are represented.
- The execute-and-compare plan covers ROCm alongside NVIDIA.
- ROCm sub-arch gating was corrected so missing HIP toolchain is reported on
  the right axis.

## Box landed (2026-06-22) — toolchain gates cleared

A Strix Halo box (Ryzen AI Max+ 395) is now available: Ubuntu 24.04 (WSL2),
ROCm **7.2.4**, LLVM/MLIR **22.1.8**. The iGPU enumerates as its native
**`gfx1151`** (RDNA 3.5; 16×16×16 WMMA, no FP8 WMMA). *(During early bring-up
WSL transiently reported `gfx1100`, the RDNA 3 discrete profile; AMD's WSL
enablement resolved that on 2026-06-23 and `rocminfo` now shows `gfx1151`. The
Stage B/C/D notes below that say "gfx1100" are accurate bring-up provenance —
same WMMA family, so the kernels are identical.)* — see
[`STRIX_HALO_EXECUTION_PLAN.md`](STRIX_HALO_EXECUTION_PLAN.md) "Bring-up status".
Cleared: `rocminfo` enumerates without `HSA_OVERRIDE`; `hipcc` compiles WMMA;
ROCm lit suite 11/11; `tessera-opt`/`tessera-rocm-opt` build clean.

**Stage A increment landed (2026-06-22):** `lower-tile-to-rocm` was emitting
`tessera_rocm.mfma` for every arch — wrong for RDNA. Added a `tessera_rocm.wmma`
op + arch-keyed selection (`gfx11xx` → WMMA, CDNA → MFMA, no-FP8-on-RDNA gate
preserved) + a `llvm.amdgcn.wmma.contract` ROCDL marker, with lit fixtures.

**Stage B verified on the box (2026-06-22):** `rocdl_emit.py` (the AMD analog of
`ptx_emit.py`) already emits `llvm.amdgcn.wmma.*` LLVM IR and `llc`-assembles it
to real `v_wmma_*` AMDGCN; now runs for real on the box (LLVM 22.1.8 AMDGPU `llc`)
and is parametrized over **gfx1100** (the box target). Added `llc_object()` — the
GEMM lowers to a real AMD GPU ELF object (`EM_AMDGPU`); `_find_llc()` now finds the
apt.llvm.org `llc`. `test_rocdl_emit.py`: 96 passed, 0 skipped. **Note:** the MLIR
`--tessera-emit-rocdl` pipeline aborts here (`tessera-to-linalg` pass unregistered
in `tessera-opt`) — a separate follow-up; Stage B rides the direct LLVM-IR emitter.

**Stage C verified on the box (2026-06-22):** a real GEMM kernel **executes on
the gfx1100 device through the C-ABI launch bridge** (`tsrLaunchKernel` →
registered `tsrGpuLauncherFn` → HIP launch) and matches `A @ B`; unregistered
kernels still report `UNIMPLEMENTED`. First non-Apple kernel through the bridge.
Mirrors the Apple G7 proof; test `test_runtime_abi_rocm_launch_bridge.py`. Fixed
the runtime CMake HIP-include bug (`tessera_runtime` now links `hip::host`) and
handled the WSL `hipGetDeviceCount==0` quirk (probe-based skip).

**Stage D verified on the box (2026-06-22):** the real RDNA **WMMA** matrix op
(`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, the same `v_wmma_f32_16x16x16_f16`
`rocdl_emit.py` emits) executes on the device and produces a numerically correct
16×16×16 `f32←f16` GEMM, routed through the C-ABI bridge — maxerr ≈ 3e-8
standalone / < 1e-2 through the bridge. Test
`tests/unit/test_rocm_wmma_execute_compare.py`. This clears the *numerical-proof*
half of the `hardware_verified` contract.

## Manifest flip landed (2026-06-22) — rocm matmul row is `hardware_verified`

The shipped runtime symbol now exists, so the `backend_manifest` matmul row was
promoted `artifact_only → hardware_verified`:
- **`runtime_symbol`** = `tessera_rocm_wmma_gemm_f16` (the C-ABI entry point in
  `libtessera_rocm_gemm.so`; HIPRTC-compiles the RDNA WMMA kernel for the device
  arch at load — no hipcc-as-compiler needed).
- **`execute_compare_fixture`** = `tests/unit/test_rocm_wmma_runtime_symbol.py`
  (dlopens the shipped symbols, compares the WMMA GEMM to numpy across f16/bf16
  and several shapes; skip-clean with no AMD GPU / HIPRTC).
- Honest dtype scope (Decision #25): the row claims **{fp16, bf16}** + WMMA (not
  the CDNA MFMA shape/descriptor); `shape_envelope` is the general tiled GEMM
  (see the kernel-generalized note below).
- `rocm_target_map`: matmul → `hardware_verified | fp16,bf16`; `artifact_only`
  32 → 31.
- Lives in `_ROCM_HARDWARE_VERIFIED` (the ROCm analog of `_APPLE_GPU_KERNELS`).

**No audit inflation:** the per-primitive `backend_kernel` axis stays **474 open
/ 0 complete** — `primitive_is_complete(matmul)` is still `False` because x86 /
apple / nvidia / cpu rows are not `hardware_verified`. Only the **rocm target
row** is hardware-verified ("complete for this target", not the universal flip).

## runtime.launch() lane wired + kernel generalized (2026-06-22)

> **Scheduler context:** how the on-GPU work scheduler (MES) maps queues onto HW
> queues and enforces priority — the layer the launch bridge ultimately submits
> into — is written up in
> [`docs/reference/isa/rdna/mes/SCHEDULER_OVERVIEW.md`](../../../reference/isa/rdna/mes/SCHEDULER_OVERVIEW.md)
> (command surface in `mes/api.json`).

- **`runtime.launch()` now dispatches `target="rocm"` matmul to the GPU.** Added
  the `rocm_wmma` executor (`runtime._execute_rocm_wmma_artifact` + a cached lib
  loader + host probe), the `(rocm, rocm_wmma)` `native_gpu` row in
  `execution_matrix._MATRIX`, and dropped `rocm` from `_UNIMPLEMENTED_TARGETS`
  (named sub-arches stay — the shipped symbol HIPRTC-compiles for whatever arch
  the device enumerates). So `../../generated/runtime_execution_matrix.md` now
  carries an honest ROCm execution row. Proven end-to-end on the box: `launch()`
  of a rocm matmul artifact runs a real WMMA GEMM, maxerr ~5e-7
  (`test_rocm_launch_execute.py`).
  - The `@jit(target="rocm")` auto-stamp is intentionally **not** wired:
    `JitFn.is_executable` reads `compile_bundle.execution_kind` (compile-time),
    which a host runtime probe can't honestly drive. `launch()` is the wired
    lane (matches how Apple G7 earned its matrix row before full jit support).
- **Kernel generalized to tiled/K-looped GEMM + bf16.** `tessera_rocm_gemm.cpp`
  now does a general tiled GEMM (any positive M/N/K, 16×16 output tiles, K-loop,
  ragged edges zero-padded) and ships a second symbol
  `tessera_rocm_wmma_gemm_bf16`. The matmul manifest row claims `{fp16, bf16}`;
  the fixture validates f16/bf16 over 16³, 64×48×32, 17³, 128×96×64, 100×33×80.

## flash_attn WMMA executes on gfx1151 (2026-06-23) — second op after matmul

`flash_attn` now executes natively on the AMD GPU — the **second op after
matmul** to run on a non-Apple backend, taking ROCm from "one op executes" to
"the two ops that matter execute." The `backend_manifest` flash_attn rocm row is
`hardware_verified`:
- **`runtime_symbol`** = `tessera_rocm_wmma_flash_attn_f16` (+ `_bf16`) in the
  shipped `libtessera_rocm_flash_attn.so`; HIPRTC-compiles the RDNA WMMA kernel
  per head_dim at load.
- **kernel** = FA-2 forward, single wave per (query-tile-of-16, b·h): both QK^T
  and P@V on 16×16×16 WMMA, online (running max/sum) softmax, scores + output
  accumulator staged in LDS, causal masking + ragged Sq/Sk (zero-pad load + −inf
  score mask + bounds-checked store). head_dim must be a multiple of 16.
- **`execute_compare_fixture`** = `tests/unit/test_rocm_flash_attn_runtime_symbol.py`
  — vs a numpy attention reference across f16/bf16, head_dim 16/32/64/128, multi
  batch/head, ragged shapes, and causal. Measured on gfx1151: maxerr ~1e-4 (f16).
  Skip-clean with no AMD GPU / HIPRTC.
- Honest scope (Decision #25): this *hand-written runtime symbol* is WMMA
  `{fp16, bf16}` **forward only**. The compiler-generated lane now covers both
  forward and **backward** (+ measured perf ladders) — see the compiler-path
  item 10 below; the hand-written symbol stays the forward oracle.

✅ **flash_attn is now ALSO compiler-generated (2026-06-23) — the second
compiler-generated op (after matmul Stage K/L).** A `tessera_rocm.flash_attn`
directive (`head_dim`, `dtype`) is expanded by the new
`generate-wmma-flash-attn-kernel` pass into a fragment-materialized FA-2 forward
`gpu.func` — LDS-staged Q (gpu.func workgroup attributions), `QK^T` on WMMA
(`tessera_rocm.wmma`), online softmax (`math.exp` → LLVM exp via the math
ConvertToLLVM interface now registered in tessera-opt), `P@V` on WMMA, scores
staged in LDS to bridge the QK^T-accumulator → P@V-A-fragment layout — a faithful
MLIR re-emission of the hand-written kernel. It lowers through Stage J + Stage I
**entirely in-process (no mlir-opt)** and **executes on gfx1151 matching a numpy
attention reference** (maxerr < 2e-2 across head_dim 16/64, causal/non-causal,
ragged Sq/Sk) — `tests/unit/test_rocm_flash_attn_compiled.py`. The remaining glue
for a `runtime.launch()` executor-table lane (a flash_attn op-metadata contract +
executor + matrix row, the same additive step matmul took at L4) is not yet
wired, so no `runtime_execution_matrix` row is claimed (adding one with no
producer would be over-claiming).

**No audit inflation:** the per-primitive `backend_kernel` axis stays open —
`primitive_is_complete(flash_attn)` is still `False` (x86/apple/nvidia/cpu rows
are not all `hardware_verified`). Only the **rocm target row** flipped.

## MLIR→hsaco→execute loop closed on gfx1151 (2026-06-23) — Stage I

The architectural gap behind the two ops above: **they execute via hand-written
HIP C++ (HIPRTC at load) and touch zero MLIR** — the Tessera IR stack (Graph →
Schedule → Tile → Target IR → ROCDL) produced artifacts that never reached
silicon. The `--tessera-emit-rocdl` reachability fix (PR #86) exposed this — it
made the MLIR route *runnable*, which revealed it dead-ended at ROCDL text.

**Stage I closes the loop on a real kernel.** A `tessera.add` Graph-IR kernel now
compiles **through the IR stack to an executing hsaco**:

```
tessera kernel --(tessera-opt --tessera-emit-rocdl)--> gpu.module(ROCDL)
  --(mlir-opt: gpu.module(convert-gpu-to-rocdl,reconcile-unrealized-casts),
     rocdl-attach-target{chip=gfx1151}, gpu-module-to-binary)--> gpu.binary{hsaco ELF}
  --(extract + hipModuleLoadData + launch, memref-descriptor ABI)--> executes
```

Measured on gfx1151: **maxerr = 0.0 vs numpy** (f32 add + mul). The kernel that
executed was produced by the compiler's lowering pipeline, not hand-written.
Fixture: `tests/unit/test_rocm_mlir_to_hsaco.py` (skip-clean without tools/GPU;
also asserts the lowered module serializes to a real `\x7fELF` hsaco with no GPU).

**Division of labour:** `tessera-opt` owns the Tessera-specific lowering
(`--tessera-emit-rocdl`); the generic `gpu.module → hsaco` serialization is 100%
upstream and rides the platform `mlir-opt` (apt LLVM 22). In-process
serialization (no shell-out, for the `runtime.launch()` path) is deferred — it
needs the MLIR ROCDL target + LLVM AMDGPU codegen + lld linked into a Tessera
tool, a heavier link addressed when the compiled path becomes a runtime lane.

**Honest scope:** the loop is proven for a **scalar element-wise** kernel (no
WMMA) — it proves *the pipeline reaches silicon*, the smallest closed loop. The
"proper" Target IR path (`tessera_rocm.wmma`) still emits **contract markers**,
not real intrinsics; a real-WMMA GEMM through the full stack is **Stage J/K**
(below). The hand-written HIPRTC kernels remain the production execution path and
become the on-silicon **oracle** the compiled path validates against.

> **Update (2026-06-24): the staging seam is now runnable too.** Both halves of
> the Tile→ROCm seam that were contract-marker-only now have executable
> lowerings: `tessera_rocm.wmma` (Stage J → real `rocdl.wmma`, the GEMM/FA
> kernels) and `tessera_rocm.async_copy` (the new `--lower-rocm-async-copy` pass
> → a cooperative global→LDS copy loop; `tessera_rocm.wait` → `gpu.barrier`).
> Proven end-to-end on gfx1151: a global→LDS→global round-trip
> (`test_rocm_async_copy_runnable.py`) AND a real **WMMA GEMM tile with A/B
> staged through async_copy** that matches `A@B`
> (`test_rocm_gemm_staged_async_copy.py`). The markers remain for the
> IR-contract path when these passes are not run. *Measured caveat:* no current
> FA/GEMM shape is staging-bound, so this is infrastructure — the pipeline-routed
> double-buffer perf payoff lands only when a staging-bound kernel uses it.

## Landed — compiler-generated execution on gfx1151

> These bullets describe **executing, execute-compare-tested, `runtime.launch()`-
> reachable** kernels on the live gfx1151 box — they are *done*, not open. (They
> previously sat under a `## Still Open` heading that had become a changelog; the
> genuinely-open work is the `## Still Open` section that now follows.)

- **Compiler-generated attention family on gfx1151:** matmul, flash_attn
  (fwd+bwd), and now **GQA/MQA** (2026-06-24) all execute. GQA = flash_attn with
  the `gqa = true` directive attr: H query heads share G<H KV heads (query head h
  reads KV head h/kv_ratio, kv_ratio=H/G; =1 MHA, =H MQA), via two trailing
  runtime args + a grouped K/V base. Same FA-2 WMMA body; validated vs a numpy
  GQA reference across MQA / GQA / MHA-equivalence, causal+non-causal
  (`tests/unit/test_rocm_gqa_compiled.py`). **GQA forward also reaches
  `runtime.launch()`** — the `rocm_flash_attn_compiled` executor detects GQA from
  the operand shapes (K/V head count < Q) and builds the gqa kernel
  (`test_rocm_gqa_launch_execute.py`). **GQA BACKWARD too** (2026-06-24): the
  `flash_attn_bwd` `gqa = true` directive emits dQ/dK/dV where `_dkdv`
  accumulates dK/dV **atomically** across the kv_ratio query-head blocks sharing
  each KV head (host pre-zeros; B*H grid) — rel-err <5e-3 vs the numpy GQA
  backward (`test_rocm_gqa_bwd_compiled.py`). `multi_head_attention` is the
  flash_attn kernel itself (full multi-head, one wave per (16-q tile, b·h)).
- **Fused GEMM epilogue** (2026-06-24): the `tessera_rocm.wmma_gemm` directive
  gained `bias` (per-output-column add, a trailing length-N memref operand) and
  `activation` (`relu`/`gelu`/`silu`) knobs that `generate-wmma-gemm-kernel`
  fuses onto the f32 accumulator **before the store** — no intermediate D
  round-trip. `gelu` is the tanh approximation; `silu` = x·σ(x). The
  transcendentals lower through the same `math` → ROCDL path the flash_attn
  softmax uses (`gelu` tanh → `__ocml_tanh_f32`, `silu` exp → `llvm.intr.exp`),
  so they execute natively. Float-only — an int8/int4 directive carrying the
  epilogue is a named error, never a silent no-op. Validated on gfx1151 vs a
  numpy gemm+bias+activation oracle across {relu,gelu,silu}×{bias,no-bias}×
  {aligned, ragged} (`test_rocm_fused_epilogue_compiled.py`, 21 GPU cases) plus
  a GPU-free codegen/lowering gate for CI (`test_rocm_fused_epilogue_codegen.py`).
  The builder lane is `_build_compiled_gemm_hsaco(..., bias, activation)`. **Also
  reaches `runtime.launch()`** — the `rocm_compiled` matmul executor takes an
  optional third bias operand + an `activation` op-kwarg (no new executor /
  matrix row); float-only, integer dtype → structured `invalid_artifact`
  (`test_rocm_fused_epilogue_launch_execute.py`).
- **Sliding-window attention** (`attn_sliding_window`, 2026-06-24): the
  `tessera_rocm.flash_attn` directive gained a `sliding_window` attr (Mistral
  causal band of width W). Query position p attends only to keys in `(p - W, p]`:
  the generated kernel takes W as a trailing runtime arg, is implicitly causal,
  **skips KV tiles entirely below the window's lower edge** (reusing the causal
  tile-skip), and the per-element mask trims the boundary. Composes with `gqa`
  (W arg follows heads/kv_ratio). Reaches `runtime.launch()` via a `window`
  kwarg on the flash_attn op (no new executor / matrix row); builder lane
  `_build_compiled_flash_attn_hsaco(..., sliding_window)`. Validated on gfx1151
  vs a numpy windowed-attention oracle across varied W / S / D incl. W≥S
  (= plain causal) and a window-differs-from-full guard
  (`test_rocm_sliding_window_compiled.py`, 6 GPU cases) + a GPU-free codegen
  gate (`test_rocm_sliding_window_codegen.py`).
- **Logit soft-capping** (Gemma-2, 2026-06-24): the `tessera_rocm.flash_attn`
  directive gained a `logit_softcap` attr. Each scaled score is passed through
  `cap·tanh(S/cap)` before masking + softmax (bounding logits to `(-cap, cap)`),
  applied in the same step that scales the score and reusing the `tanh` →
  `__ocml_tanh_f32` lowering. `cap` is a trailing f32 runtime arg; composes with
  `gqa` + `sliding_window` (cap is last). Reaches `runtime.launch()` via a
  `logit_softcap` kwarg on the flash_attn op (no new executor / matrix row);
  builder lane `_build_compiled_flash_attn_hsaco(..., logit_softcap)`. Validated
  on gfx1151 vs a numpy soft-capped-attention oracle (varied cap / S / D, causal
  + non-causal) + a softcap-differs-from-uncapped guard
  (`test_rocm_logit_softcap_compiled.py`, 4 GPU cases) + a GPU-free codegen gate
  (`test_rocm_logit_softcap_codegen.py`).
- **Linear attention** (`linear_attn`, 2026-06-25): the **first non-softmax**
  attention algorithm on RDNA — a new `tessera_rocm.linear_attn` directive +
  `generate-wmma-linear-attn-kernel` pass emitting the quadratic-parallel form
  `O = (φ(Q)φ(K)ᵀ ⊙ causal) @ V`. Structurally flash_attn forward **minus the
  online softmax**: it computes `A = φ(Q)φ(K)ᵀ` (WMMA), masks **multiplicatively**
  (masked → 0, not −∞), stages `A` in LDS, and accumulates `O += A@V` (WMMA, the
  same layout bridge), with **no final divide** (unnormalized). Feature map φ ∈
  {identity, relu, polynomial_2 (x²)} applied on the loaded Q/K fragments; square
  head dim; causal + non-causal. **Decay-masked variants** (lightning_attention =
  identity + decay; (degree-2) retention = poly2 + decay): a `decay` mode scales
  each score by `λ^(i-j)` over the causal band (per-head λ as a trailing f32
  `log_decay` arg, via `exp((i-j)·log_decay)`), matching the reference's
  `dc[i]/dc[j]` ratio for a per-head-constant decay. New `runtime.launch()` lane
  `rocm_linear_attn_compiled` (own executor + execution-matrix row, since it is a
  distinct op, unlike the flash_attn-family flags); builder
  `_build_compiled_linear_attn_hsaco(..., feature_map, decay)`. The executor
  **dispatches by op name** — `tessera.linear_attn` (feature_map from kwargs) plus
  the decay siblings `tessera.lightning_attention` (pins identity + decay) and
  `tessera.retention` (pins degree-2 x² + decay; deg≠2 is a named error) — so
  those named ops actually launch, not just `linear_attn`-with-kwargs. Validated on
  gfx1151 vs the canonical reference `O = (φ(Q)φ(K)ᵀ ⊙ tril [⊙ λ^(i-j)]) @ V`
  (`_apple_gpu_dispatch_linear_attn` math) across identity/relu/poly2 ×
  causal/non-causal × ragged + lightning/retention decay, + a
  causal-differs-from-full guard + a K/V-mismatch rejection
  (`test_rocm_linear_attn_compiled.py`)
  and a GPU-free codegen gate (`test_rocm_linear_attn_codegen.py`).
- **flash_attn**: compiler-generated forward + backward both execute on gfx1151
  with measured perf ladders, reachable through `runtime.launch()` (the
  `rocm_flash_attn_compiled` lane, #100). Landed perf rungs: `_pre`→WMMA (~3.4×
  bwd), causal tile-skip (~1.7× causal bwd), forward sQ-drop + rescale-fusion
  (~1.7× fwd). Remaining (measured low-value): KV-tile pipelining — the kernels
  are occupancy/LDS-bound, not staging-bound, so it is unlikely to pay; the
  runnable `async_copy` that would enable it now exists (#101) for when a
  staging-bound kernel appears.
- **Softmax** (`softmax`, 2026-06-25): the **first non-matmul / non-WMMA**
  compiler-generated ROCm kernel — a new `tessera_rocm.softmax` directive +
  `generate-rocm-softmax-kernel` pass emitting a **row-reduction** kernel (one
  workgroup per row, lanes stride the last axis and tree-reduce the row max then
  the row sum through LDS, two passes + an in-place divide). Numerically-stable
  `O[m,:] = exp(X[m,:] − max) / Σ exp(...)`, reductions in f32 regardless of
  storage dtype; `exp` rides the same `math` → ROCDL path the flash_attn softmax
  uses. New `runtime.launch()` lane `rocm_softmax_compiled` (own executor +
  execution-matrix row); accepts `tessera.softmax` + `tessera.softmax_safe`,
  axis=-1, f32/f16/bf16. Validated on gfx1151 vs the numpy softmax reference
  (`_apple_gpu_dispatch_softmax` math) across f32/f16/bf16 × varied M/K incl.
  K>256, ragged, and rank-3 (`test_rocm_softmax_compiled.py`) + a GPU-free
  codegen gate (`test_rocm_softmax_codegen.py`). Status `compiled`; it also
  flips the curated `softmax`×`rocm` cell to ✅ in `op_target_conformance.md`.
- **Row reduction** (`sum`/`mean`/`max`/`min`, 2026-06-25): the ROCm analog of
  the x86 AVX-512 reduction lane — a `tessera_rocm.reduce` directive +
  `generate-rocm-reduce-kernel` pass emitting a row-reduction kernel (one
  workgroup per row, lanes stride the last axis and tree-reduce through LDS;
  identity-seeded combine = +/max/min, mean divides by K). The runtime folds an
  arbitrary reduced `axis` to a `[outer, inner]` last-axis reduction by
  transposing the reduced axes to the end (matching `_apple_gpu_dispatch_reduce`);
  `keepdims` supported. New `runtime.launch()` lane `rocm_reduce_compiled`;
  handles `tessera.sum`/`mean`/`max`/`min` (+ `amax`/`amin`) by op name; f16/bf16/
  f32 storage, f32 reduce. Validated on gfx1151 vs numpy across dtype × shape ×
  axis incl. rank-3 + reduce-all + keepdims (`test_rocm_reduce_compiled.py`) + a
  GPU-free codegen gate. The CPU half (AVX-512) landed in the x86 backend
  (`avx512_reduce_f32.cpp`) — so the reduction family now has a real optimized
  kernel on both devices we have hardware for. Status `compiled`.
- **Elementwise unary math** (S2 scalar-math / stability family, 2026-06-25):
  a `tessera_rocm.unary` directive + `generate-rocm-unary-kernel` pass emitting
  a flat per-element kernel (one thread per element), the unary sibling of the
  activation lane. Covers `exp`/`log`/`sqrt`/`rsqrt`/`reciprocal`/`abs`
  (`absolute`)/`sign`/`erf`/`tanh`/`sigmoid`/`log1p`/`expm1`/`softplus`
  (softplus stable: `log1p(exp(-|x|)) + max(x,0)`); transcendentals lower
  through the `math` → ROCDL path. New `runtime.launch()` lane
  `rocm_unary_compiled`, dispatched by op name; f16/bf16/f32 storage, f32
  compute. Validated on gfx1151 vs numpy across kind × dtype × shape incl.
  rank-3 (`test_rocm_unary_compiled.py`) + a GPU-free codegen gate. The CPU
  half landed in the x86 backend as an AVX-512 kernel for the **algebraic
  subset** (`sqrt`/`rsqrt`/`reciprocal`/`abs`/`neg`/`sign`, direct intrinsics,
  no polynomial approx — `avx512_unary_f32.cpp`, validated standalone); the
  transcendentals stay numpy-reference on CPU (no fused x86 claim). Status
  `compiled`.
- **Elementwise binary arithmetic** (S2 binary-arithmetic family, 2026-06-26):
  a `tessera_rocm.binary` directive + `generate-rocm-binary-kernel` pass emitting
  a flat 2-operand per-element kernel (one thread per element), the binary
  sibling of the unary-math lane. Covers `sub`/`div`/`pow`/`maximum`/`minimum`;
  `maximum`/`minimum` are IEEE NaN-propagating (`arith.maximumf`/`minimumf`,
  matching `numpy.maximum`/`minimum`), `pow` lowers through the `math` → ROCDL
  path. New `runtime.launch()` lane `rocm_binary_compiled`, dispatched by op
  name; f16/bf16/f32 storage, f32 compute. Validated on gfx1151 vs numpy across
  kind × dtype × shape incl. rank-3 + NaN-propagation
  (`test_rocm_binary_compiled.py`) + a GPU-free codegen gate. The CPU half
  landed in the x86 backend as an AVX-512 kernel for the **direct-intrinsic
  subset** (`sub`/`div`/`maximum`/`minimum`, with NaN-propagation handled via an
  unordered-compare blend — `avx512_binary_f32.cpp`, validated standalone); `pow`
  is transcendental and stays numpy-reference on CPU (no fused x86 claim). Status
  `compiled`.
- **rmsnorm / layer_norm** (2026-06-25): the row-reduction siblings of the
  softmax kernel — a `tessera_rocm.norm` directive + `generate-rocm-norm-kernel`
  pass (one workgroup per row). rmsnorm tree-reduces Σx² in one pass; layer_norm
  uses a **stable two-pass squared-deviation** reduction (μ = Σx/K, then
  var = Σ(x−μ)²/K — never the cancellation-prone E[x²]−E[x]²). **Unweighted**
  over the last axis (the bare ops; the affine is a separate mul/add): rmsnorm
  `x / sqrt(mean(x²)+eps)`, layer_norm `(x − μ) / sqrt(var + eps)`. Reductions in
  f32 regardless of storage; `eps` is a trailing f32 runtime arg. New
  `runtime.launch()` lane `rocm_norm_compiled` (own executor + matrix row);
  accepts `tessera.rmsnorm`, `tessera.rmsnorm_safe` (tighter default eps),
  `tessera.layer_norm` by op name; f32/f16/bf16. Status `compiled`. Validated on
  gfx1151 vs the unweighted numpy reference across all three ops × f32/f16/bf16 ×
  varied M/K incl. K>256, ragged, rank-3, and a large-offset/small-variance
  layer_norm row (`test_rocm_norm_compiled.py`) + a GPU-free codegen gate
  (`test_rocm_norm_codegen.py`).
- **Standalone activations + RoPE** (`gelu`/`silu`/`relu`, `rope`, 2026-06-25):
  two pointwise compiler-generated kernels closing the activations + positional
  groups. (1) `tessera_rocm.activation` + `generate-rocm-activation-kernel` — a
  flat per-element kernel for **standalone** gelu (tanh approx) / silu (x·σ(x)) /
  relu (the same activations the GEMM fused epilogue applies, now as their own
  ops), f32 compute. Lane `rocm_activation_compiled`, dispatched by op name
  (`tessera.gelu`/`silu`/`relu`). (2) `tessera_rocm.rope` +
  `generate-rocm-rope-kernel` — interleaved-pair rotary position embedding over
  `[M,D]` (`O[2p]=e·cos−o·sin`, `O[2p+1]=e·sin+o·cos`, angle from the even-indexed
  `theta`), one workgroup per row, f32 cos/sin. Lane `rocm_rope_compiled`. Both
  f32/f16/bf16, status `compiled`; a half-precision `x` may carry an **fp32 angle
  table** (the common `nn.RotaryEmbedding` default) — theta is cast to `x`'s
  storage on the device copy, so mixed float storage is accepted (only a
  non-floating theta is rejected). Validated on gfx1151 vs the numpy references
  (`_apple_gpu_dispatch_gelu`, `_runtime_rope`) across dtype × shape incl. >1
  block, ragged, rank-3 (`test_rocm_activation_compiled.py`,
  `test_rocm_rope_compiled.py`) + a GPU-free codegen gate
  (`test_rocm_activation_rope_codegen.py`). (`relu` executes via the activation
  lane by op name but has no separate rocm target-map row.)
- **silu_mul (SwiGLU gate-multiply)** (2026-06-25): a flat 2-operand elementwise
  kernel `silu_mul(a,b) = silu(a)·b = (a/(1+exp(−a)))·b` — a direct sibling of
  the activation lane (`tessera_rocm.silu_mul` + `generate-rocm-silu-mul-kernel`,
  one thread per element). The **standalone** analog of the gate-multiply the
  fused SwiGLU MLP applies in-register. f32 compute regardless of storage. New
  `runtime.launch()` lane `rocm_silu_mul_compiled` (own executor + matrix row);
  op name `tessera.silu_mul`; f32/f16/bf16, status `compiled`. Validated on
  gfx1151 vs the numpy reference across dtype × shape incl. >1 block, multi-D
  (`test_rocm_silu_mul_compiled.py`) + a GPU-free codegen gate (in
  `test_rocm_activation_rope_codegen.py`). **Closes the norms/activations group.**
- **alibi (ALiBi positional bias)** (2026-06-25): a flat elementwise generator
  for the positional-bias tensor `bias[h,i,j] = slope[h]·(j−i)` of shape
  `[H, S, S]` — the positional sibling of the rope lane (`tessera_rocm.alibi` +
  `generate-rocm-alibi-kernel`, one thread per element, gid→(h,i,j) decode). The
  per-head `slope` comes from a length-H f32 buffer; the runtime fills the
  default `2^(-8(k+1)/H)` ramp when the caller passes none (an explicit `slopes`
  operand overrides it). f32 compute, f16/bf16/f32 output. New `runtime.launch()`
  lane `rocm_alibi_compiled` (own executor + matrix row); op name `tessera.alibi`
  with `num_heads`/`seq_len` kwargs; status `compiled`. Validated on gfx1151 vs
  the `nn.functional.alibi` numpy reference across H×S incl. >1 block, default +
  explicit slopes (`test_rocm_alibi_compiled.py`) + a GPU-free codegen gate (in
  `test_rocm_activation_rope_codegen.py`). **Closes the positional group.**
- **matmul-family chains** (`batched_gemm`, `linear_general`, `qkv_projection`,
  `factorized_matmul`, `einsum`, 2026-06-25): all five execute on the **same
  COMPILER-GENERATED WMMA GEMM kernel** (the `rocm_compiled` spine), reshaped /
  batched / split in the runtime — the matmul analog of how flash_attn GQA/MQA
  reuse the FA kernel (no new MLIR pass). `batched_gemm` loops the gemm over
  leading batch dims; `linear_general` (axis=-1) reshapes [...,K]→[M,K] + gemm +
  bias; `qkv_projection` is the packed x@W_qkv projection (the 3-way split is a
  trivial host view); `factorized_matmul` is the GPU matmul + an exact host
  rank-r SVD-truncate epilogue; `einsum` maps single-contraction two-operand
  specs to a (batched) gemm (canonicalize + transpose) and emits a stable
  *unsupported* diagnostic otherwise. Shared lane `rocm_matmul_family_compiled`
  (one executor + matrix row); f16/bf16, f32 accumulate; status `compiled`.
  Validated on gfx1151 vs numpy across dtype × shape incl. multi-batch
  (`test_rocm_matmul_family_compiled.py`). **Closes the matmul-family group.**
- **exotic-attention compositions** (`gated_attention`, `mla_decode`,
  `mla_decode_fused`, 2026-06-25): the attention analog of the matmul-family
  lane — each is built by COMPOSING already-compiled kernels, no new MLIR pass.
  `gated_attention` = the WMMA flash_attn kernel × an elementwise sigmoid-gate;
  `mla_decode` = latent K/V projections (WMMA GEMM) + flash_attn; `mla_decode_fused`
  = down/up projections (`c=x@w_dkv; K=c@w_uk; V=c@w_uv` on the WMMA GEMM) +
  flash_attn. Shared lane `rocm_exotic_attn_compiled` (one executor + matrix row);
  f16/bf16, f32 softmax+accumulate; status `compiled`. Validated on gfx1151 vs the
  numpy attention reference (`test_rocm_exotic_attn_compiled.py`).
- **recurrent DeltaNet scan** (`gated_deltanet`, `kimi_delta_attention`,
  `modified_delta_attention`, 2026-06-25): the **first RECURRENT compiled ROCm
  kernel** — a dedicated `tessera_rocm.deltanet` directive +
  `generate-rocm-deltanet-kernel` pass emitting a causal **sequential-scan**
  kernel (NOT a composition; no existing lane builds it). One workgroup per
  `(b,h)`, one thread per value-column `e` holding the state column `Ŝ[:,e]` in
  LDS — the per-step matvecs are independent per-thread `d`-loops, so the only
  barriers are the cooperative `k`/`q` load and the modified-`‖target‖`
  reduction. Realizes `_delta_attention_impl` (target = `V − α·Ŝᵀk` on erase;
  `Ŝ *= α` decay; `Ŝ += β·outer(k,target)` with the optional modified
  `/(1+‖k‖·‖target‖)` bound; `O = Q@Ŝ`; sigmoid-gate). `erase`/`modified`/gate/
  beta/decay are compile-time flags (`modified` from the op name); `d_qk`/`d_v`
  compile-time; f16/bf16/f32 storage, f32 compute; causal-only, state=None. Lane
  `rocm_deltanet_compiled`; status `compiled`. Validated on gfx1151 vs the f64
  numpy reference across all three ops × f32/f16/bf16 × flag combos
  (`test_rocm_deltanet_compiled.py`) + a GPU-free codegen gate.

## Still Open

- **The rest of the RDNA op surface stays `artifact_only`** (IR/MFMA artifact
  emits; not yet a compiler-generated executing kernel). The not-yet-executing
  groups (see `../../generated/rocm_target_map.md` for the live list):
  - **norms / activations:** *(group closed)* — `softmax`/`softmax_safe`,
    `rmsnorm(_safe)`, `layer_norm`, `gelu`, `silu`, and `silu_mul` (SwiGLU) all
    execute as compiled kernels (see Landed above).
  - **positional:** *(group closed)* — `rope` and `alibi` both execute as
    compiled kernels (see Landed above).
  - **matmul-family chains:** *(group closed)* — `batched_gemm`, `einsum`,
    `factorized_matmul`, `linear_general`, `qkv_projection` all execute on the
    shared WMMA GEMM lane (see Landed above).
  - **exotic attention:** the **composable** members
    (`gated_attention`/`mla_decode`/`mla_decode_fused`) and the **recurrent
    DeltaNet** family (`gated_deltanet`/`kimi_delta_attention`/
    `modified_delta_attention`, via the sequential-scan kernel) now execute (see
    Landed above). One genuinely-open member remains:
    - **block-sparse** — `deepseek_sparse_attention` needs a top-k block-select
      index branch + a sparse KV gather; the dense flash kernel only covers the
      `top_k == all blocks` degenerate case, so claiming it "compiled" would be
      audit inflation (Decision #25). (The sliding + compressed branches reduce
      to the compiled flash/pool lanes; the top-k gather is the open piece.)
    - **`hybrid_attention`** is a per-layer policy wrapper; its lightning / MLA /
      Kimi-Delta / gated-DeltaNet slots are now all compiled, so it's composable
      — a thin dispatch lane is the only remaining work (no new kernel).
- **CDNA (MI300X / MI325X) execution is hardware-gated** — distinct MFMA shape
  table + FP4/FP6; no device available. The named ROCm sub-arches
  (gfx90a/942/950/1100/1151/1200) stay in `_UNIMPLEMENTED_TARGETS`; the generic
  `rocm` lane covers execution via HIPRTC for the live device, and gfx1151 (the
  box's own arch) is listed there too so the classification is total (no silent
  `lookup() -> None`).
- **flash_attn KV-tile pipelining — parked (measured low-value).** The FA
  kernels are occupancy/LDS-bound, not staging-bound, so double-buffered KV
  staging is unlikely to pay on this APU; the runnable `async_copy` that would
  enable it exists (#101) for when a staging-bound kernel appears.

## Perf ladder — rung 1 landed (2026-06-22)

The GEMM kernel moved off correctness-first naive tiling onto a measured ladder
(grounded in the AMD Gluon v0→v9 tutorial, `ROCM_PATTERNS_FROM_AMD_ECOSYSTEM.md`
§B1/§B2). Rung 1 = **output-tile register blocking** (each wave computes an
MT×NT grid of 16×16 WMMA tiles, reusing fragments). Shipped tiling **2×4** is
**~2.3× over the 1×1 naive baseline** at 1024³/2048³ on gfx1151 (Ryzen AI
Max+ 395, RDNA 3.5). The Gluon lesson reproduced: `2×2` *regressed below* naive; the
non-square `2×4` won — tile shape is the lever. Measured by the device-timed
`tessera_rocm_wmma_gemm_f16_bench` symbol + `benchmarks/rocm/
benchmark_rocm_wmma_gemm.py --ladder`; see STRIX_HALO_EXECUTION_PLAN.md Stage F.

**Rung 2 — LDS staging (multi-wave workgroup): implemented, measured, did NOT
win on this APU.** A WM×WN-wave workgroup cooperatively stages A/B K-panels into
LDS, reused across waves. Numerically correct (shipped `..._lds` symbol +
fixture), but **register blocking (rung 1) still wins** on Strix Halo: LDS loses
at 512³/1024³/4096³ and edges only +6% at 2048³ — unified memory means global
bandwidth isn't the bottleneck LDS targets. Production stays rung-1 2×4; rung-2
is kept behind `benchmark_rocm_wmma_gemm.py --lds` as the substrate for rung-3
software pipelining and for discrete RDNA/CDNA where it should pay off. (The
Gluon v6 lesson generalized: measure the "obvious" optimization, don't assume.)

**Rung 3 — 2-stage software pipelining (double-buffered LDS): implemented,
measured, narrow-window win, NOT promoted.** Prefetch K-panel k+1 into a second
LDS buffer while computing panel k. Correct (shipped `..._pipe` symbol +
fixture); beats rung-1 by **~8% only in a 1024³–2048³ window** (size-dependent
best config), loses at 512³ and ≥3072³. Production stays rung-1. Reproduce with
`benchmark_rocm_wmma_gemm.py --pipe`.

**Synthesis:** the memory-staging rungs (2 LDS, 3 pipelined LDS) and the
zero-copy path all give *at most* a narrow single-digit win on this APU — unified
LPDDR5x means global bandwidth isn't the bottleneck they target, and the rung-1
register kernel is already compute/occupancy-bound (~11 of ~59 TFLOP/s f16 WMMA
peak). **Next real lever = occupancy + WMMA issue/scheduling** (VGPR-budgeted
macro-tiling, dual-issue), not staging. The rung-2/3 symbols are kept for
discrete RDNA/CDNA (where global *is* the bottleneck) and as references.

## Memory — APU zero-copy host buffers (opt-in; windowed win)

On Strix Halo host and device share the same physical LPDDR5x, so the explicit
H2D/D2H copies in the runtime symbol are physically redundant. Added an **opt-in**
zero-copy path (`TESSERA_ROCM_ZEROCOPY=1`): `hipHostRegister` device-maps the
caller's host buffers (`hipHostGetDevicePointer`) and the kernel reads/writes
them directly — no `hipMalloc`, no `hipMemcpy`. Correct everywhere (subprocess
fixture `test_zerocopy_path_matches_numpy_subprocess`); falls back to the copy
path if registration is unsupported (rc=4). This changes **end-to-end
`launch()` latency only**, not the kernel-only perf ladder.

**Measured (gfx1151/WSL, CPU-wall, end-to-end per call, copy ÷ zero-copy):**

| size | copy ms | zero-copy ms | winner |
|------|--------:|-------------:|--------|
| 256³  | 0.54 | 2.40 | copy (~4×) |
| 512³  | 0.68 | 2.77 | copy (~4×) |
| 768³  | 5.44 | 3.52 | **zero-copy 1.5×** |
| 1024³ | 7.17 | 4.15 | **zero-copy 1.7×** |
| 1536³ | 10.45 | 8.47 | **zero-copy 1.2×** |
| 2048³ | 13.86 | 10.13 | **zero-copy 1.4×** |
| 4096³ | 53.95 | 81.99 | copy (zc 1.5× slower) |

A **windowed** win (~768³–2048³), not universal: below it, `hipHostRegister`
per-call pinning overhead dominates the tiny kernel; above it, the kernel's
repeated fragment re-reads through page-mapped, **non-coherent** host memory
(`Coherent Host Access: FALSE`, XNACK off) lose locality vs device-local
staging. Both register *and* malloc are Windows-driver round-trips under WSL, so
the crossover is WSL-specific; bare-metal ROCm would differ. **Kept opt-in /
off by default** — the copy path stays the portable correctness baseline. Bench:
`tessera_rocm_wmma_gemm_f16_e2e_bench(M,N,K,iters,mt,nt,zerocopy,*ms)`. A
size-gated auto-select is a possible follow-up but premature without bare-metal
data.

## Next Work

1. ✅ **Stage B — assemble (2026-06-22):** `rocdl_emit.py` emits the WMMA GEMM
   LLVM IR and `llc -mcpu=gfx1100` lowers it to real `v_wmma_*` AMDGCN + an
   AMD GPU ELF object; verified on the box, gfx1100 + gfx1151.
2. ✅ **Stage C — launch (2026-06-22):** a GEMM executes through the C-ABI
   bridge on gfx1100 and matches `A @ B`; runtime HIP build fixed.
3. ✅ **Stage D — prove (2026-06-22):** the WMMA `f32←f16` GEMM executes through
   the bridge and matches a host reference (`test_rocm_wmma_execute_compare.py`).
4. ✅ **Ship the ROCm WMMA GEMM runtime symbol (2026-06-22):**
   `libtessera_rocm_gemm.so` exports `tessera_rocm_wmma_gemm_{f16,bf16}` (HIPRTC
   at load), with the `test_rocm_wmma_runtime_symbol.py` execute-compare fixture.
   The matmul manifest row is `hardware_verified` (rocm target).
5. ✅ **Wire the executor into `runtime.launch()` + generalize the kernel
   (2026-06-22):** the executable `("rocm", "rocm_wmma")` row is earned in the
   generated `runtime_execution_matrix` (`hip_runtime`); kernel extended to
   tiled/K-looped GEMM + bf16. (The per-primitive `backend_kernel` flip still
   needs *all* targets — out of scope for a single box.)
6. ✅ **Occupancy lever — size-adaptive macro-tile (2026-06-23):** a wider
   `(MT,NT)` sweep found a bigger register macro-tile (**3×4 = 12 WMMA tiles/
   wave**) beats production 2×4 by **+20–25% at 3072³/4096³** and never regresses
   from 1024³ up; **4×4 (16 tiles) regresses sharply — the VGPR/occupancy cliff**.
   Promoted: the shipped symbol is size-adaptive (`min(M,N,K) ≥ 1024 → 3×4`, else
   2×4). Confirms Gluon B1 "register-budget tiling is the lever," with a cliff.
   See STRIX_HALO_EXECUTION_PLAN.md Stage H. (Direct VGPR readout unavailable on
   this WSL box — rocprof v1 unsupported, rocprofv3 crashes vs HIPRTC — so the
   cliff is read off the TFLOP/s curve.)
7. ✅ **Extend hardware execution beyond matmul — `flash_attn` (2026-06-23):**
   `libtessera_rocm_flash_attn.so` exports `tessera_rocm_wmma_flash_attn_{f16,
   bf16}` (FA-2 forward, both matmuls on WMMA, online softmax, causal + ragged),
   `hardware_verified` with `test_rocm_flash_attn_runtime_symbol.py`. Second op
   after matmul to execute on ROCm. (Forward only; no perf ladder; no launch lane
   — see the flash_attn section above.)
8. ✅ **`--tessera-emit-rocdl` MLIR-graph route reachable on the box (2026-06-23):**
   the tessera-opt CMake gate excluded core Tessera IR (+ `tessera-to-linalg`)
   from *every* ROCm-backend build — the CUDA carve-out gave real NVIDIA builds
   the full route but HIP was missing the symmetric one. Extended it
   (`AND NOT TESSERA_ENABLE_HIP`) so a real HIP build links `TesseraIR`/
   `TesseraPasses`; `--tessera-emit-rocdl`/`-nvvm` now lower a tessera kernel to
   `gpu.module` + `rocdl.kernel`/`nvvm.kernel` (`test_gpu_emit_nvvm.py` 3/3). The
   hardware-free ROCm *artifact* build (HIP off) stays lean. The direct
   `rocdl_emit.py` LLVM-IR emitter remains the Stage B path; this adds the
   MLIR-graph route alongside it.
9. ✅ **MLIR→hsaco→execute loop closed — Stage I (2026-06-23):** a `tessera.add`
   Graph-IR kernel compiles through `--tessera-emit-rocdl` → `gpu-module-to-binary`
   → hsaco → `hipModuleLoadData` → executes on gfx1151, **maxerr = 0.0 vs numpy**.
   The compiler's pipeline produced the executing kernel (not hand-written HIP).
   Fixture `test_rocm_mlir_to_hsaco.py`. Scalar element-wise only — proves the
   pipeline reaches silicon. See the Stage I section above.

### Compiler-path roadmap (the real next steps — close the IR-stack/execution gap)

The hand-written HIPRTC kernels execute but bypass the IR stack; the IR stack
lowers but (for matmul/WMMA) doesn't execute. Converge them:

- ✅ **Stage J — real WMMA in the Target IR (2026-06-23).** `lower-tessera-target-to-rocdl`
  now lowers a `tessera_rocm.wmma` carrying **real RDNA fragment vectors**
  (`vector<16x{f16,bf16}>` A/B, `vector<8xf32>` acc) to the real
  `rocdl.wmma.f32.16x16x16.{f16,bf16}` op (bf16 bitcast to `<16xi16>`, the RDNA
  ABI), which `mlir-translate` lowers to `llvm.amdgcn.wmma.f32.16x16x16.*` — the
  **same intrinsic** `rocdl_emit.py` emits. Abstract/scalar WMMA (contract-level
  IR, no fragments) still lowers to the marker. Validated by
  `tests/unit/test_rocm_target_wmma_lowering.py` (5/5): the MLIR-pass LLVM-IR
  intrinsic is cross-checked against `rocdl_emit.wmma_intrinsic(dtype)` so the two
  emitters can't silently diverge — **folds the Python side-emitter (path 4) into
  the MLIR pass (path 3)**. *(Note: the ROCm lit suite under
  `Tessera_ROCM_Backend/test/` passes 12/12 on lit 18 + llvm-22 lit — the old
  `%trop`/`%t` substitution-collision concern no longer bites (modern lit sorts
  substitutions longest-first; the site config also now `insert`s `%trop` ahead
  of the built-in `%t` to be robust across lit versions). Stage J additionally
  validates via a Python fixture. The lit job stays opt-in in CI by design — it
  needs `tessera-opt` built.)*
- **Stage K — a real GEMM through the full stack vs. the oracle.** Two steps:
  - ✅ **Step 1 (2026-06-23) — chain + layout proven, oracle-matched.** A
    16×16×16 WMMA GEMM expressed at the **Target-IR level** (`tessera_rocm.wmma` +
    RDNA fragment load/store layout) compiles through Stage J (→ real `rocdl.wmma`)
    + Stage I (→ hsaco) and **executes on gfx1151, bit-identical to the
    hardware_verified hand-written oracle** (vs numpy ~2e-7, vs oracle **0.0**).
    This locks the exact fragment layout the generating pass must produce and
    proves the whole Target-IR→ROCDL→hsaco→execute chain on a real WMMA GEMM.
    Fixture `tests/unit/test_rocm_wmma_gemm_via_mlir.py`. (Also: registered the
    `vector` dialect in tessera-opt so a gpu kernel carrying WMMA fragment vectors
    parses + lowers there.)
  - ✅ **Step 2 — the generating pass: the compiler GENERATES the GEMM
    (2026-06-23).** New `tessera_rocm.wmma_gemm` matmul-directive op (m/n/k +
    name) + the `generate-wmma-gemm-kernel` pass that expands it into a
    fragment-materialized `gpu.func` (fragment loads + real `tessera_rocm.wmma` +
    accumulator stores — **the kernel body is emitted by the pass, not authored**),
    fully unrolled. The full chain `directive → generate → Stage J → Stage I →
    hsaco → launch` executes on gfx1151 **bit-identical to the hand-written
    oracle** (vs numpy ~2e-7, vs oracle **0.0**). The milestone — *the Tessera
    compiler, not a hand-written kernel, produced the executing GEMM* — is met for
    the 16×16×16 tile (Stage L1 then generalized the kernel to any runtime shape).
    Fixture `tests/unit/test_rocm_wmma_gemm_generated.py`. Wiring Graph
    `tessera.matmul` → Tile → the `wmma_gemm` directive is the remaining front-end
    glue (Stage L); the hand-written HIPRTC kernel stays the production lane +
    on-silicon oracle until the compiled path is multi-tile + perf-laddered.
- **Stage K — a real GEMM through the full stack vs. the oracle.** Graph → Tile →
  `tessera_rocm` Target IR (real WMMA) → ROCDL → hsaco → launch, execute-compare
  against **both numpy and the `hardware_verified` hand-written kernel** (the
  on-silicon oracle). Milestone: "the compiler, not a hand-written kernel,
  produced the executing GEMM."
- **Stage L — converge the compiled path to production.** Not a single change —
  a program. Stages I–K proved the compiler can *generate* a correct executing
  GEMM (16×16×16, bit-identical to the oracle). L makes that path production-grade
  and the source of truth. Concrete sub-steps (each independently landable):
  - ✅ **L1 — general-shape codegen (2026-06-23).** `generate-wmma-gemm-kernel`
    now emits a **problem-size-generic** kernel: the directive's `m`/`n`/`k` are
    the WMMA *instruction* tile (16×16×16, the only one RDNA exposes), and the
    emitted `gpu.func` takes the runtime `(M,N,K)` as `index` args, a 2-D grid of
    one wave per 16×16 output tile, an `scf.for` K-loop, and ragged-edge masking
    (clamp-and-select loads, `scf.if`-guarded stores). One compiled kernel
    computes any shape. Executes on gfx1151 vs numpy (<5e-2) **and bit-identical
    to the hand-written oracle (0.0)** across square, rectangular, and ragged
    (non-multiple-of-16) shapes — `{16³, 32³, 48×64×32, 40×24×48, 17×15×31}`.
    Fixture `tests/unit/test_rocm_wmma_gemm_general.py`; the 16³ launch still
    reduces to the Stage K single-tile case. MT=NT=1 (one tile/wave);
    register-blocked macro-tiling (3×4) is L2.
  - ✅ **L2 — register-blocked macro-tiling + perf parity (2026-06-23).** The
    `wmma_gemm` directive carries `mt`/`nt` (default 1); each wave now computes an
    `mt`×`nt` grid of 16×16 output tiles, reusing a loaded A fragment across the
    `nt` B-tiles and a B fragment across the `mt` A-tiles. To make blocking
    actually pay off, the kernel splits into an **interior fast path** (whole
    macro-tile in-bounds *and* K%16==0 → single contiguous `vector.load` for each
    A fragment, no element masking) and the **masked edge path** (clamp-and-select
    loads + `scf.if` stores) for ragged tiles. All `(mt,nt)` ∈ {1×1,2×2,2×4,3×4}
    stay bit-identical to the oracle on ragged 100×96×64
    (`test_rocm_wmma_gemm_general.py::test_register_blocked_matches_oracle`).
    **Measured on gfx1151** (`benchmarks/rocm/benchmark_rocm_compiled_gemm.py`,
    kernel-only, vs the hand-written `_bench` at the *same* `(mt,nt)`): at aligned
    sizes the compiled kernel **meets or exceeds** the hand-written at every swept
    tile — 1536³ all tiles 1.06×–2.56×; 2048³ peak **4×4 = 18.7 vs 9.0 TF/s
    (2.07×)**. The fast path is the whole win: before it, the masked-everywhere
    kernel was 0.12–0.47×. Autotuner integration (auto-select `mt`/`nt` per shape)
    rides on the existing ladder harness — the sweep script *is* the brute-force
    version.
  - ✅ **Masked ragged-edge perf parity (2026-06-23).** L2's first cut dropped any
    tile whose extent isn't divisible by 16·`mt`/16·`nt` to a per-element scalar
    masked path (3×4 at 1024/2048 → 0.44×/0.69×). Fixed by splitting the codegen
    three ways: `kAligned ? (tileFull ? fast : edge) : masked`. The new **edge
    path** (K-aligned but M/N ragged — the common case) keeps the coalesced
    `vector.load`: it loads A/B at a row/col *clamped* into range, then zeroes the
    OOB fragment with one loop-invariant vector `arith.select`, so the K-loop stays
    vector-load speed; only the once-per-kernel stores are masked. The per-element
    path is now reserved for ragged **K** (`K%16≠0`) only. Measured on gfx1151:
    **3×4 at 2048 0.69×→1.82×, at 1024 0.44×→0.93×**; aligned unchanged/better
    (1536³ 1.06×–3.82×, 2048³ 1.26×–3.93×). Ragged-M/N tiles now reach
    parity-or-better; bit-identical to the oracle preserved.
  - ✅ **Ragged-K fast path + bf16 (2026-06-23).** The K-loop is split so masking
    never sits on the hot path: a main loop over the aligned range `[0, kMain)`
    (`kMain = K` rounded down to ×16) using the fast/edge panel, then a **single
    masked tail panel** for `[kMain, K)` when `K%16≠0`. So ragged K costs one
    extra masked panel, not a masked K-loop. Measured at 2040³ (ragged K *and*
    M/N): every tile **1.42×–3.62×** the hand-written — no cliff.
    **bf16**: the directive carries `dtype` (default `f16`); the generating pass
    emits bf16 fragments + memrefs and Stage J the `rocdl.wmma.f32.16x16x16.bf16`
    intrinsic. Runtime compiled lane is dtype-keyed (hsaco cached per
    `(mt,nt,chip,dtype)`). Executes bit-identical to the hand-written **bf16**
    oracle. Fixtures: `test_rocm_wmma_gemm_general.py` (ragged-K 31/40),
    `test_rocm_compiled_launch_execute.py` (bf16 vs oracle; f32 rejected).
  - ✅ **L3 — in-process serialization (2026-06-23).** The GPU/ROCDL → LLVM-IR
    serialization spine is now linked into `tessera-opt` itself, so the WHOLE
    chain runs in ONE invocation — no `mlir-opt` shell-out (Stages I/K/L1/L2 rode
    the platform `mlir-opt` for `gpu-module-to-binary`; a runtime lane can't):
    `tessera-opt - --pass-pipeline='builtin.module(generate-wmma-gemm-kernel,
    lower-tessera-target-to-rocdl, gpu.module(convert-scf-to-cf,
    convert-gpu-to-rocdl, reconcile-unrealized-casts),
    rocdl-attach-target{chip=gfx1151}, gpu-module-to-binary)'` → `gpu.binary`
    ELF. Wiring (all gated behind a full ROCm build — the lean artifact driver
    stays lean): register `gpu-module-to-binary`/`rocdl-attach-target`/
    `convert-scf-to-cf`/`reconcile-unrealized-casts`; the LLVM-IR translations +
    `#rocdl.target` interface; the cf/arith/func/memref/vector/index/ub
    ConvertToLLVM external models (what `convert-gpu-to-rocdl` needs to lower the
    full `gpu.func` body — the missing piece vs `mlir-opt`); init the AMDGPU LLVM
    target in `main`. AMDGPU codegen comes from the shared `libLLVM`; `ld.lld`
    from the platform LLVM (the ROCDL serializer shells to it). The in-process
    hsaco executes on gfx1151 bit-identical to the oracle —
    `tests/unit/test_rocm_wmma_gemm_in_process.py`.
  - 🟢 **L4 — compiled `runtime.launch()` lane (2026-06-23, opt-in).** The
    compiled path is now a real production-dispatch lane: an artifact with
    `compiler_path="rocm_compiled"` routes through the execution matrix to
    `_execute_rocm_compiled_gemm`, which drives the Stage L3 in-process pipeline
    (tessera-opt → hsaco, cached per `(mt,nt)` since the kernel is shape-generic)
    and launches it via HIP. Same `runtime.launch()` entry point as the
    hand-written lane — only *which kernel runs* differs. Executes on gfx1151
    bit-identical to the hand-written oracle through `launch()` across
    `{16³, 64×48×32, 256³}`; size-adaptive `(mt,nt)` mirrors the oracle (3×4 once
    min≥1024, else 2×4). f16 today (bf16 is a structured `invalid_artifact`, not a
    miscompute — use `rocm_wmma`). New execution-matrix row + `KNOWN_EXECUTORS`
    entry + `tests/unit/test_rocm_compiled_launch_execute.py`.
    **Deliberately NOT flipped to default / promoted in the manifest** (Decision
    #25): the hand-written `rocm_wmma` stays the default + reference oracle/fast
    fallback. The original blocker — masked ragged-edge perf — is now **resolved**
    (see "Masked ragged-edge perf parity" above: ragged-M/N tiles reach
    parity-or-better on gfx1151). The compiled lane is therefore *ready to promote*
    on perf grounds. The earlier technical gaps are now **closed**: bf16 +
    ragged-K (above).
  - ✅ **L4 default FLIPPED + manifest promoted (2026-06-23).** `@jit(target=
    "rocm")` matmul now **executes through the compiler-generated lane by
    default**: `jit.py` stamps `compiler_path="rocm_compiled"` /
    `execution_kind="native_gpu"` / `executable=True` for a rocm single
    matmul/gemm **when the compiled lane can run on the host** (tessera-opt built
    + a usable AMD GPU — `_uses_rocm_compiled_default()`, shared by the
    `execution_kind` property so `is_executable` agrees with what `launch()`
    does). Off-device it stays `target_ir_artifact` / not-executable exactly as
    before — host-gated, no behavior change in CI. The hand-written `rocm_wmma`
    kernel is now the reference **oracle + availability fallback**:
    `_execute_rocm_compiled_gemm` degrades to it on `_RocmCompiledUnavailable` (no
    tessera-opt / no serialization spine / no GPU), but a *genuine* compiled-kernel
    failure surfaces — the fallback never masks a real bug. Manifest matmul-row
    notes + execution-matrix reasons promoted to record the compiled lane as
    default and the hand-written symbol as oracle/fallback (the `runtime_symbol`
    stays the hand-written proof anchor). Fixtures:
    `test_rocm_compiled_launch_execute.py` (default flip executes + fallback),
    `test_target_ir_contract.py` (host-gated). **Stage L is complete.**
  - ✅ **int8 + int4 WMMA in the compiled lane (2026-06-23).** gfx1151 WMMA dtype
    support re-verified on the *real device compiler* (`hipcc --offload-arch=
    gfx1151`): **f16, bf16, int8 (iu8), int4 (iu4)** all compile; **f32, tf32,
    bf8/fp8 do NOT exist** on RDNA 3.5 (FP8 WMMA is RDNA 4 / gfx1200). Enabled
    int8 + int4 in the compiled GEMM lane:
    - **int8** (`dtype="int8"`): A/B loaded `vector<16xi8>` → bitcast to the iu8
      ABI `vector<4xi32>`; i32 accumulate; D = i32. Stage J →
      `rocdl.wmma.i32.16x16x16.iu8` (signed, clamp=0 = wrap).
    - **int4** (`dtype="int4"`, opt-in via `metadata["wmma_dtype"]`): int4 values
      in int8 containers (range [-8,7]); nibble-packed in-kernel to the iu4 ABI
      `vector<2xi32>` → `rocdl.wmma.i32.16x16x16.iu4`.
    Both execute on gfx1151 **bit-EXACT vs numpy int32 matmul** (integer is exact,
    so numpy is the oracle) across aligned / ragged-M/N / ragged-K shapes —
    `test_rocm_compiled_launch_execute.py`. The codegen generalized via a
    `WmmaTypes` bundle (store/load/frag/acc/accElem + pack-kind) so f16/bf16/int8/
    int4 share the one 3-path kernel. The hand-written `runtime_symbol` (f16/bf16)
    is unchanged; int8/int4 are compiled-lane-only capabilities.
  - ✅ **int dtype perf sweep — measured (2026-06-23).** `benchmarks/rocm/
    benchmark_rocm_compiled_gemm_dtype.py` (kernel-only, best macro-tile) on
    gfx1151 at 2048³: **f16 ≈ 23.2 TFLOP/s, bf16 ≈ 23.1, int8 ≈ 21.0 TOP/s,
    int4 ≈ 23.8 TOP/s** — all within ~10%. The finding: **RDNA 3.5 WMMA runs
    iu8/iu4 at the same matrix-op rate as f16** (no low-precision FLOP-rate
    multiplier), so the compiled int paths are already compute-competitive and the
    int4 nibble-pack overhead is amortized. **Consequence:** packed-memory int4
    (2 int4/byte) would buy *memory footprint* (½) + bandwidth, **not compute**
    on this arch — so it is deliberately deferred (its large, sub-byte-strided-B
    layout is unjustified by a compute speedup that doesn't exist here). Measured,
    not assumed (Decision #25).
  - ✅ **Front-end glue wired (2026-06-23).** The Graph `tessera.matmul` → Tile →
    Target-IR lowering (`_lower_rocm_op` on `tile.mma` in `target_ir.py`) now
    EMITS the executable `tessera_rocm.wmma_gemm` directive (m=n=k=16 WMMA tile +
    dtype) alongside the abstract `tessera_rocm.mfma` marker. So a
    `@jit(target="rocm")` matmul's `target_ir` contains the directive the
    `generate-wmma-gemm-kernel` pass consumes — the directive is now produced by
    the IR stack (Decision #19), not only synthesized by the runtime. Verified
    GPU-free in `test_rocm_matmul_front_end_glue.py`: the directive appears with
    the right attrs AND the extracted directive feeds the generate pass into a
    `gpu.func` + WMMA op. (The runtime lane still synthesizes a clean directive at
    launch for the per-shape `mt`/`nt` perf choice; the canonical *lowering* now
    owns directive production.)
  The hand-written kernel stays the production default + oracle until the
  compiled lane reaches ragged-shape perf parity.

10. flash_attn follow-ups:
    - ✅ **compiler-generated forward** (2026-06-23) — the `generate-wmma-flash-
      attn-kernel` pass; executes on gfx1151 vs numpy (see the flash_attn section).
    - ✅ **forward perf ladder, measured** —
      `benchmarks/rocm/benchmark_rocm_flash_attn_compiled.py` on gfx1151.
      - rung 0 (2026-06-23): ~4.0 TFLOP/s @ D=64, ~2.4 @ 128. Occupancy is
        **LDS-limited** (decoded from the hsaco kernel descriptor: 8 waves/CU @
        D=64, 4 @ 128 — 25%/12% of the 32-wave cap; VGPRs not the limit at D=64).
      - **rung 1 (2026-06-24): drop `sQ` LDS staging** — read Q from global for
        the QK^T A-fragment instead of staging it (sQ was the 2nd-largest LDS
        buffer). LDS 7360→5312 B @ D=64 → **8→12 waves/CU (+50%)**, **~4.0→~5.3
        TFLOP/s @ D=64 (~1.3×)**. D=128 occupancy 4→6 waves but perf flat (~2.40)
        — D=128 is *also* LDS-limited (NOT VGPR-bound; VGPRs allow 24 waves, LDS
        caps at 6), so at 4–6 waves occupancy wasn't its binding constraint;
        LDS *traffic* was (see rung 2). 5/5.
      - **rung 2 (2026-06-24): fuse the online-softmax rescale into P@V** —
        instead of a separate `sAcc *= corr` pass (a full 16·D LDS read+write +
        a barrier per KV tile) then `sAcc += P@V`, do `sAcc = sAcc*corr + P@V` in
        the P@V write (each sAcc entry is written exactly once per tile). Removes
        one LDS pass + one barrier/tile. **~5.3→~6.8 TFLOP/s @ D=64 (~1.3×) AND
        ~2.4→~3.18 @ D=128 (~1.3×)** — LDS-traffic reduction helps the LDS-bound
        D=128 case too. 5/5. Cumulative forward: **~4.0→~6.8 @ D=64 (~1.7×)**,
        ~2.4→~3.18 @ D=128.
      Key insight: LDS traffic / lgkmcnt stalls (not just occupancy) are the real
      lever. Investigated but DEFERRED (measured, don't pay cleanly): (a) D=128
      occupancy is LDS-limited by `sAcc` (16·D·f32 = 8 KB) — can't shrink without
      f16 accumulation (precision loss) or register residency (D/2=64 VGPR/lane →
      spills, fewer waves); (b) multi-wave query tiles — the 16-query tile maps
      naturally to 16 lanes, so extra waves don't collaborate without a from-
      scratch warp-specialized redesign. FA forward at ~6.8 is ~34% of the GEMM
      peak (~20), reasonable given FA's softmax overhead; further gains need a
      different kernel design, not a rung.
    - ✅ **backward pass — compiler-generated (2026-06-23).** The
      `generate-wmma-flash-attn-bwd-kernel` pass expands one
      `tessera_rocm.flash_attn_bwd` directive into the textbook FA-2 backward
      as THREE fragment-materialized WMMA kernels (no stored attention matrix —
      S/P recomputed per tile): `_pre` (WMMA QK^T + online-softmax logsumexp `L`
      + `D=rowsum(O·dO)`),
      `_dkdv` (per key-tile: recompute S/P, `dP=dO@Vᵀ`, `dS=P·(dP−D)`, accumulate
      `dV+=Pᵀ@dO` and `dK+=scale·dSᵀ@Q` — P/dS staged in LDS, reread transposed),
      `_dq` (per query-tile: `dQ+=scale·dS@K`). All three use the same RDNA WMMA
      `C[m,n]=Σ_k A[m,k]B[n,k]` primitive + Stage J→I lowering as the forward.
      Executes on gfx1151 vs a numpy attention-backward reference (itself checked
      against finite differences): **rel-err ~2–4e-4** (f16 storage, f32
      accumulate) across head_dim 16/64, causal + non-causal, ragged —
      `tests/unit/test_rocm_flash_attn_bwd_compiled.py`. flash_attn (fwd+bwd) is
      now the third compiler-generated op on ROCm after matmul.
    - ✅ **backward perf ladder, measured** —
      `benchmarks/rocm/benchmark_rocm_flash_attn_bwd_compiled.py` on gfx1151
      (~10·B·H·Sq·Sk·D FLOPs).
      - rung 0 (2026-06-23, correctness-first): ~1.1–1.3 TFLOP/s @ D=64, ~0.67 @
        128. The `_pre` logsumexp was a per-lane **scalar** dot-product loop
        (O(Sq·Sk·D), zero WMMA) — the measured long pole.
      - **rung 1 (2026-06-24): `_pre`→WMMA.** Rewrote `_pre` to compute `L` with
        the forward's WMMA QK^T + online-softmax-stats path (same matrix rate as
        the forward, not serial VALU). **~3.4× faster: ~4.1–4.4 TFLOP/s @ D=64,
        ~2.1 @ 128** — now WMMA-bound like the forward (correctness unchanged:
        dQ/dK/dV rel-err ~2–4e-4, `test_rocm_flash_attn_bwd_compiled.py` 5/5).
      - **rung 2 (2026-06-24): causal tile-skip** in `_dkdv` (query loop starts
        at the key tile) and `_dq` (key loop bounded at the query tile) — the
        diagonal tile is still per-element masked, so only provably-zero tiles
        are dropped. **~1.7× for causal** (measured causal-vs-non-causal at S=
        1024/2048: 1.64–1.71×; theoretical max 2×, gap = diagonal tiles + the
        fixed `_pre`/accumulator overhead). Correctness unchanged (5/5, incl. the
        causal cases). The forward already had this bound.
      - **LDS-traffic lever investigated, NO clean win (2026-06-24, measured).**
        The forward's rung-2 (fuse a separate rescale into the accumulate) has no
        backward analog — the backward has no separate rescale pass. The dominant
        backward LDS is the `_dkdv` `dKacc`+`dVacc` accumulators (the worst
        occupancy in the suite: hsaco-decoded **3 waves/CU @ D=128**, 17408 B
        LDS). Moving them to registers (the only way to cut that LDS) **spills**:
        `_dkdv` already uses 232 VGPR @ D=64 / 207 @ D=128, and per-lane
        accumulators add D/2–D VGPRs → over the 256 ceiling. So `_dkdv` is stuck
        between an LDS-occupancy limit and a VGPR-spill limit — no clean lever
        without a from-scratch tiling redesign.
      Remaining next rungs (shared with the forward): occupancy (1 wave/block →
      multi-wave) — a real kernel restructure. Memory double-buffering is NOT a
      lever — WMMA/occupancy-bound, not staging-bound (GEMM ~20 vs FA ~4 TFLOP/s,
      FA drops at D=128 on LDS footprint).
    - ✅ **`runtime.launch()` executor-table lane (2026-06-24)** — flash_attn now
      reaches the runtime executor table like matmul. Artifact stamped
      `compiler_path="rocm_flash_attn_compiled"` → `execution_matrix` row →
      `_execute_rocm_compiled_flash_attn` (builds the FA-2 forward hsaco
      in-process via tessera-opt, HIP loads + launches `fa`). f16/bf16 storage,
      f32 softmax+accumulate; Q/K/V `[...,S,D]`, causal/scale from op kwargs.
      Executes vs a numpy attention reference through `launch()`
      (`tests/unit/test_rocm_flash_attn_launch_execute.py`, maxerr <2e-2,
      skip-clean). The matmul-L4 analog for attention.
11. Promote manifest rows only after generated dashboards agree.

## Source Material Consolidated

- `../archive/nvidia_rocm_execute_and_compare_plan.md`

