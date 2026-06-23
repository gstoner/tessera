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
- Honest scope (Decision #25): WMMA `{fp16, bf16}` forward only — **no backward,
  no perf ladder** (this is the correctness-first "rung 0" of attention, the
  analog of the naive GEMM tile). It does **not** earn a `runtime.launch()` lane
  yet: that needs flash_attn artifact plumbing (how a JIT'd attention artifact
  carries Q/K/V + dispatches), a separate piece — so no `runtime_execution_matrix`
  row is claimed (adding one with no producer would be over-claiming).

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

## Still Open

- **Other GEMM-family / attention ops on RDNA remain artifact_only** beyond
  matmul + flash_attn (CDNA MFMA shape, HIP execution gated): `multi_head_
  attention`, `gqa/mqa`, the fused chains, etc. The named ROCm sub-arches
  (gfx90a/942/950/1100/1151/1200) stay in `_UNIMPLEMENTED_TARGETS` — the generic
  `rocm` lane covers execution via HIPRTC for the live device; gfx1151 (the box's
  own arch) is listed there too so the classification is total (no silent
  `lookup() -> None`).
- **flash_attn**: no backward pass; no perf ladder; no `runtime.launch()` lane.
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
    kernel was 0.12–0.47×. Honest caveat: a tile whose extent isn't divisible by
    16·`mt`/16·`nt` has a ragged band that drops to the (slower) masked path —
    e.g. 3×4 at 1024/2048 (not ÷48) underperforms while 2×4/4×4 stay fast.
    Optimizing the masked edge path (or padding) is follow-up. Autotuner
    integration (auto-select `mt`/`nt` per shape) rides on the existing ladder
    harness — the sweep script *is* the brute-force version.
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
    fallback because the compiled path's *masked ragged-edge* tiles aren't yet
    perf-competitive (L2 caveat). Flipping the source of truth waits on the
    masked-edge optimization + dashboards agreeing. Remaining for full L4:
    masked-edge perf parity, then promote.
  Front-end glue (Graph `tessera.matmul` → Tile → the `wmma_gemm` directive) feeds
  L1. The hand-written kernel stays the production default + oracle until the
  compiled lane reaches ragged-shape perf parity.

10. flash_attn follow-ups: backward pass; a perf ladder (the forward is rung-0
    correctness-first); the `runtime.launch()` artifact lane.
11. Promote manifest rows only after generated dashboards agree.

## Source Material Consolidated

- `../archive/nvidia_rocm_execute_and_compare_plan.md`

