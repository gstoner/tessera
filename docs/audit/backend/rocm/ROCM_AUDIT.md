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

## Still Open

- The matmul WMMA path is the proven GEMM; **flash_attn / other GEMM-family ops
  on RDNA remain artifact_only** (CDNA MFMA shape, HIP execution gated). The
  named ROCm sub-arches (gfx90a/942/950/1200) stay in `_UNIMPLEMENTED_TARGETS`
  — the generic `rocm` lane covers execution via HIPRTC for the live device.
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

**Open rungs:** K-loop software pipelining over the LDS buffers (where staging
starts to earn its keep), arch-aware LDS layout. Heed Gluon's v6 double-buffer
regression.

## Next Work

1. ✅ **Stage B — assemble (2026-06-22):** `rocdl_emit.py` emits the WMMA GEMM
   LLVM IR and `llc -mcpu=gfx1100` lowers it to real `v_wmma_*` AMDGCN + an
   AMD GPU ELF object; verified on the box, gfx1100 + gfx1151.
2. ✅ **Stage C — launch (2026-06-22):** a GEMM executes through the C-ABI
   bridge on gfx1100 and matches `A @ B`; runtime HIP build fixed.
3. ✅ **Stage D — prove (2026-06-22):** the WMMA `f32←f16` GEMM executes through
   the bridge and matches a host reference (`test_rocm_wmma_execute_compare.py`).
4. ✅ **Ship the ROCm WMMA GEMM runtime symbol (2026-06-22):**
   `libtessera_rocm_gemm.so` exports `tessera_rocm_wmma_gemm_f16` (HIPRTC at
   load), with the `test_rocm_wmma_runtime_symbol.py` execute-compare fixture.
   The matmul manifest row is now `hardware_verified` (rocm target). **Still
   pending:** wire an auto-registered ROCm executor into `runtime.launch()` so a
   `runtime_execution_matrix` row is earned; then extend to tiled/K-looped GEMM +
   bf16. (The per-primitive `backend_kernel` flip needs *all* targets — out of
   scope for a single box.)
5. Register `tessera-to-linalg` into `tessera-opt` so the MLIR-graph
   `--tessera-emit-rocdl` route works (Stage B currently rides the direct emitter).
6. Promote manifest rows only after generated dashboards agree.

## Source Material Consolidated

- `../archive/nvidia_rocm_execute_and_compare_plan.md`

