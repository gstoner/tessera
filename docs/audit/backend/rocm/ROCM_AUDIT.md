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
ROCm **7.2.4**, LLVM/MLIR **22.1.8**. The iGPU enumerates as **`gfx1100`**
(RDNA3 profile; same 16×16×16 WMMA family as gfx1151, no FP8 WMMA) — see
[`STRIX_HALO_EXECUTION_PLAN.md`](STRIX_HALO_EXECUTION_PLAN.md) "Bring-up status".
Cleared: `rocminfo` enumerates without `HSA_OVERRIDE`; `hipcc` compiles WMMA for
gfx1100; ROCm lit suite 11/11; `tessera-opt`/`tessera-rocm-opt` build clean.

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

## Still Open

- No ROCm execution row exists in `../../generated/runtime_execution_matrix.md`
  yet (the Stage C launcher lives in the test harness, like Apple G7 — a shipped
  auto-registered ROCm runtime launcher is the follow-up before a matrix row).
- ROCm `backend_kernel` rows remain artifact-only until the **WMMA** op itself
  executes-and-compares (Stage D) — Stage C proved the bridge with a naive GEMM.
- **Stage D — execute-and-compare** the WMMA GEMM vs numpy oracle (bring up
  `fp32←f16`/`f16←f16` before bf16), then flip `backend_kernel` for
  `tessera.matmul` on `rocm_gfx1151`/`gfx1100`.

## Next Work

1. ✅ **Stage B — assemble (2026-06-22):** `rocdl_emit.py` emits the WMMA GEMM
   LLVM IR and `llc -mcpu=gfx1100` lowers it to real `v_wmma_*` AMDGCN + an
   AMD GPU ELF object; verified on the box, gfx1100 + gfx1151.
2. ✅ **Stage C — launch (2026-06-22):** a GEMM executes through the C-ABI
   bridge on gfx1100 and matches `A @ B`; runtime HIP build fixed.
3. **Stage D — prove:** route the Stage A/B **WMMA** GEMM through the Stage C
   bridge and execute-and-compare vs a numpy oracle (bring up `fp32←f16`/
   `f16←f16` before bf16 per the documented gfx115x bf16 bugs), then flip
   `backend_kernel` for `tessera.matmul`.
4. Ship an auto-registered ROCm runtime launcher (move the harness launcher into
   a backend lib) so a `runtime_execution_matrix` ROCm row can land.
5. Register `tessera-to-linalg` into `tessera-opt` so the MLIR-graph
   `--tessera-emit-rocdl` route works (Stage B currently rides the direct emitter).
6. Promote manifest rows only after generated dashboards agree.

## Source Material Consolidated

- `../archive/nvidia_rocm_execute_and_compare_plan.md`

