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

## Still Open

- No ROCm execution row exists in `../../generated/runtime_execution_matrix.md`.
- ROCm rows remain artifact-only or planned until real hardware execution is
  proven.
- **Stage C — HIP launch bridge:** the standalone `tessera_rocm_runtime`
  (`runtime/hip/loader.cpp`) has real `hipModuleLaunchKernel` (behind
  `TESSERA_HAS_HIP`) but is **not** registered into the core C-ABI hook
  `tsrRegisterGpuLauncher`, and there is no HIPRTC path (loads prebuilt hsaco).
- **Stage D — execute-and-compare** WMMA GEMM vs numpy oracle, then flip
  `backend_kernel` for `tessera.matmul` on `rocm_gfx1100`/`gfx1151`.

## Next Work

1. **Stage B — assemble:** emit the gfx1100 WMMA GEMM to LLVM IR
   (`mlir-translate`) → `hipcc --offload-arch=gfx1100` / `llc -mcpu=gfx1100` to a
   real object; skip-clean when hipcc absent.
2. **Stage C — launch:** register a HIP launcher into `tsrRegisterGpuLauncher`
   (landed G7 2026-06-10 — see `backend/BACKEND_AUDIT.md`); load the Stage B
   hsaco and `hipModuleLaunchKernel` the gfx1100 WMMA kernel. Add runtime ABI +
   hardware-smoke tests.
3. **Stage D — prove:** execute-and-compare oracle tests (bring up
   `fp32←f16`/`f16←f16` WMMA before bf16 per the documented gfx115x bf16 bugs).
4. Promote manifest rows only after generated dashboards agree.

## Source Material Consolidated

- `../archive/nvidia_rocm_execute_and_compare_plan.md`

