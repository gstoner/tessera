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

## Still Open

- No ROCm execution row exists in `../../generated/runtime_execution_matrix.md`.
- ROCm rows remain artifact-only or planned until real hardware execution is
  proven.
- HIP runtime launch bridge and execute-and-compare proof are still required.
- MFMA/gfx-specific proof must be validated on real AMD hardware.

## Next Work

1. Bring up ROCm hardware validation.
2. Register a HIP launcher into the C-ABI launch-bridge hook
   (`tsrRegisterGpuLauncher`, landed G7 2026-06-10 — see
   `backend/BACKEND_AUDIT.md`) for the first narrow ROCm kernel.
3. Add runtime ABI and hardware-smoke tests.
4. Add execute-and-compare oracle tests.
5. Promote manifest rows only after generated dashboards agree.

## Source Material Consolidated

- `../archive/nvidia_rocm_execute_and_compare_plan.md`

