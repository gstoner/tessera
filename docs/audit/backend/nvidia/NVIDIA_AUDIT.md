# NVIDIA Backend Audit

This document consolidates NVIDIA-specific audit material.

> **Real-hardware bring-up:** see [`BLACKWELL_SM120_EXECUTION_PLAN.md`](BLACKWELL_SM120_EXECUTION_PLAN.md)
> — the sm_120 target model is correctly grounded as **Blackwell consumer** (RTX 5070 Ti),
> not the old "Rubin placeholder" (no wgmma/tcgen05/TMEM; FP4 via `mma.sync.block_scale`).
> **Status (2026-06-24, #106):** the first real NVIDIA `backend_kernel` is **proven on
> silicon** — a sm_120 `mma.sync` bf16 matmul executes end-to-end on the RTX 5070 Ti
> (`emit_mma_sync_matmul_ptx` → PTX → assemble → CUDA launch bridge `tsrRegisterGpuLauncher`
> → execute-and-compare), under CUDA 13.3. The `mma.sync` emit path the old "Key gap" called
> for now exists. **Still open:** broaden sm_120 beyond matmul (flash-attn family), and the
> separate Hopper sm_90 (WGMMA) / datacenter sm_100 (tcgen05) emit paths — `mma.sync` ≠ WGMMA,
> so each arch needs its own proof.

## Finished

- NVIDIA target-map generation exists at
  `../../generated/nvidia_sm90_target_map.md`.
- CUDA/NVIDIA toolchain and execute-and-compare plans are documented.
- The repo distinguishes NVIDIA artifact generation from hardware execution.
- Compiler pass-order and lit-style structural work exists for NVIDIA-oriented
  paths.

## Still Open

- No NVIDIA execution row exists in `../../generated/runtime_execution_matrix.md`.
- NVIDIA rows remain artifact-only or planned until real hardware execution is
  proven.
- Execute-and-compare proof on real NVIDIA hardware is still required.
- Runtime ABI lock, smoke tests, and numerical fixtures need to land before
  promotion.

## Next Work

1. Bring up real NVIDIA hardware CI or a dedicated validation host.
2. Register a CUDA launcher into the C-ABI launch-bridge hook
   (`tsrRegisterGpuLauncher`, landed G7 2026-06-10 — see
   `backend/BACKEND_AUDIT.md`) mapping the first narrow NVIDIA kernel name to
   its native symbol.
3. Add runtime ABI and hardware-smoke tests.
4. Add execute-and-compare oracle tests.
5. Promote manifest rows only after generated dashboards agree.

## Source Material Consolidated

- `archive/nvidia_execution_audit.md`
- `../archive/nvidia_rocm_execute_and_compare_plan.md`

