# ROCm Backend Audit

ROCm backend audit tracking lives here. This theme covers HIP/ROCm execution,
gfx target maps, MFMA proof work, and execute-and-compare planning.

## Current Truth

- Start with [ROCM_AUDIT.md](ROCM_AUDIT.md) for the consolidated ROCm audit:
  what exists today, what is hardware-gated, and what must land before
  execution claims can be promoted.
- ROCm status is hardware-gated for real execute-and-compare proof.
- Generated ROCm status should be read from:
  - [../../generated/rocm_target_map.md](../../generated/rocm_target_map.md)
  - [../../generated/support_table.md](../../generated/support_table.md)
- Shared cross-target context remains in [../README.md](../README.md).

## Open Items

- Preserve the distinction between artifact generation and hardware execution.
- Add execute-and-compare proof on real ROCm hardware before promoting backend
  claims.
- Keep HIP/ROCm and MFMA target details aligned with generated dashboards and
  CI.

## Archived Source Material

- [../archive/nvidia_rocm_execute_and_compare_plan.md](../archive/nvidia_rocm_execute_and_compare_plan.md)
