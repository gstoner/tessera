# Backend Audit

Shared backend audit tracking lives here. This theme covers cross-target runtime
execution, ABI contracts, dtype policy, hardware-gated proof, and target-neutral
backend rules.

## Current Truth

- Start with [BACKEND_AUDIT.md](BACKEND_AUDIT.md) for the consolidated
  cross-target backend audit: what is executable, what is artifact-only, and
  which proof gates remain.
- Runtime execution truth is generated from the execution matrix:
  [../generated/runtime_execution_matrix.md](../generated/runtime_execution_matrix.md).
- C ABI surface truth is generated from the runtime ABI audit:
  [../generated/runtime_abi.md](../generated/runtime_abi.md).
- Hardware proof remains target-specific; Apple, NVIDIA, and ROCm have separate
  platform folders when the issue is platform-specialized.
- Metalium has shared-backend frontier notes only; no dedicated platform folder
  is created until there is enough Tenstorrent-specific audit material.

## Platform Folders

- Apple: [apple/APPLE_AUDIT.md](apple/APPLE_AUDIT.md)
- NVIDIA: [nvidia/NVIDIA_AUDIT.md](nvidia/NVIDIA_AUDIT.md)
- ROCm: [rocm/ROCM_AUDIT.md](rocm/ROCM_AUDIT.md)

## Open Items

- Keep cross-target proof language strict: compileable, artifact-only,
  executable, numerically verified, and hardware-verified are different claims.
- Ensure generated target maps remain linked from platform pages.
- Keep dtype support policy tied to backend/runtime facts, not stale prose.

## Archived Source Material

- [hardware_dtype_support_matrix.md](archive/hardware_dtype_support_matrix.md)
- [nvidia_rocm_execute_and_compare_plan.md](archive/nvidia_rocm_execute_and_compare_plan.md)
- [phase_ghi_hardware_frontier.md](archive/phase_ghi_hardware_frontier.md)
