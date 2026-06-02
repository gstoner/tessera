# NVIDIA Backend Audit

NVIDIA backend audit tracking lives here. This theme covers CUDA/NVIDIA
execution, SM target maps, CUDA toolchain proof, and execute-and-compare work.

## Current Truth

- Start with [NVIDIA_AUDIT.md](NVIDIA_AUDIT.md) for the consolidated NVIDIA
  audit: what exists today, what is hardware-gated, and what must land before
  execution claims can be promoted.
- NVIDIA status is hardware-gated for real execute-and-compare proof.
- Generated NVIDIA status should be read from:
  - [../../generated/nvidia_sm90_target_map.md](../../generated/nvidia_sm90_target_map.md)
  - [../../generated/support_table.md](../../generated/support_table.md)
- Shared cross-target context remains in [../README.md](../README.md).

## Open Items

- Preserve the distinction between artifact generation and hardware execution.
- Add execute-and-compare proof on real NVIDIA hardware before promoting
  backend claims.
- Keep CUDA toolchain/version pins aligned with generated dashboards and CI.

## Archived Source Material

- [nvidia_execution_audit.md](archive/nvidia_execution_audit.md)
- [../archive/nvidia_rocm_execute_and_compare_plan.md](../archive/nvidia_rocm_execute_and_compare_plan.md)
