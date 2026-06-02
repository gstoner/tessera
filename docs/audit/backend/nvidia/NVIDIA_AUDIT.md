# NVIDIA Backend Audit

This document consolidates NVIDIA-specific audit material.

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
2. Implement the runtime launch bridge for the first narrow NVIDIA kernel.
3. Add runtime ABI and hardware-smoke tests.
4. Add execute-and-compare oracle tests.
5. Promote manifest rows only after generated dashboards agree.

## Source Material Consolidated

- `archive/nvidia_execution_audit.md`
- `../archive/nvidia_rocm_execute_and_compare_plan.md`

