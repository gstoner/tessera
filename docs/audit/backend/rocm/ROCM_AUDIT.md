# ROCm Backend Audit

This document consolidates ROCm-specific audit material.

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
2. Implement the first narrow HIP runtime launch path.
3. Add runtime ABI and hardware-smoke tests.
4. Add execute-and-compare oracle tests.
5. Promote manifest rows only after generated dashboards agree.

## Source Material Consolidated

- `../archive/nvidia_rocm_execute_and_compare_plan.md`

