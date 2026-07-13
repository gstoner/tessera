---
classification: Backend architecture reference
authority: ROCm reader entry point
last_updated: 2026-07-13
---

# ROCm Backend

This page is the reader-facing entry point for Tessera's AMD ROCm/HIP target.
It separates the compiler and runtime architecture from the generated evidence
and from the historical bring-up record.

## Current evidence

- [Runtime execution matrix](../../audit/generated/runtime_execution_matrix.md)
  is current execution and placement truth.
- [ROCm target map](../../audit/generated/rocm_target_map.md) is the generated
  per-op target view.
- [ROCm kernel inventory](kernel-inventory.md) explains MFMA/WMMA contracts;
  it is not a mutable status ledger.

## Architecture and decisions

[ROCm audit](../../audit/backend/rocm/ROCM_AUDIT.md) owns target-specific
decisions and current deltas. [Strix Halo execution plan](../../audit/backend/rocm/STRIX_HALO_EXECUTION_PLAN.md)
records active execution work; older material remains in the audit archive.
