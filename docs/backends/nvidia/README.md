---
classification: Backend architecture reference
authority: NVIDIA / CUDA reader entry point
last_updated: 2026-07-13
---

# NVIDIA Backend

This page is the reader-facing entry point for Tessera's CUDA target, from
generic compiler-emitted CUDA through the target-specific tensor-core lanes.

## Current evidence

- [Runtime execution matrix](../../audit/generated/runtime_execution_matrix.md)
  is current execution and placement truth.
- [NVIDIA target map](../../audit/generated/nvidia_sm90_target_map.md) is the
  generated target view.
- [Kernel inventory](kernel-inventory.md) explains the SM90+ planned contract;
  [sm_120 guide](sm120-kernel-guide.md) records the separately proven consumer
  Blackwell lane.

## Architecture and decisions

[NVIDIA audit](../../audit/backend/nvidia/NVIDIA_AUDIT.md) owns current target
decisions and execution deltas. [Blackwell execution plan](../../audit/backend/nvidia/BLACKWELL_SM120_EXECUTION_PLAN.md)
is the active implementation plan; archival material stays under its audit
folder.
