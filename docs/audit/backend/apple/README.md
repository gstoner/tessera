---
last_updated: 2026-07-13
audit_role: index
---

# Apple backend audit map

Use this directory by **document role**, not by whichever file has the newest
date. Apple support has several valid execution forms, so a statement is only
meaningful when it names its target, unit, compiler form, runtime executor,
placement, and proof rung.

## Authority order

1. [`../../generated/apple_execution_inventory.md`](../../generated/apple_execution_inventory.md)
   — execution-unit status: generic dispatch, Value Target-IR, authored package
   subgraphs, and explicit reference execution.
2. [`../../generated/apple_target_map.md`](../../generated/apple_target_map.md)
   — generic per-op dispatch only; it is not a Value Target-IR or package map.
3. [`../../generated/runtime_execution_matrix.md`](../../generated/runtime_execution_matrix.md)
   — runtime executor and observed placement.
4. [`../../../backends/apple/`](../../../backends/apple/)
   — architecture, ABI, integration boundaries, and how to extend a lane.
5. [`../../../backends/apple/kernel-guide.md`](../../../backends/apple/kernel-guide.md)
   — human kernel-family and ABI guide.

## Documents in this directory

- [`APPLE_AUDIT.md`](APPLE_AUDIT.md) is the active decision and audit-delta log.
- [`APPLE_GPU_CODEGEN_PLAN.md`](APPLE_GPU_CODEGEN_PLAN.md) is a historical
  codegen plan; do not use it for current status.
- [`MPSGRAPH_RUNTIME_GLASS_JAWS.md`](MPSGRAPH_RUNTIME_GLASS_JAWS.md) is a
  MPSGraph Value Target-IR risk register.
- [`archive/`](archive/) preserves completed plans and surveys for provenance.

## Vocabulary

| Term | Meaning |
|---|---|
| generic route | Ordinary compiler/JIT dispatch for an op or region. |
| Value Target-IR call | Strict, value-preserving single Target-IR ABI call. |
| package subgraph | An authored `.mtlpackage` whole-region execution unit. |
| `native_cpu` / `native_gpu` | Where work actually executes. |
| `reference_cpu` | Correct host/reference execution; never a native-device claim. |
| artifact-only | Compiler output exists without an executable-runtime claim. |
