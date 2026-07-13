---
status: Informative
classification: Informative
authority: Reader-facing routing for current status; generated audit dashboards remain the authority for mutable execution claims.
scope: Status cards, their evidence sources, and the boundary between status, milestones, and release policy.
last_updated: 2026-07-13
---

# Status

This directory is the short, reader-facing entry point for the current state of
major Tessera capabilities. It is deliberately a routing layer, not a second
inventory of target support or benchmark results.

## How to read the status surface

| Need | Start here | Current evidence |
|---|---|---|
| GA / EBM capability and health checks | [GA / EBM status](ga_ebm.md) | [support table](../audit/generated/support_table.md) and [runtime execution matrix](../audit/generated/runtime_execution_matrix.md) |
| Visual-complex capability and health checks | [Visual-complex status](visual_complex.md) | [support table](../audit/generated/support_table.md) and [runtime execution matrix](../audit/generated/runtime_execution_matrix.md) |
| Release requirements and enforcement | [release gates](../operations/release_gates.md) | `scripts/release_gate.py` and CI configuration |

## Taxonomy

- **Status cards** describe the public surface, point to the current evidence,
  and state what a reader must not infer from a support label.
- **Generated audit dashboards** own mutable support rows, native-execution
  proof, and target-specific coverage. Refresh or check them through the
  generated-doc workflow rather than editing their conclusions by hand.
- **Milestone snapshots** preserve dated planning and implementation context in
  `docs/audit/domain/`; they are not current-status authority.
- **Release policy** belongs in `docs/operations/`, where its executable
  enforcement and operational owner can be kept together.
