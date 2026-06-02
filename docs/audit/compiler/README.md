# Compiler Audit

Current compiler audit tracking lives here. This theme covers compiler
architecture, Graph/Schedule/Tile/Target IR handoffs, lowering proof, compiler
correctness testing, and spec drift.

## Current Truth

- The compiler now has a recognizable proof spine: canonical compile, IR
  bundle, named gates, backend manifest rows, and runtime artifact metadata.
- The main open architectural issue is handoff discipline: multi-op component
  metadata, fusion groups, layout contracts, and numerical proof need to move
  through the compiler path instead of being rediscovered at runtime.
- Generated status should be read from:
  - [../op_target_conformance.md](../op_target_conformance.md)
  - [../generated/support_table.md](../generated/support_table.md)
  - [../generated/verifier_coverage.md](../generated/verifier_coverage.md)

## Open Items

- Make canonical compile component-aware instead of first-op-only.
- Promote tensor/layout/effect contracts across Graph IR to Target IR.
- Replace repeated target/runtime pattern recognition with descriptor-backed
  lowering contracts.
- Keep compiler spec claims synchronized with generated dashboards.

## Archived Source Material

- [compiler_apple_backend_end_to_end_audit_2026_06_02.md](archive/compiler_apple_backend_end_to_end_audit_2026_06_02.md)
- [compiler_correctness_testing_audit.md](archive/compiler_correctness_testing_audit.md)
- [compiler_improvement_milestone_plan_2026_05_18.md](archive/compiler_improvement_milestone_plan_2026_05_18.md)
- [compiler_layer_gap_remediation.md](archive/compiler_layer_gap_remediation.md)
- [compiler_spec_gap_audit.md](archive/compiler_spec_gap_audit.md)
- [compiler_spec_gap_matrix.md](archive/compiler_spec_gap_matrix.md)
