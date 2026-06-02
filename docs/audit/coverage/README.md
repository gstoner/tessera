# Coverage Audit

Coverage audit tracking lives here. This theme covers primitive contracts, op
coverage, partial-op uplift, KV-cache coverage, examples capability gaps, and
test/support dashboards.

## Current Truth

- Start with [COVERAGE_AUDIT.md](COVERAGE_AUDIT.md) for the consolidated
  coverage audit: finished uplift, remaining proof axes, and archived evidence.
- `primitive_coverage.py` is the registry truth for compiler primitive
  contracts.
- Generated status should be read from:
  - [../standalone_primitive_coverage.md](../standalone_primitive_coverage.md)
  - [../generated/support_table.md](../generated/support_table.md)
  - [../generated/e2e_op_coverage.md](../generated/e2e_op_coverage.md)
  - [../generated/test_coverage_by_op.md](../generated/test_coverage_by_op.md)
  - [../generated/test_coverage_classification.md](../generated/test_coverage_classification.md)

## Open Items

- Keep partial/op uplift plans tied to generated coverage dashboards.
- Treat KV-cache coverage as a cross-cutting op/runtime proof topic, not only a
  frontend feature.
- Avoid copying numeric coverage snapshots into prose docs.

## Archived Source Material

- [advanced_examples_capability_gap.md](archive/advanced_examples_capability_gap.md)
- [kv_cache_coverage_matrix.md](archive/kv_cache_coverage_matrix.md)
- [partial_ops_uplift_plan.md](archive/partial_ops_uplift_plan.md)
- [primitive_coverage_state.md](archive/primitive_coverage_state.md)
