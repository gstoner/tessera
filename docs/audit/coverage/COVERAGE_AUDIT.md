# Coverage Audit

This document consolidates primitive, op, example, KV-cache, and coverage audit
material.

## Finished

- Partial-op uplift closed the old 47 partial rows.
- E2E coverage reports 34 native-complete and 237 runnable-reference rows, with
  0 partial and 0 planned rows.
- `lowering_rule` is closed across all 432 S-series primitive entries.
- Test coverage classification reports 0 actionable direct-test-debt ops.
- Advanced examples largely moved from missing APIs to backend/hardware proof.
- KV-cache coverage has explicit target diagnostics and historical matrices.

## Still Open

- All 432 S-series primitive entries remain open on backend-kernel proof.
- Long-tail transform axes remain: batching, transpose, and sharding still have
  open counts in generated S-series status.
- Hardware-gated tests remain for a small set of EBM/manifold Langevin ops.
- Coverage prose must avoid copying stale generated counts.

## Next Work

1. Treat generated dashboards as the only count authority.
2. Close backend-kernel proof through platform backend work.
3. Prioritize remaining batching/transpose/sharding gaps by model impact.
4. Keep KV-cache status tied to runtime/conformance proof, not only Graph IR
   lowering.
5. Update example status only when generated support/e2e dashboards agree.

## Source Material Consolidated

- `archive/advanced_examples_capability_gap.md`
- `archive/kv_cache_coverage_matrix.md`
- `archive/partial_ops_uplift_plan.md`
- `archive/primitive_coverage_state.md`

