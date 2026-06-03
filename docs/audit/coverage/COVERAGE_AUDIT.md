# Coverage Audit

This document consolidates primitive, op, example, KV-cache, and coverage audit
material.

> **Counts live in `docs/audit/generated/`, never in this prose.** Per
> Decision #25/#26, every numeric coverage claim is owned by a drift-gated
> generated dashboard. This page states *qualitative* status and **links** to
> the dashboard that holds the live number — it does not copy counts (a copied
> number silently goes stale). When you need a figure, read the linked
> dashboard.

## Finished

- **Partial-op uplift closed the legacy partial rows.** E2E op coverage now
  shows no `partial` / `planned` rows — see
  [`generated/e2e_op_coverage.md`](../generated/e2e_op_coverage.md).
- **E2E op pipeline is native-complete or runnable-reference end to end** (no
  partial/planned tail). Live native-complete / runnable-reference split:
  [`generated/e2e_op_coverage.md`](../generated/e2e_op_coverage.md).
- **`lowering_rule` is closed project-wide** (0 open across all S-series
  categories) — see [`generated/s_series_status.md`](../generated/s_series_status.md).
- **No actionable direct-test-debt** (`needs_direct_test = 0`) — see
  [`generated/test_coverage_classification.md`](../generated/test_coverage_classification.md)
  for the full classification (covered-by-family / structural-only /
  hardware-gated breakdown).
- Advanced examples largely moved from missing APIs to backend/hardware proof.
- KV-cache coverage has explicit target diagnostics and historical matrices.
- **Manifold Langevin backend coverage (2026-06-02):** the EBM/manifold
  Langevin ops moved off `backend_kernel=planned` — `ebm_sphere_langevin_step`
  + `ebm_bivector_langevin_step` are now `partial` with a **real fused Apple
  GPU kernel** (sphere: dedicated MSL; bivector: the affine `ebm_langevin_step`
  kernel on grade-2 coeffs), and the two chain wrappers
  (`*_langevin_sample`) are `partial` via their numpy reference. See
  [`generated/s_series_status.md`](../generated/s_series_status.md) /
  [`generated/apple_target_map.md`](../generated/apple_target_map.md). (Their
  *distributed-mesh* axis stays Phase-G-gated — see the `hardware_gated` row in
  [`generated/test_coverage_classification.md`](../generated/test_coverage_classification.md);
  single-device kernel ≠ multi-GPU mesh.)

## Still Open

- **Backend-kernel proof is open on every S-series primitive** — a universal
  Phase-G/H/I gate (each entry needs *all* declared targets to ship real
  kernels; gated on NVIDIA/ROCm/Metalium hardware). Live open/complete counts:
  [`generated/s_series_status.md`](../generated/s_series_status.md).
- **Long-tail transform axes** — partially closed (2026-06-02).
  `batching_rule` closed for the textbook-batchable families (collective /
  recurrent / state_space / linalg decomposition+solver / sparse /
  segment_reduce) — the remaining open entries are the genuinely mesh-aware
  ones (moe / moe_transport / kv-cache state). `transpose_rule` and
  `sharding_rule` are the remaining increments (transpose closes mostly via
  `not_applicable` for non-linear families + `complete` for the linear ones;
  sharding is largely Phase-G-mesh-pending). Live open counts + per-category
  breakdown: [`generated/s_series_status.md`](../generated/s_series_status.md).
- **Hardware-gated tests** remain for a small set of EBM/manifold Langevin ops
  — see the `hardware_gated` row in
  [`generated/test_coverage_classification.md`](../generated/test_coverage_classification.md).

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

