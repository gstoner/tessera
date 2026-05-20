# Plan: enable and harden the 47 partial ops

> **Status:** Active execution plan (2026-05-20).
> Backed by the live audit data in
> `docs/audit/generated/e2e_op_coverage.md`.
> Counts in this doc are accurate at landing time; the source of
> truth for ongoing tracking is the regenerated dashboard.

## The three buckets

The 47 partial ops fall into exactly three patterns when classified
by their blocking-axis signature. The buckets are mutually exclusive
+ the patterns are stable (verified via `e2e_coverage._candidate_op_names()`
+ `support_row_for(op).cells[axis].status`).

| Bucket | Count | Blocker pattern | Family | Fix type | Promotion target |
|---|---:|---|---|---|---|
| **E1** | 26 | `graph_ir=not_applicable, tile=fused, target=fused, runtime=unknown` | 17 GA + 9 EBM | Capability registry — add per-op `OpCapability` entries | PARTIAL → **COMPLETE** |
| **E2** | 4 | `graph_ir=missing, tile=fused, target=fused, runtime=unknown` | 4 M7 (`mobius` / `stereographic` / `complex_mul` / `complex_exp`) | op_catalog + capability registry | PARTIAL → **COMPLETE** |
| **E3** | 17 | `graph_ir=missing, tile=planned, target=planned, runtime=unknown` | 17 M7 | op_catalog + Python reference wiring | PARTIAL → **RUNNABLE_REFERENCE** (not COMPLETE — no native kernel) |

## What each fix actually does

### E1 — Bucket 1 (26 ops): capability-registry hygiene

These ops **already run natively**. The Apple GPU GA/EBM benchmark
(`benchmarks/apple_gpu/benchmark_ga_ebm.py`) exercises every one
end-to-end on Apple Silicon. The audit walker reports
`tile_ir=fused + target_ir=fused` from the backend manifest, but
`runtime=unknown` because `capabilities.TARGET_CAPABILITIES` has no
explicit per-op `OpCapability` entry for them.

**Fix:** add `OpCapability(runtime_status="fused")` rows to the
`apple_gpu` target capability for every `clifford_*` + `ebm_*` op
that ships a fused MSL kernel. The list is already in
`backend_manifest._CLIFFORD_APPLE_GPU_FUSED` +
`backend_manifest._EBM_APPLE_GPU_FUSED` — we're just teaching the
capability registry what the manifest already says.

**Cost:** ~30 lines of dict entries. Lowest risk imaginable.

**Acceptance:**
- `--render` shows `complete` count rise from 8 → 34.
- `partial` count drops from 47 → 21.
- Existing GA/EBM benchmark tests stay green (we didn't break runtime
  behavior — just made the audit honest).

### E2 — Bucket 3 (4 M7 ops): op_catalog + capability

`complex_mul`, `complex_exp`, `mobius`, `stereographic` ship fused
MSL kernels (the `_M7_BACKEND_ALIASES` map handles the
`mobius → complex_mobius` translation for `mobius` /
`stereographic`). The `target_ir=fused` axis reads correctly, but
`graph_ir=missing` because these ops aren't in `op_catalog.OP_SPECS`.

**Fix:** add 4 `OP_SPECS` entries with the same shape as existing
M7-adjacent ops (`gelu`, `softmax`, etc.) + 4 `OpCapability` rows
in the `apple_gpu` capability map.

**Cost:** ~50 lines per op × 4 = ~200 LOC.

**Acceptance:**
- `complete` rises from 34 → 38.
- `partial` drops from 21 → 17.

### E3 — Bucket 2 (17 M7 ops): Graph IR registration + reference path

`cross_ratio`, `is_concyclic`, `complex_log`, `complex_sqrt`,
`complex_pow`, `complex_conjugate`, `complex_abs`, `complex_arg`,
`complex_div`, `mobius_from_three_points`, `check_cauchy_riemann`,
`dz`, `dbar`, `laplacian_2d`, `conformal_jacobian`,
`conformal_energy_on_sphere`, `complex_jit`.

Every one runs **today** via `tessera.complex.*` (the M7 Python
reference surface, ~94 focused tests passing). The audit sees them
as `graph_ir=missing` because they aren't in `op_catalog.OP_SPECS`.

**Two sub-categories:**

  * **E3a (16 ops):** real ops that need `OP_SPECS` entries + a
    `runtime_status="reference"` entry on a target capability.
    Promote to RUNNABLE_REFERENCE.
  * **E3b (1 op):** `complex_jit` is the *decorator* alias for
    `analytic_symbolic`, **not** an op. It shows up in
    `_M7_INVENTORY` for discoverability (per the
    `test_every_m7_op_is_publicly_callable` contract). The audit
    classifier should exclude decorator aliases from partial-op
    counting. **Fix:** add a small `_DECORATOR_ALIASES` denylist
    in `e2e_coverage.py`.

**Cost:** ~30 lines × 16 ops + 1 classifier fix = ~480 LOC.

**Acceptance:**
- `runnable_reference` rises from 216 → 232.
- `partial` drops from 17 → 0.
- `complete` unchanged (no new native kernels).

## What the plan deliberately does NOT promise

- **E1/E2 do not add new native kernels.** They make the audit
  honest about kernels that *already* ship. If `runtime=fused`
  doesn't match the actual runtime behavior, that's a
  pre-existing bug (not caused by the audit edit).
- **E3 does not promise COMPLETE.** Without fused kernels for the
  17 M7 long-tail ops, the highest they can reach is
  `runnable_reference`. Native kernels for any subset are
  separately-scoped follow-up work that needs its own
  Phase-D-style trigger.
- **No correctness regression risk.** The audit data flows
  one-way from `capabilities.py` + `op_catalog.OP_SPECS` into the
  dashboard. Adding entries can't break anything that was
  working; missing entries are the bug.

## Phase execution order

1. **E1** (lowest risk, highest count) — 26 ops promoted to
   COMPLETE in one focused commit.
2. **E2** — 4 M7 ops promoted to COMPLETE.
3. **E3b** — classifier fix for decorator aliases. Trivial.
4. **E3a** — 16 M7 ops promoted to RUNNABLE_REFERENCE.

Each phase regenerates `e2e_op_coverage.md` and re-runs the drift
gate. The drift gate failure messages already show before/after
counts, so each phase is observable in the test output.

## Stopping condition

After E1 + E2 + E3: `partial = 0`. Every op in the audit is in
one of `complete` / `runnable_reference` / `artifact_only` /
`planned`. The 47 partial ops are gone — not because we built new
kernels, but because the registry has caught up to the runtime
reality (E1+E2) or because the registry honestly reports
"reference-only" instead of pretending lowering is partial (E3).

This makes Phase D triggers (CSE, DBE, real optimization passes)
meaningfully measurable: a benchmark gap on a `runnable_reference`
op points at "build a kernel"; a gap on a `complete` op points at
"optimize the existing kernel."
