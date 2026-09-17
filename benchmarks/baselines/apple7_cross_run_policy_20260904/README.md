# Apple7 cross-run policy comparison — 2026-09-04

Eight independent-process `benchmark_legacy_retune.py` reports on the Apple M1
Max (apple7) and a predeclared comparison plan, used to decide whether the
default cross-run route-selection policy should change.

| File | Holds |
|---|---|
| `plan.json` | The predeclared plan: first seed, run count, reps, trials, process timeout, and the synthetic candidate slowdown factors / run index used for the sensitivity scenarios. |
| `inputs.json` | SHA-256 of the inputs the runs were bound to: `benchmark_legacy_retune.py`, `compare_cross_run_policy.py`, the Apple runtime dylib, `apple_route_selector.py` and `apple_gpu_runtime.mm`. |
| `run-00.json` … `run-07.json` | The eight raw `benchmark_legacy_retune.py` reports (`context`, `low_precision_candidates`, `runs`, `schema_version`). |
| `summary.json` | `tessera.apple.cross-run-policy-comparison.v1`: the per-scenario decisions and the recorded verdict — retain the `mean_student_t` default, `default_policy_changed: false`, `promotion_allowed: false`, scope `policy_analysis_not_sealed_device_evidence`. |

Recorded by `benchmarks/apple_gpu/compare_cross_run_policy.py`, which writes
`plan.json`, `inputs.json` and `summary.json` and invokes
`benchmarks/apple_gpu/benchmark_legacy_retune.py` once per `run-NN.json`.

Cited by `docs/audit/backend/apple/todo.md`,
`docs/audit/compiler/INTEGRATED_COMPILER_LOG.md` (Apple cross-run decision) and
`tests/unit/test_apple_cross_run_comparison.py`.
