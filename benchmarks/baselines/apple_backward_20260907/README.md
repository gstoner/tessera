# Apple attention-backward runtime evidence — 2026-09-07

Five paired attention-backward benchmark reports on the Apple M1 Max (apple7)
plus the coverage record of the broader Metal VJP contract tests they accompany.

| File | Holds |
|---|---|
| `run-1.json` … `run-5.json` | Five `benchmark_attention_backward.py` reports (`context` with compiler/runtime fingerprints, `device: apple7`, `paired_route_order_rotated: true`, `profiling_capabilities`, `runs`). |
| `coverage.json` | The test run recorded alongside: `pytest -q tests/unit/test_apple_lowp_native_contract.py -k broader_metal` — 6 passed, 12 deselected — against a float64 analytic GQA VJP oracle with fp32 bias, with its `rtol`/`atol`, runtime and test-source SHA-256s and stated limitations. |
| `resolution.json` | Accepted rows, `context_mismatch_refused`, `workspace_fallback_verified` and the per-case resolution. |

Recorded by `benchmarks/apple_gpu/benchmark_attention_backward.py`
(`run-N.json`); `coverage.json` records the pytest command it names;
`resolution.json`: recorder not found in tree.

Cited by `docs/audit/backend/apple/todo.md` and
`docs/audit/compiler/INTEGRATED_COMPILER_LOG.md` ("Raw reports and coverage").
