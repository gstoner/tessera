# Apple attention-backward mixed-runtime evidence — 2026-09-07

Refreshed runtime evidence for the bounded native package VJP with f16 and bf16
inputs and fp32 bias on the Apple M1 Max (apple7).

| File | Holds |
|---|---|
| `run-1.json` … `run-5.json` | Five `benchmark_attention_backward.py` reports (`context`, `device: apple7`, `paired_route_order_rotated: true`, `profiling_capabilities`, `runs`). |
| `coverage.json` | The test run recorded alongside: `tests/unit/test_apple_lowp_native_contract.py -k metal_differential`, 10 passed; scope "bounded native package VJP including f16 and bf16 inputs with fp32 bias"; `promotion_scope: runtime_route_only`; the runtime source SHA-256. |
| `metal_differential.xml` | The pytest JUnit log of that run (`evidence_log`). |
| `resolution.json` | Accepted rows, `context_mismatch_refused`, `workspace_fallback_verified`, per-case resolution, and the fleet packet status / tested commit. |

Recorded by `benchmarks/apple_gpu/benchmark_attention_backward.py`
(`run-N.json`); `coverage.json` and `metal_differential.xml` record the pytest
run they name; `resolution.json`: recorder not found in tree.

Cited by `docs/audit/backend/apple/todo.md` and
`docs/audit/compiler/INTEGRATED_COMPILER_LOG.md` ("refreshed runtime evidence").
