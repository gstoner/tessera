# Tessera Tests

This directory holds every test suite that ships with Tessera. New
contributors should start here.

## Quick start

| Goal | Command |
|---|---|
| Daily edit-loop sanity check (~9,600 fast tests, ~5 min, < 512 MB RAM) | `pytest tests/unit/ -m "not slow" -q` |
| Full Python suite including heavy benchmarks (~10,380 collected total; ~30 min full sweep, ~2 GB RAM) | `pytest tests/unit/ -q` |
| MLIR lit fixtures (needs `tessera-opt` on `$PATH`) | `lit tests/tessera-ir/ -v` |
| Just the autodiff slice (~213 tests, ~2 s) | `pytest tests/unit/test_autodiff_*.py tests/unit/test_conv1d_autodiff.py tests/unit/test_deferred_vjps.py tests/unit/test_sprint_*.py tests/unit/test_standalone_compiler_roadmap.py -q` |

Before running anything heavy, **read
[`MEMORY_AND_PERFORMANCE.md`](MEMORY_AND_PERFORMANCE.md)** — it tells you
exactly what each suite costs in RAM and wall clock, and explains the
`slow` marker boundary so you don't accidentally trigger the 30-minute
SuperBench / GEMM tail.

## Layout

| Directory | Purpose | Default run |
|---|---|---|
| `tests/unit/` | Fast correctness contracts for Python compiler APIs, IR emission, pass preconditions, CPU proxy execution, autodiff coverage, registry guards | `pytest -m "not slow"` |
| `tests/tessera-ir/` | FileCheck-based MLIR pass and pipeline lit fixtures | `lit tests/tessera-ir/` |
| `tests/performance/` | Deterministic roofline/proxy performance contracts (compile latency, generated-artifact size, GEMM/attention/collective timings, benchmark JSON schema) | `cmake --build . --target check-tessera-performance` or `TESSERA_RUN_PERFORMANCE_TESTS=1 ./scripts/test.sh` |
| `tests/kernel_tests/` | C++ kernel-level tests | Built via CMake when CUDA/HIP backends are enabled |
| `tests/integration/` | Cross-component integration tests | Opt-in |
| `tests/regression/` | Regression cases that locked in past bug fixes | Auto-run with unit |
| `tests/tessera_numerical_validation/` | Reference-vs-runtime comparisons for compiled CPU/mock and future hardware backends | Opt-in pytest suite |
| `tests/tessera_tests/` | Test utilities and harness fixtures | Shared infra |
| `archive/tests/` | Historical tests preserved for reference; not run | n/a |

## Markers

Declared in `pyproject.toml` `[tool.pytest.ini_options]`:

| Marker | Effect |
|---|---|
| `slow` | Excluded from `-m "not slow"`. Currently applied module-wide to `test_benchmark_gemm.py`, `test_benchmark_compiler_contract.py`, `test_operator_benchmarks_contract.py` — these are the 30-min heavy tail. |
| `performance` | Deterministic compiler performance / benchmark-proxy tests. |
| `hardware_apple_gpu` | Tests that require a Darwin host with Metal hardware. Skipped silently when collecting on non-Darwin or in CI without hardware. |
| `hardware_nvidia` | Tests that require an NVIDIA GPU with CUDA toolkit. |
| `hardware_rocm` | Tests that require an AMD GPU with the ROCm toolkit. |

The `hardware_*` markers are how `scripts/release_gate.py --target=<accel>` selects per-target tests for the release-gate hardware lane. Tests not yet using a marker still rely on `skipif(sys.platform != "darwin")`-style guards; landing the marker on each hardware test is incremental work and tracked in the tests-manifest dashboard.

See [`MEMORY_AND_PERFORMANCE.md`](MEMORY_AND_PERFORMANCE.md) for what
each marker actually costs.

## Plans and policies

| Document | Purpose |
|---|---|
| [`COMPILER_TEST_PLAN.md`](COMPILER_TEST_PLAN.md) | Authoritative test plan — Tier 0–4 CI matrix, layering by IR stage, test ownership |
| [`MEMORY_AND_PERFORMANCE.md`](MEMORY_AND_PERFORMANCE.md) | What to expect when you run the suites — peak RAM, wall clock, parallelism budget, `slow` marker rationale |

## Known pre-existing failures

None.  The previous "three tests in
`tests/unit/test_debug_env.py::TestDiffCommand` fail" note was retired
on 2026-05-20 — the `tessera-mlir diff` console entry is fully wired
and the affected tests pass on every supported host.  If you see a
fresh "pre-existing failure" claim, please open an issue: the
test-doc drift gate (`tests/unit/test_test_docs_drift.py`) is supposed
to catch stale entries in this section.

## Related

- `CLAUDE.md` "Testing" section — top-level command reference.
- `scripts/validate.sh` — CPU validation spine; runs the fast unit
  sweep + version checks + runtime + benchmark smoke.
- `scripts/test.sh` — local test driver shared with CI.
