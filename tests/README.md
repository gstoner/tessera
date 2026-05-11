# Tessera Tests

This directory holds every test suite that ships with Tessera. New
contributors should start here.

## Quick start

| Goal | Command |
|---|---|
| Daily edit-loop sanity check (2,214 tests, ~31 s, < 256 MB RAM) | `pytest tests/unit/ -m "not slow" -q` |
| Full Python suite including heavy benchmarks (~30 min, ~2 GB RAM) | `pytest tests/unit/ -q` |
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
| `tests/archive/` | Historical tests preserved for reference; not run | n/a |

## Markers

Declared in `pyproject.toml` `[tool.pytest.ini_options]`:

| Marker | Effect |
|---|---|
| `slow` | Excluded from `-m "not slow"`. Currently applied module-wide to `test_benchmark_gemm.py`, `test_benchmark_compiler_contract.py`, `test_operator_benchmarks_contract.py` — these are the 30-min heavy tail. |
| `performance` | Deterministic compiler performance / benchmark-proxy tests. |

See [`MEMORY_AND_PERFORMANCE.md`](MEMORY_AND_PERFORMANCE.md) for what
each marker actually costs.

## Plans and policies

| Document | Purpose |
|---|---|
| [`COMPILER_TEST_PLAN.md`](COMPILER_TEST_PLAN.md) | Authoritative test plan — Tier 0–4 CI matrix, layering by IR stage, test ownership |
| [`MEMORY_AND_PERFORMANCE.md`](MEMORY_AND_PERFORMANCE.md) | What to expect when you run the suites — peak RAM, wall clock, parallelism budget, `slow` marker rationale |

## Known pre-existing failures

Three tests in `tests/unit/test_debug_env.py::TestDiffCommand` fail in
the current dev environment because the `tessera-mlir diff` console
entry isn't being resolved. They reproduce on `main` and are unrelated
to the rest of the suite. Track them with the CLI's diff-subcommand fix.

## Related

- `CLAUDE.md` "Testing" section — top-level command reference.
- `scripts/validate.sh` — CPU validation spine; runs the fast unit
  sweep + version checks + runtime + benchmark smoke.
- `scripts/test.sh` — local test driver shared with CI.
