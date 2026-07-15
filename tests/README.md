# Tessera Tests

This directory holds every test suite that ships with Tessera. New
contributors should start here.

## Quick start

| Goal | Command |
|---|---|
| Hermetic CPU PR lane (14,217 selected on 2026-07-15; 6m18s on a 24-worker WSL host) | `python scripts/run_unit_tests.py -q` |
| Full Python collection, including device/performance states (~15,326 tests) | `pytest tests/unit/ tests/device/nvidia/ tests/performance/nvidia/ tests/integration/ --collect-only -q` |
| NVIDIA exact-device correctness (RTX 5070 Ti box) | `pytest tests/unit/ tests/device/nvidia/ tests/integration/ -m "hardware_nvidia and not performance" -q` |
| NVIDIA measured performance, serial (RTX 5070 Ti box) | `pytest tests/unit/ tests/device/nvidia/ tests/performance/nvidia/ tests/integration/ -m "hardware_nvidia and performance" -q -n 0` |
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
| `tests/unit/` | Transitional flattened suite; proof layer and environment are selected by markers while families migrate to dedicated directories | `python scripts/run_unit_tests.py` |
| `tests/device/nvidia/` | NVIDIA exact-device suites; selected only by the NVIDIA proof commands, with node-ID maps retained for each relocation | `pytest tests/unit tests/device/nvidia tests/integration -m hardware_nvidia -q` |
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
| `slow` | Legacy runtime/cost partition. Heavy benchmark modules use it module-wide; some device tests retain it during migration. It is not an environment marker. |
| `compiler_tool` | Requires a built external compiler tool such as `tessera-opt`, `tessera-nvidia-opt`, or `mlir-opt`. |
| `compiler_nvidia` | NVIDIA-owned compiler-artifact proof; selects the CUDA compiler lane without a filename allowlist. |
| `integration` | Crosses a child-process, package, runtime, or component boundary. |
| `performance` | Uses measured wall-clock/device timing; excluded from the parallel CPU PR lane and run serially. |
| `hardware_apple_gpu` | Tests that require a Darwin host with Metal hardware. Skipped silently when collecting on non-Darwin or in CI without hardware. |
| `hardware_nvidia` | Tests that require an NVIDIA GPU with CUDA toolkit. |
| `hardware_rocm` | Tests that require an AMD GPU with the ROCm toolkit. |

The `hardware_*` markers are how target boxes select exact-device tests. The
CPU PR expression excludes every hardware marker and `performance`; a missing
device must not turn a native proof into a reference success. CUDA's current
lane and NVIDIA-box work plan are recorded in
[`docs/audit/backend/nvidia/todo.md`](../docs/audit/backend/nvidia/todo.md).

See [`MEMORY_AND_PERFORMANCE.md`](MEMORY_AND_PERFORMANCE.md) for what
each marker actually costs.

## Plans and policies

| Document | Purpose |
|---|---|
| [`COMPILER_TEST_PLAN.md`](COMPILER_TEST_PLAN.md) | Authoritative test plan — Tier 0–4 CI matrix, layering by IR stage, test ownership |
| [`MEMORY_AND_PERFORMANCE.md`](MEMORY_AND_PERFORMANCE.md) | What to expect when you run the suites — peak RAM, wall clock, parallelism budget, `slow` marker rationale |
| [`compiler_test_architecture.md`](../docs/architecture/compiler_test_architecture.md) | Normative proof-layer, environment-state, and migration rules |

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
