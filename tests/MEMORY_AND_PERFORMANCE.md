# Tessera Test Suite — Memory and Performance Profile

Status: informative

This document characterizes what you should expect to see (peak RAM, wall
clock, parallelism budget) when running the Tessera Python test suites,
and explains the proof-layer and environment-marker boundaries.

If you only need one sentence: **run `python scripts/run_unit_tests.py`; it
selects the hermetic CPU PR state and sizes xdist to both RAM and cores.** Never
run measured performance under xdist, and never interpret a device skip as
native proof.

## Suite-by-suite footprint

Measured 2026-07-15 after the compiler test-state split. Full collection is
~15,326 tests. The CPU PR expression selected 14,217 tests on the WSL compiler
host: 12,366 passed and 1,851 capability-skipped in 6m18s with 24 loadfile
workers. Device and performance totals overlap other markers and are therefore
reported by their own collection commands, not added to the CPU total.

| Suite | Tests | Wall clock | Peak RSS | Source of pressure |
|---|---:|---:|---:|---|
| **CPU PR state** | 14,217 selected | 6m18s on 32-core/62-GB WSL (24 workers) | worker-count dependent | Hermetic semantics plus capability-skipped legacy tests; excludes slow, performance, and all hardware markers |
| **NVIDIA correctness** | 223 | NVIDIA-box measurement pending | device dependent | Exact-device execute/compare; `hardware_nvidia and not performance` |
| **NVIDIA performance** | 18 | NVIDIA-box measurement pending; serial only | device dependent | Repeated timing/resource ratchets; `hardware_nvidia and performance` |
| **Full collection** | ~15,326 | execution intentionally split by state | workload dependent | Includes CPU, external tools, all devices, measured performance, and the legacy slow tail |

The CPU lane remains larger than a pure unit suite because migration is
incremental. Compiler-tool, integration, device, performance, and audit layers
are being separated without deleting proof. The architecture ratchets prevent
new direct wall-clock tests from entering the CPU state and keep all required
entry points on the same marker expression.

## What gets allocated where

### CPU PR state (`python scripts/run_unit_tests.py`)

| Component | Approximate RSS contribution |
|---|---:|
| CPython 3.14 + pytest framework | ~50 MB |
| numpy + imported `tessera` modules + MLIR Python bindings | ~80 MB |
| Autodiff test working arrays (largest is ~`[2, 4, 12]` conv1d backward + `[2, 4, 3, 3]` group_norm backward) | < 50 MB |
| Pytree / RNG / state-tree tests | < 10 MB |
| **Total peak** | **~213 MB** |

### Slow suite (`-m slow`)

| Test | Peak transient RSS |
|---|---:|
| `test_benchmark_gemm.py::test_large_gemm_compute_bound` (fp32 8192×8192×8192) | ~1 GB (three matrices @ 256 MB each + temporary) |
| `test_superbench_compiler_smoke_emits_telemetry_summary` | ~500 MB subprocess that holds its own SuperBench state |
| `test_operator_*` (operator-bench bridge + full MLIR artifact walk) | ~500 MB subprocess transient |
| `test_benchmark_suite_exports_unified_telemetry` (`run_all_benchmarks` runs real attention + collectives) | ~300 MB |
| **Peak when running serially** | **~1.5–2 GB** (dominated by whichever single test is largest at any moment) |

Because pytest runs tests serially by default, only one heavy test is
resident at a time and the peak is dominated by the single largest one
(the 8192³ GEMM). pytest cleans up fixture state between tests, but
Python's allocator does not always return memory to the OS, so the
high-water mark can creep up over the full slow sweep — budget ~2 GB.

## Parallelism budget

With `pytest-xdist` (`pytest -n auto`), each worker holds its own state:

| Workers | CPU-state total RSS | Slow/performance total RSS |
|---:|---:|---:|
| 1 (default) | ~213 MB | ~1.5–2 GB |
| 4 | ~850 MB | ~6–8 GB |
| 8 | ~1.7 GB | ~12–16 GB |
| 10+ (typical M-series core count) | ~2.1 GB | ~15–20 GB |

**Recommendation:** use `scripts/run_unit_tests.py` so worker count is bounded
by RAM and cores. Slow and measured-performance states are dominated by
subprocesses, large arrays, compilation, or device timing; run them serially.

## Practical guidance by system RAM

| System RAM | What you can run |
|---|---|
| **4 GB** | CPU PR state with `TESSERA_TEST_WORKERS=1`; skip slow/performance |
| **8 GB** | CPU PR state plus selected slow tests serially |
| **16 GB+** | CPU PR state with a modest worker count; slow tests serially |
| **32 GB+** | Can run multiple worktrees / parallel pytest invocations |

For most contributor workflows **8 GB is plenty**.

## Daily-driver commands

| Workflow | Command | Expected RSS | Expected time |
|---|---|---:|---:|
| Edit-loop sanity check | `python scripts/run_unit_tests.py -q` | sized from host RAM/cores | 6m18s on the measured WSL host |
| Just the autodiff slice | `pytest tests/unit/test_autodiff_*.py tests/unit/test_conv1d_autodiff.py tests/unit/test_deferred_vjps.py tests/unit/test_sprint_*.py tests/unit/test_attention_family_support.py tests/unit/test_reasoning_model_support.py tests/unit/test_optimizer_mixed_precision_support.py tests/unit/test_standalone_compiler_roadmap.py -q` | < 200 MB | ~2 s |
| CUDA correctness on NVIDIA box | `pytest tests/unit -m "hardware_nvidia and not performance" -q` | device dependent | pending NVIDIA-box rebaseline |
| CUDA performance on NVIDIA box | `pytest tests/unit -m "hardware_nvidia and performance" -q -n 0` | device dependent | serial; pending NVIDIA-box rebaseline |
| MLIR lit fixtures | `lit tests/tessera-ir/ -v` (needs `tessera-opt` on PATH) | < 100 MB | < 1 s |

## What the `slow` marker covers

The original heavy benchmark modules remain slow-marked, and legacy live-device
tests may carry both `slow` and a target marker during migration. The target
marker is the environment truth; `slow` alone must not be used to select a
device lane. Representative CPU-heavy modules are:

| File | Tests | Reason |
|---|---:|---|
| `tests/unit/test_benchmark_gemm.py` | ~31 | Runs real GEMMs up to 8192³ fp32 |
| `tests/unit/test_benchmark_compiler_contract.py` | 8 | Spawns SuperBench `bench_run.py` subprocess; `run_all_benchmarks` runs real attention/collectives |
| `tests/unit/test_operator_benchmarks_contract.py` | 8 | Operator-bench bridge subprocess + full MLIR-artifact corpus walk |

The marker is **module-level**, not per-test — every test in those files
is excluded from `-m "not slow"`. To run just one heavy test by name,
use the explicit path: `pytest path/to/test_benchmark_gemm.py::TestGEMMRun::test_large_gemm_compute_bound`.

## Pre-existing failures unrelated to memory

None in the CPU PR state as of the 2026-07-15 host run. Exact CUDA execution is
tracked separately because the WSL compiler host does not expose NVIDIA
hardware or a CUDA toolkit.

## Cross-references

- `tests/COMPILER_TEST_PLAN.md` — overall test plan and CI tier matrix.
- `docs/architecture/compiler_test_architecture.md` — normative state model.
- `docs/audit/backend/nvidia/todo.md` — CUDA evaluation on the NVIDIA box.
- `pyproject.toml` `[tool.pytest.ini_options]` — marker declarations.
- `scripts/validate.sh` — CPU validation spine (calls the fast suite).
- `CLAUDE.md` "Testing" section — top-level command reference.
