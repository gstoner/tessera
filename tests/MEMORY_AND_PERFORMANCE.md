# Tessera Test Suite — Memory and Performance Profile

Status: informative

This document characterizes what you should expect to see (peak RAM, wall
clock, parallelism budget) when running the Tessera Python test suites,
and explains the `slow` marker boundary.

If you only need one sentence: **the default daily-driver sweep
(`pytest -m "not slow"`) peaks at ~125 MB resident and finishes in
~4 minutes (measured 2026-05-20).** The numbers below were measured
on Apple Silicon with Python 3.14 via `/usr/bin/time -l`; figures
will vary by ~25% across machines but the orders of magnitude hold.

## Suite-by-suite footprint

Measured 2026-06-02 against the current test tree (~6,900 fast / 777
deselected / ~7,680 total collected via `pytest --collect-only`).
Counts grew from earlier baselines as Apple GPU encode/conv2d,
MTL4 routing, optimizer-batching, descriptor, and feature-limit
suites landed:

| Suite | Tests | Wall clock | Peak RSS | Source of pressure |
|---|---:|---:|---:|---|
| **`-m "not slow"`** (default) | ~6,900 | ~4 min | **~125 MB** (measured `/usr/bin/time -l`) | Python interpreter + pytest + numpy + tessera modules; small fp64 working arrays for autodiff tests + the expanded S-series / Apple GPU encode-session / MTL4 / descriptor / feature-limit coverage |
| **`-m slow`** (heavy benchmarks) | 777 | ~30 min (serial) | ~1.5–2 GB (estimated) | fp32 8192³ GEMM in `test_benchmark_gemm.py`; SuperBench subprocess in `test_benchmark_compiler_contract.py`; operator-bench bridge in `test_operator_benchmarks_contract.py` |
| **Full suite** (`-m ""`) | ~7,680 | ~30+ min | ~1.5–2 GB | Same as `slow`; pytest is serial by default so peaks aren't additive |

The measured fast-suite numbers (~125 MB peak, ~4 min) leave a
comfortable margin on any modern machine.  Peak RSS dropped from the
2026-05 baseline of ~213 MB because every heavy test that previously
dominated steady-state allocation has been moved behind
`pytestmark = pytest.mark.slow` — the fast lane is now allocation-disciplined.
The wall-clock budget grew from ~31 s to ~4 min because the registry,
audit, Apple, and S-series coverage all expanded; the test-doc drift
gate (`tests/unit/test_test_docs_drift.py`) keeps this table in sync
with the actual measurements going forward.

## What gets allocated where

### Fast suite (`-m "not slow"`)

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

| Workers | Fast-suite total RSS | Slow-suite total RSS |
|---:|---:|---:|
| 1 (default) | ~213 MB | ~1.5–2 GB |
| 4 | ~850 MB | ~6–8 GB |
| 8 | ~1.7 GB | ~12–16 GB |
| 10+ (typical M-series core count) | ~2.1 GB | ~15–20 GB |

**Recommendation:** the fast suite is so quick (31 s serial) that
`-n auto` rarely pays off; the slow suite is dominated by subprocess
spawns and an 8192³ GEMM, neither of which parallelizes cleanly across
pytest workers. Stick to serial execution.

## Practical guidance by system RAM

| System RAM | What you can run |
|---|---|
| **4 GB** | Fast suite only (`pytest -m "not slow"`); skip slow suite |
| **8 GB** | Fast + slow comfortably, serial |
| **16 GB+** | Everything including small `-n` worker counts on the fast suite |
| **32 GB+** | Can run multiple worktrees / parallel pytest invocations |

For most contributor workflows **8 GB is plenty**.

## Daily-driver commands

| Workflow | Command | Expected RSS | Expected time |
|---|---|---:|---:|
| Edit-loop sanity check | `pytest tests/unit/ -m "not slow" -q` | < 256 MB | ~31 s |
| Just the autodiff slice | `pytest tests/unit/test_autodiff_*.py tests/unit/test_conv1d_autodiff.py tests/unit/test_deferred_vjps.py tests/unit/test_sprint_*.py tests/unit/test_attention_family_support.py tests/unit/test_reasoning_model_support.py tests/unit/test_optimizer_mixed_precision_support.py tests/unit/test_standalone_compiler_roadmap.py -q` | < 200 MB | ~2 s |
| Pre-PR full check | `pytest tests/unit/ -m "not slow" -q && pytest tests/unit/ -m slow -q` | ~2 GB peak | ~31 min |
| MLIR lit fixtures | `lit tests/tessera-ir/ -v` (needs `tessera-opt` on PATH) | < 100 MB | < 1 s |

## What the `slow` marker covers

Three test files carry `pytestmark = pytest.mark.slow` at module level:

| File | Tests | Reason |
|---|---:|---|
| `tests/unit/test_benchmark_gemm.py` | ~31 | Runs real GEMMs up to 8192³ fp32 |
| `tests/unit/test_benchmark_compiler_contract.py` | 8 | Spawns SuperBench `bench_run.py` subprocess; `run_all_benchmarks` runs real attention/collectives |
| `tests/unit/test_operator_benchmarks_contract.py` | 8 | Operator-bench bridge subprocess + full MLIR-artifact corpus walk |

The marker is **module-level**, not per-test — every test in those files
is excluded from `-m "not slow"`. To run just one heavy test by name,
use the explicit path: `pytest path/to/test_benchmark_gemm.py::TestGEMMRun::test_large_gemm_compute_bound`.

## Pre-existing failures unrelated to memory

Three tests in `tests/unit/test_debug_env.py::TestDiffCommand` have been
failing for several sessions:

- `test_identical_files_exit_zero`
- `test_differing_files_exit_one_and_show_diff`
- `test_diff_writes_output_file`

They expect a `tessera-mlir diff` CLI entry point that isn't being
resolved in the current dev environment. They reproduce on `main` before
any of the recent autodiff/S-series work and are **not** indicative of
memory or correctness issues. Track them separately when fixing the
CLI's diff subcommand.

## Cross-references

- `tests/COMPILER_TEST_PLAN.md` — overall test plan and CI tier matrix.
- `pyproject.toml` `[tool.pytest.ini_options]` — marker declarations.
- `scripts/validate.sh` — CPU validation spine (calls the fast suite).
- `CLAUDE.md` "Testing" section — top-level command reference.
