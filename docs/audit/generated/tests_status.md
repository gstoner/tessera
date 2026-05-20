<!-- AUTO-GENERATED — DO NOT EDIT BY HAND. -->
<!-- Regenerate via: python -m tessera.cli.surface_audit --surface=tests --render -->

# Tessera Tests — Status Audit

This dashboard lists every ``tests/`` subtree and its **executable status**.  It is regenerated from ``python/tessera/compiler/tests_manifest.py``.

CI guards:

* ``python -m tessera.cli.surface_audit --surface=tests --check`` — runs every ``runnable`` smoke + every ``compile_only`` collect-only smoke; ``scaffold`` / ``broken`` / ``archived`` rows are not executed.
* ``tests/unit/test_tests_manifest.py`` — drift gate that fails CI when the on-disk doc diverges from the renderer.

## Status taxonomy

| Status              | Meaning                                                       |
|---------------------|---------------------------------------------------------------|
| ``runnable``          | Runs on default venv + CPU-only CI.                            |
| ``runnable_optional`` | Runs when declared ``extras_required`` are importable.         |
| ``compile_only``      | Emits IR/artifacts but does not execute the workload.        |
| ``scaffold``          | Intentionally illustrative; not runnable today.              |
| ``broken``            | Expected to run, currently fails — followup needed.          |
| ``archived``          | Intentionally retired; in-tree for reference only.           |

## Counts

| Status | Count |
|--------|------:|
| ``runnable`` | 1 |
| ``runnable_optional`` | 0 |
| ``compile_only`` | 3 |
| ``scaffold`` | 4 |
| ``broken`` | 0 |
| ``archived`` | 2 |
| **total** | **10** |

## Entries

| Directory | Status | Entry point | Command / Reason |
|-----------|--------|-------------|------------------|
| ``tests/unit`` | ``runnable`` | ``tests/unit`` | ``python -m pytest tests/unit/ -q -m 'not slow' --collect-only --no-header`` |
| ``tests/unit/_slow_subset`` | ``compile_only`` | ``tests/unit (slow-marked tests)`` | ``python -m pytest tests/unit/ -q -m 'slow' --collect-only --no-header`` |
| ``tests/tessera-ir`` | ``compile_only`` | ``tests/tessera-ir`` | ``python -c "import pathlib; assert pathlib.Path('tests/tessera-ir').is_dir(); print('tessera-ir lit fixtures present')"`` |
| ``tests/performance`` | ``compile_only`` | ``tests/performance/test_compiler_performance_plan.py`` | ``python -m pytest tests/performance/ --collect-only -q --no-header`` |
| ``tests/kernel_tests`` | ``scaffold`` | ``tests/kernel_tests/README_TESSERA_PERF.md`` | C++ kernel-level scaffold (CUDA / HIP / ROCm).  Built via CMake when ``TESSERA_ENABLE_CUDA=ON`` or ``TESSERA_ENABLE_HIP=ON``.  Not exercised in the CPU validation spine.  Promotion to ``runnable`` is gated on Phase G (NVIDIA) / Phase H (ROCm) hardware bring-up. |
| ``tests/tessera_tests/tessera_kernels_scaffold`` | ``archived`` | ``tests/tessera_tests/tessera_kernels_scaffold/README_TESSERA_PERF.md`` | Structurally-similar scaffold to ``tests/kernel_tests/`` with the same README + ci/configs/scripts/tests layout.  Kept in-tree for reference until the kernel-tests lane is validated against real hardware (Phase G / H), at which point this directory becomes a candidate for merge or deletion. |
| ``tests/tessera_numerical_validation`` | ``scaffold`` | ``tests/tessera_numerical_validation/run_all.sh`` | Numerical validation harness (reference-vs-runtime comparisons for compiled CPU + future hardware backends).  Today the directory contains ``README.md`` + ``requirements.txt`` + ``run_all.sh`` + a ``tessera_numerics/`` Python package, but **no test_*.py files** — pytest doesn't pick up any tests here.  Modernization onto current APIs (``ts.jit``, ``fn.explain()``, ``execution_kind``, fallback_reason) is deferred until a workload genuinely needs it. |
| ``tests/integration`` | ``scaffold`` | ``tests/integration`` | Directory reserved for cross-component integration tests.  Currently empty (no test_*.py files).  Pytest skips it gracefully.  Status = scaffold so reviewers see it surface in the dashboard. |
| ``tests/regression`` | ``scaffold`` | ``tests/regression`` | Directory reserved for regression cases that lock in past bug fixes.  Currently empty (no test_*.py files).  Net-new regression tests should land under ``tests/unit/`` until the regression directory has its own ownership story. |
| ``tests/archive`` | ``archived`` | ``tests/archive`` | Historical tests preserved for reference; not run in any CI lane. |

