<!-- AUTO-GENERATED — DO NOT EDIT BY HAND. -->
<!-- Regenerate via: python -m tessera.cli.surface_audit --surface=benchmarks --render -->

# Tessera Benchmarks — Status Audit

This dashboard lists every active ``benchmarks/`` entry point and its **executable status**.  It is regenerated from ``python/tessera/compiler/benchmarks_manifest.py``.

CI guards (run as part of ``scripts/validate.sh``):

* ``python -m tessera.cli.surface_audit --surface=benchmarks --check`` — executes every ``runnable`` row and ``compile_only`` smokes; ``scaffold`` / ``broken`` / ``archived`` rows are not executed.
* ``python -m tessera.cli.claim_lint --surface=benchmarks --check`` — flags overclaim language on ``scaffold`` / ``broken`` / ``archived`` rows.

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
| ``runnable`` | 4 |
| ``runnable_optional`` | 0 |
| ``compile_only`` | 4 |
| ``scaffold`` | 1 |
| ``broken`` | 0 |
| ``archived`` | 1 |
| **total** | **10** |

## Entries

| Directory | Status | Entry point | Command / Reason |
|-----------|--------|-------------|------------------|
| ``benchmarks`` | ``runnable`` | ``benchmarks/run_all.py`` | ``PYTHONPATH=.:python python benchmarks/run_all.py --smoke --json-only --output /tmp/tessera_bench_smoke.json`` |
| ``benchmarks/baselines`` | ``runnable`` | ``benchmarks/baselines/cpu_smoke.json`` | ``PYTHONPATH=.:python python benchmarks/perf_gate.py /tmp/tessera_bench_smoke.json --baseline benchmarks/baselines/cpu_smoke.json`` |
| ``benchmarks/apple_gpu`` | ``runnable`` | ``benchmarks/apple_gpu/benchmark_ga_ebm.py`` | ``PYTHONPATH=python python benchmarks/apple_gpu/benchmark_ga_ebm.py --ci --output /tmp/tessera_ga_ebm_smoke.json`` |
| ``benchmarks/linalg`` | ``runnable`` | ``benchmarks/linalg/linalg_bench.py`` | ``PYTHONPATH=python python benchmarks/linalg/linalg_bench.py --smoke --output /tmp/tessera_linalg_smoke.json`` |
| ``benchmarks/spectral`` | ``compile_only`` | ``benchmarks/spectral/spectral_bench.py`` | ``PYTHONPATH=.:python python benchmarks/spectral/spectral_bench.py --ops fft --sizes 16 --batch 1 --repeats 1 --warmup 0 --backend numpy`` |
| ``benchmarks/Tessera_Operator_Benchmarks`` | ``compile_only`` | ``benchmarks/Tessera_Operator_Benchmarks/scripts/opbench.py`` | ``PYTHONPATH=.:python python benchmarks/Tessera_Operator_Benchmarks/scripts/opbench.py --help`` |
| ``benchmarks/Tessera_SuperBench`` | ``compile_only`` | ``benchmarks/Tessera_SuperBench/runner/bench_run.py`` | ``PYTHONPATH=.:python python benchmarks/Tessera_SuperBench/runner/bench_run.py --help`` |
| ``benchmarks/DeepScholar-Bench`` | ``scaffold`` | ``benchmarks/DeepScholar-Bench/tessera_deepscholar_model.py`` | Research sketch — imports non-existent ``tessera.models.HierarchicalReasoningModel`` and ``tessera.attention.FlashMLA``. The whole module is vocabulary borrowing against a future model surface that isn't on the canonical Tessera API. Rewrite against the current surface, or move to ``benchmarks/archive/``. |
| ``benchmarks/common`` | ``compile_only`` | ``benchmarks/common/__init__.py`` | ``PYTHONPATH=python python -c "import sys; sys.path.insert(0,'benchmarks'); from common import correctness, compiler_contract, artifact_schema; print('ok')"`` |
| ``benchmarks/archive/matrix_multiplication`` | ``archived`` | ``benchmarks/archive/matrix_multiplication/blackwell_matmul_tessera.py`` | Pre-Phase-6 matmul benchmark. Superseded by ``benchmark_gemm.py`` + ``run_all.py``. Kept in-tree for historical replay; not part of the current performance story. |

