<!-- AUTO-GENERATED — DO NOT EDIT BY HAND. -->
<!-- Regenerate via: python -m tessera.cli.surface_audit --surface=tools --render -->

# Tessera Tools — Status Audit

This dashboard lists every active project under ``tools/``. Most rows are either Python CLI helpers (``runnable``) or C++ build targets (``compile_only``).  Archived rows ship with a STATUS.md naming why they are kept in-tree but no longer treated as active compiler tool surfaces.

CI guards (run as part of ``scripts/validate.sh``):

* ``python -m tessera.cli.surface_audit --surface=tools --check`` — executes every ``runnable`` row and ``compile_only`` smokes; ``scaffold`` / ``broken`` / ``archived`` rows are not executed.
* ``python -m tessera.cli.claim_lint --surface=tools --check`` — flags overclaim language on ``scaffold`` / ``broken`` / ``archived`` rows.

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
| ``runnable`` | 3 |
| ``runnable_optional`` | 0 |
| ``compile_only`` | 2 |
| ``scaffold`` | 0 |
| ``broken`` | 0 |
| ``archived`` | 1 |
| **total** | **6** |

## Entries

| Directory | Status | Entry point | Command / Reason |
|-----------|--------|-------------|------------------|
| ``tools/tessera-opt`` | ``compile_only`` | ``tools/tessera-opt/tessera-opt.cpp`` | ``python -c "import pathlib; assert pathlib.Path('tools/tessera-opt/CMakeLists.txt').is_file(); print('tessera-opt structural smoke ok — build owned by lit lane')"`` |
| ``tools/tessera-translate`` | ``runnable`` | ``python/tessera/cli/translate.py`` | ``PYTHONPATH=python python -m tessera.cli.translate --help`` |
| ``tools/profiler`` | ``compile_only`` | ``tools/profiler/cli/tprof.cpp`` | ``python -c "import pathlib; assert pathlib.Path('tools/profiler/CMakeLists.txt').is_file(); assert pathlib.Path('tests/unit/test_tools_subprojects.py').is_file(); print('profiler build smoke owned by test_tools_subprojects + validate.sh')"`` |
| ``tools/profiler/scripts`` | ``runnable`` | ``tools/profiler/scripts/tprof_report.py`` | ``PYTHONPATH=python python tools/profiler/scripts/tprof_report.py --help`` |
| ``tools/roofline_tools`` | ``runnable`` | ``tools/roofline_tools/tools/roofline/cli_v2.py`` | ``python tools/roofline_tools/tools/roofline/cli_v2.py one --peaks tools/roofline_tools/tools/roofline/peaks/sm90_with_links.yaml --input tools/roofline_tools/tools/roofline/examples/nsight_min.csv --fmt nsight --dtype fp32 --outdir /tmp/tessera_roofline_tools_audit --export-json /tmp/tessera_roofline_tools_audit/classification.json`` |
| ``tools/CLI/Tessera_CLI_Starter_v0_1`` | ``archived`` | ``tools/CLI/Tessera_CLI_Starter_v0_1/CMakeLists.txt`` | Historical standalone CLI starter suite.  It remains in-tree for reference, but the active compiler tools are the root ``tools/tessera-opt`` and ``tools/tessera-translate`` surfaces. |

