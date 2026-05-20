<!-- AUTO-GENERATED — DO NOT EDIT BY HAND. -->
<!-- Regenerate via: python -m tessera.cli.surface_audit --surface=tools --render -->

# Tessera Tools — Status Audit

This dashboard lists every active project under ``tools/``. Most rows are either Python CLI helpers (``runnable``) or C++ build targets (``compile_only``).  Broken rows ship with a STATUS.md naming the failure mode.

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
| ``runnable`` | 2 |
| ``runnable_optional`` | 0 |
| ``compile_only`` | 3 |
| ``scaffold`` | 0 |
| ``broken`` | 1 |
| ``archived`` | 0 |
| **total** | **6** |

## Entries

| Directory | Status | Entry point | Command / Reason |
|-----------|--------|-------------|------------------|
| ``tools/tessera-opt`` | ``compile_only`` | ``tools/tessera-opt/tessera-opt.cpp`` | ``python -c "import pathlib; assert pathlib.Path('tools/tessera-opt/CMakeLists.txt').is_file(); print('tessera-opt structural smoke ok — build owned by lit lane')"`` |
| ``tools/tessera-translate`` | ``runnable`` | ``python/tessera/cli/translate.py`` | ``PYTHONPATH=python python -m tessera.cli.translate --help`` |
| ``tools/profiler`` | ``compile_only`` | ``tools/profiler/cli/tprof.cpp`` | ``python -c "import pathlib; assert pathlib.Path('tools/profiler/CMakeLists.txt').is_file(); print('profiler structural smoke ok — build owned by build lane')"`` |
| ``tools/profiler/scripts`` | ``runnable`` | ``tools/profiler/scripts/tprof_report.py`` | ``PYTHONPATH=python python tools/profiler/scripts/tprof_report.py --help`` |
| ``tools/roofline_tools`` | ``broken`` | ``tools/roofline_tools/tools/roofline/cli_v2.py`` | ``cli_v2.py`` imports ``from tprof_roofline.model import DevicePeaks, analyze``, but the bundled ``tprof_roofline/model.py`` does not export ``analyze`` — ImportError at module load.  Either rename the import to a symbol that exists, or add ``analyze`` to the model module. |
| ``tools/CLI/Tessera_CLI_Starter_v0_1`` | ``compile_only`` | ``tools/CLI/Tessera_CLI_Starter_v0_1/CMakeLists.txt`` | ``python -c "import pathlib; assert pathlib.Path('tools/CLI/Tessera_CLI_Starter_v0_1/CMakeLists.txt').is_file(); print('cli starter structural smoke ok')"`` |

