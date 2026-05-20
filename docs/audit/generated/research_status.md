<!-- AUTO-GENERATED — DO NOT EDIT BY HAND. -->
<!-- Regenerate via: python -m tessera.cli.surface_audit --surface=research --render -->

# Tessera Research — Status Audit

This dashboard lists every active project under ``research/``. Research experiments live here per ``research/README.md`` — they are not part of the production source tree.  The audit makes their runnable status explicit so a ``broken`` row gets fixed instead of silently rotting.

Per-project STATUS.md files (when present) explain the path forward.

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
| ``compile_only`` | 0 |
| ``scaffold`` | 0 |
| ``broken`` | 1 |
| ``archived`` | 0 |
| **total** | **2** |

## Entries

| Directory | Status | Entry point | Command / Reason |
|-----------|--------|-------------|------------------|
| ``research/pddl_instruct`` | ``runnable`` | ``research/pddl_instruct/tools/validator/validator.py`` | ``PYTHONPATH=python python research/pddl_instruct/tools/validator/validator.py --trace research/pddl_instruct/examples/traces/flash_trace.jsonl --out /tmp/tessera_pddl_validator_smoke.json`` |
| ``research/sandbox_compilers`` | ``broken`` | ``research/sandbox_compilers/tilec/driver.py`` | ``tilec/driver.py`` carries a SyntaxError at line 35: the ``cpu`` branch body and the subsequent ``elif`` are over-indented relative to the outer ``if`` chain. ``from .backends import codegen_cpu`` is at column 12 instead of 8.  Until the indentation is fixed, the whole module fails at import time. |

