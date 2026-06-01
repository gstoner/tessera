"""Machine-readable manifest of every active ``research/`` project.

Research lives outside the production source tree per
``research/README.md`` — it's where compiler / IR / verifier
experiments incubate before promotion.  This manifest tracks which
projects are runnable today, which are illustrative scaffolds, and
which have known breaks.

The manifest is intentionally small (one row per top-level research
project) so each row can carry a meaningful smoke command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tessera.compiler.surface_manifest import (
    ALLOWED_STATUSES,
    SurfaceEntry,
    audit_filesystem as _audit_filesystem_shared,
    render_markdown as _render_markdown_shared,
    status_counts as _status_counts_shared,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]


_ENTRIES: tuple[SurfaceEntry, ...] = (
    SurfaceEntry(
        directory="research/pddl_instruct",
        entry_point=(
            "research/pddl_instruct/tools/validator/validator.py"
        ),
        status="runnable",
        command=(
            "PYTHONPATH=python python research/pddl_instruct/tools/"
            "validator/validator.py "
            "--trace research/pddl_instruct/examples/traces/"
            "flash_trace.jsonl "
            "--out /tmp/tessera_pddl_validator_smoke.json"
        ),
        notes=(
            "Symbolic / verifier-guided reasoning experiment. "
            "Validator core does pre/effect execution + numeric "
            "constraint checks against a CoT trace.  The smoke "
            "command validates the bundled flash_trace.jsonl and "
            "emits a JSON report."
        ),
    ),
    SurfaceEntry(
        directory="research/sandbox_compilers",
        entry_point="research/sandbox_compilers/tilec/driver.py",
        status="compile_only",
        command=(
            "PYTHONPATH=research/sandbox_compilers python3 -m tilec.driver "
            "research/sandbox_compilers/examples/matmul.tss "
            "--backend c --out /tmp/tilec_smoke --dump-ir"
        ),
        notes=(
            "Frontend-to-backend compiler experiment for "
            "TileScript-style DSLs.  Indentation crash in "
            "``tilec/driver.py`` was fixed 2026-05-31 and the stale "
            "argparse ``choices`` list was extended to include the "
            "``rocm`` / ``nvidia`` branches that already existed in "
            "the body.  The smoke command parses "
            "``examples/matmul.tss``, prints the textual IR, and "
            "emits a C backend artifact (``mm.c`` + ``Makefile``) "
            "into the output directory.  ``compile_only`` because "
            "the driver emits source artifacts rather than executing "
            "them; building / running the generated C is a separate "
            "step."
        ),
    ),
)


def all_entries() -> tuple[SurfaceEntry, ...]:
    return _ENTRIES


def entries_by_status(status: str) -> tuple[SurfaceEntry, ...]:
    if status not in ALLOWED_STATUSES:
        raise ValueError(
            f"status={status!r} not in {ALLOWED_STATUSES!r}"
        )
    return tuple(e for e in _ENTRIES if e.status == status)


def status_counts() -> dict[str, int]:
    return _status_counts_shared(_ENTRIES)


def find_by_directory(directory: str) -> SurfaceEntry | None:
    target = directory.rstrip("/")
    for e in _ENTRIES:
        if e.directory == target:
            return e
    return None


def audit_filesystem(
    entries: Iterable[SurfaceEntry] | None = None,
) -> list[str]:
    rows = tuple(entries) if entries is not None else _ENTRIES
    return _audit_filesystem_shared(rows)


_SURFACE_INTRO = (
    "This dashboard lists every active project under ``research/``. "
    "Research experiments live here per ``research/README.md`` — they "
    "are not part of the production source tree.  The audit makes "
    "their runnable status explicit so a ``broken`` row gets fixed "
    "instead of silently rotting.\n\n"
    "CI guards (run as part of ``scripts/validate.sh``):\n\n"
    "* ``python -m tessera.cli.surface_audit --surface=research "
    "--check`` — executes every ``runnable`` row; ``scaffold`` / "
    "``broken`` / ``archived`` rows are not executed.\n"
    "* ``python -m tessera.cli.claim_lint --surface=research "
    "--check`` — flags overclaim language on ``scaffold`` / "
    "``broken`` / ``archived`` rows.\n\n"
    "Per-project STATUS.md files (when present) explain the path "
    "forward."
)


def render_markdown(entries: Iterable[SurfaceEntry] | None = None) -> str:
    rows = tuple(entries) if entries is not None else _ENTRIES
    return _render_markdown_shared(
        surface_title="Tessera Research — Status Audit",
        surface_intro=_SURFACE_INTRO,
        entries=rows,
        regenerate_command=(
            "python -m tessera.cli.surface_audit "
            "--surface=research --render"
        ),
    )


__all__ = [
    "all_entries",
    "audit_filesystem",
    "entries_by_status",
    "find_by_directory",
    "render_markdown",
    "status_counts",
]
