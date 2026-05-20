"""Machine-readable manifest of every active ``tools/`` directory.

Tools is the third audited surface alongside ``examples/`` and
``benchmarks/``.  The directory holds compiler / runtime / profiling
infrastructure that ships with Tessera but doesn't fit the "model
example" shape.

Most rows here are either:

* ``runnable`` — Python CLI helpers (``tprof_report.py``,
  ``tessera-translate``) that take ``--help`` and exit 0.
* ``compile_only`` — C++ binaries (``tessera-opt``, the C++ profiler)
  that exist as build targets and are exercised by their respective
  CI lanes after ``cmake --build``.
* ``archived`` — historical subprojects kept in-tree for reference,
  but no longer treated as active compiler toolchain surface.

The CLI starter (``tools/CLI/Tessera_CLI_Starter_v0_1``) is a
historical standalone scaffold.  It remains tracked for reference,
but its active tool expectations are archived.
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
    # ── Tessera-opt (C++ MLIR driver) ────────────────────────────────
    SurfaceEntry(
        directory="tools/tessera-opt",
        entry_point="tools/tessera-opt/tessera-opt.cpp",
        status="compile_only",
        command=(
            "python -c \"import pathlib; "
            "assert pathlib.Path('tools/tessera-opt/CMakeLists.txt')."
            "is_file(); "
            "print('tessera-opt structural smoke ok — build owned by lit lane')\""
        ),
        notes=(
            "MLIR opt-style driver — registers 5 dialects (tessera, "
            "tessera.neighbors, tessera.solver, tessera_apple, tpp) "
            "+ 70+ passes + 6 named lowering pipelines.  Requires "
            "MLIR/LLVM 21 — proof tests live in "
            "``tests/unit/test_tessera_opt_build.py`` and the "
            "real cmake-build/lit smoke is owned by the opt-in "
            "``lit`` CI lane."
        ),
    ),
    # ── Tessera-translate (Python CLI + C++ MLIR binary) ─────────────
    SurfaceEntry(
        directory="tools/tessera-translate",
        entry_point="python/tessera/cli/translate.py",
        status="runnable",
        command=(
            "PYTHONPATH=python python -m tessera.cli.translate --help"
        ),
        notes=(
            "Two complementary surfaces: the Python CLI ("
            "``tessera-translate``) handles StableHLO / GGUF / "
            "SafeTensors export via ``tessera.aot``; the C++ "
            "binary (``tessera-translate-mlir``, "
            "``tools/tessera-translate/tessera-translate.cpp``) "
            "handles MLIR-native translation flags.  CI smoke is "
            "the Python CLI's ``--help``; MLIR binary proof lives "
            "in ``tests/unit/test_cli_translate.py``, "
            "``tests/unit/test_tessera_opt_build.py``, and the "
            "opt-in ``lit`` CI lane."
        ),
    ),
    # ── C++ profiler ─────────────────────────────────────────────────
    SurfaceEntry(
        directory="tools/profiler",
        entry_point="tools/profiler/cli/tprof.cpp",
        status="compile_only",
        command=(
            "python -c \"import pathlib; "
            "assert pathlib.Path('tools/profiler/CMakeLists.txt').is_file(); "
            "assert pathlib.Path('tests/unit/test_tools_subprojects.py').is_file(); "
            "print('profiler build smoke owned by test_tools_subprojects + validate.sh')\""
        ),
        notes=(
            "C++ profiler (tprof) — runtime, Perfetto trace writer, "
            "and report generator.  It is build-smoked by "
            "``tests/unit/test_tools_subprojects.py`` and by "
            "``scripts/validate.sh``."
        ),
    ),
    SurfaceEntry(
        directory="tools/profiler/scripts",
        entry_point="tools/profiler/scripts/tprof_report.py",
        status="runnable",
        command=(
            "PYTHONPATH=python python tools/profiler/scripts/"
            "tprof_report.py --help"
        ),
        notes=(
            "Python helpers that wrap the tprof binary's JSON "
            "outputs: HTML report renderer, roofline annotator, "
            "Perfetto trace viewer.  CI smoke is ``--help``."
        ),
    ),
    # ── Roofline tooling ─────────────────────────────────────────────
    SurfaceEntry(
        directory="tools/roofline_tools",
        entry_point=(
            "tools/roofline_tools/tools/roofline/cli_v2.py"
        ),
        status="runnable",
        command=(
            "python tools/roofline_tools/tools/roofline/cli_v2.py one "
            "--peaks tools/roofline_tools/tools/roofline/peaks/"
            "sm90_with_links.yaml "
            "--input tools/roofline_tools/tools/roofline/examples/"
            "nsight_min.csv "
            "--fmt nsight --dtype fp32 "
            "--outdir /tmp/tessera_roofline_tools_audit "
            "--export-json /tmp/tessera_roofline_tools_audit/"
            "classification.json"
        ),
        notes=(
            "Ingests Nsight CSV + Perfetto JSON traces and emits "
            "roofline-annotated HTML.  The audit smoke runs the "
            "bundled Nsight CSV example into ``/tmp``."
        ),
    ),
    # ── CLI starter scaffold ─────────────────────────────────────────
    SurfaceEntry(
        directory="tools/CLI/Tessera_CLI_Starter_v0_1",
        entry_point="tools/CLI/Tessera_CLI_Starter_v0_1/CMakeLists.txt",
        status="archived",
        reason=(
            "Historical standalone CLI starter suite.  It remains "
            "in-tree for reference, but the active compiler tools "
            "are the root ``tools/tessera-opt`` and "
            "``tools/tessera-translate`` surfaces."
        ),
        notes=(
            "C++ CLI starter — scaffolds tessera-{compile,run,opt,"
            "inspect,profiler,autotune,new} binaries.  CMake stages "
            "into ``_build/`` (gitignored as of 2026-05-19)."
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
    "This dashboard lists every active project under ``tools/``. "
    "Most rows are either Python CLI helpers (``runnable``) or "
    "C++ build targets (``compile_only``).  Archived rows ship with "
    "a STATUS.md naming why they are kept in-tree but no longer "
    "treated as active compiler tool surfaces.\n\n"
    "CI guards (run as part of ``scripts/validate.sh``):\n\n"
    "* ``python -m tessera.cli.surface_audit --surface=tools "
    "--check`` — executes every ``runnable`` row and "
    "``compile_only`` smokes; ``scaffold`` / ``broken`` / "
    "``archived`` rows are not executed.\n"
    "* ``python -m tessera.cli.claim_lint --surface=tools "
    "--check`` — flags overclaim language on ``scaffold`` / "
    "``broken`` / ``archived`` rows."
)


def render_markdown(entries: Iterable[SurfaceEntry] | None = None) -> str:
    rows = tuple(entries) if entries is not None else _ENTRIES
    return _render_markdown_shared(
        surface_title="Tessera Tools — Status Audit",
        surface_intro=_SURFACE_INTRO,
        entries=rows,
        regenerate_command=(
            "python -m tessera.cli.surface_audit "
            "--surface=tools --render"
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
