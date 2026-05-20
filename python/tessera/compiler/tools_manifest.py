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
* ``broken`` — entries with known import/build issues.  Each ships a
  STATUS.md naming the failure mode.

The CLI starter (``tools/CLI/Tessera_CLI_Starter_v0_1``) is a
scaffold project that produces a runnable binary via CMake; its row
is ``compile_only`` because the smoke probe is the build, not an
execution.
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
            "MLIR/LLVM 21 — the real cmake-build/lit smoke is owned "
            "by the opt-in ``lit`` CI lane."
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
            "the Python CLI's ``--help``."
        ),
    ),
    # ── C++ profiler ─────────────────────────────────────────────────
    SurfaceEntry(
        directory="tools/profiler",
        entry_point="tools/profiler/cli/tprof.cpp",
        status="compile_only",
        command=(
            "python -c \"import pathlib; "
            "assert pathlib.Path('tools/profiler/CMakeLists.txt')."
            "is_file(); "
            "print('profiler structural smoke ok — build owned by build lane')\""
        ),
        notes=(
            "C++ profiler (tprof) — runtime, Perfetto trace writer, "
            "and report generator.  The real cmake-build smoke is "
            "owned by the ``build`` CI lane in validate.yml."
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
        status="broken",
        reason=(
            "``cli_v2.py`` imports ``from tprof_roofline.model "
            "import DevicePeaks, analyze``, but the bundled "
            "``tprof_roofline/model.py`` does not export "
            "``analyze`` — ImportError at module load.  Either "
            "rename the import to a symbol that exists, or add "
            "``analyze`` to the model module."
        ),
        notes=(
            "Ingests Nsight CSV + Perfetto JSON traces and emits "
            "roofline-annotated HTML.  Status will flip to "
            "runnable once the import is restored."
        ),
    ),
    # ── CLI starter scaffold ─────────────────────────────────────────
    SurfaceEntry(
        directory="tools/CLI/Tessera_CLI_Starter_v0_1",
        entry_point="tools/CLI/Tessera_CLI_Starter_v0_1/CMakeLists.txt",
        status="compile_only",
        command=(
            "python -c \"import pathlib; "
            "assert pathlib.Path("
            "'tools/CLI/Tessera_CLI_Starter_v0_1/CMakeLists.txt')."
            "is_file(); "
            "print('cli starter structural smoke ok')\""
        ),
        notes=(
            "C++ CLI starter — scaffolds tessera-{compile,run,opt,"
            "inspect,profiler,autotune,new} binaries.  CMake stages "
            "into ``_build/`` (gitignored as of 2026-05-19).  The "
            "audit's smoke is structural; the build itself is owned "
            "by a downstream CMake lane when one lands."
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
    "C++ build targets (``compile_only``).  Broken rows ship with "
    "a STATUS.md naming the failure mode."
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
