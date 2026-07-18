"""Machine-readable manifest of every ``tests/`` subtree.

Tests is the fifth audited surface alongside ``examples/``,
``benchmarks/``, ``research/``, and ``tools/``.  The directory holds
every test suite that ships with Tessera, and this manifest is the
governance record for "what status is each subtree in and how does it
get exercised in CI?".

The reuse story is identical to the four sibling surface manifests:

* The ``SurfaceEntry`` dataclass + ``ALLOWED_STATUSES`` taxonomy come
  from :mod:`tessera.compiler.surface_manifest`.
* The status taxonomy is reused verbatim — ``runnable`` for active
  fast suites, ``compile_only`` for build-only / lit-only lanes,
  ``scaffold`` for opt-in heavy or scaffold suites, ``archived`` for
  historical material, ``broken`` for known-failing rows (none today).
* The audit walker re-runs every ``runnable`` row in CI via
  ``surface_audit --surface=tests --check``; ``scaffold`` / ``broken``
  / ``archived`` rows are NOT executed and must carry a ``reason``.
* The drift gate at ``tests/unit/test_surface_audit.py`` checks the
  consolidated ``docs/audit/generated/surface_status.csv`` artifact,
  which includes the tests surface.

Apple plan + test-tree review (2026-05-20).
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
    # ── Fast Python unit suite — the daily edit-loop lane ────────────
    SurfaceEntry(
        directory="tests/unit",
        entry_point="tests/unit",
        status="runnable",
        command=(
            "python -m pytest tests/unit/ -q -m 'not slow and not performance "
            "and not hardware_apple_gpu and not hardware_nvidia "
            "and not hardware_rocm' "
            "--collect-only --no-header"
        ),
        notes=(
            "Daily hermetic CPU edit-loop suite. Measured performance and "
            "hardware-marked tests are separate proof layers.  "
            "Smoke = collect-only so the manifest check stays under "
            "a second; the real exec runs via "
            "``scripts/validate.sh`` and ``release_gate.py``."
        ),
    ),
    # ── Slow Python tail — opt-in 30-minute benchmark heavy ─────────
    SurfaceEntry(
        directory="tests/unit/_slow_subset",
        entry_point="tests/unit (slow-marked tests)",
        status="compile_only",
        command=(
            "python -m pytest tests/unit/ -q -m 'slow' "
            "--collect-only --no-header"
        ),
        notes=(
            "The ``-m slow`` partition — 778 tests dominated by "
            "``test_benchmark_gemm.py``, ``test_benchmark_compiler_contract.py``, "
            "``test_operator_benchmarks_contract.py``.  Status is "
            "``compile_only`` because the audit runs the collect-only "
            "smoke; full execution is the 30-minute opt-in lane (see "
            "``MEMORY_AND_PERFORMANCE.md``)."
        ),
    ),
    # ── MLIR lit fixtures — owned by the dedicated lit lane ─────────
    SurfaceEntry(
        directory="tests/tessera-ir",
        entry_point="tests/tessera-ir",
        status="compile_only",
        command=(
            "python -c \"import pathlib; "
            "assert pathlib.Path('tests/tessera-ir').is_dir(); "
            "print('tessera-ir lit fixtures present')\""
        ),
        notes=(
            "FileCheck-based MLIR pass and pipeline lit fixtures.  "
            "Structurally present (this manifest's smoke just confirms "
            "the directory).  Real exec needs ``lit`` + a built "
            "``tessera-opt`` binary — owned by the lit CI lane, not "
            "the manifest's surface-audit lane."
        ),
    ),
    # ── Performance test sub-suite ──────────────────────────────────
    SurfaceEntry(
        directory="tests/performance",
        entry_point="tests/performance/test_compiler_performance_plan.py",
        status="compile_only",
        command=(
            "python -m pytest tests/performance/test_compiler_performance_plan.py "
            "tests/performance/nvidia/ "
            "--collect-only -q --no-header"
        ),
        notes=(
            "Deterministic roofline / proxy performance contracts.  "
            "Run by ``TESSERA_RUN_PERFORMANCE_TESTS=1 ./scripts/test.sh`` "
            "or the ``check-tessera-performance`` CMake target.  "
            "Manifest smoke = collect-only."
        ),
    ),
    # ── C++ kernel-level tests (CUDA/HIP extension) ─────────────────
    SurfaceEntry(
        directory="tests/kernel_tests",
        entry_point="tests/kernel_tests/README_TESSERA_PERF.md",
        status="scaffold",
        reason=(
            "C++ kernel-level scaffold (CUDA / HIP / ROCm).  Built "
            "via CMake when ``TESSERA_ENABLE_CUDA=ON`` or "
            "``TESSERA_ENABLE_HIP=ON``.  Not exercised in the CPU "
            "validation spine.  Promotion to ``runnable`` is gated "
            "on Phase G (NVIDIA) / Phase H (ROCm) hardware bring-up."
        ),
        notes=(
            "Tracked separately from "
            "``tests/tessera_tests/tessera_kernels_scaffold/`` which "
            "is a structurally-similar historical scaffold; a future "
            "follow-up may merge them once both have hardware lanes."
        ),
    ),
    # ── Duplicate kernel scaffold (likely superseded) ───────────────
    SurfaceEntry(
        directory="tests/tessera_tests/tessera_kernels_scaffold",
        entry_point="tests/tessera_tests/tessera_kernels_scaffold/README_TESSERA_PERF.md",
        status="archived",
        reason=(
            "Structurally-similar scaffold to ``tests/kernel_tests/`` "
            "with the same README + ci/configs/scripts/tests layout.  "
            "Kept in-tree for reference until the kernel-tests lane "
            "is validated against real hardware (Phase G / H), at "
            "which point this directory becomes a candidate for "
            "merge or deletion."
        ),
        notes=(
            "If you're looking for an active kernel-level scaffold, "
            "use ``tests/kernel_tests/`` — that's the canonical one."
        ),
    ),
    # ── Numerical validation scaffold ───────────────────────────────
    SurfaceEntry(
        directory="tests/tessera_numerical_validation",
        entry_point="tests/tessera_numerical_validation/run_all.sh",
        status="scaffold",
        reason=(
            "Numerical validation harness (reference-vs-runtime "
            "comparisons for compiled CPU + future hardware "
            "backends).  Today the directory contains ``README.md`` "
            "+ ``requirements.txt`` + ``run_all.sh`` + a "
            "``tessera_numerics/`` Python package, but **no "
            "test_*.py files** — pytest doesn't pick up any tests "
            "here.  Modernization onto current APIs (``ts.jit``, "
            "``fn.explain()``, ``execution_kind``, fallback_reason) "
            "is deferred until a workload genuinely needs it."
        ),
        notes=(
            "Run the legacy harness via "
            "``bash tests/tessera_numerical_validation/run_all.sh`` "
            "after installing ``requirements.txt`` extras."
        ),
    ),
    # ── Integration sub-suite (currently empty) ─────────────────────
    SurfaceEntry(
        directory="tests/integration",
        entry_point="tests/integration",
        status="scaffold",
        reason=(
            "Directory reserved for cross-component integration "
            "tests.  Currently empty (no test_*.py files).  Pytest "
            "skips it gracefully.  Status = scaffold so reviewers "
            "see it surface in the dashboard."
        ),
    ),
    # ── Regression sub-suite (currently empty) ──────────────────────
    SurfaceEntry(
        directory="tests/regression",
        entry_point="tests/regression",
        status="scaffold",
        reason=(
            "Directory reserved for regression cases that lock in "
            "past bug fixes.  Currently empty (no test_*.py files).  "
            "Net-new regression tests should land under "
            "``tests/unit/`` until the regression directory has its "
            "own ownership story."
        ),
    ),
    # ── Archived material ───────────────────────────────────────────
    SurfaceEntry(
        directory="archive/tests",
        entry_point="archive/tests",
        status="archived",
        reason=(
            "Historical tests preserved for reference; not run in "
            "any CI lane."
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
    # The slow subset row points at a logical path (no `_slow_subset`
    # directory exists on disk); allow it to skip the filesystem
    # presence check.  Same trick for the "kernel_tests scaffold"
    # entry-point README which exists but is referenced as a file
    # marker rather than a runnable script.
    rows = tuple(
        e for e in rows
        if e.directory != "tests/unit/_slow_subset"
    )
    return _audit_filesystem_shared(
        rows,
        require_status_md_for=(),  # tests-tree rows don't need STATUS.md
    )


_SURFACE_INTRO = (
    "This dashboard lists every ``tests/`` subtree and its "
    "**executable status**.  It is regenerated from "
    "``python/tessera/compiler/tests_manifest.py``.\n\n"
    "CI guards:\n\n"
    "* ``python -m tessera.cli.surface_audit --surface=tests "
    "--check`` — runs every ``runnable`` smoke + every "
    "``compile_only`` collect-only smoke; ``scaffold`` / ``broken`` "
    "/ ``archived`` rows are not executed.\n"
    "* ``tests/unit/test_surface_audit.py`` — drift gate that "
    "fails CI when the consolidated surface-status artifact diverges "
    "from the renderer."
)


def render_markdown(entries: Iterable[SurfaceEntry] | None = None) -> str:
    rows = tuple(entries) if entries is not None else _ENTRIES
    return _render_markdown_shared(
        surface_title="Tessera Tests — Status Audit",
        surface_intro=_SURFACE_INTRO,
        entries=rows,
        regenerate_command=(
            "python -m tessera.cli.surface_audit "
            "--surface=tests --render"
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
