"""Machine-readable manifest of every active ``benchmarks/`` directory.

Mirrors :mod:`tessera.compiler.examples_manifest` for the benchmark
surface.  Powers two CI gates:

  * ``tessera.cli.surface_audit --surface=benchmarks --check`` runs
    each ``runnable`` row.
  * ``tessera.cli.claim_lint --surface=benchmarks --check`` scans
    each benchmark README for overclaim language on
    ``scaffold`` / ``broken`` / ``archived`` rows.

``benchmarks/archive/**`` is in scope here (unlike examples) because
some archived suites are kept in-tree for replay against historical
baselines, and we want them visible in the dashboard.
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
    # ── Top-level orchestrators ──────────────────────────────────────
    SurfaceEntry(
        directory="benchmarks",
        entry_point="benchmarks/run_all.py",
        status="runnable",
        command=(
            "PYTHONPATH=.:python python benchmarks/run_all.py "
            "--smoke --json-only --output /tmp/tessera_bench_smoke.json"
        ),
        notes=(
            "Top-level orchestrator. Sweeps GEMM / attention / "
            "collective via library modules in benchmarks/*.py. "
            "Uses a roofline-only path when no accelerator is present."
        ),
    ),
    SurfaceEntry(
        directory="benchmarks/baselines",
        entry_point="benchmarks/baselines/cpu_smoke.json",
        status="runnable",
        command=(
            "PYTHONPATH=.:python python benchmarks/perf_gate.py "
            "/tmp/tessera_bench_smoke.json --baseline "
            "benchmarks/baselines/cpu_smoke.json"
        ),
        notes=(
            "Recorded CPU smoke baseline used by perf_gate.py. The "
            "command shown gates the smoke output against it."
        ),
    ),
    # ── Apple GPU benchmark harnesses ────────────────────────────────
    SurfaceEntry(
        directory="benchmarks/apple_gpu",
        entry_point="benchmarks/apple_gpu/benchmark_ga_ebm.py",
        status="runnable",
        command=(
            "PYTHONPATH=python python benchmarks/apple_gpu/"
            "benchmark_ga_ebm.py --ci "
            "--output /tmp/tessera_ga_ebm_smoke.json"
        ),
        notes=(
            "GA + EBM end-to-end harness. ``--ci`` exits 0 on non-"
            "Darwin hosts after emitting `skipped_apple_gpu`. Apple "
            "Silicon hosts exercise 17 GA + 9 native EBM + 4 workload "
            "rows. The matched fusion sibling (``benchmark_fusion.py``) "
            "uses the same JSON schema."
        ),
    ),
    SurfaceEntry(
        directory="benchmarks/apple_gpu",
        entry_point="benchmarks/apple_gpu/benchmark_fusion.py",
        status="runnable",
        command=(
            "PYTHONPATH=python python benchmarks/apple_gpu/"
            "benchmark_fusion.py --shapes 4x4x4 "
            "--swiglu-shapes 1x4x4x4 --reps 2 "
            "--output /tmp/tessera_apple_gpu_fusion_smoke.json"
        ),
        notes=(
            "Apple GPU fusion sweep for matmul→softmax and SwiGLU. "
            "Skips cleanly on non-Darwin; on Darwin it emits a JSON "
            "row pair for fused vs sequential tiny shapes."
        ),
    ),
    # ── Linalg reference benchmark ───────────────────────────────────
    SurfaceEntry(
        directory="benchmarks/linalg",
        entry_point="benchmarks/linalg/linalg_bench.py",
        status="runnable",
        command=(
            "PYTHONPATH=python python benchmarks/linalg/linalg_bench.py "
            "--smoke --output /tmp/tessera_linalg_smoke.json"
        ),
        notes=(
            "Linalg reference benchmark — cholesky / qr / svd / "
            "tri_solve.  CPU numpy/scipy-backed; the numerical "
            "contract matches numpy to ~1e-14 (rel err).  Native "
            "backend lowering (Apple GPU MSL kernels, NVIDIA cuSOLVER "
            "bindings, ROCm hipSOLVER bindings) is a future M-series "
            "milestone — the benchmark stays useful as a correctness "
            "+ regression-bound for the reference path."
        ),
    ),
    # ── Spectral benchmark ───────────────────────────────────────────
    SurfaceEntry(
        directory="benchmarks/spectral",
        entry_point="benchmarks/spectral/spectral_bench.py",
        status="compile_only",
        command=(
            "PYTHONPATH=.:python python benchmarks/spectral/"
            "spectral_bench.py --ops fft1d --sizes 16 --batch 1 "
            "--repeats 1 --warmup 0 --backend tessera-artifact "
            "--outcsv /tmp/tessera_spectral_artifact_smoke.csv"
        ),
        notes=(
            "Spectral solver bench. The tessera-artifact backend "
            "emits IR but the Tile/Target runtime path is not "
            "executable yet — status will flip to runnable when a "
            "native FFT lowering lands.  Today the numpy backend "
            "produces correctness rows; the tessera backend produces "
            "artifact-only rows tagged as such in the JSON output."
        ),
    ),
    # ── Operator + whole-model harnesses ─────────────────────────────
    SurfaceEntry(
        directory="benchmarks/Tessera_Operator_Benchmarks",
        entry_point=(
            "benchmarks/Tessera_Operator_Benchmarks/scripts/opbench.py"
        ),
        status="compile_only",
        command=(
            "cmake -S benchmarks/Tessera_Operator_Benchmarks "
            "-B /tmp/tessera_opbench_audit_build && "
            "cmake --build /tmp/tessera_opbench_audit_build -j2 && "
            "PYTHONPATH=.:python python benchmarks/"
            "Tessera_Operator_Benchmarks/scripts/opbench.py --config "
            "benchmarks/Tessera_Operator_Benchmarks/scripts/configs/"
            "quick_sweep.yaml --bin /tmp/tessera_opbench_audit_build/"
            "opbench --backend reference --out /tmp/tessera_opbench_audit"
        ),
        notes=(
            "Operator-level C++ harness. The audit configures/builds "
            "in ``/tmp`` and runs the quick reference sweep across "
            "all seven operator groups. Deeper artifact and "
            "tessera-runtime bridge sweeps are covered by the slow "
            "operator-benchmark tests."
        ),
    ),
    SurfaceEntry(
        directory="benchmarks/Tessera_SuperBench",
        entry_point=(
            "benchmarks/Tessera_SuperBench/runner/bench_run.py"
        ),
        status="compile_only",
        command=(
            "PYTHONPATH=.:python python benchmarks/Tessera_SuperBench/"
            "runner/bench_run.py --help"
        ),
        notes=(
            "Whole-model harness (~30 min full sweep — marked "
            "``slow`` in the pytest suite). CI smoke is the "
            "``--help`` parse-only check; the real workload runs "
            "are off the critical path."
        ),
    ),
    # ── Research scaffold ────────────────────────────────────────────
    SurfaceEntry(
        directory="benchmarks/DeepScholar-Bench",
        entry_point=(
            "benchmarks/DeepScholar-Bench/tessera_deepscholar_model.py"
        ),
        status="runnable",
        command=(
            "PYTHONPATH=python python benchmarks/DeepScholar-Bench/"
            "tessera_deepscholar_model.py --output "
            "/tmp/tessera_deepscholar_smoke.json"
        ),
        notes=(
            "CPU smoke benchmark using current APIs only: "
            "``tessera.jit`` plus matmul / softmax / layer_norm over "
            "NumPy-backed text embeddings. The optional LOTUS adapter "
            "imports cleanly but remains guarded behind research extras."
        ),
    ),
    # ── Shared harness library ───────────────────────────────────────
    SurfaceEntry(
        directory="benchmarks/common",
        entry_point="benchmarks/common/__init__.py",
        status="compile_only",
        command=(
            "PYTHONPATH=python python -c "
            "\"import sys; sys.path.insert(0,'benchmarks'); "
            "from common import correctness, compiler_contract, "
            "artifact_schema; print('ok')\""
        ),
        notes=(
            "Shared benchmark harness library — correctness probes, "
            "compiler-contract checks, artifact JSON schema. Imported "
            "by every benchmark; the CI command is an import smoke."
        ),
    ),
    # ── Archived material ────────────────────────────────────────────
    SurfaceEntry(
        directory="benchmarks/archive/matrix_multiplication",
        entry_point=(
            "benchmarks/archive/matrix_multiplication/"
            "blackwell_matmul_tessera.py"
        ),
        status="archived",
        reason=(
            "Pre-Phase-6 matmul benchmark. Superseded by "
            "``benchmark_gemm.py`` + ``run_all.py``. Kept in-tree for "
            "historical replay; not part of the current performance "
            "story."
        ),
        notes="See benchmarks/archive/README.md for the deprecation note.",
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
    # Benchmarks tolerate ``archived`` rows without a STATUS.md since
    # the row's ``reason`` is sufficient and many archived dirs are
    # README-only.
    return _audit_filesystem_shared(
        rows, require_status_md_for=("scaffold", "broken"),
    )


_SURFACE_INTRO = (
    "This dashboard lists every active ``benchmarks/`` entry point "
    "and its **executable status**.  It is regenerated from "
    "``python/tessera/compiler/benchmarks_manifest.py``.\n\n"
    "CI guards (run as part of ``scripts/validate.sh``):\n\n"
    "* ``python -m tessera.cli.surface_audit --surface=benchmarks "
    "--check`` — executes every ``runnable`` row and "
    "``compile_only`` smokes; ``scaffold`` / ``broken`` / "
    "``archived`` rows are not executed.\n"
    "* ``python -m tessera.cli.claim_lint --surface=benchmarks "
    "--check`` — flags overclaim language on ``scaffold`` / "
    "``broken`` / ``archived`` rows."
)


def render_markdown(entries: Iterable[SurfaceEntry] | None = None) -> str:
    rows = tuple(entries) if entries is not None else _ENTRIES
    return _render_markdown_shared(
        surface_title="Tessera Benchmarks — Status Audit",
        surface_intro=_SURFACE_INTRO,
        entries=rows,
        regenerate_command=(
            "python -m tessera.cli.surface_audit "
            "--surface=benchmarks --render"
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
