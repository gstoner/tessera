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
    # ── Apple CPU execution-kind microbench (Apple plan B, 2026-05-20) ──
    SurfaceEntry(
        directory="benchmarks/apple_cpu",
        entry_point="benchmarks/apple_cpu/benchmark_execution_kind.py",
        status="runnable",
        command=(
            "PYTHONPATH=python python benchmarks/apple_cpu/"
            "benchmark_execution_kind.py --ci --output "
            "/tmp/tessera_apple_cpu_execution_kind.json"
        ),
        notes=(
            "Empirically proves the apple_cpu ``execution_kind`` axis "
            "in ``apple_target_map.py``: matmul tracks numpy's "
            "Accelerate-backed cblas_sgemm (proof of "
            "``accelerate_native``); layer_norm / softmax / gelu "
            "naturally track numpy (proof of ``numpy_reference``). "
            "The empirical gate is ratio-based since macOS numpy "
            "already routes through Accelerate."
        ),
    ),
    # ── Spectral correctness proof lane (Phase A3, 2026-05-20) ──────
    SurfaceEntry(
        directory="benchmarks/spectral",
        entry_point="benchmarks/spectral/spectral_correctness.py",
        status="runnable",
        command=(
            "PYTHONPATH=python python benchmarks/spectral/"
            "spectral_correctness.py --output "
            "/tmp/tessera_spectral_correctness.json"
        ),
        notes=(
            "Builds + runs the C++ ``ts-spectral-correctness`` "
            "microbench (Stockham vs naive DFT at N={64,128,256,512,"
            "1024}), scrapes its key=value output into the standard "
            "tessera.benchmark.v1 JSON schema, and exits non-zero on "
            "any size that deviates more than the 1e-3 abs tolerance.  "
            "This is the **correctness sentinel** every future native "
            "FFT lowering certifies against — runs in <1s, self-"
            "contained (no MLIR build dependency)."
        ),
    ),
    # ── Spectral benchmark ───────────────────────────────────────────
    SurfaceEntry(
        directory="benchmarks/spectral",
        entry_point="benchmarks/spectral/spectral_bench.py",
        # Phase A1 (2026-05-20): the bench now ships a
        # ``--backend tessera-runtime`` lane that calls every spectral
        # op through ``tessera.ops.registry``'s reference path,
        # producing real ``time_ms`` / ``gflops`` / ``gbs`` rows plus
        # an ``err_rel`` correctness check against the numpy
        # baseline.  Status flipped ``compile_only`` → ``runnable``.
        # The original ``tessera-artifact`` smoke continues to ship
        # for IR-emission validation (artifact_only rows in the same
        # JSON schema).
        status="runnable",
        command=(
            "PYTHONPATH=.:python python benchmarks/spectral/"
            "spectral_bench.py --ops fft1d --sizes 64,128 --batch 1 "
            "--repeats 3 --warmup 1 --backend tessera-runtime "
            "--outcsv /tmp/tessera_spectral_runtime_smoke.csv"
        ),
        notes=(
            "Spectral solver bench.  The ``tessera-runtime`` backend "
            "routes through ``tessera.ops.registry``'s numpy "
            "reference path for fft/ifft/rfft/irfft/dct/spectral_conv/"
            "stft/istft, producing real latency + correctness rows "
            "(``execution_kind=reference``).  Native FFT lowering "
            "(WGMMA/MFMA/MSL) is the next gate — when it lands the "
            "same lane reports faster numbers without any schema "
            "change.  The legacy ``--backend tessera-artifact`` "
            "smoke continues to ship IR-emission validation."
        ),
    ),
    # ── Operator + whole-model harnesses ─────────────────────────────
    SurfaceEntry(
        directory="benchmarks/Tessera_Operator_Benchmarks",
        entry_point=(
            "benchmarks/Tessera_Operator_Benchmarks/scripts/opbench.py"
        ),
        # Phase A2 (2026-05-20): the audit command actually builds AND
        # executes the C++ reference sweep end-to-end across all seven
        # operator groups (matmul / conv2d / flash_attention / reduce /
        # elementwise / softmax_layernorm / transpose_gather), producing
        # ``avg_ms`` / ``gflops`` / ``gbps`` rows in
        # ``/tmp/tessera_opbench_audit/results.csv``.  Status flipped
        # ``compile_only`` → ``runnable`` to match what the lane really
        # delivers.  The artifact-mode + tessera-runtime-bridge sweeps
        # remain slow-marked and stay off the critical path.
        status="runnable",
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
            "all seven operator groups, producing real latency + "
            "throughput numbers per row. Deeper artifact and "
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
