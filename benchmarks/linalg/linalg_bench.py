#!/usr/bin/env python3
"""Linalg reference benchmark — cholesky / qr / svd / tri_solve.

Hardware-free CPU reference path.  For each op the script:

  * Builds a deterministic well-conditioned matrix.
  * Calls the Tessera op via ``tessera.ops.*``.
  * Enforces reconstruction/solve residuals within a tolerance derived
    from the matrix size. This is not an independent implementation oracle.
  * Times N repetitions (default: 5 warmup + 25 timed).
  * Emits one row per (op, size) in the canonical benchmark schema.

The output JSON envelope mirrors ``benchmarks/benchmark_gemm.py`` so
the existing ingestion in ``tools/roofline_tools/`` and
``benchmarks/run_all.py`` can read it without changes.

Run from the repo root::

    PYTHONPATH=.:python python benchmarks/linalg/linalg_bench.py \\
        --sizes 16,64,128 --reps 5 --output /tmp/linalg_smoke.json

Status: **reference**. This entry point exercises no native package.
Consult the compiler route census for backend capabilities; this suite
does not establish their absence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "python"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

import tessera as ts  # noqa: E402  — after sys.path bootstrap


def _spd_matrix(n: int, seed: int = 0) -> np.ndarray:
    """Well-conditioned SPD matrix for cholesky / tri_solve."""

    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(np.float64)
    return A @ A.T + n * np.eye(n, dtype=np.float64)


def _general_matrix(n: int, seed: int = 0) -> np.ndarray:
    """General non-singular matrix for qr / svd."""

    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n)).astype(np.float64)


def _time_op(fn, warmup: int, reps: int) -> tuple[float, float]:
    """Return (median_ms, min_ms) over ``reps`` timed runs."""

    if warmup < 0 or reps <= 0:
        raise ValueError("warmup must be nonnegative and reps positive")
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)), float(np.min(times))


def _row_envelope(
    op: str, n: int, median_ms: float, min_ms: float, err: float
) -> dict:
    """Match the schema in benchmarks/benchmark_gemm.py."""

    tolerance = max(1e-12, 100 * n * np.finfo(np.float64).eps)
    if not np.isfinite(err) or err > tolerance:
        raise RuntimeError(f"{op} residual {err} exceeds {tolerance}")

    return {
        "backend": "cpu_reference",
        "op": f"tessera.{op}",
        "shape": [n, n],
        "dtype": "fp64",
        "latency_ms": median_ms,
        "tflops": 0.0,  # not a TFLOPs-bound op
        "memory_bw_gb_s": 0.0,
        "device": "cpu",
        "tessera_version": getattr(ts, "__version__", "unknown"),
        "metadata": {
            "runtime_status": "reference",
            "kind": "linalg",
            "size": n,
            "median_ms": median_ms,
            "min_ms": min_ms,
            "correctness_residual": err,
            "correctness_tolerance": tolerance,
            "correctness_passed": True,
            "promotion_eligible": False,
            "notes": (
                "CPU numpy/scipy-backed reference; no native package is exercised."
            ),
        },
    }


def bench_cholesky(n: int, *, warmup: int, reps: int) -> dict:
    A = _spd_matrix(n)
    L = ts.ops.cholesky(A)
    err = float(np.linalg.norm(L @ L.T - A) / max(np.linalg.norm(A), 1e-12))
    fn = lambda: ts.ops.cholesky(A)  # noqa: E731 — closure over A
    median_ms, min_ms = _time_op(fn, warmup, reps)
    return _row_envelope("cholesky", n, median_ms, min_ms, err)


def bench_qr(n: int, *, warmup: int, reps: int) -> dict:
    A = _general_matrix(n)
    Q, R = ts.ops.qr(A)
    err = float(np.linalg.norm(Q @ R - A) / max(np.linalg.norm(A), 1e-12))
    fn = lambda: ts.ops.qr(A)  # noqa: E731
    median_ms, min_ms = _time_op(fn, warmup, reps)
    return _row_envelope("qr", n, median_ms, min_ms, err)


def bench_svd(n: int, *, warmup: int, reps: int) -> dict:
    A = _general_matrix(n)
    U, s, V = ts.ops.svd(A)
    err = float(
        np.linalg.norm(U @ np.diag(s) @ V - A)
        / max(np.linalg.norm(A), 1e-12)
    )
    fn = lambda: ts.ops.svd(A)  # noqa: E731
    median_ms, min_ms = _time_op(fn, warmup, reps)
    return _row_envelope("svd", n, median_ms, min_ms, err)


def bench_tri_solve(n: int, *, warmup: int, reps: int) -> dict:
    A = _spd_matrix(n)
    L = ts.ops.cholesky(A)
    rng = np.random.default_rng(1)
    b = rng.standard_normal(n).astype(np.float64)
    x = ts.ops.tri_solve(L, b, lower=True)
    err = float(np.linalg.norm(L @ x - b) / max(np.linalg.norm(b), 1e-12))
    fn = lambda: ts.ops.tri_solve(L, b, lower=True)  # noqa: E731
    median_ms, min_ms = _time_op(fn, warmup, reps)
    return _row_envelope("tri_solve", n, median_ms, min_ms, err)


_OPS = {
    "cholesky": bench_cholesky,
    "qr": bench_qr,
    "svd": bench_svd,
    "tri_solve": bench_tri_solve,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sizes", default="16,64,128",
        help="Comma-separated NxN sizes (default: 16,64,128).",
    )
    parser.add_argument(
        "--ops", default=",".join(sorted(_OPS)),
        help="Comma-separated op subset (default: all).",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Untimed warmup iterations (default: 5).",
    )
    parser.add_argument(
        "--reps", type=int, default=25,
        help="Timed iterations (default: 25).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: print to stdout).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            "CI smoke mode: small sizes and low reps; correctness remains enforced."
        ),
    )
    args = parser.parse_args(argv)

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    if args.smoke:
        sizes = [16]
        ops = list(_OPS)
        args.warmup = 1
        args.reps = 3

    if not sizes or any(n <= 0 for n in sizes) or not ops or args.warmup < 0 or args.reps <= 0:
        parser.error("positive sizes/reps, nonempty ops and nonnegative warmup required")
    runs: list[dict] = []
    for op in ops:
        if op not in _OPS:
            raise SystemExit(f"unknown op: {op!r} (choices: {sorted(_OPS)})")
        for n in sizes:
            row = _OPS[op](n, warmup=args.warmup, reps=args.reps)
            print(
                f"  [linalg_bench] {row['op']:24s}  n={n:4d}  "
                f"median={row['latency_ms']:8.3f}ms  "
                f"err={row['metadata']['correctness_residual']:.2e}", file=sys.stderr
            )
            runs.append(row)

    envelope = {
        "schema": "tessera.benchmark.v1",
        "backend": "cpu_reference",
        "runs": runs,
        "runs_count": len(runs),
        "ops": sorted({r["op"] for r in runs}),
        "device": "cpu",
        "tessera_version": getattr(ts, "__version__", "unknown"),
        "notes": (
            "Linalg reference benchmark (cholesky / qr / svd / "
            "tri_solve).  CPU numpy/scipy-backed reference path; "
            "this entry point does not exercise native backend packages."
        ),
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        print(json.dumps(envelope, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
