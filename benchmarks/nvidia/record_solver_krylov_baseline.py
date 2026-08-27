#!/usr/bin/env python3
"""Record SM120 dense-Krylov scaling and solver-matmul ratchet rows."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
OUT = ROOT / "benchmarks/baselines/nvidia_sm120_solver_krylov_performance.json"


def _operator(order: int, algorithm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.eye(order, dtype=np.float32) * np.float32(3.5)
    index = np.arange(order - 1)
    matrix[index, index + 1] = np.float32(0.35 if algorithm == "gmres" else -0.75)
    matrix[index + 1, index] = np.float32(-0.55 if algorithm == "gmres" else -0.75)
    expected = np.sin(np.linspace(-1.0, 2.0, order, dtype=np.float32))
    return matrix, matrix @ expected, expected


def _wall_ms(fn: Any) -> float:
    start = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns() - start) / 1.0e6


def record(*, reps: int = 7, warmup: int = 2, device_reps: int = 5,
           margin: float = 2.5) -> list[dict[str, Any]]:
    from tessera import runtime as rt
    from tessera.compiler.emit.nvidia_solver_krylov import run_dense_krylov

    if rt._nvidia_device_name() != "sm_120":
        return []
    rows: list[dict[str, Any]] = []
    previous_ctas: dict[str, int] = {}
    for algorithm in ("cg", "gmres"):
        for order in (513, 1025, 2049):
            matrix, rhs, expected = _operator(order, algorithm)
            kwargs = dict(
                algorithm=algorithm, tolerance=2.0e-6, max_iterations=96,
                restart=16, reduction_ctas=0,
            )
            solution, residual, _aux, info = run_dense_krylov(matrix, rhs, **kwargs)
            true_residual = rhs - matrix @ solution
            limit = 2.0e-6 * max(1.0, float(np.linalg.norm(rhs)))
            if not np.allclose(residual, true_residual, rtol=2e-3, atol=2e-5):
                raise RuntimeError(f"{algorithm}/{order}: returned residual is not b-Ax")
            if float(np.linalg.norm(true_residual)) > limit:
                raise RuntimeError(f"{algorithm}/{order}: true residual exceeds {limit}")
            if not np.allclose(solution, expected, rtol=2e-5, atol=2e-5):
                raise RuntimeError(f"{algorithm}/{order}: independent known-solution oracle failed")
            ctas = int(info["reduction_ctas"])
            if ctas < 2 or ctas <= previous_ctas.get(algorithm, 0):
                raise RuntimeError(f"{algorithm}: CTA scaling did not grow at order {order}: {ctas}")
            previous_ctas[algorithm] = ctas

            for _ in range(warmup):
                run_dense_krylov(matrix, rhs, **kwargs)
            wall_samples = [
                _wall_ms(lambda matrix=matrix, rhs=rhs, kwargs=kwargs:
                         run_dense_krylov(matrix, rhs, **kwargs))
                for _ in range(reps)
            ]
            _x, _r, _a, measured = run_dense_krylov(
                matrix, rhs, repetitions=device_reps, **kwargs
            )
            for domain, median in (
                ("end_to_end", float(statistics.median(wall_samples))),
                ("device_event", float(measured["device_elapsed_ms"])),
            ):
                rows.append({
                    "op": f"dense_{algorithm}", "shape": f"{order}x{order}",
                    "dtype": "f32", "mode": f"cooperative_grid:{domain}",
                    "selected_route": "cooperative_grid",
                    "timing_domain": domain, "median_ms": round(median, 6),
                    "max_latency_ms": round(median * margin, 6),
                    "reduction_ctas": ctas,
                    "iterations": int(info["iterations"]),
                    "correctness_gate": "known_solution_plus_fp32_true_residual",
                    "resource_evidence": "cooperative launch geometry and deterministic CTA partials",
                })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device-reps", type=int, default=5)
    parser.add_argument("--margin", type=float, default=2.5)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    rows = record(reps=args.reps, warmup=args.warmup,
                  device_reps=args.device_reps, margin=args.margin)
    if not rows:
        print("sm_120 NVIDIA runtime unavailable; baseline unchanged")
        return 0
    args.output.write_text(json.dumps({
        "schema": "tessera.benchmark.ratchet.v1", "margin": args.margin,
        "device": "nvidia:sm_120", "rows": rows,
    }, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
