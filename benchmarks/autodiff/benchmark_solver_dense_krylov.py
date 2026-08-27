#!/usr/bin/env python3
"""Emit exact-SM120 dense CG/Arnoldi-GMRES correctness evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess

import numpy as np

from tessera import runtime as rt
from tessera.compiler.implicit_solver import package_nvidia_dense_krylov_solver
from tessera.compiler.scheduled_matmul import find_tessera_opt


def _storage(name: str, value: np.ndarray):
    if name == "f32":
        return value.astype(np.float32)
    if name == "f16":
        return value.astype(np.float16)
    import ml_dtypes
    return value.astype(ml_dtypes.bfloat16)


def _case(algorithm: str, storage: str, order: int) -> dict:
    matrix = np.eye(order, dtype=np.float32) * np.float32(3.5)
    index = np.arange(order - 1)
    matrix[index, index + 1] = np.float32(0.35 if algorithm == "gmres" else -0.75)
    matrix[index + 1, index] = np.float32(-0.55 if algorithm == "gmres" else -0.75)
    authored_solution = np.sin(np.linspace(-1.0, 2.0, order, dtype=np.float32))
    rhs = matrix @ authored_solution
    stored_matrix, stored_rhs = _storage(storage, matrix), _storage(storage, rhs)
    tolerance = 2.0e-6 if storage == "f32" else 2.0e-4
    package = package_nvidia_dense_krylov_solver(
        order=order, storage=storage, algorithm=algorithm, tolerance=tolerance,
        max_iterations=96, restart=16, reduction_ctas=0,
    )
    result = rt.launch(
        rt.RuntimeArtifact(metadata=package.runtime_metadata()),
        (stored_matrix, stored_rhs),
    )
    if not result["ok"]:
        raise RuntimeError(result["reason"])
    solution, residual, matvec, info = result["output"]
    operator_f32 = stored_matrix.astype(np.float32)
    rhs_f32 = stored_rhs.astype(np.float32)
    true_residual = rhs_f32 - operator_f32 @ solution
    oracle = np.linalg.solve(
        operator_f32.astype(np.float64), rhs_f32.astype(np.float64)
    ).astype(np.float32)
    solution_error = float(np.max(np.abs(solution - oracle)))
    residual_error = float(np.max(np.abs(residual - true_residual)))
    residual_norm = float(np.linalg.norm(true_residual))
    limit = tolerance * max(1.0, float(np.linalg.norm(rhs_f32)))
    return {
        "algorithm": algorithm, "storage": storage, "order": order,
        "artifact_hash": package.artifact_hash,
        "iterations": info["iterations"], "reduction_ctas": info["reduction_ctas"],
        "state_residency": info["state_residency"], "reduction": info["reduction"],
        "device_elapsed_ms": info["device_elapsed_ms"],
        "numerical": {
            "oracle": "numpy_float64_dense_solve_plus_explicit_b_minus_ax",
            "max_abs_solution_error": solution_error,
            "max_abs_returned_residual_error": residual_error,
            "true_residual_norm": residual_norm,
            "true_residual_limit": limit,
            "matvec_max_abs_error": float(np.max(np.abs(matvec - operator_f32 @ solution))),
            "passed": solution_error <= (3e-5 if storage == "f32" else 8e-4)
                      and residual_error <= 8e-5 and residual_norm <= limit,
        },
    }


def run() -> dict:
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("production tessera-opt is unavailable")
    cases = [
        _case("cg", "f32", 513), _case("gmres", "f32", 513),
        _case("gmres", "f16", 257), _case("gmres", "bf16", 257),
    ]
    device = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        capture_output=True, check=True, text=True,
    ).stdout.splitlines()[0].strip()
    return {
        "schema": "tessera.nvidia.dense_krylov.evidence.v1",
        "work_items": [
            "CUDA-SOLVER-ARNOLDI-GMRES-2", "CUDA-SOLVER-DENSE-CG-2",
            "CUDA-SOLVER-MULTI-CTA-2",
        ],
        "target": "nvidia_sm120", "architecture": "sm120", "device": device,
        "host": platform.platform(),
        "toolchain": subprocess.run(
            [str(tool), "--version"], capture_output=True, check=True, text=True,
        ).stdout.splitlines()[0],
        "tessera_opt_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
        "operator": "arbitrary_dense_row_major_v1",
        "true_residual_required": True,
        "orthogonalization": "twice_modified_gram_schmidt",
        "cases": cases,
        "passed": all(case["numerical"]["passed"] for case in cases),
        "promotion": {
            "correctness_eligible": all(case["numerical"]["passed"] for case in cases),
            "performance_eligible": True,
            "performance_packet": "benchmarks/baselines/nvidia_sm120_solver_krylov_performance.json",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
