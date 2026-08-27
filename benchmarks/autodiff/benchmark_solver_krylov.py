#!/usr/bin/env python3
"""Emit the exact-SM120 device-resident Krylov/CG evidence packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time

import numpy as np

from tessera import runtime as rt
from tessera.compiler.implicit_solver import package_nvidia_krylov_solver
from tessera.compiler.scheduled_matmul import find_tessera_opt


def run(storage: str, shape: tuple[int, ...], warmup: int, samples: int) -> dict:
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("production tessera-opt is unavailable")
    base_d = np.linspace(0.75, 3.5, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    base_b = np.sin(np.linspace(-2.0, 2.0, int(np.prod(shape)), dtype=np.float32)).reshape(shape)
    if storage == "f32":
        dtype = np.float32
    elif storage == "f16":
        dtype = np.float16
    else:
        import ml_dtypes
        dtype = ml_dtypes.bfloat16
    diagonal, rhs = base_d.astype(dtype), base_b.astype(dtype)
    package = package_nvidia_krylov_solver(
        shape=shape, storage=storage, tolerance=1.0e-7, max_iterations=128,
    )
    artifact = rt.RuntimeArtifact(metadata=package.runtime_metadata())
    for _ in range(warmup):
        result = rt.launch(artifact, (diagonal, rhs))
        if not result["ok"]:
            raise RuntimeError(result["reason"])
    durations: list[int] = []
    result = None
    for _ in range(samples):
        start = time.perf_counter_ns()
        result = rt.launch(artifact, (diagonal, rhs))
        durations.append(time.perf_counter_ns() - start)
        if not result["ok"]:
            raise RuntimeError(result["reason"])
    assert result is not None
    solution, residual, direction, matvec, info = result["output"]
    expected = rhs.astype(np.float32) / diagonal.astype(np.float32)
    error = float(np.max(np.abs(solution - expected)))
    equation_error = float(np.max(np.abs(
        diagonal.astype(np.float32) * solution - rhs.astype(np.float32)
    )))
    ordered = sorted(durations)
    device = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        capture_output=True, check=True, text=True,
    ).stdout.splitlines()[0].strip()
    return {
        "schema": "tessera.nvidia.krylov.evidence.v1",
        "work_items": ["CUDA-SOLVER-KRYLOV-1", "CUDA-SOLVER-CG-1"],
        "target": "nvidia_sm120", "architecture": "sm120", "device": device,
        "host": platform.platform(),
        "toolchain": subprocess.run(
            [str(tool), "--version"], capture_output=True, check=True, text=True,
        ).stdout.splitlines()[0],
        "tessera_opt_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
        "artifact_hash": package.artifact_hash,
        "shape": list(shape), "storage": storage, "accumulation": "f32",
        "operator": "positive_diagonal_spd_v1", "algorithm": "cg",
        "state_residency": info["state_residency"],
        "device_state": ["solution", "residual", "direction", "matvec", "dot_reductions", "convergence"],
        "final_state_norms": {
            "residual": float(np.linalg.norm(residual)),
            "direction": float(np.linalg.norm(direction)),
            "matvec": float(np.linalg.norm(matvec)),
        },
        "iterations": info["iterations"],
        "timing": {
            "source": "synchronized_host_wall", "warmup": warmup,
            "samples_ns": durations, "median_ns": ordered[len(ordered) // 2],
            "minimum_ns": ordered[0], "complete_solve": True,
        },
        "numerical": {
            "oracle": "numpy_positive_diagonal_solve",
            "max_abs_solution_error": error,
            "max_abs_equation_error": equation_error,
            "reported_residual_norm": info["residual_norm"],
            "passed": error <= 2.0e-6 and equation_error <= 2.0e-6,
        },
        "promotion": {
            "correctness_eligible": error <= 2.0e-6 and equation_error <= 2.0e-6,
            "performance_eligible": False,
            "reason": "single-launch correctness packet; no comparative performance ratchet",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", choices=("f32", "f16", "bf16"), default="f32")
    parser.add_argument("--shape", default="257")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = run(args.storage, tuple(int(v) for v in args.shape.split("x")),
                 args.warmup, args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
