#!/usr/bin/env python3
"""Emit a reproducible AD-SOLVER-IFT-1 / AD-RESIDUAL-EVAL-1 packet."""

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
from tessera.compiler.implicit_solver import (
    diagonal_sqrt_ift_reference,
    lower_scheduled_solver_ift,
)
from tessera.compiler.scheduled_matmul import find_tessera_opt


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _tool_version(path: Path) -> str:
    result = subprocess.run([str(path), "--version"], capture_output=True, check=True, text=True)
    return result.stdout.splitlines()[0]


def _device_name(target: str) -> str:
    if target == "x86":
        return _cpu_model()
    if target == "rocm_gfx1151":
        return rt._rocm_chip()
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, check=True, text=True,
    )
    return result.stdout.splitlines()[0].strip()


def run(target: str, shape: tuple[int, ...], warmup: int, samples: int) -> dict:
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("production tessera-opt is unavailable")
    scheduled = lower_scheduled_solver_ift(target=target, shape=shape)
    artifact = rt.RuntimeArtifact(metadata=scheduled.runtime_metadata())
    rng = np.random.default_rng(20260809)
    parameter = rng.uniform(0.25, 16.0, size=shape).astype(np.float32)
    solution = np.sqrt(parameter).astype(np.float32)
    cotangent = rng.standard_normal(shape).astype(np.float32)
    operands = (parameter, solution, cotangent)
    for _ in range(warmup):
        result = rt.launch(artifact, operands)
        if not result["ok"]:
            raise RuntimeError(result["reason"])
    durations: list[int] = []
    result = None
    for _ in range(samples):
        start = time.perf_counter_ns()
        result = rt.launch(artifact, operands)
        durations.append(time.perf_counter_ns() - start)
        if not result["ok"]:
            raise RuntimeError(result["reason"])
    assert result is not None
    expected = diagonal_sqrt_ift_reference(*operands)
    errors = [
        float(np.max(np.abs(np.asarray(actual) - reference))) for actual, reference in zip(result["output"], expected)
    ]
    sorted_ns = sorted(durations)
    return {
        "schema": "tessera.solver_ift.evidence.v1",
        "work_items": ["AD-SOLVER-IFT-1", "AD-RESIDUAL-EVAL-1"],
        "target": target,
        "architecture": scheduled.architecture,
        "device": _device_name(target),
        "host": platform.platform(),
        "toolchain": _tool_version(tool),
        "tessera_opt_sha256": _sha256(tool),
        "artifact_hash": scheduled.artifact_hash,
        "shape": list(shape),
        "dtype": "f32",
        "residual_model": scheduled.contract["residual"]["model"],
        "linear_solver": scheduled.contract["linear_solve"]["algorithm"],
        "matrix_materialized": False,
        "retained_residual_bytes": int(solution.nbytes),
        "timing": {
            "source": "synchronized_host_wall",
            "warmup": warmup,
            "samples_ns": durations,
            "median_ns": sorted_ns[len(sorted_ns) // 2],
            "minimum_ns": sorted_ns[0],
            "complete_backward": True,
        },
        "numerical": {
            "oracle": "numpy_diagonal_sqrt_ift",
            "max_abs_error_by_phase": {
                "residual": errors[0],
                "linear_solution": errors[1],
                "parameter_cotangent": errors[2],
            },
            "passed": max(errors) <= 1e-6,
        },
        "promotion": {
            "correctness_eligible": max(errors) <= 1e-6,
            "performance_eligible": target == "x86",
            "reason": (
                "native AVX-512 synchronized host timing"
                if target == "x86"
                else (
                    "WSL HIP host-wall timing is regression-only"
                    if target == "rocm_gfx1151"
                    else "CUDA host-wrapper timing is correctness/regression-only"
                )
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", choices=("x86", "rocm_gfx1151", "nvidia_sm120"),
        required=True,
    )
    parser.add_argument("--shape", default="3x257")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shape = tuple(int(dim) for dim in args.shape.split("x"))
    packet = run(args.target, shape, args.warmup, args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
