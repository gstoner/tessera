#!/usr/bin/env python3
"""Emit the exact-device NVIDIA residual-replay/general-solver packet."""

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
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.implicit_solver import compile_physical_general_solver_from_graph
from tessera.compiler.scheduled_matmul import find_tessera_opt


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _affine_graph(shape: tuple[int, ...]) -> GraphIRModule:
    ty = IRType("tensor<" + "x".join(map(str, shape)) + "xf32>")
    return GraphIRModule(functions=[GraphIRFunction(
        name="affine_residual",
        args=[IRArg("theta", ty), IRArg("x", ty)], result_types=[ty],
        body=[IROp("residual", "tessera.sub", ["%x", "%theta"],
                   [str(ty), str(ty)], str(ty))],
        return_values=["%residual"],
    )])


def run(shape: tuple[int, ...], warmup: int, samples: int) -> dict:
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("production tessera-opt is unavailable")
    package = compile_physical_general_solver_from_graph(
        _affine_graph(shape), target="nvidia_sm120", shape=shape,
        product_mode="vjp", tolerance=1.0e-7,
        max_iterations=8, restart=4,
    )
    artifact = rt.RuntimeArtifact(metadata=package.runtime_metadata())
    parameter = np.linspace(-2.0, 3.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    solution = parameter.copy()
    product = np.linspace(-1.0, 1.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    operands = (parameter, solution, product)
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
    residual, linear, parameter_product = result["output"]
    errors = {
        "residual": float(np.max(np.abs(residual))),
        "linear_solution": float(np.max(np.abs(linear - product))),
        "parameter_product": float(np.max(np.abs(parameter_product - product))),
    }
    device = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        capture_output=True, check=True, text=True,
    ).stdout.splitlines()[0].strip()
    ordered = sorted(durations)
    children = package.contract["children"]
    return {
        "schema": "tessera.general_solver.evidence.v1",
        "work_items": ["AD-SOLVER-IFT-1", "AD-RESIDUAL-EVAL-1", "CUDA-SOLVER-GENERAL-1"],
        "target": "nvidia_sm120",
        "architecture": "sm120",
        "device": device,
        "host": platform.platform(),
        "toolchain": subprocess.run(
            [str(tool), "--version"], capture_output=True, check=True, text=True,
        ).stdout.splitlines()[0],
        "tessera_opt_sha256": _sha256(tool),
        "artifact_hash": package.artifact_hash,
        "child_digests": {role: child["digest"] for role, child in children.items()},
        "shape": list(shape),
        "dtype": "f32",
        "residual_model": "affine_x_minus_theta",
        "linear_solver": "gmres",
        "matrix_free": True,
        "true_residual_check": True,
        "admitted_child_families": [
            "binary_math", "unary_math", "reduction", "comparison", "where",
            "matmul_ieee",
        ],
        "fail_closed_policies": [
            "predicate_equality_boundary", "missing_matmul_math_mode",
            "non_fp32_accumulation", "unsupported_dtype_transition",
        ],
        "timing": {
            "source": "synchronized_host_wall",
            "warmup": warmup,
            "samples_ns": durations,
            "median_ns": ordered[len(ordered) // 2],
            "minimum_ns": ordered[0],
            "complete_backward": True,
        },
        "numerical": {
            "oracle": "numpy_affine_identity_ift",
            "max_abs_error_by_phase": errors,
            "passed": max(errors.values()) <= 1.0e-6,
        },
        "promotion": {
            "correctness_eligible": max(errors.values()) <= 1.0e-6,
            "performance_eligible": False,
            "reason": "host-orchestrated GMRES and CUDA wrapper timing is correctness-only",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="37")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shape = tuple(int(dim) for dim in args.shape.split("x"))
    packet = run(shape, args.warmup, args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
