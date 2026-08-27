#!/usr/bin/env python3
"""Emit the exact-SM120 typed residual-child evidence packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess

import numpy as np

from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import (
    run_row_reduce,
    run_solver_compare,
    run_solver_matmul_ieee_f32,
    run_solver_unary,
    run_solver_where,
)
from tessera.compiler.scheduled_matmul import find_tessera_opt


def run() -> dict:
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("production tessera-opt is unavailable")
    import ml_dtypes

    storage_types = {
        "f32": np.float32, "f16": np.float16, "bf16": ml_dtypes.bfloat16,
    }
    unary_refs = (
        np.sqrt, np.reciprocal, np.exp, np.log, np.tanh,
        lambda x: 1.0 / (1.0 + np.exp(-x)), np.sin, np.cos,
    )
    errors: dict[str, dict[str, float]] = {}
    base = np.linspace(0.25, 2.0, 257, dtype=np.float32)
    for storage, dtype in storage_types.items():
        value = base.astype(dtype)
        errors[storage] = {}
        for kind, reference in enumerate(unary_refs):
            actual = run_solver_unary(value, kind).astype(np.float32)
            expected = reference(value.astype(np.float32)).astype(dtype).astype(np.float32)
            errors[storage][f"unary_{kind}"] = float(np.max(np.abs(actual - expected)))
        lhs = np.linspace(-2.0, 2.0, 259, dtype=np.float32).astype(dtype)
        rhs = np.linspace(1.5, -1.5, 259, dtype=np.float32).astype(dtype)
        comparisons = (
            np.equal, np.not_equal, np.less, np.less_equal, np.greater,
            np.greater_equal,
        )
        for kind, reference in enumerate(comparisons):
            predicate = run_solver_compare(lhs, rhs, kind)
            errors[storage][f"compare_{kind}"] = float(
                np.count_nonzero(predicate != reference(lhs, rhs))
            )
            selected = run_solver_where(predicate, lhs, rhs)
            errors[storage][f"where_{kind}"] = float(
                np.max(np.abs(
                    selected.astype(np.float32) -
                    np.where(predicate, lhs, rhs).astype(np.float32)
                ))
            )
        matrix = np.linspace(-1.0, 2.0, 3 * 257, dtype=np.float32).reshape(3, 257).astype(dtype)
        for kind, reference in (
            ("sum", np.sum), ("mean", np.mean),
            ("max", np.max), ("min", np.min),
        ):
            actual = run_row_reduce(matrix, kind)
            expected = reference(matrix.astype(np.float32), axis=1)
            errors[storage][f"reduce_{kind}"] = float(np.max(np.abs(actual - expected)))
    rng = np.random.default_rng(827)
    a = rng.standard_normal((19, 23)).astype(np.float32)
    b = rng.standard_normal((23, 17)).astype(np.float32)
    matmul = run_solver_matmul_ieee_f32(a, b)
    matmul_error = float(np.max(np.abs(
        matmul - (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
    )))
    lowp_matmul_errors: dict[str, float] = {}
    for storage, dtype in (("f16", np.float16), ("bf16", ml_dtypes.bfloat16)):
        lowp_a = rng.standard_normal((32, 48), dtype=np.float32).astype(dtype)
        lowp_b = rng.standard_normal((48, 16), dtype=np.float32).astype(dtype)
        artifact = rt.RuntimeArtifact(metadata={
            "target": "nvidia_sm120", "execution_kind": "native_gpu",
            "executable": True, "compiler_path": "nvidia_solver_matmul_compiled",
            "arg_names": ["a", "b"],
            "ops": [{"op_name": "tessera.matmul", "operands": ["a", "b"],
                     "kwargs": {"numeric_policy": {
                         "storage": storage, "accum": "fp32", "math_mode": "ieee",
                     }}}],
        })
        result = rt.launch(artifact, (lowp_a, lowp_b))
        if not result["ok"]:
            raise RuntimeError(result["reason"])
        oracle = lowp_a.astype(np.float32) @ lowp_b.astype(np.float32)
        lowp_matmul_errors[storage] = float(
            np.max(np.abs(np.asarray(result["output"]) - oracle))
        )
    limits = {"f32": 3e-5, "f16": 3e-2, "bf16": 2e-1}
    passed = all(
        value <= limits[storage]
        for storage, family in errors.items() for value in family.values()
    ) and matmul_error <= 3e-5 and all(value <= 3e-5 for value in lowp_matmul_errors.values())
    device = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        capture_output=True, check=True, text=True,
    ).stdout.splitlines()[0].strip()
    return {
        "schema": "tessera.nvidia.solver_children.evidence.v1",
        "work_items": [
            "CUDA-SOLVER-CHILDREN-1", "CUDA-SOLVER-DTYPE-1",
            "CUDA-SOLVER-LOWP-MATMUL-2",
        ],
        "target": "nvidia_sm120", "architecture": "sm120", "device": device,
        "host": platform.platform(),
        "toolchain": subprocess.run(
            [str(tool), "--version"], capture_output=True, check=True, text=True,
        ).stdout.splitlines()[0],
        "tessera_opt_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
        "families": [
            "unary", "reduction", "comparison", "where", "matmul_ieee",
            "matmul_native_lowp",
        ],
        "storage": ["f32", "f16", "bf16"], "accumulation": "f32",
        "matmul_math_mode": "ieee",
        "max_abs_error_by_storage_and_case": errors,
        "matmul_max_abs_error": matmul_error,
        "native_lowp_matmul": {
            "storage": ["f16", "bf16"], "math_mode": "ieee",
            "accumulation": "fp32", "physical_route": "mma.sync",
            "missing_storage_policy": "fail_closed",
            "max_abs_error": lowp_matmul_errors,
        },
        "passed": passed,
        "promotion": {
            "correctness_eligible": passed, "performance_eligible": False,
            "reason": "solver-child correctness matrix; no selector/performance ratchet",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
