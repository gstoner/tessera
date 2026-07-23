#!/usr/bin/env python3
"""Operation-total affine normalization comparison on local x86 and ROCm.

The timing domain is host wall time around ``runtime.launch``. It deliberately
includes validation, allocation, transfers, module load, launch, synchronization,
and result materialization; it is not presented as isolated kernel time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from tessera import runtime as rt


def _artifact(target: str, op: str, affine: bool) -> rt.RuntimeArtifact:
    operands = ["x"]
    if affine:
        operands.append("gamma")
        if op == "tessera.layer_norm":
            operands.append("beta")
    return rt.RuntimeArtifact(metadata={
        "target": target,
        "compiler_path": f"{target}_norm_compiled",
        "executable": True,
        "execution_kind": "native_gpu" if target == "rocm" else "native_cpu",
        "arg_names": operands,
        "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": operands,
                 "kwargs": {"eps": 1.0e-5}}],
    })


def _reference(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
               op: str, affine: bool) -> np.ndarray:
    if op == "tessera.layer_norm":
        center = x.mean(axis=-1, keepdims=True)
        y = (x - center) / np.sqrt(((x - center) ** 2).mean(
            axis=-1, keepdims=True) + 1.0e-5)
    else:
        y = x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1.0e-5)
    if affine:
        y = y * gamma
        if op == "tessera.layer_norm":
            y = y + beta
    return y


def _row(target: str, op: str, affine: bool, shape: tuple[int, int],
         warmup: int, reps: int) -> dict[str, object]:
    rng = np.random.default_rng(211 + shape[0] + shape[1])
    x = rng.standard_normal(shape).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, shape[1]).astype(np.float32)
    beta = rng.standard_normal(shape[1]).astype(np.float32)
    args = [x] + ([gamma] if affine else [])
    if affine and op == "tessera.layer_norm":
        args.append(beta)
    artifact = _artifact(target, op, affine)
    samples: list[float] = []
    output = None
    for iteration in range(warmup + reps):
        start = time.perf_counter_ns()
        result = rt.launch(artifact, tuple(args))
        elapsed = (time.perf_counter_ns() - start) / 1.0e6
        if not result.get("ok"):
            raise RuntimeError(str(result))
        output = np.asarray(result["output"], np.float32)
        if iteration >= warmup:
            samples.append(elapsed)
    reference = _reference(x, gamma, beta, op, affine)
    ordered = sorted(samples)
    return {
        "target": target,
        "op": op.removeprefix("tessera."),
        "affine": affine,
        "shape": list(shape),
        "timing_domain": "host_wall_operation_total",
        "reps": reps,
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max_abs_error": float(np.max(np.abs(output - reference))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["x86", "rocm"],
                        choices=["x86", "rocm"])
    parser.add_argument("--shapes", nargs="+", default=["32x128", "7x300"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    shapes = [tuple(int(v) for v in item.split("x")) for item in args.shapes]
    rows = [
        _row(target, op, affine, shape, args.warmup, args.reps)
        for target in args.targets
        for op in ("tessera.rmsnorm", "tessera.layer_norm")
        for affine in (False, True)
        for shape in shapes
    ]
    payload = {"schema": "tessera.norm_affine_cross_target.v1", "rows": rows}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
