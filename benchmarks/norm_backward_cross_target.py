#!/usr/bin/env python3
"""Operation-total affine normalization-backward comparison on x86 and ROCm.

Host wall time surrounds ``runtime.launch`` and therefore includes contract
validation, allocation, copies, module handling, launch/synchronization, and
gradient materialization. These rows are not isolated kernel timings.
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
    family = "layer_norm" if op == "tessera.layer_norm" else "rmsnorm"
    return rt.RuntimeArtifact(metadata={
        "target": target,
        "compiler_path": f"{target}_{family}_bwd_compiled",
        "executable": True,
        "execution_kind": "native_gpu" if target == "rocm" else "native_cpu",
        "autodiff_phase": "backward",
        "out_cotangent": "dy",
        "arg_names": operands + ["dy"],
        "output_names": ["dx"] + (["dgamma"] if affine else [])
                        + (["dbeta"] if affine and family == "layer_norm" else []),
        "ops": [{"op_name": op, "result": "y", "operands": operands,
                 "kwargs": {"eps": 1.0e-5}}],
    })


def _reference(x: np.ndarray, dy: np.ndarray, gamma: np.ndarray,
               op: str, affine: bool) -> tuple[np.ndarray, ...]:
    if op == "tessera.layer_norm":
        mean = x.mean(axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(((x - mean) ** 2).mean(axis=-1, keepdims=True)
                            + 1.0e-5)
        normalized = (x - mean) * inv
    else:
        inv = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1.0e-5)
        normalized = x * inv
    dz = dy * gamma if affine else dy
    mean_dz = (dz.mean(axis=-1, keepdims=True)
               if op == "tessera.layer_norm" else 0.0)
    dx = inv * (dz - mean_dz
                - normalized * (dz * normalized).mean(axis=-1, keepdims=True))
    outputs = [dx]
    if affine:
        outputs.append((dy * normalized).sum(axis=0))
        if op == "tessera.layer_norm":
            outputs.append(dy.sum(axis=0))
    return tuple(outputs)


def _row(target: str, op: str, affine: bool, shape: tuple[int, int],
         warmup: int, reps: int) -> dict[str, object]:
    rng = np.random.default_rng(307 + shape[0] + shape[1] + len(op))
    x = rng.standard_normal(shape).astype(np.float32)
    dy = rng.standard_normal(shape).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, shape[1]).astype(np.float32)
    beta = rng.standard_normal(shape[1]).astype(np.float32)
    args = [x] + ([gamma] if affine else [])
    if affine and op == "tessera.layer_norm":
        args.append(beta)
    args.append(dy)
    artifact = _artifact(target, op, affine)
    samples: list[float] = []
    output: tuple[np.ndarray, ...] = ()
    affine_baseline: tuple[np.ndarray, ...] | None = None
    bitwise_reproducible = True
    for iteration in range(warmup + reps):
        start = time.perf_counter_ns()
        result = rt.launch(artifact, tuple(args))
        elapsed = (time.perf_counter_ns() - start) / 1.0e6
        if not result.get("ok"):
            raise RuntimeError(str(result))
        output = tuple(np.asarray(value, np.float32) for value in result["output"])
        if affine:
            affine_outputs = output[1:]
            if affine_baseline is None:
                affine_baseline = tuple(value.copy() for value in affine_outputs)
            else:
                bitwise_reproducible &= all(
                    np.array_equal(got, first)
                    for got, first in zip(affine_outputs, affine_baseline)
                )
        if iteration >= warmup:
            samples.append(elapsed)
    expected = _reference(x, dy, gamma, op, affine)
    max_error = max(float(np.max(np.abs(got - want)))
                    for got, want in zip(output, expected))
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
        "max_abs_error": max_error,
        "affine_reduction": (
            "fixed_row_order_two_pass" if target == "rocm" and affine
            else "serial_cpu" if target == "x86" and affine
            else "not_applicable"
        ),
        "affine_bitwise_reproducible": bitwise_reproducible if affine else None,
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
    shapes = [tuple(int(value) for value in item.split("x"))
              for item in args.shapes]
    rows = [
        _row(target, op, affine, shape, args.warmup, args.reps)
        for target in args.targets
        for op in ("tessera.rmsnorm", "tessera.layer_norm")
        for affine in (False, True)
        for shape in shapes
    ]
    payload = {"schema": "tessera.norm_backward_cross_target.v2", "rows": rows}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
