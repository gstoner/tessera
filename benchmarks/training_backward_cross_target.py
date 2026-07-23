#!/usr/bin/env python3
"""Operation-total regression-loss and SGD backward timing on x86/ROCm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from tessera import runtime as rt


def _loss_artifact(target: str, op: str) -> rt.RuntimeArtifact:
    parameter = 0.75 if op == "huber" else 0.5
    kwargs = {"reduction": "mean"}
    if op == "huber":
        kwargs["delta"] = parameter
    if op == "smooth_l1":
        kwargs["beta"] = parameter
    return rt.RuntimeArtifact(metadata={
        "target": target,
        "compiler_path": f"{target}_regression_loss_bwd_compiled",
        "executable": True,
        "execution_kind": "native_gpu" if target == "rocm" else "native_cpu",
        "autodiff_phase": "backward",
        "arg_names": ["prediction", "target", "dy"],
        "output_names": ["d_prediction", "d_target"],
        "ops": [{
            "op_name": f"tessera.loss.{op}",
            "result": "loss",
            "operands": ["prediction", "target"],
            "kwargs": kwargs,
        }],
    })


def _sgd_artifact(target: str) -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(metadata={
        "target": target,
        "compiler_path": f"{target}_sgd_bwd_compiled",
        "executable": True,
        "execution_kind": "native_gpu" if target == "rocm" else "native_cpu",
        "autodiff_phase": "backward",
        "out_cotangent": "dy",
        "arg_names": ["parameter", "gradient", "dy"],
        "output_names": ["d_parameter", "d_gradient"],
        "ops": [{
            "op_name": "tessera.sgd",
            "result": "updated",
            "operands": ["parameter", "gradient"],
            "kwargs": {"lr": 0.125},
        }],
    })


def _measure(call, warmup: int, reps: int) -> tuple[float, float]:
    samples: list[float] = []
    for iteration in range(warmup + reps):
        start = time.perf_counter_ns()
        call()
        elapsed = (time.perf_counter_ns() - start) / 1.0e6
        if iteration >= warmup:
            samples.append(elapsed)
    ordered = sorted(samples)
    return (
        statistics.median(samples),
        ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    )


def _row(target: str, family: str, elements: int, warmup: int, reps: int
         ) -> dict[str, object]:
    rng = np.random.default_rng(20260723 + elements + len(family))
    prediction = rng.standard_normal(elements).astype(np.float32)
    target_value = rng.standard_normal(elements).astype(np.float32)
    seed = np.asarray(0.75, np.float32)
    if family == "sgd":
        artifact = _sgd_artifact(target)
        dy = seed * np.ones_like(prediction)

        def call():
            result = rt.launch(
                artifact, (prediction, target_value, dy))
            if not result.get("ok"):
                raise RuntimeError(str(result))
    else:
        artifact = _loss_artifact(target, family)

        def call():
            result = rt.launch(artifact, (prediction, target_value, seed))
            if not result.get("ok"):
                raise RuntimeError(str(result))

    median, p95 = _measure(call, warmup, reps)
    return {
        "target": target,
        "family": family,
        "elements": elements,
        "timing_domain": "host_wall_operation_total",
        "launches": 1,
        "reps": reps,
        "median_ms": median,
        "p95_ms": p95,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["x86", "rocm"])
    parser.add_argument("--elements", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [
        _row(target, family, args.elements, args.warmup, args.reps)
        for target in args.targets
        for family in ("mse", "mae", "huber", "smooth_l1", "sgd")
    ]
    payload = {
        "schema": "tessera.training_backward_cross_target.v1",
        "rows": rows,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
