#!/usr/bin/env python3
"""Operation-total loss-backward→optimizer timing on AVX-512 and gfx1151."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from tessera import runtime as rt


def _fused_artifact(
    target: str, kind: str, optimizer: str
) -> rt.RuntimeArtifact:
    parameter = 0.75 if kind == "huber" else 0.5
    adamw = optimizer == "adamw"
    operands = ["prediction", "target", "dy", "parameter"]
    outputs = ["new_parameter", "d_target"]
    kwargs = {
        "kind": kind, "parameter": parameter,
        "reduction": "mean",
    }
    if adamw:
        operands += ["moment1", "moment2"]
        outputs = ["new_parameter", "new_moment1", "new_moment2", "d_target"]
        kwargs.update({
            "lr": 0.002, "beta1": 0.8, "beta2": 0.95,
            "eps": 1.0e-7, "weight_decay": 0.01, "step": 7,
        })
    else:
        kwargs["lr"] = 0.125
    return rt.RuntimeArtifact(metadata={
        "target": target,
        "compiler_path": f"{target}_training_loss_{optimizer}_compiled",
        "executable": True,
        "execution_kind": "native_gpu" if target == "rocm" else "native_cpu",
        "arg_names": operands,
        "output_names": outputs,
        "ops": [{
            "op_name": f"tessera.training.loss_{optimizer}",
            "result": outputs, "operands": operands, "kwargs": kwargs,
        }],
    })


def _loss_artifact(target: str, kind: str) -> rt.RuntimeArtifact:
    parameter = 0.75 if kind == "huber" else 0.5
    kwargs = {"reduction": "mean"}
    if kind == "huber":
        kwargs["delta"] = parameter
    elif kind == "smooth_l1":
        kwargs["beta"] = parameter
    compiler_family = (
        "binary_loss_bwd_compiled"
        if kind == "bce" else "regression_loss_bwd_compiled"
    )
    op_name = (
        "tessera.binary_cross_entropy_loss"
        if kind == "bce" else f"tessera.loss.{kind}"
    )
    return rt.RuntimeArtifact(metadata={
        "target": target,
        "compiler_path": f"{target}_{compiler_family}",
        "executable": True,
        "execution_kind": "native_gpu" if target == "rocm" else "native_cpu",
        "autodiff_phase": "backward",
        "out_cotangent": "dy",
        "arg_names": ["prediction", "target", "dy"],
        "output_names": ["d_prediction", "d_target"],
        "ops": [{
            "op_name": op_name,
            "result": "loss",
            "operands": ["prediction", "target"],
            "kwargs": kwargs,
        }],
    })


def _sgd_artifact(target: str) -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(metadata={
        "target": target,
        "compiler_path": f"{target}_optimizer_compiled",
        "executable": True,
        "execution_kind": "native_gpu" if target == "rocm" else "native_cpu",
        "arg_names": ["parameter", "gradient"],
        "output_name": "new_parameter",
        "ops": [{
            "op_name": "tessera.sgd", "result": "new_parameter",
            "operands": ["parameter", "gradient"],
            "kwargs": {"lr": 0.125, "extras": []},
        }],
    })


def _adamw_artifact(target: str) -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(metadata={
        "target": target,
        "compiler_path": f"{target}_optimizer_compiled",
        "executable": True,
        "execution_kind": "native_gpu" if target == "rocm" else "native_cpu",
        "arg_names": ["parameter", "gradient", "moment1", "moment2"],
        "output_names": ["new_parameter", "new_moment1", "new_moment2"],
        "ops": [{
            "op_name": "tessera.adamw",
            "result": ["new_parameter", "new_moment1", "new_moment2"],
            "operands": ["parameter", "gradient", "moment1", "moment2"],
            "kwargs": {
                "lr": 0.002, "beta1": 0.8, "beta2": 0.95,
                "eps": 1.0e-7, "weight_decay": 0.01, "step": 7,
                "extras": ["m", "v"],
            },
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


def _comparison_row(
    target: str, kind: str, optimizer: str, elements: int, warmup: int,
    reps: int
) -> dict[str, object]:
    rng = np.random.default_rng(20260723 + elements + len(kind))
    prediction = rng.standard_normal(elements).astype(np.float32)
    target_value = rng.standard_normal(elements).astype(np.float32)
    parameter = rng.standard_normal(elements).astype(np.float32)
    moment1 = (0.2 * rng.standard_normal(elements)).astype(np.float32)
    moment2 = (0.01 + 0.3 * rng.random(elements)).astype(np.float32)
    dy = np.asarray(0.75, np.float32)
    fused_artifact = _fused_artifact(target, kind, optimizer)
    loss_artifact = _loss_artifact(target, kind)
    optimizer_artifact = (
        _adamw_artifact(target) if optimizer == "adamw"
        else _sgd_artifact(target)
    )
    fused_args = (
        (prediction, target_value, dy, parameter, moment1, moment2)
        if optimizer == "adamw"
        else (prediction, target_value, dy, parameter)
    )

    def fused_call():
        result = rt.launch(
            fused_artifact, fused_args
        )
        if not result.get("ok"):
            raise RuntimeError(str(result))
        return result["output"]

    def unfused_call():
        backward = rt.launch(
            loss_artifact, (prediction, target_value, dy)
        )
        if not backward.get("ok"):
            raise RuntimeError(str(backward))
        d_prediction, d_target = backward["output"]
        optimizer_args = (
            (parameter, d_prediction, moment1, moment2)
            if optimizer == "adamw" else (parameter, d_prediction)
        )
        update = rt.launch(optimizer_artifact, optimizer_args)
        if not update.get("ok"):
            raise RuntimeError(str(update))
        optimizer_output = (
            tuple(update["output"])
            if optimizer == "adamw" else (update["output"],)
        )
        return (*optimizer_output, d_target)

    fused_output = fused_call()
    unfused_output = unfused_call()
    tolerance = (
        2e-5 if target == "rocm"
        else 1e-5 if kind == "bce"
        else 2e-6
    )
    for fused_value, unfused_value in zip(fused_output, unfused_output):
        np.testing.assert_allclose(
            fused_value, unfused_value, atol=tolerance, rtol=tolerance
        )
    fused_median, fused_p95 = _measure(fused_call, warmup, reps)
    unfused_median, unfused_p95 = _measure(unfused_call, warmup, reps)
    return {
        "target": target,
        "family": kind,
        "optimizer": optimizer,
        "elements": elements,
        "timing_domain": "host_wall_operation_total",
        "fused": {
            "compiler_path": (
                f"{target}_training_loss_{optimizer}_compiled"
            ),
            "launches": 1, "median_ms": fused_median, "p95_ms": fused_p95,
        },
        "unfused": {
            "compiler_paths": [
                (
                    f"{target}_binary_loss_bwd_compiled"
                    if kind == "bce"
                    else f"{target}_regression_loss_bwd_compiled"
                ),
                f"{target}_optimizer_compiled",
            ],
            "launches": 2, "median_ms": unfused_median,
            "p95_ms": unfused_p95,
        },
        "speedup": unfused_median / fused_median,
        "cache_identity": (
            "chip,dtype,kind,parameter,reduction"
            if target == "rocm" else "runtime_loaded_avx512_abi"
        ),
        "selector_decision": (
            f"fused_when_dprediction_has_one_{optimizer}_use"
        ),
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
        _comparison_row(
            target, family, optimizer, args.elements, args.warmup, args.reps
        )
        for target in args.targets
        for family in ("mse", "mae", "huber", "smooth_l1", "bce")
        for optimizer in ("sgd", "adamw")
    ]
    payload = {
        "schema": "tessera.training_step_fusion_cross_target.v2",
        "rows": rows,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
