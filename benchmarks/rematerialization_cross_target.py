#!/usr/bin/env python3
"""Measured ROCm/AVX-512 rematerialization cost and memory trade-off packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

import tessera.compiler.emit.rocm_hip  # noqa: F401 - registers the runner
import tessera.compiler.emit.x86_llvm  # noqa: F401 - registers the runner
from tessera.compiler.emit.kernel_emitter import get_runner
from tessera.compiler.fusion_core import FusedRegion
from tessera.runtime import RuntimeArtifact, launch


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


def _row(
    target: str, size: int, epilogue: str, warmup: int, reps: int
) -> dict:
    rng = np.random.default_rng(20260723 + size + len(epilogue))
    a = rng.standard_normal((size, size)).astype(np.float32)
    b = rng.standard_normal((size, size)).astype(np.float32)
    region = FusedRegion(epilogue=(epilogue,))
    runner = get_runner(target)
    expected_tag = "x86_native" if target == "x86" else "rocm_hip"

    def recompute_once():
        value, tag = runner.run_fused_region(region, a, b)
        if tag != expected_tag:
            raise RuntimeError(
                f"{target} rematerialization benchmark used {tag!r}, "
                f"expected {expected_tag!r}"
            )
        return value

    activation = recompute_once()
    np.testing.assert_allclose(
        activation, region.reference(a, b), rtol=2e-4, atol=2e-4
    )
    recompute_ms, recompute_p95 = _measure(recompute_once, warmup, reps)

    def retained_step():
        # A backward consumer reads the already-saved activation. The copy makes
        # host-visible consumption explicit without changing its memory lifetime.
        return activation.copy()

    retained_ms, retained_p95 = _measure(retained_step, warmup, reps)
    activation_bytes = int(activation.nbytes)
    return {
        "target": target,
        "device": (
            "gfx1151" if target == "rocm"
            else "Ryzen AI MAX+ 395 AVX-512"
        ),
        "operation": f"tessera.matmul_{epilogue}",
        "remat_attribute_operation": "tessera.matmul",
        "shape": [size, size, size],
        "result_bytes": activation_bytes,
        "timing_domain": "host_wall_operation_total",
        "recompute_cost_ns": round(recompute_ms * 1.0e6),
        "recompute_median_ms": recompute_ms,
        "recompute_p95_ms": recompute_p95,
        "retained_consumer_median_ms": retained_ms,
        "retained_consumer_p95_ms": retained_p95,
        "rematerialization_runtime_ratio": (
            recompute_ms / max(retained_ms, 1.0e-12)
        ),
        "planner_peak_before_bytes": activation_bytes,
        "planner_peak_after_bytes": 0,
        "planner_peak_reduction_bytes": activation_bytes,
        "residual_policy": "save_inputs_recompute_output",
        "selector_signal": "measured_cost_ns",
    }


def _producer_family_row(
    target: str, family: str, rows: int, width: int, warmup: int, reps: int,
    *, layer: int = 0,
) -> dict:
    rng = np.random.default_rng(20260724 + rows + width + len(family))
    x = rng.standard_normal((rows, width)).astype(np.float32)
    target_values = rng.standard_normal((rows, width)).astype(np.float32)
    binary_targets = rng.integers(
        0, 2, size=(rows, width), dtype=np.int8
    ).astype(np.float32)
    specs = {
        "softmax": (
            "tessera.softmax",
            f"{target}_softmax_compiled",
            ["x"],
            (x,),
            "tessera.softmax_multiply",
        ),
        "rmsnorm": (
            "tessera.rmsnorm",
            f"{target}_norm_compiled",
            ["x"],
            (x,),
            "tessera.rmsnorm_multiply",
        ),
        "mse": (
            "tessera.loss.mse",
            f"{target}_loss_compiled",
            ["x", "target"],
            (x, target_values),
            "tessera.loss.mse_sgd",
        ),
        "huber": (
            "tessera.loss.huber",
            f"{target}_loss_compiled",
            ["x", "target"],
            (x, target_values),
            "tessera.loss.huber_adamw",
        ),
        "smooth_l1": (
            "tessera.loss.smooth_l1",
            f"{target}_loss_compiled",
            ["x", "target"],
            (x, target_values),
            "tessera.loss.smooth_l1_adamw",
        ),
        "bce": (
            "tessera.binary_cross_entropy_loss",
            f"{target}_binary_loss_compiled",
            ["x", "target"],
            (x, binary_targets),
            "tessera.loss.bce_adamw",
        ),
    }
    op_name, compiler_path, arg_names, args, consumer = specs[family]
    kwargs: dict[str, str | float] = {"reduction": "none"} if family in {
        "mse", "huber", "smooth_l1", "bce"
    } else {}
    if family == "huber":
        kwargs["delta"] = 0.75
    elif family == "smooth_l1":
        kwargs["beta"] = 0.5
    artifact = RuntimeArtifact(metadata={
        "target": target,
        "compiler_path": compiler_path,
        "executable": True,
        "arg_names": arg_names,
        "output_name": "out",
        "ops": [{
            "op_name": op_name,
            "result": "out",
            "operands": arg_names,
            "kwargs": kwargs,
        }],
    })
    expected_kind = "native_gpu" if target == "rocm" else "native_cpu"

    def recompute_once():
        result = launch(artifact, args)
        if not result.get("ok") or result.get("execution_kind") != expected_kind:
            raise RuntimeError(
                f"{target} {family} rematerialization row used "
                f"{result.get('execution_kind')!r}, expected {expected_kind!r}"
            )
        return np.asarray(result["output"], dtype=np.float32)

    activation = recompute_once()
    recompute_ms, recompute_p95 = _measure(recompute_once, warmup, reps)
    retained_ms, retained_p95 = _measure(activation.copy, warmup, reps)
    activation_bytes = int(activation.nbytes)
    return {
        "target": target,
        "device": (
            "gfx1151" if target == "rocm"
            else "Ryzen AI MAX+ 395 AVX-512"
        ),
        "producer_family": family,
        "layer": layer,
        "operation": consumer,
        "remat_attribute_operation": op_name,
        "shape": [rows, width],
        "result_bytes": activation_bytes,
        "timing_domain": "host_wall_operation_total",
        "recompute_cost_ns": round(recompute_ms * 1.0e6),
        "recompute_median_ms": recompute_ms,
        "recompute_p95_ms": recompute_p95,
        "retained_consumer_median_ms": retained_ms,
        "retained_consumer_p95_ms": retained_p95,
        "planner_peak_before_bytes": activation_bytes,
        "planner_peak_after_bytes": 0,
        "planner_peak_reduction_bytes": activation_bytes,
        "residual_policy": "save_inputs_recompute_output",
        "selector_signal": "measured_cost_ns",
    }


def _workload_policy(rows: list[dict], memory_budget_bytes: int) -> list[dict]:
    """Select exact rows by measured bytes-per-recompute-ns benefit.

    This mirrors the compiler pass's deterministic global policy but reports
    operation-total workload impact for review rather than only isolated costs.
    """
    decisions: list[dict] = []
    for target in ("x86", "rocm"):
        candidates = [
            row for row in rows
            if row["target"] == target and "producer_family" in row
        ]
        peak_before = sum(int(row["result_bytes"]) for row in candidates)
        selected: list[dict] = []
        peak_after = peak_before
        for row in sorted(
            candidates,
            key=lambda item: (
                -int(item["result_bytes"]) / max(int(item["recompute_cost_ns"]), 1),
                str(item["operation"]),
            ),
        ):
            if peak_after <= memory_budget_bytes:
                break
            selected.append(row)
            peak_after -= int(row["result_bytes"])
        decisions.append({
            "target": target,
            "memory_budget_bytes": memory_budget_bytes,
            "peak_before_bytes": peak_before,
            "peak_after_bytes": peak_after,
            "selected_operations": [row["operation"] for row in selected],
            "selected_instances": [
                {
                    "layer": int(row.get("layer", 0)),
                    "producer_family": row["producer_family"],
                    "shape": row["shape"],
                }
                for row in selected
            ],
            "selected_recompute_cost_ns": sum(
                int(row["recompute_cost_ns"]) for row in selected
            ),
            "policy": "measured_bytes_per_recompute_ns",
        })
    return decisions


def record(
    sizes: tuple[int, ...] = (64, 128, 192),
    epilogues: tuple[str, ...] = ("relu", "gelu", "silu"),
    warmup: int = 3,
    reps: int = 15,
    producer_families: tuple[str, ...] = (),
    producer_shape: tuple[int, int] = (256, 512),
    memory_budget_bytes: int = 512 * 1024,
    layers: int = 1,
) -> dict:
    if layers < 1:
        raise ValueError("rematerialization workload must contain at least one layer")
    rows = [
        _row(target, size, epilogue, warmup, reps)
        for target in ("x86", "rocm")
        for size in sizes
        for epilogue in epilogues
    ]
    rows.extend(
        _producer_family_row(
            target, family, producer_shape[0],
            producer_shape[1] + layer * 128, warmup, reps, layer=layer,
        )
        for target in ("x86", "rocm")
        for layer in range(layers)
        for family in producer_families
    )
    return {
        "schema": "tessera.rematerialization.cross-target.v1",
        "rows": rows,
        "workload_policy": _workload_policy(rows, memory_budget_bytes),
        "workload": {
            "layers": layers,
            "families_per_layer": list(producer_families),
            "shape_policy": "fixed_rows_width_plus_128_per_layer",
            "evidence": "each producer instance executed on exact target",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=(64, 128, 192))
    parser.add_argument(
        "--epilogues", nargs="+", default=("relu", "gelu", "silu"),
        choices=("relu", "gelu", "silu"),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument(
        "--producer-families", nargs="+",
        default=("softmax", "rmsnorm", "mse", "huber", "smooth_l1", "bce"),
        choices=("softmax", "rmsnorm", "mse", "huber", "smooth_l1", "bce"),
    )
    parser.add_argument("--producer-shape", type=int, nargs=2, default=(256, 512))
    parser.add_argument("--memory-budget-bytes", type=int, default=512 * 1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = record(
        tuple(args.sizes), tuple(args.epilogues), args.warmup, args.reps,
        tuple(args.producer_families), tuple(args.producer_shape),
        args.memory_budget_bytes, args.layers,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
