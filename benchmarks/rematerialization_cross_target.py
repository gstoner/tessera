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


def record(
    sizes: tuple[int, ...] = (64, 128, 192),
    epilogues: tuple[str, ...] = ("relu", "gelu", "silu"),
    warmup: int = 3,
    reps: int = 15,
) -> dict:
    return {
        "schema": "tessera.rematerialization.cross-target.v1",
        "rows": [
            _row(target, size, epilogue, warmup, reps)
            for target in ("x86", "rocm")
            for size in sizes
            for epilogue in epilogues
        ],
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = record(
        tuple(args.sizes), tuple(args.epilogues), args.warmup, args.reps
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
