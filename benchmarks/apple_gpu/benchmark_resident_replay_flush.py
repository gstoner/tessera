"""Measure ReplaySSM checkpoint folding in end-to-end and device domains."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from tessera import runtime as rt
from tessera._apple_gpu_dispatch import (
    clear_dispatch_telemetry,
    set_dispatch_telemetry_enabled,
)
from tessera.cache import SSMStateHandle
from tessera.compiler.apple_route_selector import live_apple_device_tag


def _shape(text: str) -> tuple[int, int, int]:
    values = tuple(int(value) for value in text.lower().split("x"))
    if len(values) != 3 or min(values) <= 0:
        raise ValueError(f"shape must be BxDxN, got {text!r}")
    return values  # type: ignore[return-value]


def _inputs(seed: int, tokens: int, B: int, D: int, N: int):
    rng = np.random.default_rng(seed)
    a = -np.abs(rng.standard_normal(D))
    delta = np.abs(rng.standard_normal((tokens, B, D))) * 0.2
    x = rng.standard_normal((tokens, B, D))
    b = rng.standard_normal((tokens, B, N))
    return a, delta, x, b


def _fill(handle: Any, delta: Any, x: Any, b: Any) -> None:
    for token in range(delta.shape[0]):
        handle.append(delta[token], x[token], b[token], auto_flush=False)


def run_benchmark(shapes: list[str], *, tokens: int, warmup: int, reps: int, runs: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    device = live_apple_device_tag()
    set_dispatch_telemetry_enabled(True)
    try:
        for run in range(1, runs + 1):
            for case, spec in enumerate(shapes):
                B, D, N = _shape(spec)
                a, delta, x, b = _inputs(7100 + run * 101 + case, tokens, B, D, N)
                oracle = SSMStateHandle(B, D, N, a, capacity=tokens)
                _fill(oracle, delta, x, b)
                expected = oracle.materialize_state()
                for route in ("reference_fold", "resident_native"):
                    wall: list[int] = []
                    device_times: list[int] = []
                    correct = True
                    available = True
                    provenance = "reference_cpu"
                    native_proof = route == "resident_native"
                    resources: Any = None
                    for iteration in range(warmup + reps):
                        if route == "reference_fold":
                            handle = SSMStateHandle(B, D, N, a, capacity=tokens)
                        else:
                            handle = rt.apple_gpu_resident_ssm_replay_state_handle(
                                B, D, N, a, capacity=tokens, async_slots=2
                            )
                            if not handle.resident_inputs:
                                handle.close()
                                available = False
                                native_proof = False
                                break
                        _fill(handle, delta, x, b)
                        clear_dispatch_telemetry()
                        start = time.perf_counter_ns()
                        handle.flush()
                        elapsed = time.perf_counter_ns() - start
                        correct = correct and bool(
                            np.allclose(handle.materialize_state(), expected, rtol=4e-4, atol=4e-4)
                        )
                        if route == "resident_native":
                            provenance = handle.last_flush_execution
                            telemetry = handle.last_flush_telemetry or {}
                            native_proof = native_proof and provenance == "native_gpu"
                            if iteration >= warmup:
                                value = telemetry.get("device_time_ns")
                                if isinstance(value, int):
                                    device_times.append(value)
                                resources = telemetry.get("resources")
                            handle.close()
                        if iteration >= warmup:
                            wall.append(elapsed)
                    rows.append(
                        {
                            "run": run,
                            "device": device,
                            "shape": spec,
                            "tokens": tokens,
                            "route": route,
                            "available": available,
                            "native_proof": native_proof,
                            "correctness": correct if available else None,
                            "timing_domain_end_to_end_ns": (int(statistics.median(wall)) if wall else None),
                            "timing_domain_device_ns": (
                                int(statistics.median(device_times)) if len(device_times) == reps else None
                            ),
                            "device_time_coverage": (
                                (len(device_times) / reps if reps > 0 else 0.0) if route == "resident_native" else 0.0
                            ),
                            "resources": resources,
                        }
                    )
    finally:
        set_dispatch_telemetry_enabled(False)
    return {
        "schema": "tessera.apple.resident_replay_flush.v1",
        "device": device,
        "os": platform.platform(),
        "runs": runs,
        "warmup": warmup,
        "reps": reps,
        "rows": rows,
        "timing_domains": ["end_to_end", "device"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", default=["1x128x64", "1x256x128"])
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(args.shapes, tokens=args.tokens, warmup=args.warmup, reps=args.reps, runs=args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
