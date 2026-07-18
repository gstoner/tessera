"""APPLE-REPLAY-1 paired fused-block vs resident-ring selector corpus."""
from __future__ import annotations

import argparse
import json
import statistics
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from tessera import runtime as rt
from tessera._apple_gpu_dispatch import (
    clear_dispatch_telemetry, read_dispatch_telemetry,
    set_dispatch_telemetry_enabled)
from tessera.cache import SSMStateHandle
from tessera.compiler.apple_route_selector import live_apple_device_tag


def _shape(text: str) -> tuple[int, int, int]:
    values = tuple(int(v) for v in text.lower().split("x"))
    if len(values) != 3 or min(values) <= 0:
        raise ValueError(f"shape must be BxDxN, got {text!r}")
    return values  # type: ignore[return-value]


def _inputs(seed: int, tokens: int, B: int, D: int, N: int):
    rng = np.random.default_rng(seed)
    a = -np.abs(rng.standard_normal(D)).astype(np.float64)
    delta = np.abs(rng.standard_normal((tokens, B, D))) * 0.2
    x = rng.standard_normal((tokens, B, D))
    b = rng.standard_normal((tokens, B, N))
    c = rng.standard_normal((tokens, B, N))
    return a, delta, x, b, c


def run_benchmark(shapes: list[str], *, tokens: int, warmup: int,
                  reps: int, runs: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    device = live_apple_device_tag()
    set_dispatch_telemetry_enabled(True)
    try:
        for run in range(1, runs + 1):
            for case, spec in enumerate(shapes):
                B, D, N = _shape(spec)
                a, delta, x, b, c = _inputs(6100 + run * 101 + case,
                                             tokens, B, D, N)
                oracle = SSMStateHandle(B, D, N, a).decode_block(delta, x, b, c)
                for route in ("fused_block", "resident_ring"):
                    wall: list[int] = []
                    device: list[int] = []
                    native = True
                    correct = True
                    resources: Any = None
                    scope: str | None = None
                    for iteration in range(warmup + reps):
                        clear_dispatch_telemetry()
                        start = time.perf_counter_ns()
                        if route == "fused_block":
                            handle = rt.apple_gpu_fused_ssm_state_handle(
                                B, D, N, a, capacity=tokens + 4)
                            got = handle.decode_block(delta, x, b, c)
                            provenance = handle.last_block_execution
                            telemetry = read_dispatch_telemetry()
                        else:
                            handle = rt.apple_gpu_resident_ssm_replay_state_handle(
                                B, D, N, a, capacity=tokens + 4, async_slots=2)
                            future = handle.submit_block_async(delta, x, b, c)
                            got = future.wait()
                            provenance = "native_gpu" if handle.resident_inputs else "reference_cpu"
                            telemetry = handle.last_submission_telemetry or {}
                            handle.close()
                        elapsed = time.perf_counter_ns() - start
                        native = native and provenance == "native_gpu"
                        correct = correct and bool(np.allclose(
                            got, oracle, rtol=3e-4, atol=3e-4))
                        resources = telemetry.get("resources")
                        scope = telemetry.get("device_time_scope") or (
                            "ordered_async_command_buffer" if route == "resident_ring"
                            else "single_block_dispatch")
                        if iteration >= warmup:
                            wall.append(elapsed)
                            value = telemetry.get("device_time_ns")
                            if isinstance(value, int):
                                device.append(value)
                    rows.append({
                        "run": run, "device": device, "shape": spec, "tokens": tokens,
                        "route": route,
                        "timing_domain_end_to_end_ns": int(statistics.median(wall)),
                        "timing_domain_device_ns": (
                            int(statistics.median(device)) if len(device) == reps else None),
                        "device_time_coverage": len(device) / reps,
                        "native_proof": native, "correctness": correct,
                        "device_time_scope": scope, "resources": resources,
                    })
    finally:
        set_dispatch_telemetry_enabled(False)

    decisions = []
    for spec in shapes:
        for domain, field in (
            ("end_to_end", "timing_domain_end_to_end_ns"),
            ("device", "timing_domain_device_ns"),
        ):
            winners = []
            for run in range(1, runs + 1):
                candidates = [r for r in rows if r["shape"] == spec and r["run"] == run
                              and r["native_proof"] and r["correctness"]
                              and r[field] is not None]
                winners.append(min(candidates, key=lambda r: r[field])["route"]
                               if len(candidates) == 2 else None)
            decisions.append({
                "shape": spec, "tokens": tokens, "timing_domain": domain,
                "run_winners": winners,
                "stable_winner": winners[0] if winners and len(set(winners)) == 1
                and winners[0] is not None else None,
            })
    return {
        "schema": "tessera.apple.resident_replay.v1", "device": device,
        "os": platform.platform(), "runs": runs,
        "warmup": warmup, "reps": reps, "rows": rows,
        "decisions": decisions,
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
    report = run_benchmark(args.shapes, tokens=args.tokens, warmup=args.warmup,
                           reps=args.reps, runs=args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["decisions"], indent=2))


if __name__ == "__main__":
    main()
