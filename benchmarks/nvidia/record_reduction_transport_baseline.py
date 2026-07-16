"""Record TEST-5 repeated-median rows for NVIDIA reductions and MoE transport.

Each route is measured in two non-interchangeable domains: end-to-end host
calls (allocation, transfer, launch, result) and CUDA-event kernel time with
the same generated production kernel and resident operands.  This recorder
never writes a baseline unless an sm_120 CUDA device is actually executing it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
OUT = ROOT / "benchmarks/baselines/nvidia_sm120_reduction_transport.json"


def _median(fn: Callable[[], float], reps: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    return float(statistics.median(fn() for _ in range(reps)))


def _wall(fn: Callable[[], Any]) -> float:
    start = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns() - start) / 1e6


def benchmark_cases() -> list[tuple[str, str, str, str, Callable[[], Any], Callable[[], float]]]:
    """Return selected production routes with end-to-end and event thunks."""
    from tessera.compiler.emit import nvidia_cuda as nv

    rng = np.random.default_rng(20260715)
    reductions = [
        ("reduction_sum", "257x1025", "f32", "sum"),
        ("reduction_mean", "255x1023", "f16", "mean"),
        ("reduction_max", "129x769", "f32", "max"),
    ]
    cases = []
    for op, shape, dtype, kind in reductions:
        m, k = (int(v) for v in shape.split("x"))
        x = (rng.standard_normal((m, k)) * .25).astype(
            np.float16 if dtype == "f16" else np.float32)
        # A preflight computes the oracle-independent output before timing.
        nv.run_row_reduce(x, kind)
        cases.append((op, shape, dtype, "generated_row_reduce",
                      lambda x=x, kind=kind: nv.run_row_reduce(x, kind),
                      lambda x=x, kind=kind: nv.measure_row_reduce_device(x, kind)))

    tokens, hidden, experts, out = 257, 193, 5, 127
    x = (rng.standard_normal((tokens, hidden)) * .25).astype(np.float32)
    slots = np.arange(tokens, dtype=np.int32)[::-1]
    partials = (rng.standard_normal((tokens, hidden)) * .25).astype(np.float32)
    weights = rng.random(tokens, dtype=np.float32)
    groups = np.array([51, 0, 73, 61, 72], dtype=np.int64)
    expert_weights = (rng.standard_normal((experts, hidden, out)) * .1).astype(np.float32)
    cases.extend([
        ("moe_dispatch", "257x193", "f32", "generated_gather",
         lambda: nv.run_moe_dispatch_f32(x, slots),
         lambda: nv.measure_moe_dispatch_device(x, slots)),
        ("moe_combine", "257x193", "f32", "generated_combine",
         lambda: nv.run_moe_combine_f32(partials, slots, weights, tokens),
         lambda: nv.measure_moe_combine_device(partials, slots, weights, tokens)),
        ("grouped_gemm", "257x193x127x5", "f32", "generated_grouped",
         lambda: nv.run_grouped_gemm_f32(x, expert_weights, groups),
         lambda: nv.measure_grouped_gemm_device(x, expert_weights, groups)),
    ])
    return cases


def record(*, reps: int = 20, warmup: int = 3, device_reps: int = 100,
           margin: float = 2.0) -> list[dict[str, Any]]:
    from tessera import runtime as rt
    if rt._nvidia_device_name() != "sm_120":
        return []
    rows: list[dict[str, Any]] = []
    for op, shape, dtype, route, end_to_end, device in benchmark_cases():
        end_ms = _median(lambda: _wall(end_to_end), reps, warmup)
        device_ms = _median(lambda: device(), reps, warmup)
        for domain, median in (("end_to_end", end_ms), ("device_event", device_ms)):
            rows.append({
                "op": op, "shape": shape, "dtype": dtype,
                "mode": f"{route}:{domain}", "selected_route": route,
                "timing_domain": domain, "median_ms": round(median, 6),
                "max_latency_ms": round(median * margin, 6),
                # Kept with the row so the required NCU capture cannot be
                # confused with a timing-domain or selector verdict.
                "resource_evidence": "ncu production capture required",
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device-reps", type=int, default=100)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    rows = record(reps=args.reps, warmup=args.warmup,
                  device_reps=args.device_reps, margin=args.margin)
    if not rows:
        print("sm_120 NVIDIA runtime unavailable; baseline unchanged")
        return 0
    args.output.write_text(json.dumps({
        "schema": "tessera.benchmark.ratchet.v1", "margin": args.margin,
        "device": "nvidia:sm_120", "rows": rows,
    }, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
