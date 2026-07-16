"""Record dual-domain repeated-median NVIDIA Flash-Attention backward rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
OUT = ROOT / "benchmarks/baselines/nvidia_sm120_attention_backward.json"
SHAPES = ((1, 8, 128, 64), (1, 8, 257, 64))


def _median(fn: Callable[[], float], reps: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    return float(statistics.median(fn() for _ in range(reps)))


def _wall(fn: Callable[[], Any]) -> float:
    start = time.perf_counter_ns(); fn()
    return (time.perf_counter_ns() - start) / 1e6


def compiler_fingerprint() -> str:
    nvcc = subprocess.run(["/usr/local/cuda/bin/nvcc", "--version"],
                          check=True, capture_output=True, text=True).stdout
    return "sha256:" + hashlib.sha256(nvcc.encode()).hexdigest()


def record(*, reps: int = 20, warmup: int = 3, device_reps: int = 20,
           margin: float = 2.0) -> list[dict[str, Any]]:
    from tessera import runtime as rt
    from tessera.compiler.emit.nvidia_cuda import (
        measure_flash_attention_backward_device, run_flash_attention_backward)
    if rt._nvidia_device_name() != "sm_120":
        return []
    rng = np.random.default_rng(20260716)
    fingerprint = compiler_fingerprint()
    rows: list[dict[str, Any]] = []
    for b, h, s, d in SHAPES:
        arrays = [rng.standard_normal((b, h, s, d), dtype=np.float32)
                  for _ in range(4)]
        do, q, k, v = arrays
        scale = d ** -0.5
        start = time.perf_counter_ns()
        run_flash_attention_backward(do, q, k, v, scale=scale, causal=True)
        cold_ms = (time.perf_counter_ns() - start) / 1e6
        end = _median(lambda: _wall(lambda: run_flash_attention_backward(
            do, q, k, v, scale=scale, causal=True)), reps, warmup)
        device = _median(lambda: measure_flash_attention_backward_device(
            do, q, k, v, scale=scale, causal=True, reps=device_reps),
            reps, warmup)
        common = {
            "op": "flash_attention_backward", "shape": f"{b}x{h}x{s}x{d}",
            "dtype": "f32", "selected_route": "generated_atomic_vjp",
            "compiler_fingerprint": fingerprint, "cold_compile_ms": round(cold_ms, 6),
            "cache_state": "warm", "warmup": warmup, "reps": reps,
            "resource_key": "tsr_flash_bwd",
        }
        for domain, median in (("end_to_end", end), ("device_event", device)):
            rows.append({**common, "timing_domain": domain,
                         "mode": f"generated_atomic_vjp:{domain}",
                         "median_ms": round(median, 6),
                         "max_latency_ms": round(median * margin, 6)})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device-reps", type=int, default=20)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    rows = record(reps=args.reps, warmup=args.warmup,
                  device_reps=args.device_reps, margin=args.margin)
    if not rows:
        print("sm_120 NVIDIA runtime unavailable; baseline unchanged")
        return 0
    args.output.write_text(json.dumps({
        "schema": "tessera.benchmark.ratchet.v1", "device": "nvidia:sm_120",
        "margin": args.margin, "rows": rows,
    }, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
