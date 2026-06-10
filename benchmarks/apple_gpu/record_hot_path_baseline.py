"""P2 (2026-06-09) — record the Apple GPU hot-path perf-ratchet baseline.

Times the named hot paths (matmul, fused epilogue chains, conv2d, the
3-op decode chain) through the SAME runtime dispatchers production code
uses, then writes `benchmarks/baselines/apple_gpu_hot_paths.json` with
per-row thresholds = median * margin. The CI ratchet (`perf_gate.
evaluate_ratchet`, locked by `tests/unit/test_apple_gpu_perf_ratchet.py`)
re-times and fails on regressions past the margin.

Run on the machine that hosts CI:
    python benchmarks/apple_gpu/record_hot_path_baseline.py [--margin 2.0]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

OUT = ROOT / "benchmarks" / "baselines" / "apple_gpu_hot_paths.json"


def _median_ms(fn, reps: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def hot_path_cases(rt):
    """(op, shape, dtype, thunk) per named hot path — module-level so the
    ratchet test re-times the identical work."""
    rng = np.random.default_rng(0)
    f32 = np.float32
    a = rng.standard_normal((512, 512)).astype(f32)
    b = rng.standard_normal((512, 512)).astype(f32)
    ms_a = rng.standard_normal((64, 64)).astype(f32)
    ms_b = rng.standard_normal((64, 256)).astype(f32)
    ms_c = rng.standard_normal((256, 64)).astype(f32)
    x = rng.standard_normal((64, 128)).astype(f32)
    wg = rng.standard_normal((128, 256)).astype(f32)
    wu = rng.standard_normal((128, 256)).astype(f32)
    wd = rng.standard_normal((256, 128)).astype(f32)
    cx = rng.standard_normal((1, 32, 32, 16)).astype(f32)
    cw = rng.standard_normal((3, 3, 16, 32)).astype(f32)
    return [
        ("matmul", "512x512x512", "f32",
         lambda: rt._apple_gpu_dispatch_matmul("tessera.matmul", [a, b], np)),
        ("matmul_softmax", "64x64x256", "f32",
         lambda: rt._apple_gpu_dispatch_matmul_softmax([ms_a, ms_b], np)),
        ("matmul_gelu", "64x64x256", "f32",
         lambda: rt._apple_gpu_dispatch_matmul_gelu([ms_a, ms_b], np)),
        ("matmul_rmsnorm", "64x64x256", "f32",
         lambda: rt._apple_gpu_dispatch_matmul_rmsnorm([ms_a, ms_b], 1e-5, np)),
        ("matmul_softmax_matmul", "64x64x256x64", "f32",
         lambda: rt._apple_gpu_dispatch_matmul_softmax_matmul([ms_a, ms_b, ms_c], np)),
        ("swiglu", "64x128x256", "f32",
         lambda: rt._apple_gpu_dispatch_swiglu(x, wg, wu, wd, np)),
        ("conv2d", "1x32x32x16_3x3x16x32", "f32",
         lambda: rt._apple_gpu_dispatch_conv2d([cx, cw], {}, np)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", type=float, default=2.0,
                        help="threshold = median * margin (CI noise headroom)")
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()
    if sys.platform != "darwin":
        print("apple_gpu hot-path baseline requires Darwin; skipping")
        return 0
    from tessera import runtime as rt

    rows = []
    for op, shape, dtype, thunk in hot_path_cases(rt):
        med = _median_ms(thunk, reps=args.reps)
        rows.append({
            "op": op, "shape": shape, "dtype": dtype, "mode": "fused",
            "median_ms": round(med, 4),
            "max_latency_ms": round(med * args.margin, 4),
        })
        print(f"{op:24s} {shape:20s} median {med:8.3f} ms  cap {med * args.margin:8.3f} ms")
    OUT.write_text(json.dumps({
        "schema": "tessera.benchmark.ratchet.v1",
        "margin": args.margin,
        "rows": rows,
    }, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
