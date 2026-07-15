#!/usr/bin/env python3
"""Correctness-gated f32 GEMM tile retuning with device and end-to-end timing."""

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

from tessera import runtime as rt  # noqa: E402

TILES = ((1, 1), (2, 2), (2, 4), (3, 4), (4, 4),
         (4, 6), (6, 4), (4, 8), (8, 4))
SHAPES = ((256, 256, 256), (512, 512, 512), (1024, 1024, 1024),
          (128, 512, 256), (512, 1024, 512), (1025, 1537, 1009))


def _median_samples(fn, trials: int) -> tuple[float, list[float]]:
    samples = []
    for _ in range(trials):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(samples), samples


def _device_ms(hip, hsaco: bytes, shape: tuple[int, int, int],
               tile: tuple[int, int], trials: int, iterations: int) -> list[float]:
    import ctypes

    m, n, k = shape
    mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        raise RuntimeError("f32 module load failed")
    if hip.hipModuleGetFunction(ctypes.byref(fn), mod, b"g") != 0:
        raise RuntimeError("f32 symbol g missing")
    ptrs = [ctypes.c_void_p() for _ in range(3)]
    sizes = (m * k, k * n, m * n)
    for ptr, count in zip(ptrs, sizes, strict=True):
        if hip.hipMalloc(ctypes.byref(ptr), 4 * count) != 0:
            raise RuntimeError("f32 hipMalloc failed")

    def mr(ptr, count):
        return [ctypes.c_void_p(ptr.value), ctypes.c_void_p(ptr.value),
                ctypes.c_int64(0), ctypes.c_int64(count), ctypes.c_int64(1)]

    args = (mr(ptrs[0], sizes[0]) + mr(ptrs[1], sizes[1]) +
            mr(ptrs[2], sizes[2]) +
            [ctypes.c_int64(m), ctypes.c_int64(n), ctypes.c_int64(k)])
    array = (ctypes.c_void_p * len(args))()
    for index, value in enumerate(args):
        array[index] = ctypes.cast(ctypes.byref(value), ctypes.c_void_p)
    tm, tn = tile
    count = ((m + tm - 1) // tm) * ((n + tn - 1) // tn)
    grid = (count + 255) // 256

    def launch():
        rc = hip.hipModuleLaunchKernel(
            fn, grid, 1, 1, 256, 1, 1, 0, None, array, None)
        if rc:
            raise RuntimeError(f"f32 launch failed: {rc}")

    for _ in range(3):
        launch()
    hip.hipDeviceSynchronize()
    samples = []
    for _ in range(trials):
        start, stop = ctypes.c_void_p(), ctypes.c_void_p()
        hip.hipEventCreate(ctypes.byref(start)); hip.hipEventCreate(ctypes.byref(stop))
        hip.hipEventRecord(start, None)
        for _ in range(iterations):
            launch()
        hip.hipEventRecord(stop, None); hip.hipEventSynchronize(stop)
        elapsed = ctypes.c_float()
        hip.hipEventElapsedTime(ctypes.byref(elapsed), start, stop)
        hip.hipEventDestroy(start); hip.hipEventDestroy(stop)
        samples.append(float(elapsed.value) / iterations)
    for ptr in ptrs:
        hip.hipFree(ptr)
    hip.hipModuleUnload(mod)
    return samples


def run(trials: int, iterations: int) -> dict:
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("live ROCm device required")
    rows = []
    for shape in SHAPES:
        m, n, k = shape
        rng = np.random.default_rng(sum(shape))
        a = (rng.standard_normal((m, k), dtype=np.float32) * 0.125).astype(np.float32)
        b = (rng.standard_normal((k, n), dtype=np.float32) * 0.125).astype(np.float32)
        reference = a @ b
        print(f"f32 {m}x{n}x{k}", flush=True)
        for tile in TILES:
            hsaco = rt._build_compiled_gemm_f32_hsaco(*tile)
            actual = rt._rocm_f32_gemm(a, b, np, tile=tile)
            max_error = float(np.max(np.abs(actual - reference)))
            if max_error > 2e-3:
                raise AssertionError(f"{shape} {tile} max error {max_error}")
            device = _device_ms(hip, hsaco, shape, tile, trials, iterations)
            e2e_median, e2e = _median_samples(
                lambda: rt._rocm_f32_gemm(a, b, np, tile=tile), trials)
            device_median = statistics.median(device)
            row = {"shape": list(shape), "tile": list(tile),
                   "device_trials_ms": device, "device_median_ms": device_median,
                   "e2e_trials_ms": e2e, "e2e_median_ms": e2e_median,
                   "tflops": 2 * m * n * k / (device_median * 1e9),
                   "max_abs_error": max_error}
            rows.append(row)
            print(f"  {tile}: device={device_median:.4f}ms "
                  f"e2e={e2e_median:.3f}ms {row['tflops']:.3f} TF/s", flush=True)
    winners = []
    for shape in SHAPES:
        candidates = [row for row in rows if row["shape"] == list(shape)]
        device = min(candidates, key=lambda row: row["device_median_ms"])
        e2e = min(candidates, key=lambda row: row["e2e_median_ms"])
        winners.append({"shape": list(shape), "device_tile": device["tile"],
                        "e2e_tile": e2e["tile"],
                        "device_ms": device["device_median_ms"],
                        "e2e_ms": e2e["e2e_median_ms"]})
    return {"schema": "tessera.rocm.f32_retune.v1", "evidence_arch": "gfx1151",
            "trials": trials, "iterations": iterations, "rows": rows,
            "winners": winners, "all_correct": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.trials, args.iterations)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
