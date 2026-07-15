#!/usr/bin/env python3
"""Correctness-gated grouped-f32 GEMM schedule ratchet for gfx1151.

Device rows keep modules and buffers resident and use HIP events.  End-to-end
rows include allocation, copies, module load, launch, and result copy through
the production runtime helper.  Trial order rotates and reverses so a tile is
not systematically advantaged by temperature or clock drift.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from tessera import runtime as rt  # noqa: E402

TILES = (1, 2, 4, 8, 16)
CASES = (
    ("balanced_small", (16,) * 8, 256, 256),
    ("transition_64k", (16,) * 8, 512, 512),
    ("transition_64k_high_k", (32,) * 8, 1024, 256),
    ("balanced_model", (32,) * 8, 512, 512),
    ("ragged_model", (3, 11, 19, 29, 37, 43, 51, 63), 512, 512),
    ("wide_ffn", (16,) * 8, 256, 1024),
    ("narrow_down", (16,) * 8, 1024, 256),
)


def _hsaco(tile_n: int) -> bytes:
    chip = rt._rocm_chip()
    directive = (
        'module {\n  "tessera_rocm.grouped_gemm"() '
        f'{{name = "grouped_gemm", tn = {tile_n} : i64}} : () -> ()\n}}\n')
    return rt._build_rocm_elementwise_hsaco(
        "generate-rocm-moe-kernel", directive,
        rt._rocm_grouped_gemm_hsaco_cache, (chip, tile_n))


class _Resident:
    def __init__(self, hip, x, weights, offsets, tile_n):
        self.hip = hip
        self.tile_n = tile_n
        self.t, self.k = map(int, x.shape)
        self.e, _, self.n = map(int, weights.shape)
        self.mod, self.fn = ctypes.c_void_p(), ctypes.c_void_p()
        hsaco = _hsaco(tile_n)
        if hip.hipModuleLoadData(ctypes.byref(self.mod), hsaco):
            raise RuntimeError("grouped GEMM module load failed")
        if hip.hipModuleGetFunction(
                ctypes.byref(self.fn), self.mod, b"grouped_gemm"):
            raise RuntimeError("grouped GEMM symbol missing")
        out = np.zeros((self.t, self.n), np.float32)
        self.devs = [
            rt._rocm_dev_in(hip, x.reshape(-1), 4 * x.size),
            rt._rocm_dev_in(hip, weights.reshape(-1), 4 * weights.size),
            rt._rocm_dev_in(hip, offsets, 4 * offsets.size),
            rt._rocm_dev_in(hip, out.reshape(-1), 4 * out.size),
        ]
        sizes = (x.size, weights.size, offsets.size, out.size)
        args = []
        for ptr, size in zip(self.devs, sizes, strict=True):
            args += [ctypes.c_void_p(ptr.value), ctypes.c_void_p(ptr.value),
                     ctypes.c_int64(0), ctypes.c_int64(size), ctypes.c_int64(1)]
        args += [ctypes.c_int64(v) for v in (self.t, self.k, self.n, self.e)]
        self.args = args
        self.array = (ctypes.c_void_p * len(args))()
        for index, value in enumerate(args):
            self.array[index] = ctypes.cast(ctypes.byref(value), ctypes.c_void_p)
        work = self.t * (self.n // tile_n)
        self.grid = (work + 255) // 256

    def launch(self):
        rc = self.hip.hipModuleLaunchKernel(
            self.fn, self.grid, 1, 1, 256, 1, 1, 0, None, self.array, None)
        if rc:
            raise RuntimeError(f"grouped GEMM launch failed: {rc}")

    def close(self):
        for ptr in self.devs:
            self.hip.hipFree(ptr)
        self.hip.hipModuleUnload(self.mod)


def _timed_device(hip, resident: _Resident, iterations: int) -> float:
    start, stop = ctypes.c_void_p(), ctypes.c_void_p()
    hip.hipEventCreate(ctypes.byref(start)); hip.hipEventCreate(ctypes.byref(stop))
    hip.hipEventRecord(start, None)
    for _ in range(iterations):
        resident.launch()
    hip.hipEventRecord(stop, None); hip.hipEventSynchronize(stop)
    elapsed = ctypes.c_float()
    hip.hipEventElapsedTime(ctypes.byref(elapsed), start, stop)
    hip.hipEventDestroy(start); hip.hipEventDestroy(stop)
    return float(elapsed.value) / iterations


def _order(trial: int) -> tuple[int, ...]:
    values = list(TILES)
    if trial & 1:
        values.reverse()
    shift = (trial // 2) % len(values)
    return tuple(values[shift:] + values[:shift])


def run(trials: int, iterations: int) -> dict:
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):
        raise RuntimeError("live ROCm device required")
    rows = []
    for name, sizes, k, n in CASES:
        gs = np.asarray(sizes, np.int64)
        t, e = int(gs.sum()), int(gs.size)
        rng = np.random.default_rng(t + k + n)
        x = (rng.standard_normal((t, k), dtype=np.float32) * 0.05).astype(np.float32)
        weights = (rng.standard_normal((e, k, n), dtype=np.float32) * 0.05).astype(np.float32)
        offsets = np.ascontiguousarray(
            np.concatenate((np.zeros(1, np.int64), np.cumsum(gs))), np.int32)
        ref = np.empty((t, n), np.float32)
        off = 0
        for expert, count in enumerate(gs):
            count = int(count)
            ref[off:off + count] = x[off:off + count] @ weights[expert]
            off += count
        resident = {tile: _Resident(hip, x, weights, offsets, tile)
                    for tile in TILES if n % tile == 0}
        for session in resident.values():
            for _ in range(3):
                session.launch()
        hip.hipDeviceSynchronize()
        device = {tile: [] for tile in resident}
        e2e = {tile: [] for tile in resident}
        errors = {}
        for tile in resident:
            got = rt._rocm_grouped_gemm_native(x, weights, gs, np, tile_n=tile)
            errors[tile] = float(np.max(np.abs(got - ref)))
            if errors[tile] > 2e-3:
                raise AssertionError(f"{name} tn={tile} error={errors[tile]}")
        for trial in range(trials):
            for tile in _order(trial):
                if tile not in resident:
                    continue
                device[tile].append(_timed_device(hip, resident[tile], iterations))
                begin = time.perf_counter_ns()
                rt._rocm_grouped_gemm_native(x, weights, gs, np, tile_n=tile)
                e2e[tile].append((time.perf_counter_ns() - begin) / 1e6)
        baseline = statistics.median(device[1])
        for tile in resident:
            dmed, emed = statistics.median(device[tile]), statistics.median(e2e[tile])
            paired = [device[1][i] / device[tile][i] for i in range(trials)]
            row = {
                "case": name, "shape": [t, k, n], "experts": e,
                "group_sizes": list(map(int, gs)), "tile_n": tile,
                "device_trials_ms": device[tile], "device_median_ms": dmed,
                "e2e_trials_ms": e2e[tile], "e2e_median_ms": emed,
                "paired_speedup_vs_tn1": statistics.median(paired),
                "paired_win_rate_vs_tn1": sum(v > 1.0 for v in paired) / trials,
                "tflops": 2 * t * k * n / (dmed * 1e9),
                "max_abs_error": errors[tile],
            }
            rows.append(row)
            print(f"{name:16s} tn={tile:2d} device={dmed:8.4f} ms "
                  f"e2e={emed:8.3f} ms speedup={baseline / dmed:5.2f}x",
                  flush=True)
        for session in resident.values():
            session.close()
    winners = []
    for name, *_ in CASES:
        candidates = [r for r in rows if r["case"] == name]
        winners.append({
            "case": name,
            "device_tile_n": min(candidates, key=lambda r: r["device_median_ms"])["tile_n"],
            "e2e_tile_n": min(candidates, key=lambda r: r["e2e_median_ms"])["tile_n"],
        })
    return {"schema": "tessera.rocm.grouped_gemm_retune.v1",
            "evidence_arch": "gfx1151", "trials": trials,
            "iterations": iterations, "rows": rows, "winners": winners,
            "all_correct": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.trials, args.iterations)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
