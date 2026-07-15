#!/usr/bin/env python3
"""Repeated-median ROCm KV/MoE transport ratchet (resident and end-to-end)."""

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

CASES = ((16, 128), (128, 128), (16, 1024), (128, 1024), (16, 4096))


def _mr(ptr, size):
    cv = ctypes.c_void_p
    return [cv(ptr.value), cv(ptr.value), ctypes.c_int64(0),
            ctypes.c_int64(size), ctypes.c_int64(1)]


class _Kernel:
    def __init__(self, hip, hsaco, symbol, args, work):
        self.hip = hip
        self.mod, self.fn = ctypes.c_void_p(), ctypes.c_void_p()
        if hip.hipModuleLoadData(ctypes.byref(self.mod), hsaco):
            raise RuntimeError("transport module load failed")
        if hip.hipModuleGetFunction(ctypes.byref(self.fn), self.mod, symbol):
            raise RuntimeError(f"transport symbol {symbol!r} missing")
        self.args = args
        self.array = (ctypes.c_void_p * len(args))()
        for i, value in enumerate(args):
            self.array[i] = ctypes.cast(ctypes.byref(value), ctypes.c_void_p)
        self.grid = max((work + 255) // 256, 1)

    def launch(self):
        rc = self.hip.hipModuleLaunchKernel(
            self.fn, self.grid, 1, 1, 256, 1, 1, 0, None, self.array, None)
        if rc:
            raise RuntimeError(f"transport launch failed: {rc}")

    def close(self):
        self.hip.hipModuleUnload(self.mod)


def _event_ms(hip, kernel, iterations):
    start, stop = ctypes.c_void_p(), ctypes.c_void_p()
    hip.hipEventCreate(ctypes.byref(start)); hip.hipEventCreate(ctypes.byref(stop))
    hip.hipEventRecord(start, None)
    for _ in range(iterations):
        kernel.launch()
    hip.hipEventRecord(stop, None); hip.hipEventSynchronize(stop)
    elapsed = ctypes.c_float()
    hip.hipEventElapsedTime(ctypes.byref(elapsed), start, stop)
    hip.hipEventDestroy(start); hip.hipEventDestroy(stop)
    return float(elapsed.value) / iterations


def _wall_ms(fn):
    start = time.perf_counter_ns(); fn()
    return (time.perf_counter_ns() - start) / 1e6


def _paired_summary(old, new):
    ratios = [a / b for a, b in zip(old, new, strict=True)]
    return {"old_trials_ms": old, "new_trials_ms": new,
            "old_median_ms": statistics.median(old),
            "new_median_ms": statistics.median(new),
            "paired_speedup": statistics.median(ratios),
            "paired_win_rate": sum(r > 1 for r in ratios) / len(ratios)}


def _gather_case(hip, rows, width, trials, iterations, rng):
    src_rows = 2048
    src = rng.standard_normal((src_rows, width), dtype=np.float32)
    ids = np.ascontiguousarray(rng.choice(src_rows, rows, replace=False), np.int64)
    flat_idx = np.ascontiguousarray(
        (ids[:, None] * width + np.arange(width, dtype=np.int64)).reshape(-1))
    out = np.zeros((rows, width), np.float32)
    devs = [rt._rocm_dev_in(hip, src.reshape(-1), 4 * src.size),
            rt._rocm_dev_in(hip, ids, 8 * ids.size),
            rt._rocm_dev_in(hip, flat_idx, 8 * flat_idx.size),
            rt._rocm_dev_in(hip, out.reshape(-1), 4 * out.size),
            rt._rocm_dev_in(hip, out.reshape(-1), 4 * out.size)]
    dsrc, dids, dflat, dout_old, dout_new = devs
    old_args = (_mr(dsrc, src.size) + [ctypes.c_int64(src.size)]
                + _mr(dflat, flat_idx.size) + _mr(dout_old, out.size)
                + [ctypes.c_int64(out.size)])
    new_args = (_mr(dsrc, src.size) + _mr(dids, ids.size)
                + _mr(dout_new, out.size)
                + [ctypes.c_int64(src_rows), ctypes.c_int64(rows),
                   ctypes.c_int64(width)])
    oldk = _Kernel(hip, rt._build_compiled_gather_hsaco(), b"g", old_args, out.size)
    newk = _Kernel(hip, rt._build_compiled_row_gather_hsaco(), b"rg", new_args, out.size)
    for _ in range(3): oldk.launch(); newk.launch()
    hip.hipDeviceSynchronize()
    d_old, d_new, e_old, e_new = [], [], [], []
    legacy = lambda: rt._rocm_gather(src.reshape(-1), flat_idx,
                                     np.zeros(out.size, np.float32))
    current = lambda: rt._rocm_row_gather(src, ids, np)
    got = current()
    if not np.array_equal(got, src[ids]):
        raise AssertionError("row gather mismatch")
    for trial in range(trials):
        order = ((oldk, d_old, legacy, e_old), (newk, d_new, current, e_new))
        if trial & 1: order = tuple(reversed(order))
        for kernel, ds, fn, es in order:
            ds.append(_event_ms(hip, kernel, iterations)); es.append(_wall_ms(fn))
    oldk.close(); newk.close()
    for ptr in devs: hip.hipFree(ptr)
    return {"operation": "row_gather", "shape": [rows, width],
            "index_bytes_old": int(flat_idx.nbytes),
            "index_bytes_new": int(ids.nbytes),
            "device": _paired_summary(d_old, d_new),
            "end_to_end": _paired_summary(e_old, e_new), "exact": True}


def _combine_case(hip, rows, width, trials, iterations, rng):
    out_rows = max(rows // 2, 1)
    src = rng.standard_normal((rows, width), dtype=np.float32)
    ids = np.ascontiguousarray(rng.integers(0, out_rows, rows), np.int64)
    weights = np.ascontiguousarray(rng.random(rows), np.float32)
    scaled = np.ascontiguousarray(src * weights[:, None], np.float32)
    out = np.zeros((out_rows, width), np.float32)
    devs = [rt._rocm_dev_in(hip, out.reshape(-1), 4 * out.size),
            rt._rocm_dev_in(hip, out.reshape(-1), 4 * out.size),
            rt._rocm_dev_in(hip, scaled.reshape(-1), 4 * scaled.size),
            rt._rocm_dev_in(hip, src.reshape(-1), 4 * src.size),
            rt._rocm_dev_in(hip, ids, 8 * ids.size),
            rt._rocm_dev_in(hip, weights, 4 * weights.size)]
    do, dn, dscaled, dsrc, dids, dw = devs
    old_args = (_mr(do, out.size) + [ctypes.c_int64(out_rows)]
                + _mr(dscaled, scaled.size) + _mr(dids, ids.size)
                + [ctypes.c_int64(rows), ctypes.c_int64(width)])
    new_args = (_mr(dn, out.size) + _mr(dsrc, src.size) + _mr(dids, ids.size)
                + _mr(dw, weights.size) + [ctypes.c_int64(out_rows),
                   ctypes.c_int64(rows), ctypes.c_int64(width)])
    oldk = _Kernel(hip, rt._build_compiled_scatter_hsaco(1), b"sc", old_args, src.size)
    chip = rt._rocm_chip()
    directive = ('module {\n  "tessera_rocm.scatter"() '
                 '{name = "wsc", mode = "weighted_add"} : () -> ()\n}\n')
    hsaco = rt._build_rocm_elementwise_hsaco(
        "generate-rocm-scatter-kernel", directive,
        rt._rocm_weighted_scatter_hsaco_cache, (chip,))
    newk = _Kernel(hip, hsaco, b"wsc", new_args, src.size)
    for _ in range(3):
        hip.hipMemset(do, 0, 4 * out.size); oldk.launch()
        hip.hipMemset(dn, 0, 4 * out.size); newk.launch()
    hip.hipDeviceSynchronize()
    d_old, d_new, e_old, e_new = [], [], [], []
    def legacy():
        value = np.zeros_like(out)
        rt._rocm_scatter(value, src * weights[:, None], ids, out_rows, width, 1, np)
        return value
    def current():
        value = np.zeros_like(out)
        rt._rocm_weighted_scatter_add(value, src, ids, weights, out_rows, width, np)
        return value
    if not np.allclose(legacy(), current(), rtol=2e-6, atol=2e-6):
        raise AssertionError("weighted combine mismatch")
    for trial in range(trials):
        order = ((oldk, do, d_old, legacy, e_old),
                 (newk, dn, d_new, current, e_new))
        if trial & 1: order = tuple(reversed(order))
        for kernel, target, ds, fn, es in order:
            hip.hipMemset(target, 0, 4 * out.size)
            ds.append(_event_ms(hip, kernel, iterations)); es.append(_wall_ms(fn))
    oldk.close(); newk.close()
    for ptr in devs: hip.hipFree(ptr)
    return {"operation": "weighted_scatter_add", "shape": [rows, width],
            "device": _paired_summary(d_old, d_new),
            "end_to_end": _paired_summary(e_old, e_new), "correct": True}


def run(trials, iterations):
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0): raise RuntimeError("live ROCm device required")
    rng = np.random.default_rng(1151)
    rows_out = []
    for rows, width in CASES:
        for fn in (_gather_case, _combine_case):
            row = fn(hip, rows, width, trials, iterations, rng)
            rows_out.append(row)
            d, e = row["device"], row["end_to_end"]
            print(f'{row["operation"]:22s} {rows:4d}x{width:4d} '
                  f'device={d["paired_speedup"]:.2f}x '
                  f'e2e={e["paired_speedup"]:.2f}x', flush=True)
    return {"schema": "tessera.rocm.transport_retune.v1",
            "evidence_arch": "gfx1151", "trials": trials,
            "iterations": iterations, "rows": rows_out, "all_correct": True}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--trials", type=int, default=9)
    p.add_argument("--iterations", type=int, default=50); p.add_argument("--output", required=True)
    a = p.parse_args(); result = run(a.trials, a.iterations)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__": main()
