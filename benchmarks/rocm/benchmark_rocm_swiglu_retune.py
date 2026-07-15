#!/usr/bin/env python3
"""Kernel-only and end-to-end ratchet for ROCm grouped SwiGLU."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from tessera import runtime as rt  # noqa: E402
from tessera.stdlib import moe  # noqa: E402

CASES = ((8, 16, 256, 512), (8, 32, 512, 1024))  # E, tokens/expert, H, F


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


F32 = _load("_f32_bench", ROOT / "benchmarks/rocm/benchmark_rocm_f32_retune.py")
GROUPED = _load("_grouped_bench", ROOT / "benchmarks/rocm/benchmark_rocm_grouped_gemm_retune.py")


def _legacy(x, wg, wu, wd, gs):
    out = np.empty((x.shape[0], wd.shape[2]), np.float32)
    off = 0
    for expert, count in enumerate(gs):
        count = int(count); xe = x[off:off + count]
        gate = rt._rocm_f32_gemm(xe, wg[expert], np)
        up = rt._rocm_f32_gemm(xe, wu[expert], np)
        hidden = (gate / (1 + np.exp(-gate)) * up).astype(np.float32)
        out[off:off + count] = rt._rocm_f32_gemm(hidden, wd[expert], np)
        off += count
    return out


def run(trials, iterations):
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0): raise RuntimeError("live ROCm device required")
    rows = []
    for experts, per, hidden, ffn in CASES:
        total = experts * per; gs = np.full(experts, per, np.int64)
        rng = np.random.default_rng(total + hidden + ffn)
        x = (rng.standard_normal((total, hidden), dtype=np.float32) * .03).astype(np.float32)
        wg = (rng.standard_normal((experts, hidden, ffn), dtype=np.float32) * .03).astype(np.float32)
        wu = (rng.standard_normal((experts, hidden, ffn), dtype=np.float32) * .03).astype(np.float32)
        wd = (rng.standard_normal((experts, ffn, hidden), dtype=np.float32) * .03).astype(np.float32)
        expected = moe.grouped_swiglu(x, wg, wu, wd, gs)
        got = rt._moe_grouped_swiglu_native(x, wg, wu, wd, gs, np)
        max_error = float(np.max(np.abs(got - expected)))
        np.testing.assert_allclose(got, expected, rtol=2e-4, atol=2e-5)

        offsets = np.ascontiguousarray(
            np.concatenate((np.zeros(1, np.int64), np.cumsum(gs))), np.int32)
        gate_tile = rt._rocm_grouped_gemm_tile_n(total, ffn)
        down_tile = rt._rocm_grouped_gemm_tile_n(total, hidden)
        gate_session = GROUPED._Resident(hip, x, wg, offsets, gate_tile)
        down_input = np.zeros((total, ffn), np.float32)
        down_session = GROUPED._Resident(hip, down_input, wd, offsets, down_tile)
        old_gate = F32._device_ms(
            hip, rt._build_compiled_gemm_f32_hsaco(
                *rt._rocm_f32_gemm_tile(per, ffn, hidden)),
            (per, ffn, hidden), rt._rocm_f32_gemm_tile(per, ffn, hidden),
            trials, iterations)
        old_down = F32._device_ms(
            hip, rt._build_compiled_gemm_f32_hsaco(
                *rt._rocm_f32_gemm_tile(per, hidden, ffn)),
            (per, hidden, ffn), rt._rocm_f32_gemm_tile(per, hidden, ffn),
            trials, iterations)
        new_device, old_device, new_e2e, old_e2e = [], [], [], []
        for trial in range(trials):
            new_device.append(
                2 * GROUPED._timed_device(hip, gate_session, iterations)
                + GROUPED._timed_device(hip, down_session, iterations))
            old_device.append(experts * (2 * old_gate[trial] + old_down[trial]))
            order = (False, True) if not (trial & 1) else (True, False)
            for new in order:
                start = time.perf_counter_ns()
                (rt._moe_grouped_swiglu_native(x, wg, wu, wd, gs, np)
                 if new else _legacy(x, wg, wu, wd, gs))
                value = (time.perf_counter_ns() - start) / 1e6
                (new_e2e if new else old_e2e).append(value)
        gate_session.close(); down_session.close()
        dr = [a/b for a, b in zip(old_device, new_device, strict=True)]
        er = [a/b for a, b in zip(old_e2e, new_e2e, strict=True)]
        row = {"shape": [experts, total, hidden, ffn],
               "legacy_launches": 3 * experts, "grouped_launches": 3,
               "legacy_device_trials_ms": old_device,
               "grouped_device_trials_ms": new_device,
               "legacy_device_median_ms": statistics.median(old_device),
               "grouped_device_median_ms": statistics.median(new_device),
               "paired_device_speedup": statistics.median(dr),
               "paired_device_win_rate": sum(x > 1 for x in dr)/trials,
               "legacy_e2e_trials_ms": old_e2e, "grouped_e2e_trials_ms": new_e2e,
               "legacy_e2e_median_ms": statistics.median(old_e2e),
               "grouped_e2e_median_ms": statistics.median(new_e2e),
               "paired_e2e_speedup": statistics.median(er),
               "paired_e2e_win_rate": sum(x > 1 for x in er)/trials,
               "max_abs_error": max_error}
        rows.append(row)
        print(f"E{experts} T{total} H{hidden} F{ffn}: device "
              f"{row['paired_device_speedup']:.2f}x, e2e "
              f"{row['paired_e2e_speedup']:.2f}x", flush=True)
    return {"schema": "tessera.rocm.swiglu_retune.v1",
            "evidence_arch": "gfx1151", "trials": trials,
            "iterations": iterations, "rows": rows, "all_correct": True}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--trials", type=int, default=9)
    p.add_argument("--iterations", type=int, default=30); p.add_argument("--output", required=True)
    a = p.parse_args(); result = run(a.trials, a.iterations)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__": main()
