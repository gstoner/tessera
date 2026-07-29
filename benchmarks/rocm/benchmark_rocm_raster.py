#!/usr/bin/env python3
"""ROCM-RASTER-1 gfx1151 correctness and interleaved HIP-event timing."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from tessera import runtime as rt  # noqa: E402
from tessera.compiler.rocm_schedule import select_rocm_gemm_schedule  # noqa: E402
from tessera.compiler.tile_rasterization import is_bijection  # noqa: E402

MATRIX_PATH = ROOT / "benchmarks/rocm/benchmark_rocm_gemm_schedule_matrix.py"
SPEC = importlib.util.spec_from_file_location("rocm_schedule_matrix", MATRIX_PATH)
assert SPEC and SPEC.loader
MATRIX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MATRIX
SPEC.loader.exec_module(MATRIX)

SHAPES = (
    ("square", 1024, 1024, 1024),
    ("wide", 256, 4096, 1024),
    ("tall", 4096, 256, 1024),
    ("ragged", 1025, 1537, 1009),
)
RASTERS = (
    ("row_major", 1),
    ("column_major", 1),
    ("grouped_m", 4),
    ("grouped_m", 8),
    ("grouped_n", 4),
    ("grouped_n", 8),
)


def _pmc_capability() -> dict[str, object]:
    command = ["/opt/rocm/core/bin/rocprofv3-avail", "list", "--pmc"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return {"supported": False, "command": command, "error": str(exc)}
    text = (result.stdout + result.stderr).strip()
    supported = (
        result.returncode == 0
        and "not supported" not in text.lower()
        and bool(text)
    )
    return {
        "supported": supported,
        "command": command,
        "returncode": result.returncode,
        "output": text,
        "requested_domains": ["MALL", "L2/TCP/TCC"],
    }


def run(*, trials: int, iterations: int, warmup: int,
        quick: bool = False) -> dict[str, object]:
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("live ROCm device required")
    shapes = SHAPES[:2] if quick else SHAPES
    rows: list[dict[str, object]] = []
    all_correct = True
    timing_available = True
    timing_error: str | None = None

    for family, m, n, k in shapes:
        case = MATRIX.Case(family, m, n, k)
        a, b, _ = MATRIX._inputs(case)
        reference = MATRIX._reference(case, a, b, None)
        assert reference is not None
        devices = []
        try:
            for order, group in RASTERS:
                schedule = select_rocm_gemm_schedule(
                    m, n, k, dtype="f16", arch="gfx1151",
                    raster_order=order, raster_group=group)
                hsaco = rt._build_compiled_gemm_hsaco(
                    schedule.mt, schedule.nt, "f16", schedule=schedule)
                grid_m = (m + 16 * schedule.mt - 1) // (16 * schedule.mt)
                grid_n = (n + 16 * schedule.nt - 1) // (16 * schedule.nt)
                if not is_bijection(order, grid_m, grid_n, group=group):
                    raise AssertionError(
                        f"{order}/{group} is not bijective on {grid_m}x{grid_n}")
                device = MATRIX.DeviceCase(
                    hip, hsaco, case, schedule.macro_tile, a, b, None)
                metrics = MATRIX._error_metrics(device.download(), reference)
                correct = MATRIX._correct(case, metrics)
                all_correct &= correct
                devices.append((order, group, schedule, device, metrics, correct))

            samples = {(order, group): [] for order, group in RASTERS}
            if timing_available:
                try:
                    # Warm every code object before rotating the timed order.
                    for *_, device, _metrics, _correct in devices:
                        device.measure(trials=1, iterations=1, warmup=warmup)
                    for round_index in range(trials):
                        rotated = (
                            devices[round_index % len(devices):]
                            + devices[:round_index % len(devices)]
                        )
                        for (order, group, _schedule, device, _metrics,
                             _correct) in rotated:
                            sample = device.measure(
                                trials=1, iterations=iterations, warmup=0)[0]
                            samples[(order, group)].append(sample)
                except RuntimeError as exc:
                    timing_available = False
                    timing_error = str(exc)

            baseline = (
                statistics.median(samples[("row_major", 1)])
                if samples[("row_major", 1)] else None
            )
            for order, group, schedule, _device, metrics, correct in devices:
                values = samples[(order, group)]
                median = statistics.median(values) if values else None
                paired = (
                    [row / candidate for row, candidate in zip(
                        samples[("row_major", 1)], values, strict=True)]
                    if values else []
                )
                rows.append({
                    "family": family,
                    "shape": [m, n, k],
                    "macro_tile": list(schedule.macro_tile),
                    "raster_order": order,
                    "raster_group": group,
                    "trials_ms": values,
                    "median_ms": median,
                    "speedup_vs_row_major": (
                        baseline / median
                        if baseline is not None and median is not None else None
                    ),
                    "paired_median_speedup_vs_row_major": (
                        statistics.median(paired) if paired else None
                    ),
                    "paired_win_rate_vs_row_major": (
                        sum(value > 1.0 for value in paired) / len(paired)
                        if paired else None
                    ),
                    "correct": correct,
                    **metrics,
                })
        finally:
            for *_, device, _metrics, _correct in devices:
                device.close()

    pmc = _pmc_capability()
    return {
        "schema": "tessera.rocm.raster.gfx1151.v1",
        "sync_key": "RASTER-CONTRACT-2026-07-28",
        "architecture": "gfx1151",
        "method": {
            "timing": "resident buffers, HIP events",
            "ordering": "round-robin rotated interleaving",
            "trials": trials,
            "iterations_per_trial": iterations,
            "warmup_launches": warmup,
            "promotion_gate": (
                "correct, >=3% paired-median speedup, >=75% paired win rate, "
                "and exact-device MALL/L2 counter support"
            ),
            "device_event_timing_available": timing_available,
            "device_event_timing_error": timing_error,
        },
        "pmc": pmc,
        "rows": rows,
        "all_correct": all_correct,
        "promotion": (
            "eligible_for_review"
            if all_correct and pmc["supported"] and timing_available
            else "blocked_retain_row_major"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        trials=args.trials, iterations=args.iterations,
        warmup=args.warmup, quick=args.quick)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "all_correct": result["all_correct"],
        "pmc_supported": result["pmc"]["supported"],
        "promotion": result["promotion"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
