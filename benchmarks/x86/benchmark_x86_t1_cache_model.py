#!/usr/bin/env python3
"""Calibrate the T1 CPU cache-hierarchy model against tiled AVX-512 GEMM."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from tessera.compiler.autotune_v2 import GEMMWorkload, TuningConfig  # noqa: E402
from tessera.compiler.reuse_distance_cost import (  # noqa: E402
    CacheLevel,
    estimate_gemm_reuse_distance_hierarchy,
)

SCHEMA = "tessera.x86.t1_cache_model.v1"
WORKLOADS = ((256, 256, 256), (384, 512, 256), (511, 383, 257))  # M,N,K
CONFIGS = (
    (1, 64, 64),
    (4, 64, 64),
    (8, 64, 64),
    (16, 32, 32),
    (16, 64, 64),
    (16, 128, 32),
    (16, 128, 64),
    (32, 64, 64),
    (32, 128, 64),
    (64, 128, 64),
)


def _cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.partition(":")[2].strip()
    return platform.processor() or "unknown"


def _parse_size(text: str) -> int:
    text = text.strip().upper()
    scale = 1
    if text.endswith("K"):
        scale, text = 1024, text[:-1]
    elif text.endswith("M"):
        scale, text = 1024 * 1024, text[:-1]
    return int(text) * scale


def _cache_capacities(cpu: int) -> list[tuple[str, int]]:
    root = Path(f"/sys/devices/system/cpu/cpu{cpu}/cache")
    result: dict[int, tuple[str, int]] = {}
    for index in root.glob("index*"):
        if (index / "type").read_text().strip() == "Instruction":
            continue
        level = int((index / "level").read_text())
        size = _parse_size((index / "size").read_text())
        result[level] = (f"L{level}" if level > 1 else "L1D", size)
    return [result[level] for level in sorted(result)]


def _copy_bandwidth(size: int, *, trials: int = 9) -> tuple[float, list[float]]:
    # Count both the read and write side of the link. Repetition amortizes the
    # Python call boundary while keeping the selected working set resident.
    elements = max(1024, size // 8)
    source = np.arange(elements, dtype=np.float32)
    destination = np.empty_like(source)
    repeats = max(4, math.ceil((64 * 1024 * 1024) / source.nbytes))
    samples = []
    for _ in range(trials):
        start = time.perf_counter_ns()
        for _ in range(repeats):
            np.copyto(destination, source)
        elapsed = time.perf_counter_ns() - start
        samples.append((2.0 * source.nbytes * repeats) / elapsed)
    return statistics.median(samples), samples  # bytes/ns == GB/s


def _max_frequency_hz(cpu: int) -> tuple[float, str]:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq")
    if path.is_file():
        return float(path.read_text().strip()) * 1000.0, str(path)
    mhz = []
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("cpu MHz"):
            mhz.append(float(line.partition(":")[2]))
    if not mhz:
        raise RuntimeError("no CPU frequency evidence is available")
    return max(mhz) * 1e6, "/proc/cpuinfo cpu MHz maximum"


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + end - 1) / 2.0
        for index in order[cursor:end]:
            ranks[index] = average
        cursor = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    x, y = _rank(left), _rank(right)
    x_mean, y_mean = statistics.mean(x), statistics.mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y)
    )
    return numerator / denominator if denominator else 0.0


def run(*, library_path: Path, trials: int) -> dict[str, Any]:
    if trials < 7:
        raise ValueError("T1 calibration requires at least seven trials")
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [0]
    cpu = allowed[0]
    original_affinity = set(allowed)
    affinity_status = "unsupported"
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {cpu})
            affinity_status = f"pinned_cpu_{cpu}"
        except OSError as error:
            affinity_status = f"denied:{error.errno}"
    try:
        capacities = _cache_capacities(cpu)
        level_records = []
        levels = []
        for name, capacity in capacities:
            bandwidth, samples = _copy_bandwidth(max(4096, capacity // 2))
            levels.append(CacheLevel(name, capacity, bandwidth))
            level_records.append(
                {
                    "name": name,
                    "capacity_bytes": capacity,
                    "bandwidth_gbps": bandwidth,
                    "bandwidth_samples_gbps": samples,
                    "source": f"sysfs capacity + resident np.copyto on cpu {cpu}",
                }
            )
        dram_size = max(64 * 1024 * 1024, capacities[-1][1] * 2)
        dram_bw, dram_samples = _copy_bandwidth(dram_size)
        frequency_hz, frequency_source = _max_frequency_hz(cpu)
        peak_tflops = frequency_hz * 32.0 / 1e12

        library = ctypes.CDLL(str(library_path))
        function = library.tessera_x86_avx512_gemm_f32_tiled
        pointer = ctypes.POINTER(ctypes.c_float)
        function.argtypes = [
            pointer, pointer, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            pointer, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ]
        function.restype = ctypes.c_int
        rng = np.random.default_rng(0x71CA)
        workload_rows: list[dict[str, Any]] = []
        for m, n, k in WORKLOADS:
            a = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=np.float32)
            b = np.ascontiguousarray(rng.standard_normal((k, n)), dtype=np.float32)
            expected = a @ b
            candidate_rows: list[dict[str, Any]] = []
            calls = []
            for bm, bn, bk in CONFIGS:
                output = np.empty((m, n), dtype=np.float32)

                def call(
                    output=output, bm=bm, bn=bn, bk=bk,
                ) -> None:
                    status = function(
                        a.ctypes.data_as(pointer), b.ctypes.data_as(pointer),
                        m, n, k, output.ctypes.data_as(pointer), bm, bn, bk,
                    )
                    if status != 0:
                        raise RuntimeError(f"tiled GEMM declined {(bm, bn, bk)}")

                call()
                maximum_error = float(np.max(np.abs(output - expected)))
                if maximum_error > 3e-4:
                    raise RuntimeError(
                        f"tiled GEMM {(bm, bn, bk)} error {maximum_error}"
                    )
                config = TuningConfig(bm, bn, bk)
                estimate = estimate_gemm_reuse_distance_hierarchy(
                    GEMMWorkload(m, n, k, dtype="fp32", arch="zen5-avx512"),
                    config,
                    peak_tflops=peak_tflops,
                    dram_bw_gbps=dram_bw,
                    cache_levels=tuple(levels),
                )
                row: dict[str, Any] = {
                    "config": config.to_dict(),
                    "correct": True,
                    "max_abs_error": maximum_error,
                    "samples_ns": [],
                    "predicted_ms": estimate.latency_ms,
                    "predicted_compute_ms": estimate.compute_ms,
                    "predicted_memory_ms": estimate.memory_ms,
                    "predicted_dram_bytes": estimate.dram_bytes,
                    "level_hit_rates": {
                        level.name: level.hit_rate for level in estimate.levels
                    },
                    "model_exact": estimate.exact,
                }
                candidate_rows.append(row)
                calls.append(call)
            for call in calls:
                call()
            for trial in range(trials):
                indices = range(len(calls)) if trial % 2 == 0 else range(len(calls) - 1, -1, -1)
                for index in indices:
                    start = time.perf_counter_ns()
                    calls[index]()
                    candidate_rows[index]["samples_ns"].append(
                        time.perf_counter_ns() - start
                    )
            for row in candidate_rows:
                row["median_ns"] = int(statistics.median(row["samples_ns"]))
            measured = [float(row["median_ns"]) for row in candidate_rows]
            predicted = [float(row["predicted_ms"]) for row in candidate_rows]
            measured_winner = min(range(len(measured)), key=measured.__getitem__)
            predicted_winner = min(range(len(predicted)), key=predicted.__getitem__)
            workload_rows.append(
                {
                    "shape_mnk": [m, n, k],
                    "candidates": candidate_rows,
                    "spearman_rho": _spearman(predicted, measured),
                    "winner_match": measured_winner == predicted_winner,
                    "measured_winner": candidate_rows[measured_winner]["config"],
                    "predicted_winner": candidate_rows[predicted_winner]["config"],
                }
            )
        correlations = [row["spearman_rho"] for row in workload_rows]
        winner_matches = sum(bool(row["winner_match"]) for row in workload_rows)
        retain = statistics.median(correlations) > 0.25 and winner_matches >= 1
        return {
            "schema": SCHEMA,
            "work_item": "T1-X86-CACHE-MODEL",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "architecture": "zen5-avx512",
            "device": _cpu_model(),
            "host": platform.platform(),
            "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
            "timing_domain": "steady_host_wall_ns",
            "affinity": affinity_status,
            "trials": trials,
            "hardware_inputs": {
                "cache_levels": level_records,
                "dram_bandwidth_gbps": dram_bw,
                "dram_bandwidth_samples_gbps": dram_samples,
                "dram_bandwidth_source": f"np.copyto working set {dram_size} bytes",
                "single_core_peak_tflops": peak_tflops,
                "peak_source": f"{frequency_source} * 16 f32 lanes * FMA2",
                "frequency_hz": frequency_hz,
            },
            "rows": workload_rows,
            "median_spearman_rho": statistics.median(correlations),
            "winner_matches": winner_matches,
            "winner_trials": len(workload_rows),
            "promotion_threshold": "median rho > 0.25 and at least one exact winner",
            "verdict": "retain_for_pruning" if retain else "reject_for_ranking",
            "selector_changed": False,
        }
    finally:
        if hasattr(os, "sched_setaffinity"):
            try:
                os.sched_setaffinity(0, original_affinity)
            except OSError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.library.is_file():
        raise SystemExit("--library must name the built x86 shared image")
    result = run(library_path=args.library, trials=args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"T1 median rho={result['median_spearman_rho']:.4f}, "
        f"winner matches={result['winner_matches']}/{result['winner_trials']}, "
        f"verdict={result['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
