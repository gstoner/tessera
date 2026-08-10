#!/usr/bin/env python3
"""Record exact-host AVX-512 EGGROLL W2 correctness and timing evidence."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
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

from tessera import rng  # noqa: E402
from tessera.stdlib.es import low_rank_correction  # noqa: E402

SCHEMA = "tessera.eggroll.es_low_rank.zen5_avx512.v1"
SHAPES = ((4, 3, 8, 6), (8, 7, 257, 193), (16, 32, 1024, 1024))


def _cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.partition(":")[2].strip()
    return platform.processor() or "unknown"


def run(*, library_path: Path, trials: int) -> dict[str, Any]:
    if trials < 7:
        raise ValueError("EGGROLL evidence requires at least seven trials")
    library = ctypes.CDLL(str(library_path))
    function = library.tessera_x86_avx512_es_low_rank_correction_f32
    f32p = ctypes.POINTER(ctypes.c_float)
    i64p = ctypes.POINTER(ctypes.c_int64)
    function.argtypes = [
        f32p, i64p, i64p, f32p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_float, ctypes.c_int32,
    ]
    function.restype = ctypes.c_int
    key = rng.RNGKey(
        seed_high=0x0123456789ABCDEF,
        seed_low=0x1234567890ABCDEF,
    )
    key_storage = np.array([key.seed_high, key.seed_low], dtype=np.uint64).view(
        np.int64
    )
    generator = np.random.default_rng(0xE660)
    rows = []
    for population, row_count, in_dim, out_dim in SHAPES:
        values = generator.standard_normal(
            (population, row_count, in_dim), dtype=np.float32
        )
        members = np.arange(8, 8 + population, dtype=np.int64)
        output = np.empty((population, row_count, out_dim), dtype=np.float32)

        def native_call() -> None:
            status = function(
                values.ctypes.data_as(f32p), members.ctypes.data_as(i64p),
                key_storage.ctypes.data_as(i64p), output.ctypes.data_as(f32p),
                population, row_count, in_dim, out_dim, 3,
                ctypes.c_float(0.02), 1,
            )
            if status != 0:
                raise RuntimeError(f"AVX-512 EGGROLL call failed: {status}")

        native_call()
        expected = low_rank_correction(
            values, members, key, out_dim=out_dim, rank=1, sigma=0.02,
            epoch=3, antithetic=True,
        )
        maximum_error = float(np.max(np.abs(output - expected)))
        relative_error = float(
            np.max(np.abs(output - expected) / np.maximum(np.abs(expected), 1e-7))
        )
        if maximum_error > 3e-5:
            raise RuntimeError(f"AVX-512 EGGROLL error {maximum_error}")
        for _ in range(5):
            native_call()
        samples = []
        for _ in range(trials):
            start = time.perf_counter_ns()
            native_call()
            samples.append(time.perf_counter_ns() - start)
        rows.append(
            {
                "shape": {
                    "population": population,
                    "rows_per_member": row_count,
                    "in_dim": in_dim,
                    "out_dim": out_dim,
                    "rank": 1,
                },
                "correct": True,
                "max_abs_error": maximum_error,
                "max_rel_error": relative_error,
                "samples_ns": samples,
                "median_ns": int(statistics.median(samples)),
            }
        )
    return {
        "schema": SCHEMA,
        "work_item": "EGGROLL-W2-X86",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "zen5-avx512",
        "device": _cpu_model(),
        "host": platform.platform(),
        "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
        "algorithm": "member_factor_cache_plus_avx512_rank1_dot_outer",
        "rng": "splitmix64-philox4x32-boxmuller.v1",
        "dtype": {"storage": "fp32", "accum": "fp32"},
        "timing_domain": "steady_host_wall_ns",
        "trials": trials,
        "rows": rows,
        "all_correct": all(row["correct"] for row in rows),
        "verdict": {
            "correctness": "retain",
            "performance": "retain_avx512_f32",
            "selector_eligible": True,
            "vnni": "fail_closed_pending_quantization_scale_and_saturation_contract",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=21)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.library.is_file():
        raise SystemExit("--library must name the x86 shared image")
    result = run(library_path=args.library, trials=args.trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["rows"]:
        print(f"{row['shape']}: {row['median_ns'] / 1e6:.4f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
