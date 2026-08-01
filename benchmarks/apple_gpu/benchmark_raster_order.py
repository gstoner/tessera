"""Record matched Apple7 raster-order evidence for the Tile simdgroup GEMM.

The MSL source, tile shape, storage, threadgroup count, inputs, warmup, and
number of timed executions are identical within each pair.  Only the shared
``raster_order``/``raster_group`` block-id mapping changes.  A missing Metal
counter capability is recorded as ``false`` rather than inferred.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from tessera.compiler.apple_target import AppleGPUArch, AppleGPUTargetProfile
from tessera.compiler.msl_gemm_emit import (
    dispatch_apple_simdgroup_tile_f16,
    materialize_apple_simdgroup_tile_msl,
)


DEFAULT_SHAPES = ("256x256x256", "512x128x512", "128x512x128", "257x193x129")


def _shape(spec: str) -> tuple[int, int, int]:
    try:
        shape = tuple(int(part) for part in spec.lower().split("x"))
    except ValueError as exc:
        raise ValueError(f"shape must be MxKxN: {spec!r}") from exc
    if len(shape) != 3 or any(dimension <= 0 for dimension in shape):
        raise ValueError(f"shape must be positive MxKxN: {spec!r}")
    return shape


def _sample(a: np.ndarray, b: np.ndarray, *, order: str, group: int, reps: int) -> dict[str, Any]:
    artifact = materialize_apple_simdgroup_tile_msl(
        AppleGPUTargetProfile(AppleGPUArch.APPLE7), "fp16", 32, 32, 16,
        raster_order=order, raster_group=group,
    )
    # Compile/cache the exact source before timing its device dispatches.
    warm_output, warm_native, warm_record = dispatch_apple_simdgroup_tile_f16(
        artifact, a, b, return_provenance=True,
    )
    del warm_output
    samples: list[int] = []
    output = None
    native = warm_native
    record = warm_record
    for _ in range(reps):
        output, ran, record = dispatch_apple_simdgroup_tile_f16(
            artifact, a, b, return_provenance=True,
        )
        native &= ran
        if record.device_time_ns is not None and record.device_time_ns > 0:
            samples.append(record.device_time_ns)
    reference = a.astype(np.float32) @ b.astype(np.float32)
    return {
        "raster_order": artifact.raster_order,
        "raster_group": artifact.raster_group,
        "native_gpu": native,
        "numerically_validated": bool(
            output is not None and np.allclose(output, reference, rtol=1e-2, atol=1e-2)
        ),
        "device_time_ns_samples": samples,
        "device_time_ns_median": int(statistics.median(samples)) if len(samples) == reps else None,
        "counter_sampling_supported": record.counter_sampling_supported,
        "counter_timestamp_delta": record.counter_timestamp_delta,
        "source_sha256": record.source_sha256,
    }


def characterize(
    shapes: tuple[tuple[int, int, int], ...], *, group: int, reps: int, rounds: int,
) -> dict[str, Any]:
    rows = []
    for m, k, n in shapes:
        rng = np.random.default_rng(m * 1_000_003 + k * 1_009 + n)
        a = (rng.standard_normal((m, k), dtype=np.float32) * 0.1).astype(np.float16)
        b = (rng.standard_normal((k, n), dtype=np.float32) * 0.1).astype(np.float16)
        rows.append({
            "shape": f"{m}x{k}x{n}",
            "tile": "32x32x16",
            "storage_dtype": "fp16",
            "accumulation_dtype": "fp32",
            "baseline_runs": [
                _sample(a, b, order="row_major", group=1, reps=reps) for _ in range(rounds)
            ],
            "candidate_runs": [
                _sample(a, b, order="grouped_m", group=group, reps=reps) for _ in range(rounds)
            ],
        })
    return {
        "schema_version": 1,
        "target": "apple7",
        "timing_domain": "Metal GPU timestamp",
        "repetitions": reps,
        "independent_warm_runs": rounds,
        "paired_variable": "raster_order/raster_group only",
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=list(DEFAULT_SHAPES))
    parser.add_argument("--group", type=int, default=2)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.group < 1 or args.reps < 2 or args.rounds < 2:
        parser.error("--group must be >= 1, --reps must be >= 2, and --rounds must be >= 2")
    payload = characterize(
        tuple(_shape(spec) for spec in args.shapes), group=args.group,
        reps=args.reps, rounds=args.rounds,
    )
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
