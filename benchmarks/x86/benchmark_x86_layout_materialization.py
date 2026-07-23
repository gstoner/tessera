#!/usr/bin/env python3
"""AVX-512 operation-total row/column-major physical ABI comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np

import tessera.compiler.emit.x86_llvm  # noqa: F401
from tessera.compiler.emit.kernel_emitter import get_runner
from tessera.compiler.fusion_core import FusedRegion


def _measure(call, warmup: int, reps: int) -> tuple[float, float]:
    samples = []
    for iteration in range(warmup + reps):
        start = time.perf_counter_ns()
        call()
        elapsed = (time.perf_counter_ns() - start) / 1.0e6
        if iteration >= warmup:
            samples.append(elapsed)
    ordered = sorted(samples)
    return (
        statistics.median(samples),
        ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    )


def record(size: int = 128, warmup: int = 5, reps: int = 30) -> dict:
    rng = np.random.default_rng(20260723)
    # This is the transpose-elimination case: upstream already produced
    # Fortran-order physical storage. A row-major binding must repack it.
    a = np.asfortranarray(rng.standard_normal((size, size)).astype(np.float32))
    b = np.asfortranarray(rng.standard_normal((size, size)).astype(np.float32))
    runner = get_runner("x86")
    regions = {
        "repack_row_major": FusedRegion(
            epilogue=("relu",), a_layout="row_major", b_layout="row_major"
        ),
        "preserve_column_major": FusedRegion(
            epilogue=("relu",), a_layout="col_major", b_layout="col_major"
        ),
    }

    def invoke(region):
        value, tag = runner.run_fused_region(region, a, b)
        if tag != "x86_native":
            raise RuntimeError(f"expected x86_native, got {tag}")
        return value

    expected = regions["repack_row_major"].reference(a, b)
    for region in regions.values():
        np.testing.assert_allclose(
            invoke(region), expected, rtol=2e-4, atol=2e-4
        )
    rows: list[dict[str, Any]] = []
    for name, region in regions.items():
        median, p95 = _measure(lambda: invoke(region), warmup, reps)
        rows.append({
            "route": name,
            "input_order": "fortran",
            "abi_order": region.a_layout,
            "shape": [size, size, size],
            "median_ms": median,
            "p95_ms": p95,
            "timing_domain": "host_wall_operation_total",
            "correct": True,
        })
    row = float(rows[0]["median_ms"])
    col = float(rows[1]["median_ms"])
    return {
        "schema": "tessera.x86.layout-materialization.v1",
        "device": "Ryzen AI MAX+ 395 AVX-512",
        "rows": rows,
        "column_major_speedup_vs_repack": row / col,
        "default_policy": (
            "enable_column_major_for_existing_fortran_inputs"
            if col <= row * 1.03
            else "retain_row_major_default_column_major_opt_in"
        ),
        "packed_storage_policy": (
            "preserve_metadata_but_keep_terminal_packing_opt_in_until_"
            "a_compiled_packed_consumer_exists"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = record(args.size, args.warmup, args.reps)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
