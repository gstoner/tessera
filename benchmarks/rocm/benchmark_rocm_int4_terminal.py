#!/usr/bin/env python3
"""Operation-total gfx1151 packet for terminal packed INT4 execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import numpy as np

from tessera import runtime as rt


def _packed_wmma_artifact(shape: tuple[int, int, int]) -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(metadata={
        "target": "rocm",
        "compiler_path": "rocm_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": ["a", "b"],
        "output_name": "c",
        "wmma_dtype": "int4",
        "wmma_inputs_packed": True,
        "logical_mnk": shape,
        "ops": [{
            "op_name": "tessera.matmul",
            "result": "c",
            "operands": ["a", "b"],
            "kwargs": {},
        }],
    })


def _measure(
    call: Callable[[], dict[str, Any]], warmup: int, reps: int
) -> tuple[float, float, dict[str, Any]]:
    samples: list[float] = []
    result: dict[str, Any] = {}
    for iteration in range(warmup + reps):
        start = time.perf_counter_ns()
        result = call()
        elapsed = (time.perf_counter_ns() - start) / 1.0e6
        if iteration >= warmup:
            samples.append(elapsed)
    ordered = sorted(samples)
    return (
        statistics.median(samples),
        ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        result,
    )


def record(warmup: int = 5, reps: int = 30) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20260723)
    for shape in ((16, 16, 16), (33, 17, 31), (128, 128, 128)):
        m, n, k = shape
        a = rng.integers(-8, 8, size=(m, k), dtype=np.int8)
        b = rng.integers(-8, 8, size=(k, n), dtype=np.int8)
        a_packed = rt._rocm_int4_storage_convert(a, a.size, "pack", np)
        b_packed = rt._rocm_int4_storage_convert(b, b.size, "pack", np)
        artifact = _packed_wmma_artifact(shape)
        median, p95, result = _measure(
            lambda: rt.launch(artifact, (a_packed, b_packed)), warmup, reps
        )
        if not result or not result.get("ok"):
            raise RuntimeError(str(result))
        reference = a.astype(np.int32) @ b.astype(np.int32)
        exact = bool(np.array_equal(result["output"], reference))
        if not exact:
            raise RuntimeError(f"packed INT4 WMMA mismatch for {shape}")
        rows.append({
            "shape": list(shape),
            "logical_input_bytes": int(a.nbytes + b.nbytes),
            "packed_input_bytes": int(a_packed.nbytes + b_packed.nbytes),
            "compression_ratio": (
                (a.nbytes + b.nbytes) /
                (a_packed.nbytes + b_packed.nbytes)
            ),
            "pack_route": "compiled_gfx1151",
            "consumer_route": "compiled_rocdl_wmma_iu4_packed_memory",
            "host_repack_in_consumer": False,
            "exact_integer_result": exact,
            "timing_domain": "host_wall_operation_total_packed_consumer",
            "median_ms": median,
            "p95_ms": p95,
        })
    return {
        "schema": "tessera.rocm.int4-terminal.v1",
        "target": "gfx1151",
        "signedness": "signed_twos_complement",
        "nibble_order": "low_logical_index_in_low_nibble",
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = record(args.warmup, args.reps)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
