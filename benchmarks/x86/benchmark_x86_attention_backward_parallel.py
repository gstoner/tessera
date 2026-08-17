#!/usr/bin/env python3
"""Compare deterministic parallel and retained serial AVX-512 attention VJPs."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np


_PTR = ctypes.POINTER(ctypes.c_float)
_ARGS = [_PTR] * 6 + [ctypes.c_int64] * 7 + [ctypes.c_float, ctypes.c_int,
                                             ctypes.c_int64, ctypes.c_float] + [_PTR] * 3


def _symbol(path: Path):
    library = ctypes.CDLL(str(path))
    function = library.tessera_x86_flash_attn_bwd_f32
    function.argtypes = _ARGS
    function.restype = None
    return library, function


def _ptr(array: np.ndarray | None):
    return None if array is None else array.ctypes.data_as(_PTR)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(path: Path, arrays: tuple[np.ndarray, ...], shape: tuple[int, ...],
         *, warmup: int, iterations: int) -> tuple[dict[str, object], tuple[np.ndarray, ...]]:
    library, function = _symbol(path)
    _ = library  # Keep the image live for the duration of the samples.
    dout, q, k, v = arrays
    b, hq, hkv, sq, sk, d, dv = shape
    outputs = (
        np.empty_like(q), np.empty_like(k), np.empty_like(v)
    )

    def launch() -> None:
        function(
            _ptr(dout), _ptr(q), _ptr(k), _ptr(v), None, None,
            b, hq, hkv, sq, sk, d, dv, np.float32(1.0 / np.sqrt(d)),
            1, 0, np.float32(0.0), *(_ptr(value) for value in outputs),
        )

    for _ in range(warmup):
        launch()
    samples: list[int] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        launch()
        samples.append(time.perf_counter_ns() - start)
    return ({
        "artifact_sha256": _digest(path),
        "samples_ns": samples,
        "median_ns": int(statistics.median(samples)),
        "minimum_ns": min(samples),
        "timing_domain": "synchronous_host_steady_clock",
    }, outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-lib", type=Path)
    parser.add_argument("--baseline-lib", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    build = Path(os.environ.get("TESSERA_BUILD_DIR", "build"))
    candidate = args.candidate_lib or (
        build / "src/compiler/codegen/tessera_x86_backend/libtessera_x86_elementwise.so"
    )
    shape = (2, 8, 2, 128, 128, 64, 64)
    rng = np.random.default_rng(0x501)
    arrays = tuple(
        (rng.standard_normal(extents) * 0.2).astype(np.float32)
        for extents in (
            (shape[0], shape[1], shape[3], shape[6]),
            (shape[0], shape[1], shape[3], shape[5]),
            (shape[0], shape[2], shape[4], shape[5]),
            (shape[0], shape[2], shape[4], shape[6]),
        )
    )
    candidate_row, candidate_outputs = _run(
        candidate, arrays, shape, warmup=args.warmup, iterations=args.iterations
    )
    packet: dict[str, object] = {
        "schema": "tessera.x86.attention_backward.parallel.v1",
        "architecture": "zen5_avx512",
        "host": platform.platform(),
        "processor": platform.processor(),
        "shape": list(shape),
        "policy": "contiguous_query_rows_private_dkdv_fixed_worker_reduction",
        "workspace_budget_bytes": 256 << 20,
        "candidate": candidate_row,
        "selector_eligible": False,
        "selector_ineligible_reason": "WSL host-wall comparison; clean bare-metal packet required",
    }
    if args.baseline_lib:
        baseline_row, baseline_outputs = _run(
            args.baseline_lib, arrays, shape,
            warmup=args.warmup, iterations=args.iterations,
        )
        maximum_error = max(
            float(np.max(np.abs(lhs - rhs)))
            for lhs, rhs in zip(candidate_outputs, baseline_outputs)
        )
        packet["baseline"] = baseline_row
        packet["speedup_median"] = (
            int(baseline_row["median_ns"]) / int(candidate_row["median_ns"])
        )
        packet["maximum_absolute_difference"] = maximum_error
    encoded = json.dumps(packet, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
