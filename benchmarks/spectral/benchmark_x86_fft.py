#!/usr/bin/env python3
"""Compare Tessera's production AVX-512 FFT ABI with SciPy's CPU FFT.

Results use the same complex64 inputs, batch, warmup, repeat count, and
synchronized host-wall timing. One JSON row per size is emitted only after a
numerical comparison succeeds.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
from scipy import fft as scipy_fft

from tessera import runtime as rt


def _sizes(value: str) -> tuple[int, ...]:
    result = tuple(int(part) for part in value.split(","))
    if not result or any(size <= 0 or size & (size - 1) for size in result):
        raise argparse.ArgumentTypeError("sizes must be positive powers of two")
    return result


def _measure(call, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        call()
    begin = time.perf_counter()
    for _ in range(repeats):
        call()
    return (time.perf_counter() - begin) * 1000.0 / repeats


def _production_c2c(values, inverse: bool):
    """Run the production C2C ABI including its plan-owned inverse scale."""
    transformed = rt._x86_fft_c2c_rows(values, inverse, np)
    if inverse:
        transformed = transformed / np.float32(values.shape[-1])
    return transformed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        type=_sizes,
        default=_sizes("64,256,1024,4096,16384,65536"),
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if args.batch <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error("batch/repeats must be positive and warmup non-negative")
    if not rt._x86_elementwise_available():
        parser.error("production x86 native FFT image is unavailable")

    rng = np.random.default_rng(0xF17)
    transforms = (
        ("c2c_forward", False, scipy_fft.fft),
        ("c2c_inverse", True, scipy_fft.ifft),
    )
    for n in args.sizes:
        values = (
            rng.standard_normal((args.batch, n))
            + 1j * rng.standard_normal((args.batch, n))
        ).astype(np.complex64)
        for transform, inverse, scipy_call in transforms:
            expected = scipy_call(values, axis=-1, workers=1).astype(np.complex64)
            actual = _production_c2c(values, inverse)
            maximum = float(np.max(np.abs(actual - expected)))
            tolerance = 2.0e-4 * max(1.0, float(np.max(np.abs(expected))))
            if not maximum <= tolerance:
                raise RuntimeError(
                    f"x86 {transform} numerical comparison failed at N={n}: "
                    f"max_abs_error={maximum} tolerance={tolerance}"
                )

            tessera_ms = _measure(
                lambda: _production_c2c(values, inverse),
                args.warmup,
                args.repeats,
            )
            scipy_ms = _measure(
                lambda: scipy_call(values, axis=-1, workers=1),
                args.warmup,
                args.repeats,
            )
            operations = 5.0 * args.batch * n * math.log2(n)
            print(
                json.dumps(
                    {
                        "schema": "tessera.fft_comparison.v1",
                        "target": "x86_avx512",
                        "baseline": "scipy.fft_workers_1",
                        "dtype": "complex64",
                        "transform": transform,
                        "n": n,
                        "batch": args.batch,
                        "warmup": args.warmup,
                        "repeats": args.repeats,
                        "tessera_ms": tessera_ms,
                        "baseline_ms": scipy_ms,
                        "latency_ratio": tessera_ms / scipy_ms,
                        "tessera_gflops": operations / (tessera_ms * 1.0e6),
                        "baseline_gflops": operations / (scipy_ms * 1.0e6),
                        "max_abs_error": maximum,
                        "timing_domain": "host_wall",
                    },
                    sort_keys=True,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
