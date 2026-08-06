#!/usr/bin/env python3
"""Measure named x86 FFT candidates without mutating the production selector.

Prime lengths compare cached Bluestein with Rader. Power-of-two large lengths
compare the production in-place DIT lane with a Bailey six-step decomposition.
Every row is emitted only after comparison with scipy.fft succeeds.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable

import numpy as np
from scipy import fft as scipy_fft

from tessera import runtime as rt
from tessera.compiler.spectral_plan import mixed_radix_sequence, next_power_of_two


def _sizes(value: str) -> tuple[int, ...]:
    result = tuple(int(part) for part in value.split(",") if part)
    if not result or any(size <= 1 for size in result):
        raise argparse.ArgumentTypeError("sizes must be integers greater than one")
    return result


def _median_ms(call: Callable[[], object], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        begin = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - begin) * 1.0e-6)
    return float(statistics.median(samples))


def _cold_ms(call: Callable[[], object]) -> float:
    begin = time.perf_counter_ns()
    call()
    return (time.perf_counter_ns() - begin) * 1.0e-6


def _check(name: str, actual: np.ndarray, expected: np.ndarray, n: int) -> float:
    error = float(np.max(np.abs(actual - expected)))
    tolerance = 3.0e-4 * max(1.0, float(np.max(np.abs(expected))))
    if not math.isfinite(error) or error > tolerance:
        raise RuntimeError(
            f"{name} numerical comparison failed at N={n}: "
            f"max_abs_error={error} tolerance={tolerance}"
        )
    return error


def _bluestein_uncached(rows: np.ndarray) -> np.ndarray:
    """Pre-cache production shape retained as the comparison baseline."""
    n = int(rows.shape[1])
    batch = int(rows.shape[0])
    m = next_power_of_two(2 * n - 1)
    j = np.arange(n, dtype=np.int64)
    phase_index = (j * j) % (2 * n)
    chirp = np.exp(-1j * np.pi * phase_index.astype(np.float64) / n).astype(
        np.complex64
    )
    padded = np.zeros((batch, m), np.complex64)
    padded[:, :n] = rows * chirp[None, :]
    kernel = np.zeros(m, np.complex64)
    conjugate = np.conj(chirp)
    kernel[:n] = conjugate
    if n > 1:
        kernel[m - n + 1 :] = conjugate[1:][::-1]
    transformed = rt._x86_fft_c2c_rows(padded, False, np)
    kernel_fft = rt._x86_fft_c2c_rows(kernel[None, :], False, np)[0]
    convolved = rt._x86_fft_c2c_rows(
        transformed * kernel_fft[None, :], True, np
    ) / np.float32(m)
    return (convolved[:, :n] * chirp[None, :]).astype(np.complex64)


def _prime_rows(
    sizes: tuple[int, ...], batch: int, warmup: int, repeats: int, rng: np.random.Generator
) -> None:
    for n in sizes:
        if not rt._is_prime(n):
            raise ValueError(f"Rader comparison requires prime sizes; got {n}")
        values = (
            rng.standard_normal((batch, n)) + 1j * rng.standard_normal((batch, n))
        ).astype(np.complex64)
        expected = scipy_fft.fft(values, axis=-1, workers=1).astype(np.complex64)
        for algorithm, plan_cache, call in (
            (
                "bluestein_uncached_baseline",
                None,
                lambda: _bluestein_uncached(values),
            ),
            (
                "bluestein_cached",
                rt._x86_bluestein_plan,
                lambda: rt._x86_bluestein_rows(values, False, np),
            ),
            (
                "rader_candidate",
                rt._x86_rader_plan,
                lambda: rt._x86_rader_rows(values, False, np),
            ),
        ):
            if plan_cache is not None:
                plan_cache.cache_clear()
            cold = _cold_ms(call)
            actual = call()
            error = _check(algorithm, actual, expected, n)
            warm = _median_ms(call, warmup, repeats)
            convolution_n = (
                next_power_of_two(2 * n - 1)
                if algorithm != "rader_candidate"
                else next_power_of_two(2 * (n - 1) - 1)
            )
            print(
                json.dumps(
                    {
                        "schema": "tessera.fft_algorithm_comparison.v1",
                        "target": "x86_avx512",
                        "family": "prime_fft",
                        "algorithm": algorithm,
                        "selector_state": (
                            "candidate_only"
                            if algorithm == "rader_candidate"
                            else "historical_baseline"
                            if algorithm == "bluestein_uncached_baseline"
                            else "production"
                        ),
                        "n": n,
                        "batch": batch,
                        "convolution_n": convolution_n,
                        "workspace_bytes": batch * convolution_n * 8,
                        "cold_ms": cold,
                        "warm_median_ms": warm,
                        "max_abs_error": error,
                        "timing_domain": "host_wall",
                    },
                    sort_keys=True,
                )
            )


def _large_rows(
    sizes: tuple[int, ...], batch: int, warmup: int, repeats: int, rng: np.random.Generator
) -> None:
    for n in sizes:
        if n & (n - 1):
            raise ValueError(f"six-step comparison requires powers of two; got {n}")
        values = (
            rng.standard_normal((batch, n)) + 1j * rng.standard_normal((batch, n))
        ).astype(np.complex64)
        expected = scipy_fft.fft(values, axis=-1, workers=1).astype(np.complex64)
        n1, n2, _ = rt._x86_six_step_plan(n, False)
        for algorithm, call in (
            ("production_dit", lambda: rt._x86_fft_c2c_rows(values, False, np)),
            ("bailey_six_step_candidate", lambda: rt._x86_six_step_rows(values, False, np)),
        ):
            actual = call()
            error = _check(algorithm, actual, expected, n)
            warm = _median_ms(call, warmup, repeats)
            print(
                json.dumps(
                    {
                        "schema": "tessera.fft_algorithm_comparison.v1",
                        "target": "x86_avx512",
                        "family": "large_power_of_two_fft",
                        "algorithm": algorithm,
                        "selector_state": "candidate_only"
                        if algorithm.endswith("candidate")
                        else "production",
                        "n": n,
                        "batch": batch,
                        "n1": n1,
                        "n2": n2,
                        "workspace_bytes": batch * n * 8,
                        "warm_median_ms": warm,
                        "max_abs_error": error,
                        "timing_domain": "host_wall",
                    },
                    sort_keys=True,
                )
            )


def _mixed_rows(
    sizes: tuple[int, ...], batch: int, warmup: int, repeats: int, rng: np.random.Generator
) -> None:
    for n in sizes:
        radices = mixed_radix_sequence(n)
        if radices is None or n & (n - 1) == 0:
            raise ValueError(f"mixed-radix comparison requires a composite direct plan; got {n}")
        values = (
            rng.standard_normal((batch, n)) + 1j * rng.standard_normal((batch, n))
        ).astype(np.complex64)
        expected = scipy_fft.fft(values, axis=-1, workers=1).astype(np.complex64)
        for algorithm, call, workspace in (
            (
                "bluestein_cached",
                lambda: rt._x86_bluestein_rows(values, False, np),
                batch * next_power_of_two(2 * n - 1) * 8,
            ),
            (
                "mixed_radix_avx512_candidate",
                lambda: rt._x86_mixed_radix_rows(values, False, np),
                batch * 2 * n * 8,
            ),
        ):
            actual = call()
            error = _check(algorithm, actual, expected, n)
            warm = _median_ms(call, warmup, repeats)
            print(
                json.dumps(
                    {
                        "schema": "tessera.fft_algorithm_comparison.v1",
                        "target": "x86_avx512",
                        "family": "mixed_radix_fft",
                        "algorithm": algorithm,
                        "selector_state": "candidate_only"
                        if algorithm.endswith("candidate")
                        else "production",
                        "n": n,
                        "batch": batch,
                        "radix_sequence": list(radices),
                        "workspace_bytes": workspace,
                        "warm_median_ms": warm,
                        "max_abs_error": error,
                        "timing_domain": "host_wall",
                    },
                    sort_keys=True,
                )
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prime-sizes", type=_sizes, default=_sizes("127,257,509,1009"))
    parser.add_argument(
        "--large-sizes", type=_sizes, default=_sizes("65536,262144,1048576")
    )
    parser.add_argument(
        "--mixed-sizes", type=_sizes, default=_sizes("255,768,3072,5120")
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if args.batch <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error("batch/repeats must be positive and warmup non-negative")
    if not rt._x86_elementwise_available():
        parser.error("production x86 native FFT image is unavailable")
    rng = np.random.default_rng(0xB41E7)
    _prime_rows(args.prime_sizes, args.batch, args.warmup, args.repeats, rng)
    _mixed_rows(args.mixed_sizes, args.batch, args.warmup, args.repeats, rng)
    _large_rows(args.large_sizes, args.batch, args.warmup, args.repeats, rng)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
