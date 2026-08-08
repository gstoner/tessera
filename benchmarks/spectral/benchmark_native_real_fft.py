#!/usr/bin/env python3
"""Compare native packed-real FFT packages with their legacy full-C2C route."""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import statistics
import time
from typing import Any, Callable

import numpy as np


def _samples(fn: Callable[[], Any], iterations: int) -> tuple[list[int], Any]:
    result = fn()
    values: list[int] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        result = fn()
        values.append(time.perf_counter_ns() - start)
    return values, result


def _row(target: str, n: int, batch: int, iterations: int) -> dict[str, Any]:
    from tessera import runtime as rt
    from tessera.compiler.scheduled_fft import lower_scheduled_fft

    rng = np.random.default_rng(1901 + n + batch)
    real = rng.standard_normal((batch, n)).astype(np.float32)
    artifact = lower_scheduled_fft(
        target=target, op_name="tessera.rfft", input_shape=real.shape
    )
    contract = artifact.to_metadata()
    digest = artifact.schedule_digest

    if target == "x86":
        lib = rt._load_x86_elementwise()
        if lib is None:
            raise RuntimeError("x86 native package is unavailable")
        packed = np.empty((batch, n // 2 + 1), np.complex64)

        def native() -> np.ndarray:
            rc = lib.tessera_x86_fft_r2c_packed_f32(
                digest.encode("ascii"),
                real.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                packed.view(np.float32).ctypes.data_as(
                    ctypes.POINTER(ctypes.c_float)
                ),
                batch,
                n,
            )
            if rc:
                raise RuntimeError(f"x86 packed RFFT failed rc={rc}")
            return packed

        strategy = "radix2" if n & (n - 1) == 0 else str(contract["strategy"])

        def legacy() -> np.ndarray:
            full = rt._x86_transform_rows(real.astype(np.complex64), False, strategy, np)
            return full[:, : n // 2 + 1]

    else:
        from tessera.compiler.emit.spectral_candidates import (
            run_rocm_packed_real_rows,
            run_rocm_stockham_rows,
        )

        def native() -> np.ndarray:
            return run_rocm_packed_real_rows(
                real, inverse=False, logical_n=n, artifact_digest=digest
            )

        def legacy() -> np.ndarray:
            full = run_rocm_stockham_rows(
                real.astype(np.complex64), inverse=False, artifact_digest=digest
            )
            return full[:, : n // 2 + 1]

    for _ in range(3):
        native()
        legacy()
    native_ns, actual = _samples(native, iterations)
    legacy_ns, old = _samples(legacy, iterations)
    reference = np.fft.rfft(real, axis=-1).astype(np.complex64)
    return {
        "operation": "rfft",
        "target": target,
        "n": n,
        "batch": batch,
        "artifact_digest": digest,
        "tile_digest": artifact.tile_digest,
        "real_transform_policy": contract["real_transform_policy"],
        "logical_length": contract["length"],
        "physical_length": contract["physical_length"],
        "strategy": contract["strategy"],
        "native_samples_ns": native_ns,
        "legacy_samples_ns": legacy_ns,
        "native_median_ns": int(statistics.median(native_ns)),
        "legacy_median_ns": int(statistics.median(legacy_ns)),
        "speedup_vs_full_complex": statistics.median(legacy_ns)
        / statistics.median(native_ns),
        "native_max_abs_error": float(np.max(np.abs(actual - reference))),
        "legacy_max_abs_error": float(np.max(np.abs(old - reference))),
    }


def _inverse_row(target: str, n: int, batch: int, iterations: int) -> dict[str, Any]:
    from tessera import runtime as rt
    from tessera.compiler.scheduled_fft import lower_scheduled_fft

    rng = np.random.default_rng(2901 + n + batch)
    original = rng.standard_normal((batch, n)).astype(np.float32)
    spectrum = np.fft.rfft(original, axis=-1).astype(np.complex64)
    artifact = lower_scheduled_fft(
        target=target,
        op_name="tessera.irfft",
        input_shape=spectrum.shape,
        n=n,
    )
    contract = artifact.to_metadata()
    digest = artifact.schedule_digest
    if target == "x86":
        lib = rt._load_x86_elementwise()
        if lib is None:
            raise RuntimeError("x86 native package is unavailable")
        packed = np.empty((batch, n), np.float32)

        def native() -> np.ndarray:
            rc = lib.tessera_x86_fft_c2r_packed_f32(
                digest.encode("ascii"),
                spectrum.view(np.float32).ctypes.data_as(
                    ctypes.POINTER(ctypes.c_float)
                ),
                packed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch,
                n,
            )
            if rc:
                raise RuntimeError(f"x86 packed IRFFT failed rc={rc}")
            return packed

        def legacy() -> np.ndarray:
            full = np.zeros((batch, n), np.complex64)
            full[:, : n // 2 + 1] = spectrum
            full[:, n // 2 + 1 :] = np.conj(spectrum[:, 1 : n // 2])[:, ::-1]
            return (rt._x86_transform_rows(full, True, "radix2", np).real / n).astype(
                np.float32
            )

    else:
        from tessera.compiler.emit.spectral_candidates import (
            run_rocm_packed_real_rows,
            run_rocm_stockham_rows,
        )

        def native() -> np.ndarray:
            return run_rocm_packed_real_rows(
                spectrum, inverse=True, logical_n=n, artifact_digest=digest
            )

        def legacy() -> np.ndarray:
            full = np.zeros((batch, n), np.complex64)
            full[:, : n // 2 + 1] = spectrum
            full[:, n // 2 + 1 :] = np.conj(spectrum[:, 1 : n // 2])[:, ::-1]
            return run_rocm_stockham_rows(
                full, inverse=True, artifact_digest=digest
            ).real.astype(np.float32)

    for _ in range(3):
        native()
        legacy()
    native_ns, actual = _samples(native, iterations)
    legacy_ns, old = _samples(legacy, iterations)
    return {
        "operation": "irfft",
        "target": target,
        "n": n,
        "batch": batch,
        "artifact_digest": digest,
        "tile_digest": artifact.tile_digest,
        "real_transform_policy": contract["real_transform_policy"],
        "logical_length": contract["length"],
        "physical_length": contract["physical_length"],
        "strategy": contract["strategy"],
        "native_samples_ns": native_ns,
        "legacy_samples_ns": legacy_ns,
        "native_median_ns": int(statistics.median(native_ns)),
        "legacy_median_ns": int(statistics.median(legacy_ns)),
        "speedup_vs_full_complex": statistics.median(legacy_ns)
        / statistics.median(native_ns),
        "native_max_abs_error": float(np.max(np.abs(actual - original))),
        "legacy_max_abs_error": float(np.max(np.abs(old - original))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("x86", "rocm"), required=True)
    parser.add_argument("--lengths", default="64,256,1024")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    lengths = [int(value) for value in args.lengths.split(",")]
    if any(length < 2 or length % 2 for length in lengths):
        parser.error("packed-real comparison requires positive even lengths")
    rows = [
        row
        for n in lengths
        for row in (
            _row(args.target, n, args.batch, args.iterations),
            _inverse_row(args.target, n, args.batch, args.iterations),
        )
    ]
    print(
        json.dumps(
            {
                "schema": "tessera.native_real_fft_benchmark.v1",
                "host": platform.platform(),
                "processor": platform.processor(),
                "timing_domain": "synchronized_host_wall",
                "warm_state": True,
                "selector_eligible": False,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
