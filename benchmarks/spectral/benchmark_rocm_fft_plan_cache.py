#!/usr/bin/env python3
"""Compare the legacy per-call ROCm FFT path with the persistent native plan.

The timing domain is synchronized host wall: both entry points include host to
device copies, kernel execution, synchronization, and device to host copies.
That makes the result suitable for assessing allocation/plan-cache overhead,
but not for promoting a kernel-level selector when HIP events are unavailable.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np

from tessera.compiler.emit import spectral_candidates as sc


def _cptr(array: np.ndarray) -> ctypes.c_void_p:
    return array.ctypes.data_as(ctypes.c_void_p)


def _measure(call, warmup: int, repeats: int) -> tuple[float, float]:
    start = time.perf_counter_ns()
    call()
    cold_ms = (time.perf_counter_ns() - start) * 1.0e-6
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - start) * 1.0e-6)
    return cold_ms, float(statistics.median(samples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="257,509,1009")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()
    sizes = tuple(int(value) for value in args.sizes.split(",") if value)
    if not sizes or min(sizes) <= 0 or args.batch <= 0:
        raise ValueError("sizes and batch must be positive")

    lib = sc._amd_lib()
    if lib is None:
        raise RuntimeError("prebuilt libtessera_spectral_rocm.so is unavailable")
    package_digest = hashlib.sha256(Path(str(lib._name)).read_bytes()).hexdigest()
    rng = np.random.default_rng(20260805)
    for n in sizes:
        values = (
            rng.standard_normal((args.batch, n))
            + 1j * rng.standard_normal((args.batch, n))
        ).astype(np.complex64)
        expected = np.fft.fft(values, axis=-1).astype(np.complex64)
        legacy_out = np.empty_like(values)
        cached_out = np.empty_like(values)
        digest = hashlib.sha256(f"rocm-fft-plan-cache:{n}".encode()).hexdigest()

        def legacy() -> None:
            rc = lib.ts_fft_stockham_amd_hostptr_batch(
                _cptr(values), _cptr(legacy_out), args.batch, n, -1
            )
            if rc:
                raise RuntimeError(f"legacy ROCm FFT failed rc={rc}")

        handle = ctypes.c_void_p()
        rc = lib.ts_fft_plan_create_for_artifact_amd(
            n, -1, digest.encode(), ctypes.byref(handle)
        )
        if rc or not handle.value:
            raise RuntimeError(f"persistent ROCm FFT plan creation failed rc={rc}")

        def cached() -> None:
            rc = lib.ts_fft_plan_execute_hostptr_batch_amd(
                handle, _cptr(values), _cptr(cached_out), args.batch
            )
            if rc:
                raise RuntimeError(f"persistent ROCm FFT failed rc={rc}")

        try:
            legacy_cold, legacy_warm = _measure(legacy, args.warmup, args.repeats)
            cached_cold, cached_warm = _measure(cached, args.warmup, args.repeats)
            legacy_error = float(np.max(np.abs(legacy_out - expected)))
            cached_error = float(np.max(np.abs(cached_out - expected)))
            tolerance = 3.0e-4 * max(1.0, float(np.max(np.abs(expected))))
            if max(legacy_error, cached_error) > tolerance:
                raise RuntimeError(
                    f"numerical comparison failed at N={n}: "
                    f"legacy={legacy_error} cached={cached_error} tol={tolerance}"
                )
            print(
                json.dumps(
                    {
                        "schema": "tessera.rocm_fft_plan_cache.v1",
                        "target": "rocm_gfx1151",
                        "package_abi": lib.ts_fft_package_abi_amd().decode(),
                        "package_sha256": package_digest,
                        "library": str(lib._name),
                        "n": n,
                        "batch": args.batch,
                        "artifact_digest": digest,
                        "workspace_elems": int(
                            lib.ts_fft_plan_workspace_elems_amd(handle)
                        ),
                        "legacy_cold_ms": legacy_cold,
                        "legacy_warm_median_ms": legacy_warm,
                        "cached_cold_ms": cached_cold,
                        "cached_warm_median_ms": cached_warm,
                        "warm_speedup": legacy_warm / cached_warm,
                        "legacy_max_abs_error": legacy_error,
                        "cached_max_abs_error": cached_error,
                        "timing_domain": "host_wall_synchronized",
                        "selector_eligible": False,
                    },
                    sort_keys=True,
                )
            )
        finally:
            lib.ts_fft_plan_destroy_amd(handle)


if __name__ == "__main__":
    main()
