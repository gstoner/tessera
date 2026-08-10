#!/usr/bin/env python3
"""Record the native AVX-512 mixed-radix codelet envelope on one host."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "tessera.x86.fft_mixed_codelets.v1"
# Covers the largest direct radix, a repeated radix-17 transform, and the
# composite audio/model sizes used by the existing selector packet.
LENGTHS = (68, 225, 289, 768, 3072, 5120)


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or "unknown"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _median_call_ns(call, trials: int) -> tuple[list[int], int]:
    samples: list[int] = []
    for _ in range(trials):
        start = time.perf_counter_ns()
        call()
        samples.append(time.perf_counter_ns() - start)
    return samples, int(statistics.median(samples))


def run(*, library_path: Path, trials: int, batch: int) -> dict[str, Any]:
    if trials < 7 or batch < 1:
        raise ValueError("FFT codelet evidence requires trials >= 7 and batch >= 1")
    library = ctypes.CDLL(str(library_path))
    function = library.tessera_x86_fft_mixed_c2c_f32
    pointer = ctypes.POINTER(ctypes.c_float)
    function.argtypes = [pointer, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
    function.restype = ctypes.c_int
    rng = np.random.default_rng(0xA512)
    rows = []
    for length in LENGTHS:
        source = (
            rng.standard_normal((batch, length))
            + 1j * rng.standard_normal((batch, length))
        ).astype(np.complex64)
        reference = np.fft.fft(source, axis=-1).astype(np.complex64)
        native = source.copy().view(np.float32).reshape(batch, 2 * length)
        status = function(
            native.ctypes.data_as(pointer), batch, length, 0
        )
        if status != 0:
            raise RuntimeError(f"mixed-radix ABI declined N={length}: {status}")
        maximum_error = float(np.max(np.abs(native.view(np.complex64) - reference)))
        tolerance = 6e-4 * max(1.0, float(np.max(np.abs(reference))))
        if maximum_error > tolerance:
            raise RuntimeError(
                f"mixed-radix N={length} error {maximum_error} exceeds {tolerance}"
            )

        native_work = source.copy().view(np.float32).reshape(batch, 2 * length)

        def native_call() -> None:
            native_work[:] = source.view(np.float32).reshape(batch, 2 * length)
            if function(native_work.ctypes.data_as(pointer), batch, length, 0) != 0:
                raise RuntimeError("native mixed-radix call failed during timing")

        def numpy_call() -> None:
            np.fft.fft(source, axis=-1)

        # Warm plan/twiddle/workspace caches and the reference library.
        for _ in range(5):
            native_call()
            numpy_call()
        native_samples, native_median = _median_call_ns(native_call, trials)
        numpy_samples, numpy_median = _median_call_ns(numpy_call, trials)
        rows.append(
            {
                "length": length,
                "batch": batch,
                "correct": True,
                "max_abs_error": maximum_error,
                "tolerance": tolerance,
                "native_samples_ns": native_samples,
                "numpy_samples_ns": numpy_samples,
                "native_median_ns": native_median,
                "numpy_median_ns": numpy_median,
                "native_over_numpy": native_median / numpy_median,
            }
        )
    return {
        "schema": SCHEMA,
        "work_item": "X86-FFT-MIXED-CODELETS-1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "zen5-avx512",
        "device": _cpu_model(),
        "host": platform.platform(),
        "timing_domain": "steady_host_wall_ns",
        "timing_policy": "warm serial calls; input restoration included for native",
        "library_path": str(library_path),
        "library_sha256": _digest(library_path),
        "codelet_radices": [2, 3, 4, 5, 7, 9, 11, 13, 15, 17],
        "batch": batch,
        "trials": trials,
        "rows": rows,
        "all_correct": all(row["correct"] for row in rows),
        "selector_changed": False,
        "verdict": "retain specialized codelets; selector remains work-gated",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--library",
        type=Path,
        default=Path(os.environ.get("TESSERA_X86_ELEMENTWISE_LIB", "")),
    )
    parser.add_argument("--trials", type=int, default=31)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.library.is_file():
        raise SystemExit("--library must name the built x86 shared image")
    report = run(library_path=args.library, trials=args.trials, batch=args.batch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["rows"]:
        print(
            f"N={row['length']:5d} native={row['native_median_ns'] / 1e6:.4f} ms "
            f"native/numpy={row['native_over_numpy']:.3f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
