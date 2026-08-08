#!/usr/bin/env python3
"""Same-process packed-real versus retired full-complex TSOL comparison.

The retired symbol is deliberately benchmark-only: production Schedule→Tile
artifacts cannot select it.  Keeping both policies in one native image avoids
mixing Python composition overhead or different process state into the verdict.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _git(*command: str) -> str:
    result = subprocess.run(
        ["git", *command], capture_output=True, check=False, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _cpu_identity() -> dict[str, Any]:
    fields: dict[str, str] = {}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() not in fields:
                fields[key.strip()] = value.strip()
    return {
        "vendor_id": fields.get("vendor_id", "unknown"),
        "family": fields.get("cpu family", "unknown"),
        "model": fields.get("model", "unknown"),
        "model_name": fields.get("model name", platform.processor() or "unknown"),
        "microcode": fields.get("microcode", "unknown"),
    }


def _governors() -> list[str]:
    values: set[str] = set()
    for path in Path("/sys/devices/system/cpu").glob(
        "cpu[0-9]*/cpufreq/scaling_governor"
    ):
        try:
            values.add(path.read_text(encoding="utf-8").strip())
        except OSError:
            pass
    return sorted(values)


def _time(fn: Callable[[], None], iterations: int) -> tuple[int, list[int]]:
    start = time.perf_counter_ns()
    fn()
    cold = time.perf_counter_ns() - start
    for _ in range(3):
        fn()
    samples: list[int] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - start)
    return cold, samples


def _configure(lib: Any) -> None:
    pointer = ctypes.POINTER(ctypes.c_float)
    args = [
        ctypes.c_char_p,
        pointer,
        ctypes.c_int64,
        pointer,
        ctypes.c_int64,
        pointer,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    for name in (
        "tessera_x86_spectral_conv_f32",
        "tessera_x86_spectral_conv_full_complex_comparison_f32",
    ):
        function = getattr(lib, name)
        function.restype = ctypes.c_int
        function.argtypes = args


def _row(input_n: int, kernel_n: int, batch: int, iterations: int) -> dict[str, Any]:
    from tessera import runtime as rt
    from tessera.compiler.scheduled_spectral import lower_scheduled_spectral

    lib = rt._load_x86_elementwise()
    if lib is None:
        raise RuntimeError("x86 native package is unavailable")
    if not hasattr(lib, "tessera_x86_spectral_conv_full_complex_comparison_f32"):
        raise RuntimeError("x86 package lacks the comparison-only full-complex symbol")
    _configure(lib)
    rng = np.random.default_rng(5300 + input_n + kernel_n + batch)
    x = rng.standard_normal((batch, input_n)).astype(np.float32)
    kernel = rng.standard_normal((batch, kernel_n)).astype(np.float32)
    output_n = input_n + kernel_n - 1
    fft_n = 1 << (output_n - 1).bit_length()
    packed_output = np.empty((batch, output_n), np.float32)
    full_output = np.empty_like(packed_output)
    artifact = lower_scheduled_spectral(
        target="x86",
        op_name="tessera.spectral_conv",
        input_shapes=(x.shape, kernel.shape),
    )
    digest = artifact.schedule_digest.encode("ascii")
    pointer = ctypes.POINTER(ctypes.c_float)

    def invoke(name: str, output: np.ndarray) -> None:
        rc = getattr(lib, name)(
            digest,
            x.ctypes.data_as(pointer),
            input_n,
            kernel.ctypes.data_as(pointer),
            kernel_n,
            output.ctypes.data_as(pointer),
            batch,
            fft_n,
        )
        if rc:
            raise RuntimeError(f"{name} failed rc={rc}")

    packed = lambda: invoke("tessera_x86_spectral_conv_f32", packed_output)
    full = lambda: invoke(
        "tessera_x86_spectral_conv_full_complex_comparison_f32", full_output
    )
    packed_cold, packed_samples = _time(packed, iterations)
    full_cold, full_samples = _time(full, iterations)
    reference = np.stack(
        [np.convolve(x[row], kernel[row], mode="full") for row in range(batch)]
    ).astype(np.float32)
    packed_median = int(statistics.median(packed_samples))
    full_median = int(statistics.median(full_samples))
    metadata = artifact.to_metadata()
    return {
        "operation": "spectral_conv",
        "shape": {"batch": batch, "input_n": input_n, "kernel_n": kernel_n},
        "fft_length": fft_n,
        "schedule_digest": artifact.schedule_digest,
        "tile_digest": artifact.tile_digest,
        "child_digests": [child["schedule_digest"] for child in metadata["child_ffts"]],
        "packed_policy": sorted(
            {child["real_transform_policy"] for child in metadata["child_ffts"]}
        ),
        "retired_policy": "full_complex_hermitian_comparison_only",
        "packed_cold_ns": packed_cold,
        "full_complex_cold_ns": full_cold,
        "packed_samples_ns": packed_samples,
        "full_complex_samples_ns": full_samples,
        "packed_median_ns": packed_median,
        "full_complex_median_ns": full_median,
        "packed_speedup": full_median / packed_median,
        "packed_max_abs_error": float(np.max(np.abs(packed_output - reference))),
        "full_complex_max_abs_error": float(np.max(np.abs(full_output - reference))),
        "policy_max_abs_difference": float(np.max(np.abs(packed_output - full_output))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="256x65,1024x257,4096x513")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--event-map", type=Path)
    args = parser.parse_args()
    shapes = [tuple(int(part) for part in item.split("x")) for item in args.shapes.split(",")]
    if any(len(shape) != 2 or min(shape) <= 0 for shape in shapes):
        parser.error("--shapes must contain input_n x kernel_n pairs")
    version = f"{platform.release()} {platform.version()}".lower()
    dirty = bool(_git("status", "--porcelain"))
    # `git status --porcelain` is intentionally conservative: any output means the
    # packet was not captured from an auditable clean checkout.
    affinity = sorted(os.sched_getaffinity(0))
    governors = _governors()
    is_wsl = "microsoft" in version
    ineligibility = []
    if is_wsl:
        ineligibility.append("WSL is not a bare-metal timing environment")
    if dirty:
        ineligibility.append("worktree is not clean")
    if len(affinity) != 1:
        ineligibility.append("process affinity is not pinned to one logical CPU")
    if not governors:
        ineligibility.append("CPU frequency governor is not observable")
    event_map = None
    event_map_digest = None
    if args.event_map is None:
        ineligibility.append("model-specific event map was not supplied")
    else:
        event_map_bytes = args.event_map.read_bytes()
        event_map = json.loads(event_map_bytes)
        event_map_digest = hashlib.sha256(event_map_bytes).hexdigest()
        if not bool(event_map.get("eligible_for_promotion")):
            ineligibility.append("model-specific event map is promotion-ineligible")
    payload = {
        "schema": "tessera.tsol_packed_composite_benchmark.v1",
        "target": "x86",
        "architecture": "zen5-avx512",
        "cpu": _cpu_identity(),
        "host": platform.platform(),
        "is_wsl": is_wsl,
        "affinity": affinity,
        "governors": governors,
        "event_map_digest": event_map_digest,
        "event_map_schema": event_map.get("schema") if event_map else None,
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "clean_worktree": not dirty,
        "timing": {
            "domain": "steady_clock_host_wall",
            "synchronized": True,
            "warmup_iterations": 3,
            "measured_iterations": args.iterations,
        },
        "selector_eligible": not ineligibility,
        "selector_ineligibility": ineligibility,
        "rows": [_row(a, b, args.batch, args.iterations) for a, b in shapes],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
