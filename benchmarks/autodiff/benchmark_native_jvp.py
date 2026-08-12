#!/usr/bin/env python3
"""Emit a reproducible native-JVP correctness/timing packet.

Timing is end-to-end parent-package wall time.  The packet never labels it as
device time; clean-host promotion requires an architecture-owned timing source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import time

import numpy as np

import tessera


@tessera.jit(target="x86", autodiff="jvp", wrt=("x",))
def _x86_sum(x):
    return tessera.ops.sum(x, axis=-1)


@tessera.jit(target="x86", autodiff="jvp", wrt=("x",))
def _x86_rmsnorm(x):
    return tessera.ops.rmsnorm(x, eps=1e-5)


@tessera.jit(target="x86", autodiff="jvp", wrt=("x",))
def _x86_rfft(x):
    return tessera.ops.rfft(x, axis=-1)


@tessera.jit(target="rocm", autodiff="jvp", wrt=("x",))
def _rocm_sum(x):
    return tessera.ops.sum(x, axis=-1)


@tessera.jit(target="rocm", autodiff="jvp", wrt=("x",))
def _rocm_rmsnorm(x):
    return tessera.ops.rmsnorm(x, eps=1e-5)


@tessera.jit(target="rocm", autodiff="jvp", wrt=("x",))
def _rocm_rfft(x):
    return tessera.ops.rfft(x, axis=-1)


def _references(family: str, x: np.ndarray, dx: np.ndarray):
    if family == "sum":
        return x.sum(-1), dx.sum(-1)
    if family == "rfft":
        return np.fft.rfft(x, axis=-1), np.fft.rfft(dx, axis=-1)
    inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)
    primal = x * inv
    tangent = dx * inv - x * np.mean(x * dx, axis=-1, keepdims=True) * inv**3
    return primal, tangent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("x86", "rocm"), required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    funcs = {
        "x86": {"sum": _x86_sum, "rmsnorm": _x86_rmsnorm, "rfft": _x86_rfft},
        "rocm": {"sum": _rocm_sum, "rmsnorm": _rocm_rmsnorm, "rfft": _rocm_rfft},
    }[args.target]
    rng = np.random.default_rng(20260811)
    x = rng.normal(size=(7, 64)).astype(np.float32)
    dx = rng.normal(size=x.shape).astype(np.float32)
    rows = []
    for family, compiled in funcs.items():
        cold_start = time.perf_counter_ns()
        compiled.native_jvp(x, tangents=dx)  # build + warm
        cold_ns = time.perf_counter_ns() - cold_start
        samples = []
        output = None
        for _ in range(args.iterations):
            start = time.perf_counter_ns()
            output = compiled.native_jvp(x, tangents=dx)
            samples.append(time.perf_counter_ns() - start)
        assert output is not None
        expected = _references(family, x, dx)
        rows.append({
            "family": family,
            "artifact_hash": compiled.last_jvp_execution["artifact_hash"],
            "paired_jvp_ir_digest": compiled.last_jvp_execution["paired_jvp_ir_digest"],
            "schedule_program_digest": compiled.last_jvp_execution["schedule_program_digest"],
            "tile_program_digest": compiled.last_jvp_execution["tile_program_digest"],
            "compiler_path": compiled.last_jvp_execution["compiler_path"],
            "execution_kind": compiled.last_jvp_execution["execution_kind"],
            "execution_mode": compiled.last_jvp_execution["execution_mode"],
            "max_abs_error_primal": float(np.max(np.abs(output[0] - expected[0]))),
            "max_abs_error_tangent": float(np.max(np.abs(output[1] - expected[1]))),
            "cold_parent_package_ns": cold_ns,
            "host_wall_ns_samples": samples,
        })
    is_wsl = "microsoft" in platform.release().lower()
    opt = os.environ.get("TESSERA_OPT", "")
    build_directory = os.environ.get("TESSERA_BUILD_DIR")
    opt_path = Path(opt) if opt else (
        Path(build_directory) / "tools/tessera-opt/tessera-opt"
        if build_directory else None
    )
    tool_digest = (
        hashlib.sha256(opt_path.read_bytes()).hexdigest()
        if opt_path is not None and opt_path.is_file()
        else None
    )
    cpu_model = platform.processor()
    if not cpu_model:
        try:
            cpu_model = next(
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            )
        except (OSError, StopIteration):
            cpu_model = "unknown"
    packet = {
        "schema": "tessera.native_jvp.evidence.v2",
        "target": args.target,
        "architecture": "gfx1151" if args.target == "rocm" else "zen5_avx512",
        "host": platform.platform(),
        "processor": cpu_model,
        "toolchain": {
            "tessera_opt": str(opt_path) if opt_path is not None else None,
            "tessera_opt_sha256": tool_digest,
            "build_directory": build_directory,
        },
        "timing": {
            "source": "synchronized_parent_package_host_wall",
            "provenance": "time.perf_counter_ns around completed parent launch",
            "device_clock_calibrated": False,
            "profiler_activity_available": False,
        },
        "timing_promotion_eligible": not is_wsl and args.target == "x86",
        "correctness_promotion_eligible": args.target == "rocm" or not is_wsl,
        "warm_state": True,
        "shape": [7, 64],
        "dtype": "f32",
        "rows": rows,
    }
    payload = json.dumps(packet, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
