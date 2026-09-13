"""Fixed-count, fresh-process staged GEMM comparison; diagnostic, not admission."""

import argparse
import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
from tests._support.rocm_build import rocm_gemm_lib_path


def worker():
    agents = subprocess.check_output(["rocminfo"], text=True)
    if not re.search(r"Name:\s+gfx1201\b", agents):
        raise RuntimeError("requires owning gfx1201 device")
    path = rocm_gemm_lib_path()
    if path is None:
        raise RuntimeError("build shipped ROCm GEMM runtime")
    lib = ct.CDLL(str(path))
    timer = lib.tessera_rocm_bench_last_timer_source
    timer.argtypes, timer.restype = [], ct.c_int
    rows = []
    for shape in [(128, 128, 128), (511, 513, 509), (1024, 1024, 1024)]:
        rng = np.random.default_rng(731)
        m, n, k = shape
        a = (rng.standard_normal((m, k)) * 0.25).astype(np.float16)
        b = (rng.standard_normal((k, n)) * 0.25).astype(np.float16)
        expected = a.astype(np.float32) @ b.astype(np.float32)
        for variant in ("register", "lds", "pipe"):
            staged = variant != "register"
            suffix = "_" + variant if staged else ""
            fn = getattr(lib, "tessera_rocm_wmma_gemm_f16" + suffix)
            fn.argtypes = [ct.c_void_p] * 3 + [ct.c_int] * (7 if staged else 3)
            fn.restype = ct.c_int
            output = np.empty((m, n), np.float32)
            extra = [2, 2, 2, 4] if staged else []
            started = time.perf_counter_ns()
            rc = fn(a.ctypes.data, b.ctypes.data, output.ctypes.data, *shape, *extra)
            cold_api_ms = (time.perf_counter_ns() - started) / 1e6
            if rc:
                raise RuntimeError(f"{variant} correctness launch failed: {rc}")
            error = float(np.max(np.abs(output - expected)))
            # Same absolute error budget across candidates; nonfinite refuses.
            if not math.isfinite(error) or error > 0.01:
                raise RuntimeError(f"{variant} failed numerical budget: {error}")
            bench = getattr(lib, "tessera_rocm_wmma_gemm_f16_bench" + suffix)
            bench.argtypes = [ct.c_int] * (8 if staged else 6) + [ct.POINTER(ct.c_double)]
            bench.restype = ct.c_int
            value = ct.c_double()
            rc = bench(*shape, 100, *([2, 2, 2, 4] if staged else [1, 1]), ct.byref(value))
            source = timer()
            if rc or source not in (0, 1) or not math.isfinite(value.value) or value.value <= 0:
                raise RuntimeError(f"{variant} timing failed: {rc}, {source}, {value.value}")
            rows.append(
                dict(
                    shape=shape,
                    variant=variant,
                    max_abs_error=error,
                    cold_api_ms=cold_api_ms,
                    per_launch_ms=value.value,
                    clock="hip_event_checked_against_wall" if source == 0 else "host_wall_launch_and_sync",
                    iterations=100,
                )
            )
    return dict(pid=os.getpid(), runtime_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=rows)


def record(output):
    runs = []
    with tempfile.TemporaryDirectory() as directory:
        for index in range(5):
            child = Path(directory) / f"{index}.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    __name__ if __name__ != "__main__" else "benchmarks.rocm.record_gfx1201_staged",
                    "--worker",
                    "--output",
                    str(child),
                ],
                check=True,
            )
            runs.append(json.loads(child.read_text()))
    if len({run["runtime_sha256"] for run in runs}) != 1:
        raise RuntimeError("runtime image changed between measurement processes")
    summary = []
    for index, row in enumerate(runs[0]["rows"]):
        matches = [run["rows"][index] for run in runs]
        if any((r["shape"], r["variant"], r["clock"]) != (row["shape"], row["variant"], row["clock"]) for r in matches):
            raise RuntimeError("cross-process measurement modalities differ")
        summary.append(
            dict(
                shape=row["shape"],
                variant=row["variant"],
                clock=row["clock"],
                median_ms=statistics.median(r["per_launch_ms"] for r in matches),
                run_ms=[r["per_launch_ms"] for r in matches],
            )
        )
    output.write_text(
        json.dumps(
            dict(
                schema=1,
                chip="gfx1201",
                os_release=platform.release(),
                runs=runs,
                summary=summary,
                promotion_eligible=False,
                missing_gates=[
                    "exact HSACO and profiler kernel attribution",
                    "independent kernel-clock validation",
                    "clean native-Linux promotion run",
                    "production package candidate binding",
                ],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        args.output.write_text(json.dumps(worker()) + "\n")
    else:
        record(args.output)
