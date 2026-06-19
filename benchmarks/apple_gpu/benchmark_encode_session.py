"""apple_gpu R2 command-buffer batching benchmark.

Compares a dependent chain of N device-resident bmms run **per-op** (R1 — one
synchronous run + one CPU↔GPU sync per op) against the same chain **batched**
into one command buffer (R2 — a single commit + wait). This is the direct
measurement of the per-op synchronization overhead the GPU-resident architecture
removes. Same JSON schema as ``benchmarks/benchmark_gemm.py``.

Shape spec: ``NxS`` (chain length × square matrix dim).

Usage:
    python benchmarks/apple_gpu/benchmark_encode_session.py \\
        --shapes 8x64 32x64 64x128 --reps 30
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tessera import runtime as R
from tessera.runtime import DeviceTensor, AppleGPUEncodeSession


def _parse(spec: str):
    a, b = spec.lower().split("x")
    return int(a), int(b)


def _time(fn, reps):
    fn()
    s = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        s.append((time.perf_counter_ns() - t0) / 1e6)
    return statistics.median(s), statistics.stdev(s) if reps > 1 else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=["8x64", "32x64", "64x128"])
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if sys.platform != "darwin" or not AppleGPUEncodeSession().available:
        if args.output is not None:
            args.output.write_text(json.dumps(
                {"runs": [], "skipped_apple_gpu": "no metal encode session"},
                indent=2, sort_keys=True))
        print("apple_gpu encode-session benchmark: skipping (no Metal)", file=sys.stderr)
        return 0

    version = "dev"
    try:
        import importlib.metadata
        version = importlib.metadata.version("tessera")
    except Exception:
        pass

    rows: list[dict[str, Any]] = []
    for shape in args.shapes:
        N, S = _parse(shape)
        rng = np.random.RandomState(0)
        dts = [DeviceTensor.from_numpy(rng.randn(1, S, S).astype(np.float32))
               for _ in range(N)]

        def per_op():
            acc = dts[0]
            for nxt in dts[1:]:
                acc = R._apple_gpu_bmm_device(acc, nxt)
            return acc.numpy()

        def batched():
            with AppleGPUEncodeSession() as s:
                acc = dts[0]
                for nxt in dts[1:]:
                    acc = s.bmm(acc, nxt)
            # Read the result only AFTER the session's __exit__ commits the
            # command buffer — the deferred-encode result handles are not valid
            # device memory until commit (reading inside the `with` segfaults).
            return acc.numpy()

        # Probe the actual number of command-buffer commits the batched path
        # performs: a long chain auto-flushes across several command buffers to
        # stay within Metal's per-command-buffer capacity, so syncs can exceed 1.
        probe = AppleGPUEncodeSession()
        with probe:
            acc = dts[0]
            for nxt in dts[1:]:
                acc = probe.bmm(acc, nxt)
        _ = acc.numpy()
        batched_syncs = probe.commits

        for mode, fn in (("per_op_r1", per_op), ("batched_r2", batched)):
            ms, stdev_ms = _time(fn, args.reps)
            rows.append({
                "backend": "apple_gpu",
                "op": "bmm_chain",
                "shape": shape,
                "dtype": "f32",
                "mode": mode,
                "chain_len": N,
                "syncs": (N - 1) if mode == "per_op_r1" else batched_syncs,
                "reps": args.reps,
                "latency_ms": ms,
                "stdev_ms": stdev_ms,
                "tflops": 0.0,
                "memory_bw_gb_s": 0.0,
                "device": "apple_silicon_metal",
                "tessera_version": version,
            })

    output = json.dumps({"runs": rows}, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(output)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
