"""apple_gpu fusion benchmark — Phase 8.4.6.

Times the Phase 8.4.3 / 8.4.5 fused MSL kernels against the equivalent
sequential per-op pipeline on a sweep of representative shapes. Outputs
a JSON summary suitable for ingestion by ``tools/roofline_tools/`` (same
schema as ``benchmarks/benchmark_gemm.py``).

Usage:
    python benchmarks/apple_gpu/benchmark_fusion.py \\
        --shapes 8x16x32 8x16x64 16x32x128 \\
        --reps 50 \\
        --output apple_gpu_fusion.json

Output schema (matches ``benchmark_gemm.py`` for downstream tooling):

    {"backend": "apple_gpu",
     "op": "matmul_softmax",
     "shape": "MxKxN",
     "dtype": "f32",
     "mode": "fused" | "sequential",
     "latency_ms": <float, median across reps>,
     "tflops": <float>,
     "memory_bw_gb_s": <float>,
     "device": "<MTLDevice name>",
     "tessera_version": "..."}

Compares ``fused`` vs ``sequential`` for each shape so the speedup is
visible as a row pair. The benchmark is best-effort — runs only on
Darwin with Metal active; on other platforms it skips with a clear
message and exits 0.
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

import tessera as ts


def _parse_shape(spec: str) -> tuple[int, int, int]:
    parts = spec.lower().split("x")
    if len(parts) != 3:
        raise ValueError(f"shape must be MxKxN, got {spec!r}")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def _bench_fused(M: int, K: int, N: int, reps: int) -> tuple[float, float]:
    """Fused matmul -> softmax through the Phase 8.4.3 kernel."""

    @ts.jit(target="apple_gpu")
    def fused(a, b):
        return ts.ops.softmax(ts.ops.matmul(a, b))

    rng = np.random.RandomState(0)
    A = rng.randn(M, K).astype(np.float32) * 0.5
    B = rng.randn(K, N).astype(np.float32) * 0.5
    fused(A, B)  # warm up: cache compile + MTL pipeline
    samples_ms = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fused(A, B)
        samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
    return statistics.median(samples_ms), statistics.stdev(samples_ms) if reps > 1 else 0.0


def _bench_sequential(M: int, K: int, N: int, reps: int) -> tuple[float, float]:
    """Per-op matmul + softmax through two Phase 8.3 / 8.4.2 kernels."""

    @ts.jit(target="apple_gpu")
    def matmul_only(a, b):
        return ts.ops.matmul(a, b)

    @ts.jit(target="apple_gpu")
    def softmax_only(x):
        return ts.ops.softmax(x)

    rng = np.random.RandomState(1)
    A = rng.randn(M, K).astype(np.float32) * 0.5
    B = rng.randn(K, N).astype(np.float32) * 0.5
    softmax_only(matmul_only(A, B))  # warm up
    samples_ms = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        out = matmul_only(A, B)
        softmax_only(out)
        samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
    return statistics.median(samples_ms), statistics.stdev(samples_ms) if reps > 1 else 0.0


def _flops_matmul_softmax(M: int, K: int, N: int) -> int:
    # 2 * M * K * N for the matmul, ~3 * M * N for the softmax (max + exp + sum).
    return 2 * M * K * N + 3 * M * N


def _bytes_matmul_softmax(M: int, K: int, N: int, elem_bytes: int) -> int:
    return elem_bytes * (M * K + K * N + M * N)


def _device_name() -> str:
    if sys.platform != "darwin":
        return "non-darwin-fallback"
    # Best-effort device name probe via ctypes — the runtime shim doesn't
    # expose it, so we just label by platform marker. roofline_tools is
    # OK with a coarse label.
    return "apple_silicon_metal"


def _tessera_version() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("tessera")
    except Exception:
        return "dev"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["8x16x32", "8x16x64", "16x32x128", "32x64x256"],
        help="MxKxN shapes for matmul -> softmax",
    )
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path (stdout if omitted)",
    )
    args = parser.parse_args(argv)

    if sys.platform != "darwin":
        print(
            "apple_gpu benchmark: skipping (non-Darwin host — no Metal device)",
            file=sys.stderr,
        )
        return 0

    rows: list[dict[str, Any]] = []
    device = _device_name()
    version = _tessera_version()

    for shape in args.shapes:
        M, K, N = _parse_shape(shape)
        flops = _flops_matmul_softmax(M, K, N)
        rw_bytes = _bytes_matmul_softmax(M, K, N, elem_bytes=4)

        for mode, fn in (("fused", _bench_fused), ("sequential", _bench_sequential)):
            ms, stdev_ms = fn(M, K, N, args.reps)
            sec = ms / 1000.0
            tflops = (flops / sec) / 1e12 if sec > 0 else 0.0
            mem_bw = (rw_bytes / sec) / 1e9 if sec > 0 else 0.0
            rows.append({
                "backend": "apple_gpu",
                "op": "matmul_softmax",
                "shape": shape,
                "dtype": "f32",
                "mode": mode,
                "reps": args.reps,
                "latency_ms": ms,
                "stdev_ms": stdev_ms,
                "tflops": tflops,
                "memory_bw_gb_s": mem_bw,
                "device": device,
                "tessera_version": version,
            })

    payload = {"runs": rows}
    output = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(output)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
