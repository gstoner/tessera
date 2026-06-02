"""apple_gpu package-lane benchmark — PK8e.

Compares the two Apple-GPU execution lanes for the *same* jitted function:

  - ``package`` — ``@jit(target="apple_gpu", dispatch_via_package=True)``
    executes through the Tessera-authored ``.mtlpackage``
    (MTL4MachineLearningCommandEncoder dispatch, per-shape cached pipeline).
  - ``live``    — ``@jit(target="apple_gpu")`` runs the normal MPS / custom-MSL
    envelope.

Both are timed at **steady state** (warmed up: the package authored + its
PK1-PK7 pipeline prepared, the live pipeline compiled) so the row pair shows
the per-dispatch cost of each lane, not one-time setup. The package lane's
*cold* cost (first call = author + compile + prepare) is reported separately
as ``cold_author_ms`` so the amortization story is honest.

Ops:
  - matmul              C[M,N] = A[M,K] @ B[K,N]
  - matmul_softmax      softmax(A @ B)

Usage:
    python benchmarks/apple_gpu/benchmark_package_lane.py \\
        --shapes 8x16x32 64x64x64 256x256x256 --reps 50 \\
        --output apple_gpu_package_lane.json

Output schema (matches ``benchmark_gemm.py`` / ``benchmark_fusion.py``):

    {"backend": "apple_gpu", "op": "matmul" | "matmul_softmax",
     "shape": "MxKxN", "dtype": "f32", "mode": "package" | "live",
     "reps": N, "latency_ms": <median>, "stdev_ms": <float>,
     "cold_author_ms": <float|null>, "tflops": <float>,
     "memory_bw_gb_s": <float>, "device": "...", "tessera_version": "..."}

Best-effort: runs only on Darwin with Metal + packaged-ML available; on
other hosts it writes an empty/skipped payload and exits 0.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

import tessera as ts


def _parse_shape(spec: str) -> tuple[int, int, int]:
    parts = spec.lower().split("x")
    if len(parts) != 3:
        raise ValueError(f"shape must be MxKxN, got {spec!r}")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def _time_reps(call: Callable[[], Any], reps: int) -> tuple[float, float]:
    samples_ms = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        call()
        samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
    return (
        statistics.median(samples_ms),
        statistics.stdev(samples_ms) if reps > 1 else 0.0,
    )


# ── op factories: each returns (live_fn, package_fn) jitted the same way ──

def _matmul_factory():
    @ts.jit(target="apple_gpu")
    def live(a, b):
        return ts.ops.matmul(a, b)

    @ts.jit(target="apple_gpu", dispatch_via_package=True)
    def pkg(a, b):
        return ts.ops.matmul(a, b)

    return live, pkg


def _matmul_softmax_factory():
    @ts.jit(target="apple_gpu")
    def live(a, b):
        return ts.ops.softmax(ts.ops.matmul(a, b))

    @ts.jit(target="apple_gpu", dispatch_via_package=True)
    def pkg(a, b):
        return ts.ops.softmax(ts.ops.matmul(a, b))

    return live, pkg


_OPS = {
    "matmul": (_matmul_factory, lambda M, K, N: 2 * M * K * N),
    "matmul_softmax": (
        _matmul_softmax_factory, lambda M, K, N: 2 * M * K * N + 3 * M * N),
}


def _bench_op(op: str, M: int, K: int, N: int, reps: int) -> list[dict]:
    live, pkg = _OPS[op][0]()
    rng = np.random.RandomState(0)
    A = (rng.randn(M, K).astype(np.float32) * 0.5)
    B = (rng.randn(K, N).astype(np.float32) * 0.5)

    flops = _OPS[op][1](M, K, N)
    rw_bytes = 4 * (M * K + K * N + M * N)

    # Live lane: warm up (compile + pipeline), then time.
    live(A, B)
    live_ms, live_sd = _time_reps(lambda: live(A, B), reps)

    # Package lane: time the COLD first call (author + compile + prepare),
    # then warm steady-state.
    t0 = time.perf_counter_ns()
    pkg(A, B)
    cold_ms = (time.perf_counter_ns() - t0) / 1e6
    pkg_ms, pkg_sd = _time_reps(lambda: pkg(A, B), reps)

    shape = f"{M}x{K}x{N}"
    rows = []
    for mode, ms, sd, cold in (
        ("live", live_ms, live_sd, None),
        ("package", pkg_ms, pkg_sd, cold_ms),
    ):
        sec = ms / 1000.0
        rows.append({
            "backend": "apple_gpu",
            "op": op,
            "shape": shape,
            "dtype": "f32",
            "mode": mode,
            "reps": reps,
            "latency_ms": ms,
            "stdev_ms": sd,
            "cold_author_ms": cold,
            "tflops": (flops / sec) / 1e12 if sec > 0 else 0.0,
            "memory_bw_gb_s": (rw_bytes / sec) / 1e9 if sec > 0 else 0.0,
            "device": _device_name(),
            "tessera_version": _tessera_version(),
        })
    return rows


def _device_name() -> str:
    return "apple_silicon_metal" if sys.platform == "darwin" \
        else "non-darwin-fallback"


def _tessera_version() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("tessera")
    except Exception:
        return "dev"


def _packaged_ml_ok() -> tuple[bool, str]:
    try:
        from tessera.apple_mlpkg import (
            packaged_ml_available,
            packaged_ml_skip_reason,
        )
    except Exception as exc:  # pragma: no cover
        return False, f"apple_mlpkg import failed: {exc}"
    if not packaged_ml_available():
        return False, packaged_ml_skip_reason() or "packaged ML unavailable"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes", nargs="+",
        default=["8x16x32", "32x32x32", "64x64x64", "128x128x128",
                 "256x256x256"],
        help="MxKxN shapes")
    parser.add_argument(
        "--ops", nargs="+", default=["matmul", "matmul_softmax"],
        choices=sorted(_OPS.keys()))
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    ok, reason = (False, "non-Darwin host — no Metal device") \
        if sys.platform != "darwin" else _packaged_ml_ok()
    if not ok:
        payload = {"runs": [], "skipped_apple_gpu": reason}
        if args.output is not None:
            args.output.write_text(json.dumps(payload, indent=2,
                                               sort_keys=True))
        print(f"apple_gpu package-lane benchmark: skipping ({reason})",
              file=sys.stderr)
        return 0

    rows: list[dict[str, Any]] = []
    for op in args.ops:
        for shape in args.shapes:
            M, K, N = _parse_shape(shape)
            rows.extend(_bench_op(op, M, K, N, args.reps))

    payload = {"runs": rows}
    output = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(output)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
