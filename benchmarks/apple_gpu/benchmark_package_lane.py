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

Output uses the strict-v2 source-report schema consumed by
``apple_route_selector``.  Every route row retains interleaved paired-trial
medians, exact producer context, numerical/native proof, and resource evidence;
the complete-call timing domain is selectable while unavailable device timing
is preserved as explicitly ineligible by the sealer.

    {"backend": "apple_gpu", "op": "matmul" | "matmul_softmax",
     "shape": "MxKxN", "dtype": "f32", "mode": "package" | "live",
     "reps": N, "latency_ms": <median>, "stdev_ms": <float>,
     "cold_author_ms": <float|null>, "tflops": <float>,
     "memory_bw_gb_s": <float>, "device": "...", "tessera_version": "...",
     "route": "package" | "live", "native_dispatched": <bool>,
     "numerically_validated": <bool>}

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
from tessera.compiler.apple_route_selector import live_apple_route_context


def _parse_shape(spec: str) -> tuple[int, int, int]:
    parts = spec.lower().split("x")
    if len(parts) != 3:
        raise ValueError(f"shape must be MxKxN, got {spec!r}")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def _trial_median_ns(call: Callable[[], Any], reps: int) -> int:
    samples = []
    for _ in range(reps):
        start = time.perf_counter_ns()
        call()
        samples.append(time.perf_counter_ns() - start)
    return int(statistics.median(samples))


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


def _bench_op(op: str, M: int, K: int, N: int, reps: int, trials: int) -> list[dict]:
    live, pkg = _OPS[op][0]()
    rng = np.random.RandomState(0)
    A = (rng.randn(M, K).astype(np.float32) * 0.5)
    B = (rng.randn(K, N).astype(np.float32) * 0.5)

    flops = _OPS[op][1](M, K, N)
    rw_bytes = 4 * (M * K + K * N + M * N)

    # Live lane: warm up (compile + pipeline), then time.
    live_out = live(A, B)

    # Package lane: time the COLD first call (author + compile + prepare),
    # then warm steady-state.
    t0 = time.perf_counter_ns()
    pkg_out = pkg(A, B)
    cold_ms = (time.perf_counter_ns() - t0) / 1e6
    paired: dict[str, list[int]] = {"live": [], "package": []}
    calls = {"live": lambda: live(A, B), "package": lambda: pkg(A, B)}
    for trial in range(trials):
        order = ("live", "package") if trial % 2 == 0 else ("package", "live")
        for route in order:
            paired[route].append(_trial_median_ns(calls[route], reps))

    shape = f"{M}x{K}x{N}"
    rows = []
    # A route is eligible to influence generic JIT selection only after it has
    # both an oracle comparison and concrete evidence that its package pipeline
    # was actually prepared rather than silently falling back to the live lane.
    package_native = bool(getattr(pkg, "_package_pipeline_cache", {}))
    package_ok = bool(np.allclose(pkg_out, live_out, rtol=1e-4, atol=1e-5))
    live_ok = bool(np.allclose(live_out, pkg_out, rtol=1e-4, atol=1e-5))
    device = live_apple_route_context().device
    for mode, cold, native, correct in (
        ("live", None, True, live_ok),
        ("package", cold_ms, package_native, package_ok),
    ):
        samples_ns = paired[mode]
        median_ns = int(statistics.median(samples_ns))
        ms = median_ns / 1e6
        sd = statistics.stdev(samples_ns) / 1e6 if len(samples_ns) > 1 else 0.0
        sec = ms / 1000.0
        resources = ({
            "api": "MTL4MachineLearningCommandEncoder",
            "package_pipeline_cache_key": [op, [M, K, N]],
            "package_pipeline_prepared": package_native,
            "private_intermediates": op == "matmul_softmax",
        } if mode == "package" else {
            "api": "apple_gpu_live_runtime",
            "package_pipeline_cache_key": None,
            "package_pipeline_prepared": False,
            "private_intermediates": False,
        })
        rows.append({
            "backend": "apple_gpu",
            "op": op,
            "shape": shape,
            "dtype": "f32",
            "mode": mode,
            "route": mode,
            "native_dispatched": native,
            "numerically_validated": correct,
            "reps": reps * trials,
            "latency_ms": ms,
            "stdev_ms": sd,
            "cold_author_ms": cold,
            "tflops": (flops / sec) / 1e12 if sec > 0 else 0.0,
            "memory_bw_gb_s": (rw_bytes / sec) / 1e9 if sec > 0 else 0.0,
            "device": device,
            "tessera_version": _tessera_version(),
            "telemetry": {
                "end_to_end_median_ns": median_ns,
                "device_time_median_ns": None,
                "device_time_samples": 0,
                "device_time_coverage": 0.0,
                "paired_trial_end_to_end_medians_ns": samples_ns,
                "paired_trial_device_medians_ns": None,
                "timing_source": "host_steady_state_complete_call",
                "counter_sampling_supported": False,
                "counter_timestamp_delta_median": None,
                "resources": resources,
            },
        })
    return rows


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
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    ok, reason = (False, "non-Darwin host — no Metal device") \
        if sys.platform != "darwin" else _packaged_ml_ok()
    if not ok:
        payload = {"schema_version": 1, "runs": [], "skipped_apple_gpu": reason}
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
            rows.extend(_bench_op(op, M, K, N, args.reps, args.trials))

    payload = {"schema_version": 1,
               "selection_scope": "package_subgraph",
               "context": live_apple_route_context().as_mapping(),
               "runs": rows}
    output = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(output)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
