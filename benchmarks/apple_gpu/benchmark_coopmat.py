"""F2d cooperative-matrix vs scalar synthesis benchmark (Apple GPU).

Times the SYNTHESIZED ``matmul -> pointwise-epilogue`` kernel two ways on a
sweep of shapes, at f32 and f16:

  - ``scalar``  — the per-thread / threadgroup-tiled kernel (fp32 FMA in the
    general ALU; never touches the matrix units).
  - ``coopmat`` — the ``simdgroup_matrix`` MMA kernel (f16 multiply / fp32
    accumulate — the M-series matrix-unit path) with the epilogue fused after.

On an M-series Mac the coopmat kernel measures ~55-98x the scalar kernel and
recovers the f16 throughput the scalar kernel cannot (the scalar kernel shows
~1.0x f16/f32 because it is compute-bound on the fp32 ALU; coopmat shows
~1.4-1.8x because the matrix units run f16 at ~2x). Same JSON schema as
``benchmark_fusion.py`` so ``tools/roofline_tools/`` can ingest it.

Best-effort: runs only on Darwin with Metal active; skips with exit 0 otherwise.

    python benchmarks/apple_gpu/benchmark_coopmat.py --reps 40
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

from tessera.compiler.fusion import (
    FusedRegion,
    run_fused_region,
    run_fused_region_coopmat,
    run_fused_region_coopmat_reduce,
)

REGION = FusedRegion(("gelu",))   # pointwise → both the scalar and coopmat paths apply


def _time(fn, reps: int) -> float:
    fn()
    fn()  # warm up: MSL compile + pipeline cache
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return statistics.median(samples)


def _parse_shape(spec: str) -> tuple[int, int, int]:
    parts = spec.lower().split("x")
    if len(parts) != 3:
        raise ValueError(f"shape must be MxKxN, got {spec!r}")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def _reduce_vs_compose(shapes: list[str], reps: int) -> list[dict[str, Any]]:
    """F2d-v2.2 verdict harness — for each shape, time the fused coopmat-reduce
    softmax kernel against (a) the pointwise coopmat kernel on the *identical*
    matmul and (b) the production MPS-compose matmul->softmax path.

    The (reduce / pointwise) ratio isolates the cause: the reduction phase is
    free; the fused kernel pays 6-12x in the matmul because fusing the reduction
    forces BM=8 / one simdgroup (the full N-wide row must stay resident in
    threadgroup memory to be reduced).  A cooperative reduction tree can't relax
    that, so v2.2 is structurally dominated by compose.  See OPTIMIZING_COMPILER_PLAN.md.
    """
    from tessera.runtime import _apple_gpu_dispatch_matmul_softmax
    soft = FusedRegion(epilogue=(), reduction="softmax")
    gelu = FusedRegion(epilogue=("gelu",))
    rows: list[dict[str, Any]] = []
    for shape in shapes:
        M, K, N = _parse_shape(shape)
        A = (np.random.RandomState(0).randn(M, K) * 0.3).astype(np.float16)
        B = (np.random.RandomState(1).randn(K, N) * 0.3).astype(np.float16)
        _o, exr = run_fused_region_coopmat_reduce(soft, A, B)
        if exr != "metal_runtime":
            continue
        tr = _time(lambda: run_fused_region_coopmat_reduce(soft, A, B), reps)
        tp = _time(lambda: run_fused_region_coopmat(gelu, A, B), reps)
        tc = _time(lambda: _apple_gpu_dispatch_matmul_softmax([A, B], np), reps)
        rows.append({
            "backend": "apple_gpu", "op": "matmul_softmax", "shape": shape,
            "dtype": "f16", "reps": reps,
            "fused_reduce_ms": tr, "pointwise_coopmat_ms": tp, "compose_ms": tc,
            "reduce_over_pointwise": tr / tp if tp > 0 else 0.0,
            "compose_speedup": tr / tc if tc > 0 else 0.0,
            "device": "apple_silicon_metal", "tessera_version": "dev",
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=None)
    parser.add_argument("--reps", type=int, default=40)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=("coopmat", "reduce-vs-compose"),
                        default="coopmat",
                        help="coopmat: scalar-vs-coopmat pointwise sweep (default). "
                             "reduce-vs-compose: F2d-v2.2 fused-reduce vs "
                             "pointwise-coopmat vs MPS-compose.")
    args = parser.parse_args(argv)

    if args.mode == "reduce-vs-compose":
        if sys.platform != "darwin":
            print("apple_gpu coopmat benchmark: skipping (non-Darwin host)",
                  file=sys.stderr)
            return 0
        shapes = args.shapes or ["512x512x512", "1024x256x512",
                                 "2048x1024x256", "2048x512x128"]
        rs = _reduce_vs_compose(shapes, args.reps)
        output = json.dumps({"runs": rs}, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.write_text(output)
        else:
            print(output)
        return 0

    args.shapes = args.shapes or ["2048x1024x512", "2048x256x512", "2048x64x768"]

    if sys.platform != "darwin":
        print("apple_gpu coopmat benchmark: skipping (non-Darwin host)",
              file=sys.stderr)
        if args.output is not None:
            args.output.write_text(json.dumps(
                {"runs": [], "skipped_apple_gpu": "non-Darwin host"}, indent=2))
        return 0

    rows: list[dict[str, Any]] = []
    for shape in args.shapes:
        M, K, N = _parse_shape(shape)
        flop = 2 * M * K * N + 3 * M * N
        for dt, dt_name in ((np.float32, "f32"), (np.float16, "f16")):
            A = (np.random.RandomState(0).randn(M, K) * 0.3).astype(dt)
            B = (np.random.RandomState(1).randn(K, N) * 0.3).astype(dt)
            for mode, runner in (("scalar", run_fused_region),
                                 ("coopmat", run_fused_region_coopmat)):
                _out, ex = runner(REGION, A, B)
                if ex != "metal_runtime":
                    continue
                ms = _time(lambda: runner(REGION, A, B), args.reps)
                sec = ms / 1000.0
                rows.append({
                    "backend": "apple_gpu",
                    "op": "matmul_gelu",
                    "shape": shape,
                    "dtype": dt_name,
                    "mode": mode,
                    "reps": args.reps,
                    "latency_ms": ms,
                    "tflops": (flop / sec) / 1e12 if sec > 0 else 0.0,
                    "device": "apple_silicon_metal",
                    "tessera_version": "dev",
                })

    output = json.dumps({"runs": rows}, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(output)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
