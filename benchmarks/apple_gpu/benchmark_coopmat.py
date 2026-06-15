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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+",
                        default=["2048x1024x512", "2048x256x512", "2048x64x768"])
    parser.add_argument("--reps", type=int, default=40)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

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
