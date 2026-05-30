"""apple_gpu Gumbel-max sampler benchmark.

Times the GPU Gumbel-max sampler (`tessera_apple_gpu_gumbel_argmax_f32` —
per-row vocab argmax on-GPU) against the equivalent host numpy argmax, over a
sweep of (batch, vocab) shapes. The GPU path's win grows with the batch size
(many concurrent decode streams sampling at once). Same JSON schema as
``benchmarks/benchmark_gemm.py``.

Shape spec: ``BxV`` (batch × vocab).

Usage:
    python benchmarks/apple_gpu/benchmark_gumbel_sampler.py \\
        --shapes 1x128000 8x128000 64x128000 256x32000 --reps 50
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
from tessera import rng as TR


def _parse(spec: str):
    parts = spec.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"shape must be BxV, got {spec!r}")
    return int(parts[0]), int(parts[1])


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
    parser.add_argument("--shapes", nargs="+",
                        default=["1x128000", "8x128000", "64x128000", "256x32000"])
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if sys.platform != "darwin":
        if args.output is not None:
            args.output.write_text(json.dumps(
                {"runs": [], "skipped_apple_gpu": "non-Darwin host"},
                indent=2, sort_keys=True))
        print("apple_gpu gumbel benchmark: skipping (non-Darwin)", file=sys.stderr)
        return 0

    version = "dev"
    try:
        import importlib.metadata
        version = importlib.metadata.version("tessera")
    except Exception:
        pass

    rows: list[dict[str, Any]] = []
    for shape in args.shapes:
        B, V = _parse(shape)
        rng = np.random.RandomState(0)
        logits = rng.randn(B, V).astype(np.float32)
        key = TR.RNGKey.from_seed(0)
        gumbel = R._gumbel_noise_from_key((B, V), key, np)

        def gpu():
            return R._apple_gpu_gumbel_sample(logits, np, key=key, temperature=1.0)

        def host():
            return np.argmax(logits + gumbel, axis=-1)

        for mode, fn in (("gpu", gpu), ("host_numpy", host)):
            ms, stdev_ms = _time(fn, args.reps)
            rows.append({
                "backend": "apple_gpu",
                "op": "gumbel_sample",
                "shape": shape,
                "dtype": "f32",
                "mode": mode,
                "reps": args.reps,
                "latency_ms": ms,
                "stdev_ms": stdev_ms,
                "tflops": 0.0,
                "memory_bw_gb_s": (B * V * 4 / (ms / 1000.0)) / 1e9 if ms > 0 else 0.0,
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
