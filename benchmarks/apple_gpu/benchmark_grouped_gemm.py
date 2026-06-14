"""Isolated grouped-GEMM / MoE-block benchmark (DeepGEMM keystone, 2026-06).

Times ``grouped_gemm`` and ``moe_swiglu_block`` **standalone** — through the same
production runtime dispatchers, but without the MegaMoE overlap harness that
conflates the all-to-all collective with compute. This lets a regression pin to
the primitive itself, and exercises the low-precision quant paths
(f32 / fp8 / nvfp4 / fp8xfp4) the dense GEMM benchmark doesn't cover.

Emits the stable benchmark JSON schema (``op``, ``shape``, ``dtype``,
``latency_ms``, ``tflops``, ``memory_bw_gb_s``, ``device``, ``tessera_version``)
so ``tools/roofline_tools`` reads it directly. Off Darwin/Metal it reports the
analytical roofline latency (``mode="roofline"``) instead of a measured value.

    python benchmarks/apple_gpu/benchmark_grouped_gemm.py [--out grouped.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def _median_ms(fn, reps: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _grouped_flops(T: int, K: int, N: int) -> int:
    # One (T,K)x(K,N) contraction's worth of MACs, summed over tokens.
    return 2 * T * K * N


def _moe_flops(T: int, K: int, F: int, M: int) -> int:
    # gate + up (T,K)->(T,F) twice, then down (T,F)->(T,M).
    return 2 * T * K * F * 2 + 2 * T * F * M


def cases(rng):
    f32 = np.float32
    T, K, N, E = 64, 128, 256, 4
    x = rng.standard_normal((T, K)).astype(f32)
    w = rng.standard_normal((E, K, N)).astype(f32)
    gs = np.array([T // E] * E, dtype=np.int64)
    F, M = 256, 128
    wg = rng.standard_normal((E, K, F)).astype(f32)
    wu = rng.standard_normal((E, K, F)).astype(f32)
    wd = rng.standard_normal((E, F, M)).astype(f32)
    rows = []
    # grouped_gemm across the quant lanes (dequant-on-host today).
    for quant in (None, "fp8_e4m3", "nvfp4", "fp8xfp4"):
        kw = {"grouped_kind": "contiguous"}
        if quant is not None:
            kw["quant"] = quant
        rows.append((
            "grouped_gemm", f"{T}x{K}x{N}_E{E}",
            "f32" if quant is None else quant,
            _grouped_flops(T, K, N), dict(kw),
            ("grouped_gemm", (x, w, gs))))
    # moe_swiglu_block (f32 fused fast path).
    rows.append((
        "moe_swiglu_block", f"{T}x{K}x{F}x{M}_E{E}", "f32",
        _moe_flops(T, K, F, M), {"grouped_kind": "contiguous"},
        ("moe_swiglu_block", (x, wg, wu, wd, gs))))
    return rows


def _roofline_ms(flops: int) -> float:
    # Conservative analytical fallback when no Metal device is present.
    peak_tflops = 10.0  # generic M-series f32 lane
    return flops / (peak_tflops * 1e12) * 1e3


def run() -> list[dict]:
    from tessera import __version__ as ver  # noqa: F401  (best-effort)
    rng = np.random.default_rng(0)
    darwin = sys.platform == "darwin"
    rt = None
    if darwin:
        from tessera import runtime as rt  # type: ignore
    out = []
    for op, shape, dtype, flops, kw, (kind, operands) in cases(rng):
        if rt is not None:
            if kind == "grouped_gemm":
                thunk = lambda o=operands, k=kw: rt._apple_gpu_dispatch_grouped_gemm(o, k, np)
            else:
                thunk = lambda o=operands, k=kw: rt._apple_gpu_dispatch_moe_swiglu_block(o, k, np)
            latency_ms = _median_ms(thunk)
            # moe_swiglu_block runs the composed grouped-GEMM + silu_mul lane by
            # default (the fused MSL kernel is ~9.6× slower; opt-in via
            # TESSERA_APPLE_MOE_FUSED=1).
            mode = "composed" if op == "moe_swiglu_block" else "fused"
        else:
            latency_ms = _roofline_ms(flops)
            mode = "roofline"
        sec = max(latency_ms * 1e-3, 1e-12)
        out.append({
            "schema": "tessera.benchmark.v1",
            "backend": "apple_gpu",
            "op": op, "shape": shape, "dtype": dtype, "mode": mode,
            "latency_ms": round(latency_ms, 4),
            "tflops": round(flops / sec / 1e12, 4),
            "memory_bw_gb_s": 0.0,
            "device": "apple_gpu" if darwin else "cpu_roofline",
            "tessera_version": "experimental",
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = run()
    for r in rows:
        print(f"{r['op']:18s} {r['shape']:22s} {r['dtype']:10s} "
              f"{r['latency_ms']:8.3f} ms  {r['tflops']:7.2f} TFLOP/s  ({r['mode']})")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
