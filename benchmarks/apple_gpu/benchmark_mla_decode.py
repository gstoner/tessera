"""apple_gpu MLA-decode (decoupled RoPE) benchmark — Tier-2.

Times `tessera_apple_gpu_mla_decode_rope_f32` (the on-GPU decoupled-RoPE MLA
decode — RoPE + concat assembled on host, fused attention via the `bsmm`
matmul→softmax→matmul kernel) against an equivalent vectorized numpy baseline,
over DeepSeek-shaped per-decode-step configs. Outputs the same JSON schema as
``benchmarks/benchmark_gemm.py`` for ingestion by ``tools/roofline_tools/``.

Shape spec: ``BxHxSqxSkvxDnxDrxDv``
  B   batch
  H   query heads
  Sq  query positions (1 = a single decode step)
  Skv KV context length
  Dn  no-position-encoding head dim (d_nope)
  Dr  RoPE-carrying head dim (d_rope, even)
  Dv  value head dim

Usage:
    python benchmarks/apple_gpu/benchmark_mla_decode.py \\
        --shapes 1x16x1x512x128x64x128 1x32x1x1024x128x64x128 \\
        --reps 30 --style interleaved --output apple_gpu_mla.json

Best-effort: runs only on Darwin with Metal active; on other platforms it
writes an empty run set and exits 0.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tessera import runtime as R


def _parse_shape(spec: str) -> tuple[int, int, int, int, int, int, int]:
    parts = spec.lower().split("x")
    if len(parts) != 7:
        raise ValueError(f"shape must be BxHxSqxSkvxDnxDrxDv, got {spec!r}")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def _make(B, H, Sq, Skv, Dn, Dr, Dv, base=10000.0):
    rng = np.random.RandomState(0)
    f = lambda *s: (rng.randn(*s) * 0.3).astype(np.float32)
    Qn, Qr = f(B, H, Sq, Dn), f(B, H, Sq, Dr)
    Kn, Kr = f(B, H, Skv, Dn), f(B, Skv, Dr)
    V = f(B, H, Skv, Dv)
    half = Dr // 2
    inv = base ** (-(np.arange(half, dtype=np.float64) * 2.0 / Dr))
    cosQ = np.cos(np.arange(Sq)[:, None] * inv[None, :]).astype(np.float32)
    sinQ = np.sin(np.arange(Sq)[:, None] * inv[None, :]).astype(np.float32)
    cosK = np.cos(np.arange(Skv)[:, None] * inv[None, :]).astype(np.float32)
    sinK = np.sin(np.arange(Skv)[:, None] * inv[None, :]).astype(np.float32)
    return Qn, Qr, Kn, Kr, V, cosQ, sinQ, cosK, sinK


def _rope_np(x, cos, sin, style):
    dr = x.shape[-1]
    half = dr // 2
    out = np.empty_like(x)
    if style == "interleaved":
        a, b = x[..., 0::2], x[..., 1::2]
        out[..., 0::2] = a * cos - b * sin
        out[..., 1::2] = a * sin + b * cos
    else:
        a, b = x[..., :half], x[..., half:]
        out[..., :half] = a * cos - b * sin
        out[..., half:] = b * cos + a * sin
    return out


def _numpy_mla(Qn, Qr, Kn, Kr, V, cosQ, sinQ, cosK, sinK, style):
    B, H, Sq, Dn = Qn.shape
    Dr = Qr.shape[-1]
    Skv, Dv = Kn.shape[-2], V.shape[-1]
    dh = Dn + Dr
    scale = 1.0 / math.sqrt(dh)
    QrR = _rope_np(Qr, cosQ[None, None], sinQ[None, None], style)
    KrR = _rope_np(Kr, cosK[None], sinK[None], style)
    Qfull = np.concatenate([Qn, QrR], -1)
    Kfull = np.concatenate([Kn, np.broadcast_to(KrR[:, None], (B, H, Skv, Dr))], -1)
    s = np.einsum("bhqd,bhkd->bhqk", Qfull, Kfull) * scale
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    attn = e / e.sum(-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", attn, V)


def _flops(B, H, Sq, Skv, Dn, Dr, Dv) -> int:
    dh = Dn + Dr
    per_head = 2 * Sq * Skv * dh + 3 * Sq * Skv + 2 * Sq * Skv * Dv
    return B * H * per_head


def _bytes(B, H, Sq, Skv, Dn, Dr, Dv, elem=4) -> int:
    return elem * (B * H * Sq * (Dn + Dr) + B * H * Skv * Dn + B * Skv * Dr
                   + B * H * Skv * Dv + B * H * Sq * Dv)


def _time(fn, reps) -> tuple[float, float]:
    fn()  # warm up
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return statistics.median(samples), statistics.stdev(samples) if reps > 1 else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes", nargs="+",
        default=["1x16x1x512x128x64x128", "1x32x1x1024x128x64x128",
                 "1x8x4x256x64x32x64"],
        help="BxHxSqxSkvxDnxDrxDv configs",
    )
    parser.add_argument("--style", choices=["interleaved", "half"],
                        default="interleaved")
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if sys.platform != "darwin":
        if args.output is not None:
            args.output.write_text(json.dumps(
                {"runs": [], "skipped_apple_gpu": "non-Darwin host"},
                indent=2, sort_keys=True))
        print("apple_gpu MLA benchmark: skipping (non-Darwin host)", file=sys.stderr)
        return 0

    version = "dev"
    try:
        import importlib.metadata
        version = importlib.metadata.version("tessera")
    except Exception:
        pass

    rows: list[dict[str, Any]] = []
    for shape in args.shapes:
        B, H, Sq, Skv, Dn, Dr, Dv = _parse_shape(shape)
        data = _make(B, H, Sq, Skv, Dn, Dr, Dv)
        flops = _flops(B, H, Sq, Skv, Dn, Dr, Dv)
        rw = _bytes(B, H, Sq, Skv, Dn, Dr, Dv)

        def gpu():
            return R._apple_gpu_mla_decode_rope(*data, np, rotation_style=args.style)

        def npref():
            return _numpy_mla(*data, args.style)

        for mode, fn in (("gpu", gpu), ("numpy", npref)):
            ms, stdev_ms = _time(fn, args.reps)
            sec = ms / 1000.0
            rows.append({
                "backend": "apple_gpu",
                "op": "mla_decode_rope",
                "shape": shape,
                "dtype": "f32",
                "mode": mode,
                "style": args.style,
                "reps": args.reps,
                "latency_ms": ms,
                "stdev_ms": stdev_ms,
                "tflops": (flops / sec) / 1e12 if sec > 0 else 0.0,
                "memory_bw_gb_s": (rw / sec) / 1e9 if sec > 0 else 0.0,
                "device": "apple_silicon_metal",
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
