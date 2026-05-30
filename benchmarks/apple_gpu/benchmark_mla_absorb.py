"""apple_gpu MLA weight-absorption benchmark — Tier-2 follow-on.

Compares the weight-absorbed MLA decode (`tessera_apple_gpu_mla_absorb_decode_f32`
— attention against the cached latent, up-proj weights absorbed) against the
explicit-K decoupled-RoPE decode (`tessera_apple_gpu_mla_decode_rope_f32` —
materializes per-head K/V) on DeepSeek-shaped decode configs, and reports the
KV-cache size win (latent c_kv + shared k_rope vs per-head K/V).

Same JSON schema as ``benchmarks/benchmark_gemm.py`` plus two extra fields:
``cache_bytes_per_token`` and ``cache_ratio_vs_explicit``.

Shape spec: ``BxHxSqxSkvxDnxDrxDvxDl``  (Dl = latent dim).

Usage:
    python benchmarks/apple_gpu/benchmark_mla_absorb.py \\
        --shapes 1x16x1x512x128x64x128x512 1x32x1x1024x128x64x128x512 \\
        --reps 30 --output apple_gpu_mla_absorb.json
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


def _parse(spec: str):
    parts = spec.lower().split("x")
    if len(parts) != 8:
        raise ValueError(f"shape must be BxHxSqxSkvxDnxDrxDvxDl, got {spec!r}")
    return tuple(int(p) for p in parts)


def _make(B, H, Sq, Skv, dn, dr, dv, Dl, base=10000.0):
    rng = np.random.RandomState(0)
    f = lambda *s: (rng.randn(*s) * 0.3).astype(np.float32)
    q_nope, q_rope = f(B, H, Sq, dn), f(B, H, Sq, dr)
    c_kv, k_rope = f(B, Skv, Dl), f(B, Skv, dr)
    Wuk, Wuv = f(H, Dl, dn), f(H, Dl, dv)
    half = dr // 2
    inv = base ** (-(np.arange(half, dtype=np.float64) * 2.0 / dr))
    cosQ = np.cos(np.arange(Sq)[:, None] * inv[None, :]).astype(np.float32)
    sinQ = np.sin(np.arange(Sq)[:, None] * inv[None, :]).astype(np.float32)
    cosK = np.cos(np.arange(Skv)[:, None] * inv[None, :]).astype(np.float32)
    sinK = np.sin(np.arange(Skv)[:, None] * inv[None, :]).astype(np.float32)
    return q_nope, q_rope, c_kv, k_rope, Wuk, Wuv, cosQ, sinQ, cosK, sinK


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
    parser.add_argument(
        "--shapes", nargs="+",
        default=["1x16x1x512x128x64x128x512", "1x32x1x1024x128x64x128x512",
                 "1x8x4x256x64x32x64x256"],
        help="BxHxSqxSkvxDnxDrxDvxDl configs",
    )
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if sys.platform != "darwin":
        if args.output is not None:
            args.output.write_text(json.dumps(
                {"runs": [], "skipped_apple_gpu": "non-Darwin host"},
                indent=2, sort_keys=True))
        print("apple_gpu MLA absorb benchmark: skipping (non-Darwin)", file=sys.stderr)
        return 0

    version = "dev"
    try:
        import importlib.metadata
        version = importlib.metadata.version("tessera")
    except Exception:
        pass

    rows: list[dict[str, Any]] = []
    for shape in args.shapes:
        B, H, Sq, Skv, dn, dr, dv, Dl = _parse(shape)
        d = _make(B, H, Sq, Skv, dn, dr, dv, Dl)
        q_nope, q_rope, c_kv, k_rope, Wuk, Wuv, cQ, sQ, cK, sK = d
        Wuk_t = np.ascontiguousarray(np.swapaxes(Wuk, 1, 2))
        # explicit per-head K_nope / V derived from the latent
        Kn = np.einsum("bsl,hld->bhsd", c_kv, Wuk).astype(np.float32)
        V = np.einsum("bsl,hld->bhsd", c_kv, Wuv).astype(np.float32)

        # cache footprint per token (bytes, f32)
        cache_latent = (Dl + dr) * 4                        # shared across heads
        cache_explicit = H * (dn + dr + dv) * 4             # per-head K_nope+k_rope+V
        dh = dn + dr
        flops = B * H * (2 * Sq * Skv * dh + 3 * Sq * Skv + 2 * Sq * Skv * dv)

        def absorbed():
            return R._apple_gpu_mla_absorb_decode(q_nope, q_rope, c_kv, k_rope,
                                                  Wuk_t, Wuv, cQ, sQ, cK, sK, np)

        def explicit():
            return R._apple_gpu_mla_decode_rope(q_nope, q_rope, Kn, k_rope, V,
                                                cQ, sQ, cK, sK, np)

        for mode, fn in (("absorbed", absorbed), ("explicit", explicit)):
            ms, stdev_ms = _time(fn, args.reps)
            sec = ms / 1000.0
            rows.append({
                "backend": "apple_gpu",
                "op": "mla_absorb_decode",
                "shape": shape,
                "dtype": "f32",
                "mode": mode,
                "reps": args.reps,
                "latency_ms": ms,
                "stdev_ms": stdev_ms,
                "tflops": (flops / sec) / 1e12 if sec > 0 else 0.0,
                "cache_bytes_per_token": cache_latent if mode == "absorbed" else cache_explicit,
                "cache_ratio_vs_explicit": (cache_explicit / cache_latent
                                            if mode == "absorbed" else 1.0),
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
