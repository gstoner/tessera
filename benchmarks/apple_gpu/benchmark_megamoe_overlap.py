"""apple_gpu distributed-MegaMoE comm/compute overlap benchmark.

Measures the REAL wall-clock benefit of the async-pipelined expert-parallel MoE
forward (``megamoe_layer_pipelined``) against the sequential-chunked overlap
path (``megamoe_layer_overlapped``) on a sweep of modeled interconnect
latencies. The expert FFN runs on the Apple GPU via the fused
``moe_swiglu_block`` kernel; because the Metal command buffer runs
asynchronously and the ``ctypes.CDLL`` call releases the GIL, chunk c's GPU
compute overlaps chunk c+1's DISPATCH all-to-all — so the dispatch comm is
hidden under compute.

``comm_latency_s`` models the per-all-to-all interconnect transfer cost (the
single-machine mock collective has none; real multi-device comm does). Sweeping
it shows the overlap win grow with the comm:compute ratio.

Usage:
    python benchmarks/apple_gpu/benchmark_megamoe_overlap.py \\
        --latencies 0 4 8 12 16 --num-chunks 4 --world-size 2 --reps 3 \\
        --output megamoe_overlap.json

Output schema (best-effort superset of benchmark_gemm.py):

    {"backend": "apple_gpu",
     "op": "megamoe_overlap",
     "shape": "TxKxExF",
     "dtype": "f32",
     "mode": "seq_chunked" | "async_pipelined",
     "num_chunks": <int>,
     "world_size": <int>,
     "comm_latency_ms": <float>,
     "latency_ms": <float, best across reps>,
     "speedup_vs_seq_chunked": <float>,
     "tessera_version": "..."}

Best-effort: runs only on Darwin with Metal active; skips with a clear message
and exits 0 elsewhere.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np


def _gpu_available() -> bool:
    try:
        from tessera import _apple_gpu_backend as agb
        from tessera import _jit_boundary as jb
        return bool(agb.is_available() and jb.is_available())
    except Exception:  # noqa: BLE001
        return False


def _wall(fn, reps: int) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latencies", type=float, nargs="+", default=[0, 4, 8, 12, 16],
                    help="modeled per-all-to-all interconnect latencies (ms)")
    ap.add_argument("--num-chunks", type=int, default=4)
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--shape", type=str, default="4096x256x8x256",
                    help="TxKxExF (tokens x model-dim x experts x hidden)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    if not _gpu_available():
        print("apple_gpu runtime unavailable — skipping (exit 0).")
        return 0

    import tessera
    from tessera.distributed.moe import (
        MoEConfig,
        megamoe_layer_overlapped,
        megamoe_layer_pipelined,
    )

    T, K, E, Fd = (int(v) for v in args.shape.split("x"))
    N = K
    rng = np.random.default_rng(0)
    x = rng.standard_normal((T, K)).astype(np.float32)
    Wr = rng.standard_normal((K, E)).astype(np.float32)
    Wg = rng.standard_normal((E, K, Fd)).astype(np.float32)
    Wu = rng.standard_normal((E, K, Fd)).astype(np.float32)
    Wd = rng.standard_normal((E, Fd, N)).astype(np.float32)
    cfg = MoEConfig(num_experts=E, top_k=2, capacity_factor=4.0)
    ws, nc = args.world_size, args.num_chunks
    ver = getattr(tessera, "__version__", "unknown")

    # Warm the fused kernel (one-time MSL compile).
    megamoe_layer_pipelined(x, Wr, Wg, Wu, Wd, world_size=ws, config=cfg, num_chunks=nc)

    rows = []
    print(f"shape={args.shape}  world_size={ws}  num_chunks={nc}  reps={args.reps}")
    print(f"{'comm/a2a':>10} {'seq_chunked':>13} {'async_pipe':>12} {'speedup':>9}")
    for lat_ms in args.latencies:
        lat = lat_ms / 1e3
        seq = _wall(lambda lat=lat: megamoe_layer_overlapped(
            x, Wr, Wg, Wu, Wd, world_size=ws, config=cfg, num_chunks=nc,
            comm_latency_s=lat), args.reps)
        pll = _wall(lambda lat=lat: megamoe_layer_pipelined(
            x, Wr, Wg, Wu, Wd, world_size=ws, config=cfg, num_chunks=nc,
            comm_latency_s=lat), args.reps)
        speedup = seq / pll if pll else 0.0
        print(f"{lat_ms:>9.0f}m {seq*1e3:>11.1f}m {pll*1e3:>10.1f}m {speedup:>8.2f}x")
        for mode, lat_s in (("seq_chunked", seq), ("async_pipelined", pll)):
            rows.append({
                "backend": "apple_gpu", "op": "megamoe_overlap",
                "shape": args.shape, "dtype": "f32", "mode": mode,
                "num_chunks": nc, "world_size": ws, "comm_latency_ms": lat_ms,
                "latency_ms": round(lat_s * 1e3, 3),
                "speedup_vs_seq_chunked": round(speedup, 3) if mode == "async_pipelined" else 1.0,
                "tessera_version": ver,
            })

    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {len(rows)} rows → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
