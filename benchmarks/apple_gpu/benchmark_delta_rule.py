"""Track L L2.2 — measured perf: gated delta rule recurrent (L1.1) vs chunked
UT-transform (L2.1) on Apple GPU.  Establishes where the perf actually is before
any cooperative-parallel kernel rewrite (the perf-ratchet entry: measure first).

Run: PYTHONPATH=python python3 benchmarks/apple_gpu/benchmark_delta_rule.py
Emits the standard benchmark JSON schema (backend/op/shape/dtype/latency_ms/...).
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np


def _normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _bench(fn, *, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3   # ms/iter


def main() -> int:
    try:
        from tessera import _apple_gpu_backend as agb
    except Exception as exc:  # noqa: BLE001
        print(f"apple_gpu backend unavailable: {exc}")
        return 0
    if not agb.is_available():
        print("apple_gpu runtime unavailable — skipping")
        return 0

    # B*H = 256 threadgroups → high GPU occupancy (where the lane-0 serial
    # sections cost the most; L2.2 cooperative parallelization wins here).
    B, H, D = 8, 32, 16
    chunk = 32
    rng = np.random.default_rng(0)
    rows = []
    print(f"{'shape (B,H,S,D)':>20} | {'recur ms':>9} | {'L2.1 ms':>9} | "
          f"{'L2.2 ms':>9} | {'L2.2/rec':>8} | {'L2.2/L2.1':>9}")
    print("-" * 78)
    for S in (256, 512, 1024):
        Q = _normalize(rng.standard_normal((B, H, S, D))).astype(np.float32)
        K = _normalize(rng.standard_normal((B, H, S, D))).astype(np.float32)
        V = rng.standard_normal((B, H, S, D)).astype(np.float32)
        beta = (1.0 / (1.0 + np.exp(-rng.standard_normal((B, H, S))))).astype(np.float32)
        decay = (1.0 / (1.0 + np.exp(-(rng.standard_normal((B, H, S)) + 2)))).astype(np.float32)

        rec_ms = _bench(lambda: agb.gpu_gated_delta_rule(Q, K, V, beta, decay, erase=True))
        l21_ms = _bench(lambda: agb.gpu_gated_delta_rule_chunked(
            Q, K, V, beta, decay, chunk=chunk, erase=True, coop=False))
        l22_ms = _bench(lambda: agb.gpu_gated_delta_rule_chunked(
            Q, K, V, beta, decay, chunk=chunk, erase=True, coop=True))
        print(f"{f'({B},{H},{S},{D})':>20} | {rec_ms:9.3f} | {l21_ms:9.3f} | "
              f"{l22_ms:9.3f} | {rec_ms/l22_ms:7.2f}x | {l21_ms/l22_ms:8.2f}x")
        for op, ms in (("gated_delta_rule", rec_ms),
                       ("gated_delta_rule_chunked_lane0", l21_ms),
                       ("gated_delta_rule_chunked_coop", l22_ms)):
            rows.append({"backend": "apple_gpu", "op": op, "shape": [B, H, S, D],
                         "dtype": "f32", "latency_ms": round(ms, 5),
                         "tflops": None, "memory_bw_gb_s": None,
                         "device": "apple_gpu", "tessera_version": "track-l"})

    out = "tessera_benchmarks_delta_rule.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
