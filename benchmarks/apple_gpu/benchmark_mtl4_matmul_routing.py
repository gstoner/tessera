"""MTL4 vs MPS fp16/bf16 matmul routing benchmark (P7, 2026-06-01).

Measures end-to-end (Python → C ABI → GPU → back) wall time for the
native MPP ``matmul2d`` tensor-op against the MPS GEMM, so the default
routing decision in ``runtime._mtl4_route_matmul2d_{f16,bf16}`` stays
data-driven and re-runnable rather than a one-off claim.

Headline finding (M-series, macOS 26.5, Metal 4):
  * **fp16**: MPS has a well-tuned GEMM and wins on square shapes
    (~0.7-0.9x for MTL4) and most decode shapes — EXCEPT the M==1
    GEMV decode step, where MPS is slow and MTL4 wins a robust
    ~3.2-3.4x. So the default fp16 route is strictly **M==1 only**.
  * **bf16**: MPS has no native bf16 GEMM (falls back to fp32
    conversion), so MTL4 wins broadly — routed by default (P5).

Run::

    PYTHONPATH=python python3 benchmarks/apple_gpu/benchmark_mtl4_matmul_routing.py

Emits a table; exits 0 always (informational). Skips cleanly when
Metal 4 is unavailable.
"""

from __future__ import annotations

import ctypes
import time

import numpy as np


def _bench(fn, *args, warmup=5, iters=20):
    for _ in range(warmup):
        fn(*args)
    best = float("inf")
    for _ in range(iters):
        t = time.perf_counter()
        fn(*args)
        best = min(best, (time.perf_counter() - t) * 1e3)
    return best


def main() -> int:
    from tessera import runtime as rt

    caps = rt._mtl4_caps_cached()
    if not (caps.get("command_queue") and caps.get("compiler")):
        print("[mtl4-routing] Metal 4 unavailable on this host — skipping.")
        return 0

    u16 = ctypes.POINTER(ctypes.c_uint16)
    mps_f16 = rt._apple_gpu_mps_matmul_f16()

    def mps_mm(A, B):
        M, K = A.shape
        _, N = B.shape
        C = np.empty((M, N), np.float16)
        mps_f16(A.ctypes.data_as(u16), B.ctypes.data_as(u16),
                C.ctypes.data_as(u16),
                ctypes.c_int32(M), ctypes.c_int32(N), ctypes.c_int32(K))
        return C

    def mtl4_mm(A, B):
        C, ran = rt.apple_gpu_mtl4_matmul2d_f16(A, B, np)
        assert ran
        return C.astype(np.float16)

    print("fp16 — MTL4 matmul2d vs MPS GEMM (best-of-20, ms)")
    print(f"{'shape':>22} {'MPS':>9} {'MTL4':>9} {'MTL4/MPS':>9}  winner")
    shapes = [
        ("M=1 K=N=4096 (decode)", 1, 4096, 4096),
        ("M=8 K=N=4096", 8, 4096, 4096),
        ("M=64 K=N=4096", 64, 4096, 4096),
        ("512^2", 512, 512, 512),
        ("1024^2", 1024, 1024, 1024),
        ("2048^2", 2048, 2048, 2048),
    ]
    for label, M, K, N in shapes:
        rng = np.random.default_rng(M * 131 + K)
        A = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
        B = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
        t_mps = _bench(mps_mm, A, B)
        t_mtl4 = _bench(mtl4_mm, A, B)
        sp = t_mps / t_mtl4
        print(f"{label:>22} {t_mps:>9.3f} {t_mtl4:>9.3f} "
              f"{sp:>8.2f}x  {'MTL4' if sp > 1.0 else 'MPS'}")

    print()
    print("Decision (P7): default fp16 route = M==1 only "
          "(set TESSERA_APPLE_GPU_MTL4_F16=all to force-route every "
          "fp16 matmul for comparison; =0 to disable).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
