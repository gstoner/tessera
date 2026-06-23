"""benchmark_rocm_wmma_gemm.py — device-timed ROCm WMMA GEMM perf ladder.

Times the shipped ``libtessera_rocm_gemm.so`` WMMA kernel on the AMD GPU via the
``tessera_rocm_wmma_gemm_f16_bench`` C-ABI entry point (hipEvent-timed,
buffers reused, **kernel-only** — no H2D/D2H in the timed loop). Emits the
stable benchmark JSON schema (Decision #12): ``backend, op, shape, dtype,
latency_ms, tflops, memory_bw_gb_s, device, tessera_version``.

Two modes:
  * default — the production tiling (the shipped symbol's MTxNT), one row/size.
  * ``--ladder`` — sweep output-tile blocking factors (MTxNT) per size, so the
    perf-ladder rung deltas are reproducible (this is how 2x4 was chosen over
    the 1x1 naive baseline; 2x2 regressed — see STRIX_HALO_EXECUTION_PLAN.md).

Honest gating: with no AMD GPU / GEMM lib this prints a clear note and emits an
empty result set (exit 0) — it never fabricates GPU numbers. There is no
roofline fallback here on purpose; a perf dashboard must report measured silicon.

Usage::

    python benchmarks/rocm/benchmark_rocm_wmma_gemm.py
    python benchmarks/rocm/benchmark_rocm_wmma_gemm.py --ladder \
        --output rocm_gemm_ladder.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
GEMM_LIB = (REPO_ROOT / "build" / "src" / "compiler" / "codegen"
            / "Tessera_ROCM_Backend" / "runtime" / "hip"
            / "libtessera_rocm_gemm.so")
ROCM_LIB_DIR = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "lib")

# The shipped production output-tile blocking (mirror of kProdMT/kProdNT in
# tessera_rocm_gemm.cpp). Used for the default (non-ladder) row.
PROD_MT, PROD_NT = 2, 4

DEFAULT_SIZES = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
LADDER_TILINGS = [(1, 1), (1, 2), (1, 4), (2, 2), (2, 4), (4, 2), (4, 4)]
# rung-2 LDS configs (WM, WN waves, MT, NT register tiles/wave) to compare
# against the rung-1 production register tiling under --lds.
LDS_CONFIGS = [(2, 2, 2, 4), (4, 1, 2, 4), (2, 2, 1, 4), (4, 2, 1, 2)]
# rung-3 2-stage-pipelined configs to compare under --pipe (the two that win the
# 1024³–2048³ window on gfx1151).
PIPE_CONFIGS = [(4, 1, 1, 4), (2, 2, 2, 4), (2, 1, 2, 4), (2, 2, 1, 4)]


def _load_bench() -> Optional[ctypes.CDLL]:
    if not GEMM_LIB.is_file():
        return None
    for dep in ("libamdhip64.so", "libhiprtc.so"):
        p = os.path.join(ROCM_LIB_DIR, dep)
        if os.path.isfile(p):
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    try:
        lib = ctypes.CDLL(str(GEMM_LIB), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return None
    fn = getattr(lib, "tessera_rocm_wmma_gemm_f16_bench", None)
    if fn is None:
        return None
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                   ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    fn.restype = ctypes.c_int
    lds = getattr(lib, "tessera_rocm_wmma_gemm_f16_bench_lds", None)
    if lds is not None:
        lds.argtypes = [ctypes.c_int] * 4 + [ctypes.c_int] * 4 + \
            [ctypes.POINTER(ctypes.c_double)]
        lds.restype = ctypes.c_int
    return lib


def _device_name() -> str:
    # Best-effort: parse rocminfo's marketing name; fall back to a generic tag.
    try:
        import subprocess
        out = subprocess.run(["rocminfo"], capture_output=True, text=True,
                             timeout=10).stdout
        for line in out.splitlines():
            if "Marketing Name" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "amd_gpu"


def _bench_one(lib, M, N, K, iters, mt, nt) -> Optional[float]:
    """Best-of-3 average per-launch ms for one (size, tiling); None on error."""
    best = None
    for _ in range(3):
        ms = ctypes.c_double(0.0)
        rc = lib.tessera_rocm_wmma_gemm_f16_bench(M, N, K, iters, mt, nt,
                                                  ctypes.byref(ms))
        if rc != 0:
            return None
        if best is None or ms.value < best:
            best = ms.value
    return best


def _bench_staged(lib, sym, M, N, K, iters, wm, wn, mt, nt) -> Optional[float]:
    """Best-of-3 ms for a staged (LDS/pipe) symbol; None if missing/errs."""
    fn = getattr(lib, sym, None)
    if fn is None:
        return None
    best = None
    for _ in range(3):
        ms = ctypes.c_double(0.0)
        if fn(M, N, K, iters, wm, wn, mt, nt, ctypes.byref(ms)) != 0:
            return None
        if best is None or ms.value < best:
            best = ms.value
    return best


def _bench_lds(lib, M, N, K, iters, wm, wn, mt, nt) -> Optional[float]:
    return _bench_staged(lib, "tessera_rocm_wmma_gemm_f16_bench_lds",
                         M, N, K, iters, wm, wn, mt, nt)


def _bench_pipe(lib, M, N, K, iters, wm, wn, mt, nt) -> Optional[float]:
    return _bench_staged(lib, "tessera_rocm_wmma_gemm_f16_bench_pipe",
                         M, N, K, iters, wm, wn, mt, nt)


def _row(M, N, K, ms, mt, nt, device, version) -> dict:
    sec = ms * 1e-3
    flops = 2 * M * N * K
    bytes_accessed = 2 * (M * K + K * N) + 4 * (M * N)  # f16 in, f32 out
    return {
        "backend": "rocm",
        "op": "matmul",
        "shape": [M, N, K],
        "dtype": "fp16",
        "latency_ms": ms,
        "tflops": flops / sec / 1e12,
        "memory_bw_gb_s": bytes_accessed / sec / 1e9,
        "device": device,
        "tessera_version": version,
        "tile_mt_nt": [mt, nt],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ladder", action="store_true",
                    help="sweep MTxNT output-tile blocking per size (rung 1)")
    ap.add_argument("--lds", action="store_true",
                    help="compare rung-2 LDS-staged configs vs the rung-1 "
                         "production tiling (reproduces the 'LDS does not win on "
                         "Strix Halo' finding)")
    ap.add_argument("--pipe", action="store_true",
                    help="compare rung-3 2-stage-pipelined configs vs the rung-1 "
                         "production tiling (rung-3 edges +8%% only in the "
                         "1024³–2048³ window on Strix Halo)")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--output", type=str, default=None,
                    help="write the JSON result rows here")
    args = ap.parse_args()

    try:
        import tessera
        version = getattr(tessera, "__version__", "0.0.0")
    except Exception:
        version = "0.0.0"

    lib = _load_bench()
    if lib is None:
        print("no AMD GPU / libtessera_rocm_gemm.so — build it with "
              "`ninja -C build tessera_rocm_gemm` on a ROCm host. "
              "(No numbers fabricated.)")
        if args.output:
            Path(args.output).write_text(json.dumps([], indent=2))
        return 0

    device = _device_name()

    if args.lds:
        # rung-1 (register) vs rung-2 (LDS) comparison. Finding on Strix Halo:
        # register blocking wins; single-buffer LDS staging is a wash/regression
        # (unified memory — global bandwidth isn't the bottleneck LDS targets).
        rows = []
        print(f"device: {device}   iters: {args.iters}   mode: rung1-vs-LDS")
        for (M, N, K) in DEFAULT_SIZES:
            print(f"\n{M}x{N}x{K}:")
            ms = _bench_one(lib, M, N, K, args.iters, PROD_MT, PROD_NT)
            if ms:
                r = _row(M, N, K, ms, PROD_MT, PROD_NT, device, version)
                r["rung"] = "1_register"
                rows.append(r)
                print(f"  rung1 reg {PROD_MT}x{PROD_NT}:    {ms:8.4f} ms   "
                      f"{r['tflops']:6.2f} TFLOP/s  <- production")
            for (wm, wn, mt, nt) in LDS_CONFIGS:
                ms = _bench_lds(lib, M, N, K, args.iters, wm, wn, mt, nt)
                if ms is None:
                    continue
                r = _row(M, N, K, ms, mt, nt, device, version)
                r["rung"] = "2_lds"
                r["lds_waves_wm_wn"] = [wm, wn]
                rows.append(r)
                print(f"  rung2 LDS {wm}x{wn}w {mt}x{nt}t: {ms:8.4f} ms   "
                      f"{r['tflops']:6.2f} TFLOP/s")
        if args.output:
            Path(args.output).write_text(json.dumps(rows, indent=2))
            print(f"\nwrote {len(rows)} rows -> {args.output}")
        return 0

    if args.pipe:
        # rung-1 (register) vs rung-3 (2-stage pipelined, double-buffered LDS).
        # Finding on Strix Halo: rung-3 edges rung-1 by ~+8% only in the
        # 1024³–2048³ window; loses at 512³ and ≥3072³. Production stays rung-1.
        rows = []
        print(f"device: {device}   iters: {args.iters}   mode: rung1-vs-pipe")
        for (M, N, K) in DEFAULT_SIZES:
            print(f"\n{M}x{N}x{K}:")
            ms = _bench_one(lib, M, N, K, args.iters, PROD_MT, PROD_NT)
            if ms:
                r = _row(M, N, K, ms, PROD_MT, PROD_NT, device, version)
                r["rung"] = "1_register"
                rows.append(r)
                print(f"  rung1 reg {PROD_MT}x{PROD_NT}:     {ms:8.4f} ms   "
                      f"{r['tflops']:6.2f} TFLOP/s  <- production")
            for (wm, wn, mt, nt) in PIPE_CONFIGS:
                ms = _bench_pipe(lib, M, N, K, args.iters, wm, wn, mt, nt)
                if ms is None:
                    continue
                r = _row(M, N, K, ms, mt, nt, device, version)
                r["rung"] = "3_pipe"
                r["pipe_waves_wm_wn"] = [wm, wn]
                rows.append(r)
                print(f"  rung3 pipe {wm}x{wn}w {mt}x{nt}t: {ms:8.4f} ms   "
                      f"{r['tflops']:6.2f} TFLOP/s")
        if args.output:
            Path(args.output).write_text(json.dumps(rows, indent=2))
            print(f"\nwrote {len(rows)} rows -> {args.output}")
        return 0

    tilings = LADDER_TILINGS if args.ladder else [(PROD_MT, PROD_NT)]
    rows = []
    print(f"device: {device}   iters: {args.iters}   "
          f"mode: {'ladder' if args.ladder else 'production'}")
    for (M, N, K) in DEFAULT_SIZES:
        print(f"\n{M}x{N}x{K}:")
        for (mt, nt) in tilings:
            ms = _bench_one(lib, M, N, K, args.iters, mt, nt)
            if ms is None:
                print(f"  {mt}x{nt}: (kernel error)")
                continue
            r = _row(M, N, K, ms, mt, nt, device, version)
            rows.append(r)
            tag = "  <- production" if (mt, nt) == (PROD_MT, PROD_NT) else ""
            print(f"  {mt}x{nt}: {ms:8.4f} ms   {r['tflops']:6.2f} TFLOP/s   "
                  f"{r['memory_bw_gb_s']:7.1f} GB/s{tag}")

    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {len(rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
