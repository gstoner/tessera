"""benchmark_rocm_gemm_pipeline_vs_direct.py — Fork-A A/B: route the compiler-
generated WMMA GEMM through the typed Tile chain vs. emit it directly.

Two lowerings of the SAME `tessera_rocm.wmma_gemm` directive:

  direct : generate-wmma-gemm-kernel            -> lower-tessera-target-to-rocdl
  fork_a : generate-wmma-gemm-kernel=via-tile
             (emits view/pack/mma/unpack/store)
             -> rocm-wave-lds-pipeline -> rocm-wave-lds-legality
             -> lower-tile-to-rocm{arch} -> lower-tessera-target-to-rocdl

Fork-A routes the matrix op through the Tile-IR seam (tile.mma) and the wave/LDS
pipeline, then lowers it to tessera_rocm.wmma. This harness proves the migrated
producer on-device: both are validated vs numpy and timed with a synchronized
host clock, matching GEMM_PERF_LADDER.md's 8.02 TFLOP/s timer modality. The HIP
event API on this WSL /dev/dxg host reports zero/garbage while returning success,
so event timing is not evidence. The comparison is pinned to the production
MT=2, NT=4 schedule rather than selecting a different tile per lane.

Honest gating: no GPU / tools -> a clear note + empty result set (exit 0).

Usage::

    python benchmarks/rocm/benchmark_rocm_gemm_pipeline_vs_direct.py --size 2048
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESSERA_OPT = REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt"
CHIP = os.environ.get("TESSERA_ROCM_CHIP", "gfx1151")
TILES = [(2, 4)]
BASELINE_TFLOPS = 8.02


def _directive(mt, nt):
    return ('module {\n  "tessera_rocm.wmma_gemm"() {name = "gemm", m = 16 : '
            f'i64, n = 16 : i64, k = 16 : i64, mt = {mt} : i64, nt = {nt} : '
            'i64, dtype = "f16"} : () -> ()\n}\n')


def _find_mlir_opt():
    if env := os.environ.get("TESSERA_MLIR_OPT"):
        return env if Path(env).is_file() else None
    for c in ("/usr/lib/llvm-23/bin/mlir-opt",
              "/opt/homebrew/opt/llvm@23/bin/mlir-opt"):
        if Path(c).is_file():
            return c
    return shutil.which("mlir-opt")


def _extract_hsaco(text):
    i = text.index('bin = "') + len('bin = "')
    out = bytearray(); j = i
    hexd = "0123456789abcdefABCDEF"
    simple = {"\\": 0x5C, '"': 0x22, "n": 0x0A, "t": 0x09, "r": 0x0D}
    while j < len(text):
        c = text[j]
        if c == '"':
            break
        if c == "\\":
            nx = text[j + 1:j + 3]
            if len(nx) == 2 and nx[0] in hexd and nx[1] in hexd:
                out.append(int(nx, 16)); j += 3; continue
            if text[j + 1] in simple:
                out.append(simple[text[j + 1]]); j += 2; continue
        out.append(ord(c)); j += 1
    return bytes(out)


# The tessera-opt front half of each lowering (everything up to ROCDL).
def _front(mode):
    if mode == "direct":
        return ["--generate-wmma-gemm-kernel", "--lower-tessera-target-to-rocdl"]
    return ["--generate-wmma-gemm-kernel=via-tile=true",
            "--rocm-wave-lds-pipeline", "--rocm-wave-lds-legality",
            f"--lower-tile-to-rocm=arch={CHIP}",
            "--lower-tessera-target-to-rocdl"]


def _build(mlir_opt, mode, mt, nt):
    g = subprocess.run([str(TESSERA_OPT), "-", *_front(mode)],
                       input=_directive(mt, nt), capture_output=True, text=True)
    if g.returncode != 0:
        raise RuntimeError(g.stderr)
    pl = ("builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
          f"reconcile-unrealized-casts),rocdl-attach-target{{chip={CHIP}}},"
          "gpu-module-to-binary)")
    s = subprocess.run([mlir_opt, f"--pass-pipeline={pl}"],
                       input=g.stdout, capture_output=True, text=True)
    if s.returncode != 0:
        raise RuntimeError(s.stderr)
    return _extract_hsaco(s.stdout)


def _load_hip():
    try:
        hip = ctypes.CDLL("libamdhip64.so", mode=ctypes.RTLD_LOCAL)
    except OSError:
        return None
    hip.hipInit.argtypes = [ctypes.c_uint]
    hip.hipMalloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    hip.hipFree.argtypes = [ctypes.c_void_p]
    hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_size_t, ctypes.c_int]
    hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                         ctypes.c_void_p, ctypes.c_char_p]
    hip.hipModuleUnload.argtypes = [ctypes.c_void_p]
    hip.hipModuleLaunchKernel.argtypes = (
        [ctypes.c_void_p] + [ctypes.c_uint] * 6
        + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    )
    hip.hipDeviceSynchronize.argtypes = []
    return hip


def _mr(p, size):
    return [ctypes.c_void_p(p.value), ctypes.c_void_p(p.value),
            ctypes.c_int64(0), ctypes.c_int64(size), ctypes.c_int64(1)]


def _run(hip, hsaco, M, N, K, mt, nt, iters, check):
    import numpy as np
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        return None
    fn = ctypes.c_void_p()
    if hip.hipModuleGetFunction(ctypes.byref(fn), mod, b"gemm") != 0:
        return None
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((M, K)) * 0.2).astype(np.float16)
    B = (rng.standard_normal((K, N)) * 0.2).astype(np.float16)
    da, db, dd = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    for d, nb in ((da, 2 * M * K), (db, 2 * K * N), (dd, 4 * M * N)):
        if hip.hipMalloc(ctypes.byref(d), nb) != 0:
            return None
    hip.hipMemcpy(da, A.ctypes.data_as(ctypes.c_void_p), 2 * M * K, 1)
    hip.hipMemcpy(db, B.ctypes.data_as(ctypes.c_void_p), 2 * K * N, 1)
    args = (_mr(da, M * K) + _mr(db, K * N) + _mr(dd, M * N)
            + [ctypes.c_int64(M), ctypes.c_int64(N), ctypes.c_int64(K)])
    arr = (ctypes.c_void_p * len(args))()
    for i, a in enumerate(args):
        arr[i] = ctypes.cast(ctypes.byref(a), ctypes.c_void_p)
    launch = hip.hipModuleLaunchKernel
    gx, gy = (N + 16 * nt - 1) // (16 * nt), (M + 16 * mt - 1) // (16 * mt)
    for _ in range(3):
        if launch(fn, gx, gy, 1, 32, 1, 1, 0, None, arr, None) != 0:
            return None
    hip.hipDeviceSynchronize()
    maxerr = None
    if check:
        D = np.zeros((M, N), np.float32)
        hip.hipMemcpy(D.ctypes.data_as(ctypes.c_void_p), dd, 4 * M * N, 2)
        ref = (A.astype(np.float32) @ B.astype(np.float32))
        maxerr = float(np.max(np.abs(D - ref)) / (np.max(np.abs(ref)) + 1e-6))
    hip.hipDeviceSynchronize()
    started = time.perf_counter()
    for _ in range(iters):
        if launch(fn, gx, gy, 1, 32, 1, 1, 0, None, arr, None) != 0:
            return None
    hip.hipDeviceSynchronize()
    elapsed_ms = (time.perf_counter() - started) * 1e3 / iters
    for d in (da, db, dd):
        hip.hipFree(d)
    hip.hipModuleUnload(mod)
    return elapsed_ms, maxerr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()
    M = N = K = args.size

    if not TESSERA_OPT.is_file():
        print("[note] tessera-opt not built — nothing to compare.",
              file=sys.stderr)
        if args.output:
            Path(args.output).write_text("[]\n")
        return 0
    mlir_opt = _find_mlir_opt()
    hip = _load_hip()
    if mlir_opt is None or hip is None or hip.hipInit(0) != 0:
        print("[note] no mlir-opt / AMD GPU — refusing to fabricate numbers.",
              file=sys.stderr)
        if args.output:
            Path(args.output).write_text("[]\n")
        return 0

    flop = 2.0 * M * N * K
    rows = []
    print(f"# Fork-A A/B: pipeline-routed vs direct WMMA GEMM, {M}x{N}x{K}, "
          f"{args.iters} iters x {args.trials} interleaved trials, {CHIP}")
    print(f"# {'mode':>8} {'best tile':>10} {'TFLOP/s':>9} {'rel-err':>10}")
    builds = {}
    for mode in ("direct", "fork_a"):
        for mt, nt in TILES:
            try:
                builds[(mode, mt, nt)] = _build(mlir_opt, mode, mt, nt)
            except RuntimeError as e:
                print(f"  {mode} {mt}x{nt}: build failed: {str(e)[:80]}",
                      file=sys.stderr)

    samples = {key: [] for key in builds}
    errors = {key: None for key in builds}
    for trial in range(args.trials):
        # Reverse the order every trial. On this APU, running one lane for all
        # samples before the other measures boost/thermal order as much as IR.
        modes = ("direct", "fork_a") if trial % 2 == 0 else ("fork_a", "direct")
        for mode in modes:
            for mt, nt in TILES:
                hsaco = builds.get((mode, mt, nt))
                if hsaco is None:
                    continue
                result = _run(hip, hsaco, M, N, K, mt, nt, args.iters,
                              check=(trial == 0))
                if result is None:
                    continue
                ms, err = result
                if err is not None and err > 1e-2:
                    print(f"  {mode} {mt}x{nt}: CORRECTNESS FAIL "
                          f"rel-err={err:.2e}", file=sys.stderr)
                    continue
                samples[(mode, mt, nt)].append(ms)
                if err is not None:
                    errors[(mode, mt, nt)] = err

    best = {}
    for mode in ("direct", "fork_a"):
        best_ms, best_tile, best_err = None, None, None
        for mt, nt in TILES:
            timings = samples.get((mode, mt, nt), [])
            if not timings:
                continue
            ms = statistics.median(timings)
            err = errors[(mode, mt, nt)]
            if best_ms is None or ms < best_ms:
                best_ms, best_tile, best_err = ms, (mt, nt), err
        if best_ms is None:
            print(f"  {mode}: no usable build", file=sys.stderr)
            continue
        tf = flop / (best_ms / 1e3) / 1e12
        best[mode] = tf
        print(f"  {mode:>8} {str(best_tile):>10} {tf:>9.2f} {best_err:>10.2e}")
        rows.append({
            "backend": "rocm", "op": "gemm", "shape": [M, N, K], "dtype": "f16",
            "latency_ms": best_ms, "tflops": tf, "memory_bw_gb_s": None,
            "device": CHIP, "tessera_version": f"fork-a-ab:{mode}",
            "mt": best_tile[0], "nt": best_tile[1],
            "path": "compiler-generated", "lowering": mode,
            "timer_source": "host_wall",
            "iterations_per_trial": args.iters,
            "trials": args.trials,
            "aggregation": "median_interleaved",
        })
    if "direct" in best and "fork_a" in best:
        ratio = best["fork_a"] / best["direct"]
        print(f"# fork_a / direct = {ratio:.3f}x "
              f"({'parity (APU noise)' if 0.85 <= ratio <= 1.18 else 'CHECK'})")
        baseline_ratio = best["fork_a"] / BASELINE_TFLOPS
        print(f"# fork_a / 8.02 TFLOP/s baseline = {baseline_ratio:.3f}x")
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
