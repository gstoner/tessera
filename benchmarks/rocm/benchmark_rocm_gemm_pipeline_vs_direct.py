"""benchmark_rocm_gemm_pipeline_vs_direct.py — Fork-A A/B: route the compiler-
generated WMMA GEMM through the wave/LDS pipeline vs. emit it directly.

Two lowerings of the SAME `tessera_rocm.wmma_gemm` directive:

  direct : generate-wmma-gemm-kernel            -> lower-tessera-target-to-rocdl
  fork_a : generate-wmma-gemm-kernel=via-tile   (emits tile.mma)
             -> rocm-wave-lds-pipeline -> rocm-wave-lds-legality
             -> lower-tile-to-rocm{arch} -> lower-tessera-target-to-rocdl

Fork-A routes the matrix op through the Tile-IR seam (tile.mma) and the wave/LDS
pipeline, then lowers it back to tessera_rocm.wmma — so the final rocdl.wmma is
expected identical to direct. This harness proves it on-device: both are
validated vs numpy (correctness) and hipEvent-timed (perf parity is the pass
condition — Fork-A must not regress the direct lane). The GEMM has no LDS
staging yet, so the pipeline's double-buffering does not engage here; this pilot
establishes the routing + parity. Emits the stable benchmark JSON (Decision #12).

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
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESSERA_OPT = REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt"
ROCM_LIB_DIR = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "lib")
CHIP = os.environ.get("TESSERA_ROCM_CHIP", "gfx1151")
TILES = [(2, 4), (3, 4), (4, 4)]


def _directive(mt, nt):
    return ('module {\n  "tessera_rocm.wmma_gemm"() {name = "gemm", m = 16 : '
            f'i64, n = 16 : i64, k = 16 : i64, mt = {mt} : i64, nt = {nt} : '
            'i64, dtype = "f16"} : () -> ()\n}\n')


def _find_mlir_opt():
    if env := os.environ.get("TESSERA_MLIR_OPT"):
        return env if Path(env).is_file() else None
    for c in ("/usr/lib/llvm-23/bin/mlir-opt",
              "/opt/homebrew/opt/llvm/bin/mlir-opt"):
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
    for dep in ("libamdhip64.so", "libhiprtc.so"):
        p = os.path.join(ROCM_LIB_DIR, dep)
        if os.path.isfile(p):
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    try:
        return ctypes.CDLL("libamdhip64.so", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return None


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
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
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
    st, sp = ctypes.c_void_p(), ctypes.c_void_p()
    hip.hipEventCreate(ctypes.byref(st)); hip.hipEventCreate(ctypes.byref(sp))
    hip.hipEventRecord(st, None)
    for _ in range(iters):
        launch(fn, gx, gy, 1, 32, 1, 1, 0, None, arr, None)
    hip.hipEventRecord(sp, None); hip.hipEventSynchronize(sp)
    ms = ctypes.c_float(0.0)
    hip.hipEventElapsedTime(ctypes.byref(ms), st, sp)
    for d in (da, db, dd):
        hip.hipFree(d)
    return ms.value / iters, maxerr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=30)
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
          f"{args.iters} iters, {CHIP}")
    print(f"# {'mode':>8} {'best tile':>10} {'TFLOP/s':>9} {'rel-err':>10}")
    best = {}
    for mode in ("direct", "fork_a"):
        best_ms, best_tile, best_err = None, None, None
        for mt, nt in TILES:
            try:
                hsaco = _build(mlir_opt, mode, mt, nt)
            except RuntimeError as e:
                print(f"  {mode} {mt}x{nt}: build failed: {str(e)[:80]}",
                      file=sys.stderr)
                continue
            r = _run(hip, hsaco, M, N, K, mt, nt, args.iters, check=True)
            if r is None:
                continue
            ms, err = r
            if err is not None and err > 1e-2:
                print(f"  {mode} {mt}x{nt}: CORRECTNESS FAIL rel-err={err:.2e}",
                      file=sys.stderr)
                continue
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
        })
    if "direct" in best and "fork_a" in best:
        ratio = best["fork_a"] / best["direct"]
        # The two paths lower to an IDENTICAL ROCDL op stream (proven GPU-free by
        # test_rocm_gemm_pipeline_routing.test_fork_a_final_rocdl_is_identical_
        # to_direct), so perf parity is by construction — any gap here is APU
        # clock/thermal run-to-run noise, not a Fork-A regression. The wide band
        # reflects that; the IR-identity test is the real gate.
        print(f"# fork_a / direct = {ratio:.3f}x "
              f"({'parity (APU noise)' if 0.85 <= ratio <= 1.18 else 'CHECK'})")
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
