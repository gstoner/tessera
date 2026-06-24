"""benchmark_rocm_compiled_gemm_dtype.py — device-timed dtype sweep of the
COMPILER-generated ROCm WMMA GEMM (f16 / bf16 / int8 / int4).

Companion to ``benchmark_rocm_compiled_gemm.py`` (which sweeps the macro-tile for
f16). This sweeps the *dtype* the compiler generates (`dtype=` on the
`tessera_rocm.wmma_gemm` directive) and reports throughput per dtype, picking the
best macro-tile among a small set. int4 values are int8 containers ([-8,7]),
nibble-packed in-kernel.

Key finding it documents (gfx1151 / RDNA 3.5): the iu8/iu4 WMMA runs at the SAME
matrix-op rate as f16 — there is no low-precision FLOP-rate multiplier on RDNA
3.5 — so the compiled int paths are already compute-competitive with f16, and
int4's win over int8 is memory footprint (half), not speed.

Kernel-only timing (buffers reused, no H2D/D2H in the loop). Honest gating: no
GPU / tools -> a clear note + empty result set (exit 0), never fabricated
numbers. Emits the stable benchmark JSON schema (Decision #12).

Usage::

    python benchmarks/rocm/benchmark_rocm_compiled_gemm_dtype.py --size 2048
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

# (dtype tag, input element bytes). int4/int8 -> i32 out; f16/bf16 -> f32 out.
DTYPES = [("f16", 2), ("bf16", 2), ("int8", 1), ("int4", 1)]
TILES = [(2, 4), (3, 4), (4, 4)]


def _find_mlir_opt():
    if env := os.environ.get("TESSERA_MLIR_OPT"):
        return env if Path(env).is_file() else None
    for c in ("/usr/lib/llvm-22/bin/mlir-opt",
              "/opt/homebrew/opt/llvm/bin/mlir-opt"):
        if Path(c).is_file():
            return c
    return shutil.which("mlir-opt")


def _directive(dt, mt, nt):
    return (
        'module {\n  "tessera_rocm.wmma_gemm"() {name = "gemm", m = 16 : i64, '
        f'n = 16 : i64, k = 16 : i64, mt = {mt} : i64, nt = {nt} : i64, '
        f'dtype = "{dt}"}} : () -> ()\n}}\n'
    )


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


def _build(mlir_opt, dt, mt, nt):
    g = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-wmma-gemm-kernel",
         "--lower-tessera-target-to-rocdl"],
        input=_directive(dt, mt, nt), capture_output=True, text=True)
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


def _time(hip, hsaco, M, N, K, mt, nt, ab, iters):
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        return None
    fn = ctypes.c_void_p()
    if hip.hipModuleGetFunction(ctypes.byref(fn), mod, b"gemm") != 0:
        return None
    da, db, dd = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    for d, nb in ((da, ab * M * K), (db, ab * K * N), (dd, 4 * M * N)):
        if hip.hipMalloc(ctypes.byref(d), nb) != 0:
            return None
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
    return ms.value / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()
    M = N = K = args.size

    if not TESSERA_OPT.is_file():
        print("[note] tessera-opt not built — nothing to time.", file=sys.stderr)
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
    print(f"# compiled GEMM dtype sweep, {M}x{N}x{K}, {args.iters} iters, {CHIP}")
    print(f"# {'dtype':>6} {'best tile':>10} {'TOP/s (TFLOP/s for f16)':>24}")
    for dt, ab in DTYPES:
        best_ms, best_tile = None, None
        for mt, nt in TILES:
            try:
                hsaco = _build(mlir_opt, dt, mt, nt)
            except RuntimeError as e:
                print(f"  {dt}: build failed: {str(e)[:80]}", file=sys.stderr)
                continue
            ms = _time(hip, hsaco, M, N, K, mt, nt, ab, args.iters)
            if ms is not None and (best_ms is None or ms < best_ms):
                best_ms, best_tile = ms, (mt, nt)
        if best_ms is None:
            print(f"  {dt}: launch failed", file=sys.stderr)
            continue
        tops = flop / (best_ms / 1e3) / 1e12
        print(f"  {dt:>6} {str(best_tile):>10} {tops:>20.2f}")
        rows.append({
            "backend": "rocm", "op": "gemm", "shape": [M, N, K], "dtype": dt,
            "latency_ms": best_ms, "tflops": tops, "memory_bw_gb_s": None,
            "device": CHIP, "tessera_version": "int-dtype-sweep",
            "mt": best_tile[0], "nt": best_tile[1], "path": "compiler-generated",
        })
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
