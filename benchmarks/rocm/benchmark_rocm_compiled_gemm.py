"""benchmark_rocm_compiled_gemm.py — device-timed perf of the COMPILER-generated
ROCm WMMA GEMM (Stage L2), swept over the register-blocked macro-tile (mt x nt).

Where ``benchmark_rocm_wmma_gemm.py`` times the *hand-written* HIPRTC kernel via
its ``_bench`` C-ABI symbol, this times the kernel the Tessera compiler GENERATES:

    "tessera_rocm.wmma_gemm"{m=n=k=16, mt, nt}
      --(tessera-opt --generate-wmma-gemm-kernel --lower-tessera-target-to-rocdl)
      --(mlir-opt finish-lower + rocdl-attach-target{gfx1151} + gpu-module-to-binary)
      --> hsaco  --(hipModuleLoadData + hipEvent-timed launch loop)--> TFLOP/s

For each (mt, nt) it reports the compiled kernel's TFLOP/s AND the hand-written
``tessera_rocm_wmma_gemm_f16_bench`` at the same (mt, nt), plus the ratio — so
the L2 convergence target ("the generated kernel reaches the hand-written
kernel's TFLOP/s") is a measured, side-by-side number, not a claim. Kernel-only
timing (buffers reused, no H2D/D2H in the loop), mirroring the hand-written
harness so the comparison is apples-to-apples.

This is the autotuner *sweep over the compiled kernel* — same brute-force ladder
the hand-written harness used to pick 2x4/3x4, now driving the generated path.

Honest gating: with no AMD GPU / tools / GEMM lib it prints a clear note and
emits an empty result set (exit 0) — it never fabricates GPU numbers. Emits the
stable benchmark JSON schema (Decision #12).

Usage::

    python benchmarks/rocm/benchmark_rocm_compiled_gemm.py
    python benchmarks/rocm/benchmark_rocm_compiled_gemm.py --size 2048 \
        --iters 50 --output rocm_compiled_gemm.json
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
GEMM_LIB = (REPO_ROOT / "build" / "src" / "compiler" / "codegen"
            / "Tessera_ROCM_Backend" / "runtime" / "hip"
            / "libtessera_rocm_gemm.so")
ROCM_LIB_DIR = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "lib")
CHIP = os.environ.get("TESSERA_ROCM_CHIP", "gfx1151")

# The macro-tiles to sweep: the L1 baseline (1x1), small (2x4), and the
# hand-written oracle's measured-best large tile (3x4) + the 4x4 occupancy cliff.
SWEEP_TILES = [(1, 1), (2, 2), (2, 4), (3, 4), (4, 4)]


def _find_mlir_opt():
    if env := os.environ.get("TESSERA_MLIR_OPT"):
        return env if Path(env).is_file() else None
    for c in ("/usr/lib/llvm-23/bin/mlir-opt",
              "/opt/homebrew/opt/llvm@23/bin/mlir-opt"):
        if Path(c).is_file():
            return c
    return shutil.which("mlir-opt")


def _directive(mt, nt):
    return (
        'module {\n'
        '  "tessera_rocm.wmma_gemm"() {name = "gemm", m = 16 : i64, '
        'n = 16 : i64, k = 16 : i64, '
        f'mt = {mt} : i64, nt = {nt} : i64}} : () -> ()\n'
        '}\n'
    )


def _extract_hsaco(text: str) -> bytes:
    i = text.index('bin = "') + len('bin = "')
    out = bytearray()
    j = i
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


def _build_hsaco(mlir_opt, mt, nt) -> bytes:
    gen = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-wmma-gemm-kernel",
         "--lower-tessera-target-to-rocdl"],
        input=_directive(mt, nt), capture_output=True, text=True)
    if gen.returncode != 0:
        raise RuntimeError(f"tessera-opt failed for {mt}x{nt}: {gen.stderr}")
    pipeline = (
        "builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
        f"reconcile-unrealized-casts),rocdl-attach-target{{chip={CHIP}}},"
        "gpu-module-to-binary)"
    )
    ser = subprocess.run([mlir_opt, f"--pass-pipeline={pipeline}"],
                         input=gen.stdout, capture_output=True, text=True)
    if ser.returncode != 0:
        raise RuntimeError(f"mlir-opt serialize failed for {mt}x{nt}: {ser.stderr}")
    return _extract_hsaco(ser.stdout)


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


def _time_compiled(hip, hsaco, M, N, K, mt, nt, iters):
    """hipEvent-timed mean ms per launch of the compiled kernel (buffers reused)."""
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        return None
    fn = ctypes.c_void_p()
    if hip.hipModuleGetFunction(ctypes.byref(fn), mod, b"gemm") != 0:
        return None
    da, db, dd = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    for d, nb in ((da, 2 * M * K), (db, 2 * K * N), (dd, 4 * M * N)):
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
    gx = (N + 16 * nt - 1) // (16 * nt)
    gy = (M + 16 * mt - 1) // (16 * mt)

    def fire():
        return launch(fn, gx, gy, 1, 32, 1, 1, 0, None, arr, None)

    for _ in range(3):  # warmup
        if fire() != 0:
            return None
    hip.hipDeviceSynchronize()
    start, stop = ctypes.c_void_p(), ctypes.c_void_p()
    hip.hipEventCreate(ctypes.byref(start))
    hip.hipEventCreate(ctypes.byref(stop))
    hip.hipEventRecord(start, None)
    for _ in range(iters):
        fire()
    hip.hipEventRecord(stop, None)
    hip.hipEventSynchronize(stop)
    ms = ctypes.c_float(0.0)
    hip.hipEventElapsedTime(ctypes.byref(ms), start, stop)
    hip.hipEventDestroy(start)
    hip.hipEventDestroy(stop)
    for d in (da, db, dd):
        hip.hipFree(d)
    return ms.value / iters


def _time_handwritten(lib, M, N, K, mt, nt, iters):
    fn = lib.tessera_rocm_wmma_gemm_f16_bench
    fn.argtypes = [ctypes.c_int] * 6 + [ctypes.POINTER(ctypes.c_double)]
    fn.restype = ctypes.c_int
    avg = ctypes.c_double(0.0)
    rc = fn(M, N, K, iters, mt, nt, ctypes.byref(avg))
    return avg.value if rc == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1024,
                    help="square M=N=K problem size")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()
    M = N = K = args.size

    if not TESSERA_OPT.is_file():
        print("[note] tessera-opt not built — nothing to time. "
              "Build: ninja -C build tessera-opt", file=sys.stderr)
        if args.output:
            Path(args.output).write_text("[]\n")
        return 0
    mlir_opt = _find_mlir_opt()
    if mlir_opt is None:
        print("[note] mlir-opt not found — set TESSERA_MLIR_OPT.", file=sys.stderr)
        if args.output:
            Path(args.output).write_text("[]\n")
        return 0
    hip = _load_hip()
    if hip is None or hip.hipInit(0) != 0:
        print("[note] no usable AMD GPU — refusing to fabricate perf numbers.",
              file=sys.stderr)
        if args.output:
            Path(args.output).write_text("[]\n")
        return 0
    lib = ctypes.CDLL(str(GEMM_LIB), mode=ctypes.RTLD_GLOBAL) \
        if GEMM_LIB.is_file() else None

    flop = 2.0 * M * N * K
    rows = []
    print(f"# compiled-vs-handwritten WMMA GEMM, {M}x{N}x{K}, "
          f"{args.iters} iters, {CHIP}")
    print(f"# {'mt x nt':>8} {'compiled TF/s':>14} {'handwritten TF/s':>17} "
          f"{'ratio':>7}")
    for mt, nt in SWEEP_TILES:
        try:
            hsaco = _build_hsaco(mlir_opt, mt, nt)
        except RuntimeError as e:
            print(f"  {mt}x{nt}: build failed: {e}", file=sys.stderr)
            continue
        cms = _time_compiled(hip, hsaco, M, N, K, mt, nt, args.iters)
        if cms is None:
            print(f"  {mt}x{nt}: compiled launch failed", file=sys.stderr)
            continue
        ctf = flop / (cms / 1e3) / 1e12
        hms = _time_handwritten(lib, M, N, K, mt, nt, args.iters) if lib else None
        htf = (flop / (hms / 1e3) / 1e12) if hms else None
        ratio = (ctf / htf) if htf else None
        print(f"  {mt}x{nt:>5} {ctf:>14.2f} "
              f"{(f'{htf:.2f}' if htf else 'n/a'):>17} "
              f"{(f'{ratio:.2f}x' if ratio else 'n/a'):>7}")
        rows.append({
            "backend": "rocm", "op": "gemm",
            "shape": [M, N, K], "dtype": "f16",
            "latency_ms": cms, "tflops": ctf,
            "memory_bw_gb_s": None, "device": CHIP,
            "tessera_version": "L2",
            "mt": mt, "nt": nt,
            "handwritten_tflops": htf,
            "compiled_vs_handwritten": ratio,
            "path": "compiler-generated",
        })

    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"# wrote {args.output} ({len(rows)} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
