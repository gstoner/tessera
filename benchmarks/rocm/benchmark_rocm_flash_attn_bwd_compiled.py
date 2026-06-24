"""benchmark_rocm_flash_attn_bwd_compiled.py — device-timed perf ladder of the
COMPILER-generated ROCm WMMA flash-attention BACKWARD (fa_pre + fa_dkdv + fa_dq).

Companion to benchmark_rocm_flash_attn_compiled.py (forward). Builds the three
backward kernels per head_dim (tessera-opt, in-process) and hipEvent-times the
full backward (pre -> dkdv -> dq, one timed region) across (head_dim, seqlen),
reporting TFLOP/s. FA backward FLOPs ~= 5 matmuls of 2*B*H*Sq*Sk*D each (S, dP,
dV, dK, dQ) -> ~10*B*H*Sq*Sk*D (the scalar logsumexp pre-pass is O(Sq*Sk*D) and
counted in wall-clock but not the FLOP rate; non-causal full attention).

Honest gating: no GPU / tools -> a clear note + empty result set (exit 0). The
compiled backward is correctness-first (recompute S/P per tile, no double-
buffering / causal loop-bound) — this ladder characterizes THAT kernel; it is
not a tuned attention-backward kernel. Emits the stable benchmark JSON
(Decision #12).

Usage::

    python benchmarks/rocm/benchmark_rocm_flash_attn_bwd_compiled.py
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

# (head_dim, B, H, Sq, Sk). Non-causal full attention for clean FLOP accounting.
LADDER = [
    (64, 4, 8, 512, 512),
    (64, 2, 8, 1024, 1024),
    (128, 2, 8, 1024, 1024),
    (64, 1, 8, 2048, 2048),
]


def _find_mlir_opt():
    if env := os.environ.get("TESSERA_MLIR_OPT"):
        return env if Path(env).is_file() else None
    for c in ("/usr/lib/llvm-22/bin/mlir-opt",
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


def _build(mlir_opt, head_dim):
    directive = ('module {\n  "tessera_rocm.flash_attn_bwd"() {name = "fa", '
                 f'head_dim = {head_dim} : i64, dtype = "f16"}} : () -> ()\n}}\n')
    g = subprocess.run(
        [str(TESSERA_OPT), "-", "--generate-wmma-flash-attn-bwd-kernel",
         "--lower-tessera-target-to-rocdl"],
        input=directive, capture_output=True, text=True)
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


def _arr(args):
    a = (ctypes.c_void_p * len(args))()
    for i, v in enumerate(args):
        a[i] = ctypes.cast(ctypes.byref(v), ctypes.c_void_p)
    return a


def _time(hip, hsaco, D, B, H, Sq, Sk, iters):
    mod = ctypes.c_void_p()
    if hip.hipModuleLoadData(ctypes.byref(mod), hsaco) != 0:
        return None
    fns = {}
    for nm in (b"fa_pre", b"fa_dkdv", b"fa_dq"):
        fn = ctypes.c_void_p()
        if hip.hipModuleGetFunction(ctypes.byref(fn), mod, nm) != 0:
            return None
        fns[nm] = fn
    nQ, nKV, nL = B * H * Sq * D, B * H * Sk * D, B * H * Sq
    d = {}
    for name, nb in (("Q", 2 * nQ), ("K", 2 * nKV), ("V", 2 * nKV),
                     ("dO", 2 * nQ), ("O", 4 * nQ), ("L", 4 * nL),
                     ("Dd", 4 * nL), ("dQ", 4 * nQ), ("dK", 4 * nKV),
                     ("dV", 4 * nKV)):
        p = ctypes.c_void_p()
        if hip.hipMalloc(ctypes.byref(p), nb) != 0:
            return None
        d[name] = p
    Sqc, Skc = ctypes.c_int64(Sq), ctypes.c_int64(Sk)
    sc, cau = ctypes.c_float(1.0 / (D ** 0.5)), ctypes.c_int64(0)
    launch = hip.hipModuleLaunchKernel
    launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                       + [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p])
    gyt, gqt, gkt = B * H, (Sq + 15) // 16, (Sk + 15) // 16
    pre = (_mr(d["Q"], nQ) + _mr(d["K"], nKV) + _mr(d["dO"], nQ)
           + _mr(d["O"], nQ) + _mr(d["L"], nL) + _mr(d["Dd"], nL)
           + [Sqc, Skc, sc, cau])
    dkdv = (_mr(d["Q"], nQ) + _mr(d["K"], nKV) + _mr(d["V"], nKV)
            + _mr(d["dO"], nQ) + _mr(d["L"], nL) + _mr(d["Dd"], nL)
            + _mr(d["dK"], nKV) + _mr(d["dV"], nKV) + [Sqc, Skc, sc, cau])
    dq = (_mr(d["Q"], nQ) + _mr(d["K"], nKV) + _mr(d["V"], nKV)
          + _mr(d["dO"], nQ) + _mr(d["L"], nL) + _mr(d["Dd"], nL)
          + _mr(d["dQ"], nQ) + [Sqc, Skc, sc, cau])
    pa, da, qa = _arr(pre), _arr(dkdv), _arr(dq)

    def run():
        return (launch(fns[b"fa_pre"], gqt, gyt, 1, 32, 1, 1, 0, None, pa, None)
                | launch(fns[b"fa_dkdv"], gkt, gyt, 1, 32, 1, 1, 0, None, da,
                         None)
                | launch(fns[b"fa_dq"], gqt, gyt, 1, 32, 1, 1, 0, None, qa,
                         None))
    for _ in range(3):
        if run() != 0:
            return None
    hip.hipDeviceSynchronize()
    st, sp = ctypes.c_void_p(), ctypes.c_void_p()
    hip.hipEventCreate(ctypes.byref(st)); hip.hipEventCreate(ctypes.byref(sp))
    hip.hipEventRecord(st, None)
    for _ in range(iters):
        run()
    hip.hipEventRecord(sp, None); hip.hipEventSynchronize(sp)
    ms = ctypes.c_float(0.0)
    hip.hipEventElapsedTime(ctypes.byref(ms), st, sp)
    for p in d.values():
        hip.hipFree(p)
    return ms.value / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

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

    rows = []
    print(f"# compiled flash_attn BACKWARD ladder, {args.iters} iters, {CHIP} "
          f"(correctness-first kernels)")
    print(f"# {'D':>4} {'B*H':>5} {'Sq':>6} {'Sk':>6} {'ms':>9} {'TFLOP/s':>9}")
    cache: dict[int, bytes] = {}
    for D, B, H, Sq, Sk in LADDER:
        try:
            hsaco = cache.get(D) or _build(mlir_opt, D)
            cache[D] = hsaco
        except RuntimeError as e:
            print(f"  D={D}: build failed: {str(e)[:80]}", file=sys.stderr)
            continue
        ms = _time(hip, hsaco, D, B, H, Sq, Sk, args.iters)
        if ms is None:
            print(f"  D={D} {B}x{H} {Sq}x{Sk}: launch failed", file=sys.stderr)
            continue
        flop = 10.0 * B * H * Sq * Sk * D
        tf = flop / (ms / 1e3) / 1e12
        print(f"  {D:>4} {B*H:>5} {Sq:>6} {Sk:>6} {ms:>9.3f} {tf:>9.2f}")
        rows.append({
            "backend": "rocm", "op": "flash_attn_bwd",
            "shape": [B, H, Sq, Sk, D], "dtype": "f16",
            "latency_ms": ms, "tflops": tf, "memory_bw_gb_s": None,
            "device": CHIP, "tessera_version": "fa-bwd-ladder",
            "path": "compiler-generated",
        })
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
