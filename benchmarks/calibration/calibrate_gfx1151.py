#!/usr/bin/env python3
"""P1b calibration sweep for gfx1151 (Strix Halo / Radeon 8060S).

Measures the two fields `target_perf.py`'s radeon_8060s profile deliberately
left absent ("calibrate rather than pick a convention"):

  * ``peak_tflops.bf16:matrix`` / ``peak_tflops.fp16:matrix`` — a
    register-resident WMMA microbenchmark: 8 independent accumulator chains per
    wave (ILP to cover WMMA latency), zero memory traffic in the timed loop.
    This is the *achievable device ceiling* in the TileSight Table-3 sense
    (measured-vs-spec), not a kernel-quality number.
  * ``dram_bw_gbps`` — a grid-stride vector<4xf32> copy over 256 MiB,
    counting read+write bytes.

Timing is WALL-CLOCK around launch + hipDeviceSynchronize (median of reps,
warmup discarded): WSL /dev/dxg device events return garbage while reporting
success (measured 2026-07; see the rocm-gemm-roofline note), so device-event
timing is deliberately not used. Wall-clock over long kernels (>0.2 s) makes
launch overhead negligible.

Output: a `calibration_corpus` JSON (target_perf.apply_corpus schema v1)
written to benchmarks/baselines/, then verified by loading it and asking
SchedulePlanner.for_target("rocm_gfx1151") — the call that raised before this
sweep existed.

Compile/launch machinery mirrors tests/unit/test_rocm_gemm_staged_async_copy.py
(tessera-rocm-opt → mlir-opt gpu-module-to-binary → ctypes HIP).
"""

from __future__ import annotations

import ctypes
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO))

from tests._support.rocm_build import rocm_opt_path  # noqa: E402

CHIP = os.environ.get("TESSERA_ROCM_CHIP", "gfx1151")
WS = "#gpu.address_space<workgroup>"


# ── compile/launch machinery (pattern-proven in the staged-WMMA device test) ──

def find_mlir_opt():
    for c in ("/usr/lib/llvm-23/bin/mlir-opt",
              "/opt/homebrew/opt/llvm@23/bin/mlir-opt"):
        if Path(c).is_file():
            return c
    import shutil
    return os.environ.get("TESSERA_MLIR_OPT") or shutil.which("mlir-opt")


def extract_hsaco(text: str) -> bytes:
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


def hip_lib():
    rocm_lib = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "lib")
    for dep in ("libamdhip64.so", "libhiprtc.so"):
        p = os.path.join(rocm_lib, dep)
        if os.path.isfile(p):
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_LOCAL)
            except OSError:
                pass
    return ctypes.CDLL("libamdhip64.so", mode=ctypes.RTLD_LOCAL)


def compile_kernel(mlir: str, extra_rocm_passes: list[str] | None = None) -> bytes:
    rocm_opt = rocm_opt_path()
    mlir_opt = find_mlir_opt()
    if rocm_opt is None or mlir_opt is None:
        raise SystemExit("need tessera-rocm-opt + mlir-opt built")
    passes = (extra_rocm_passes or []) + ["--lower-tessera-target-to-rocdl"]
    lowered = subprocess.run([str(rocm_opt), "-", *passes],
                             input=mlir, capture_output=True, text=True)
    if lowered.returncode != 0:
        raise RuntimeError(f"rocm-opt failed:\n{lowered.stderr[:800]}")
    pl = ("builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
          f"reconcile-unrealized-casts),rocdl-attach-target{{chip={CHIP}}},"
          "gpu-module-to-binary)")
    ser = subprocess.run([mlir_opt, f"--pass-pipeline={pl}"],
                         input=lowered.stdout, capture_output=True, text=True)
    if ser.returncode != 0 or "gpu.binary" not in ser.stdout:
        raise RuntimeError(f"serialize failed:\n{ser.stderr[:800]}")
    hsaco = extract_hsaco(ser.stdout)
    assert hsaco[:4] == b"\x7fELF"
    return hsaco


def memref_args(ptr, n):
    return [ctypes.c_void_p(ptr.value), ctypes.c_void_p(ptr.value),
            ctypes.c_int64(0), ctypes.c_int64(n), ctypes.c_int64(1)]


class Launcher:
    def __init__(self, hip, hsaco: bytes, name: bytes):
        self.hip = hip
        self.mod = ctypes.c_void_p()
        if hip.hipModuleLoadData(ctypes.byref(self.mod), hsaco) != 0:
            raise SystemExit("no usable AMD GPU (hipModuleLoadData failed)")
        self.fn = ctypes.c_void_p()
        assert hip.hipModuleGetFunction(ctypes.byref(self.fn), self.mod, name) == 0
        self.launch = hip.hipModuleLaunchKernel
        self.launch.argtypes = ([ctypes.c_void_p] + [ctypes.c_uint] * 6
                                + [ctypes.c_uint, ctypes.c_void_p,
                                   ctypes.c_void_p, ctypes.c_void_p])

    def run_timed(self, grid, block, args, reps=5) -> float:
        arr = (ctypes.c_void_p * len(args))()
        for i, x in enumerate(args):
            arr[i] = ctypes.cast(ctypes.byref(x), ctypes.c_void_p)
        times = []
        for rep in range(reps + 1):  # first is warmup
            t0 = time.perf_counter()
            assert self.launch(self.fn, grid, 1, 1, block, 1, 1,
                               0, None, arr, None) == 0
            assert self.hip.hipDeviceSynchronize() == 0
            t1 = time.perf_counter()
            if rep > 0:
                times.append(t1 - t0)
        return statistics.median(times)


# ── kernels ──────────────────────────────────────────────────────────────────

def wmma_peak_mlir(elem: str) -> str:
    """8 independent WMMA accumulator chains per wave, register-resident."""
    chains = 8
    inits = "\n".join(
        f"      %z{k} = arith.constant dense<0.0> : vector<8xf32>"
        for k in range(chains))
    iter_args = ", ".join(f"%acc{k} = %z{k}" for k in range(chains))
    types = ", ".join(["vector<8xf32>"] * chains)
    body = "\n".join(
        f"        %n{k} = tessera_rocm.wmma %a, %b, %acc{k}"
        f" : vector<16x{elem}>, vector<16x{elem}>, vector<8xf32>"
        f" -> vector<8xf32>"
        for k in range(chains))
    yields = ", ".join(f"%n{k}" for k in range(chains))
    results = ", ".join(f"%r{k}" for k in range(chains))
    adds = "\n".join(
        f"      %s{k} = arith.addf %s{k - 1}, %r{k} : vector<8xf32>"
        for k in range(1, chains))
    return f"""
module {{
  gpu.module @m {{
    gpu.func @wmma_peak(%out: memref<?xf32>, %iters: i64) kernel {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %a = arith.constant dense<1.0> : vector<16x{elem}>
      %b = arith.constant dense<1.0> : vector<16x{elem}>
{inits}
      %n = arith.index_cast %iters : i64 to index
      {results} = scf.for %i = %c0 to %n step %c1
          iter_args({iter_args}) -> ({types}) {{
{body}
        scf.yield {yields} : {types}
      }}
      %s0 = arith.addf %r0, %r0 : vector<8xf32>
{adds}
      %tid = gpu.thread_id x
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %base = arith.muli %bid, %bdim : index
      %gid = arith.addi %base, %tid : index
      %v = vector.extract %s{chains - 1}[0] : f32 from vector<8xf32>
      memref.store %v, %out[%gid] : memref<?xf32>
      gpu.return
    }}
  }}
}}
"""


def copy_bw_mlir() -> str:
    """Grid-stride vector<4xf32> copy: dst[i] = src[i]."""
    return """
module {
  gpu.module @m {
    gpu.func @copy_bw(%src: memref<?xf32>, %dst: memref<?xf32>, %n4: i64) kernel {
      %c4 = arith.constant 4 : index
      %tid = gpu.thread_id x
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %gdim = gpu.grid_dim x
      %base = arith.muli %bid, %bdim : index
      %gid = arith.addi %base, %tid : index
      %stride = arith.muli %gdim, %bdim : index
      %n = arith.index_cast %n4 : i64 to index
      scf.for %i = %gid to %n step %stride {
        %addr = arith.muli %i, %c4 : index
        %v = vector.load %src[%addr] : memref<?xf32>, vector<4xf32>
        vector.store %v, %dst[%addr] : memref<?xf32>, vector<4xf32>
      }
      gpu.return
    }
  }
}
"""


# ── measurements ─────────────────────────────────────────────────────────────

def measure_wmma_peak(hip, elem: str) -> float:
    hsaco = compile_kernel(wmma_peak_mlir(elem))
    lch = Launcher(hip, hsaco, b"wmma_peak")
    blocks, threads = 320, 256           # 8 waves/block -> 2560 waves total
    waves = blocks * (threads // 32)
    chains, flop_per_wmma = 8, 2 * 16 * 16 * 16
    nfloat = blocks * threads
    out = ctypes.c_void_p()
    assert hip.hipMalloc(ctypes.byref(out), 4 * nfloat) == 0
    # Auto-scale iters to ~0.3 s using a probe run.
    iters = 2000
    args = memref_args(out, nfloat) + [ctypes.c_int64(iters)]
    t = lch.run_timed(blocks, threads, args, reps=2)
    iters = max(2000, int(iters * 0.3 / max(t, 1e-6)))
    args = memref_args(out, nfloat) + [ctypes.c_int64(iters)]
    t = lch.run_timed(blocks, threads, args, reps=5)
    hip.hipFree(out)
    tflops = waves * iters * chains * flop_per_wmma / t / 1e12
    print(f"  wmma {elem}: {tflops:.2f} TFLOP/s "
          f"(iters={iters}, waves={waves}, t={t * 1e3:.1f} ms)")
    return tflops


def measure_dram_bw(hip) -> float:
    hsaco = compile_kernel(copy_bw_mlir())
    lch = Launcher(hip, hsaco, b"copy_bw")
    nfloat = 64 * 1024 * 1024            # 256 MiB per buffer
    nbytes = 4 * nfloat
    src, dst = ctypes.c_void_p(), ctypes.c_void_p()
    assert hip.hipMalloc(ctypes.byref(src), nbytes) == 0
    assert hip.hipMalloc(ctypes.byref(dst), nbytes) == 0
    hip.hipMemset(src, 1, nbytes)
    blocks, threads = 2048, 256
    args = (memref_args(src, nfloat) + memref_args(dst, nfloat)
            + [ctypes.c_int64(nfloat // 4)])
    t = lch.run_timed(blocks, threads, args, reps=7)
    for d in (src, dst):
        hip.hipFree(d)
    gbps = 2 * nbytes / t / 1e9
    print(f"  dram copy: {gbps:.1f} GB/s (read+write, 256 MiB, t={t * 1e3:.2f} ms)")
    return gbps


def main() -> None:
    hip = hip_lib()
    if hip.hipInit(0) != 0:
        raise SystemExit("hipInit failed — no ROCm host")
    print(f"calibrating {CHIP} (wall-clock; WSL device events unusable)")

    results: dict[str, float] = {}
    results["dram_bw_gbps"] = round(measure_dram_bw(hip), 1)
    for elem, key in (("f16", "peak_tflops.fp16:matrix"),
                      ("bf16", "peak_tflops.bf16:matrix")):
        try:
            results[key] = round(measure_wmma_peak(hip, elem), 2)
        except RuntimeError as e:
            print(f"  wmma {elem}: NOT MEASURED ({str(e)[:120]}...)")

    corpus = {
        "kind": "calibration_corpus",
        "version": 1,
        "measured_on": date.today().isoformat(),
        "host": "strix-halo-wsl2",
        "method": ("wall-clock median-of-reps (WSL /dev/dxg device events "
                   "return garbage); WMMA peak = 8-chain register-resident "
                   "microbenchmark; BW = grid-stride vector<4xf32> copy, "
                   "read+write bytes over 256 MiB"),
        "devices": {"radeon_8060s": results},
    }
    out_path = (REPO / "benchmarks" / "baselines" /
                f"rocm_gfx1151_calibration_{date.today().isoformat().replace('-', '_')}.json")
    out_path.write_text(json.dumps(corpus, indent=2) + "\n")
    print(f"corpus written: {out_path.relative_to(REPO)}")

    # Verification: the corpus loads, and the call that raised now succeeds.
    from tessera.compiler import target_perf
    updated = target_perf.load_corpus(out_path)
    print(f"apply_corpus updated: {updated}")
    if "peak_tflops.bf16:matrix" in results:
        from tessera.compiler.schedule_planner import SchedulePlanner
        planner = SchedulePlanner.for_target("rocm_gfx1151")
        print(f"SchedulePlanner.for_target('rocm_gfx1151') -> OK ({planner!r})")


if __name__ == "__main__":
    main()
