"""Matched-kernel comparison for the SDK27 low-precision MPP ``matmul2d`` lane.

On the owning Mac (macOS 27, Metal 4.1) times, per shape and with the SAME
command model and the runtime's Metal 4 counter-heap device clock where a lane
has one:

  * ``mps_f16``            -- MPS fp16 GEMM, the production incumbent (wall clock only)
  * ``mtl4_matmul2d_f16``  -- MetalPerformancePrimitives fp16 on the matrix units
  * ``mtl4_matmul2d_lowp`` -- the same op with fp8 e4m3 / e5m2, fp4 e2m1, and half x e4m3
  * ``mtl4_matmul_sg_f32`` -- the existing hand-written ``simdgroup_matrix`` kernel (f32 in)
  * fused epilogue (bias + gelu) vs decomposed (plain + separate bias/act dispatch),
    for fp16 and for e4m3

and the host-side packing cost a caller pays to produce the low-precision
operands (numpy/ml_dtypes conversion per element count), which is what makes
"FP8 is faster" honest or not. Emits Decision #12 JSON rows (+ ``route`` and
the timing provenance) to ``--out``. Correctness is NOT re-proved here; see
``tests/unit/test_apple_gpu_lowp_*.py``. Informational: exits 0, skips off-Mac.

    PYTHONPATH=python python3 benchmarks/apple_gpu/benchmark_lowp_matmul2d.py --reps 12
"""
from __future__ import annotations

import argparse
import ctypes
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

Q = 256  # fp4 stride quantum in elements; all shapes below are multiples


def _device_name() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:  # noqa: BLE001
        return platform.machine()


def _codes(rng, rows, cols, fmt):
    if fmt == "fp4_e2m1":
        n = rng.integers(0, 16, size=(rows, cols * 2), dtype=np.uint8)
        return (n[:, ::2] | (n[:, 1::2] << 4)).astype(np.uint8)
    c = rng.integers(0, 256, size=(rows, cols), dtype=np.uint8)
    if fmt == "fp8_e4m3":
        c = (c & 0x87) | ((c >> 3) & 0xF) % 12 << 3
    else:
        c = (c & 0x83) | ((c >> 2) & 0x1F) % 24 << 2
    return c.astype(np.uint8)


class Timer:
    """Wall + device time (Metal 4 counter heap via the runtime telemetry)."""

    def __init__(self, rt):
        from tessera import _apple_gpu_dispatch as d
        self.d = d
        en = d.bind_registered("tessera_apple_gpu_dispatch_telemetry_set_enabled")
        if en is not None:
            en(1)
        self.enabled = en is not None

    def run(self, fn, dispatches=1, warmup=3, reps=12):
        for _ in range(warmup):
            fn()
        wall, dev, src = [], [], set()
        for _ in range(reps):
            t0 = time.perf_counter_ns()
            d_ns = fn()  # fn returns summed device ns for its dispatches (or None)
            wall.append((time.perf_counter_ns() - t0) / 1e6)
            if d_ns is not None:
                dev.append(d_ns / 1e6)
        return {
            "wall_ms_median": statistics.median(wall), "wall_ms_min": min(wall),
            "device_ms_median": statistics.median(dev) if dev else None,
            "device_ms_min": min(dev) if dev else None,
            "n_dispatches": dispatches, "reps": reps,
        }

    def device_ns(self):
        t = self.d.read_dispatch_telemetry()
        v = t.get("device_time_ns")
        return (int(v), t.get("timing_source")) if v is not None and v > 0 else (None, None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="512,1024,2048")
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--out", default="benchmarks/baselines/apple_macos27_20260914/lowp_matmul2d.json")
    args = ap.parse_args(argv)
    if platform.system() != "Darwin":
        print("[lowp-matmul2d] not Darwin -- skipping"); return 0
    from tessera import runtime as rt
    from tessera import __version__
    if not rt.apple_gpu_mtl4_matmul2d_lowp_available():
        print("[lowp-matmul2d] low-precision matmul2d lane unavailable (needs macOS 27 + SDK27 dylib) -- skipping")
        return 0
    timer = Timer(rt)
    device = _device_name()
    mps_f16 = rt._apple_gpu_mps_matmul_f16()
    u16 = ctypes.POINTER(ctypes.c_uint16)
    rng = np.random.default_rng(0)
    rows = []
    sources = {}

    def row(op, shape, dtype, route, stats, extra=None):
        M, N, K = shape
        lat = stats["device_ms_median"] if stats["device_ms_median"] is not None else stats["wall_ms_median"]
        r = {
            "backend": "apple_gpu", "op": op, "shape": [M, N, K], "dtype": dtype,
            "latency_ms": lat, "tflops": 2.0 * M * N * K / (lat * 1e-3) / 1e12,
            "memory_bw_gb_s": None, "device": device, "tessera_version": __version__,
            "route": route, "latency_source": "device_counter_heap" if stats["device_ms_median"] is not None else "wall",
            **stats,
        }
        if extra:
            r.update(extra)
        rows.append(r)
        print(f"{op:18s} {str(shape):18s} {dtype:16s} {route:30s} "
              f"dev {lat:8.3f} ms  wall {stats['wall_ms_median']:8.3f} ms  {r['tflops']:6.2f} TFLOP/s")

    for n in (int(x) for x in args.shapes.split(",")):
        M = N = K = n
        A32 = (rng.standard_normal((M, K)) * 0.25).astype(np.float32)
        B32 = (rng.standard_normal((K, N)) * 0.25).astype(np.float32)
        A16, B16 = A32.astype(np.float16), B32.astype(np.float16)
        bias = rng.standard_normal(N).astype(np.float32)
        shape = (M, N, K)

        # ---- packing cost (host): what a caller pays to produce each operand
        pack = {}
        t0 = time.perf_counter_ns(); A32.astype(np.float16); B32.astype(np.float16)
        pack["pack_f16_ms"] = (time.perf_counter_ns() - t0) / 1e6
        try:
            import ml_dtypes
            for fmt, dt in (("fp8_e4m3", ml_dtypes.float8_e4m3fn), ("fp8_e5m2", ml_dtypes.float8_e5m2),
                            ("fp4_e2m1", ml_dtypes.float4_e2m1fn)):
                t0 = time.perf_counter_ns()
                a = A32.astype(dt); b = B32.astype(dt)
                if fmt == "fp4_e2m1":  # nibble-pack
                    av = a.view(np.uint8); (av[:, ::2] | (av[:, 1::2] << 4))
                    bv = b.view(np.uint8); (bv[:, ::2] | (bv[:, 1::2] << 4))
                pack[f"pack_{fmt}_ms"] = (time.perf_counter_ns() - t0) / 1e6
        except ImportError:
            pack["pack_note"] = "ml_dtypes unavailable; low-precision host pack not timed"

        # ---- MPS fp16 incumbent (classic command buffer; wall clock)
        if mps_f16 is not None:
            C16 = np.empty((M, N), np.float16)
            def f_mps():
                mps_f16(A16.ctypes.data_as(u16), B16.ctypes.data_as(u16), C16.ctypes.data_as(u16),
                        ctypes.c_int32(M), ctypes.c_int32(N), ctypes.c_int32(K))
                return None
            row("matmul", shape, "f16->f16", "mps_matmul_f16", timer.run(f_mps, reps=args.reps), pack)

        # ---- MPP fp16
        def f_mpp16():
            C, ran = rt.apple_gpu_mtl4_matmul2d_f16(A16, B16, np)
            assert ran
            ns, src = timer.device_ns(); sources["mtl4"] = src
            return ns
        row("matmul", shape, "f16->f32", "mtl4_matmul2d_f16", timer.run(f_mpp16, reps=args.reps), pack)

        # ---- MPP low precision
        for fmt, both in (("fp8_e4m3", True), ("fp8_e5m2", True), ("fp4_e2m1", True), ("fp8_e4m3", False)):
            bits = 4 if fmt == "fp4_e2m1" else 8
            Ac = A16 if not both else _codes(rng, M, K * bits // 8, fmt)
            Bc = _codes(rng, K, N * bits // 8, fmt)
            def f_lowp(Ac=Ac, Bc=Bc, fmt=fmt, both=both):
                rt.apple_gpu_mtl4_matmul2d_lowp(Ac, Bc, np, fmt=fmt, M=M, N=N, K=K,
                                                a_dtype="lowp" if both else "f16")
                ns, _ = timer.device_ns(); return ns
            dtype = f"{fmt}->f32" if both else f"f16x{fmt}->f32"
            row("matmul", shape, dtype, f"mtl4_matmul2d_lowp:{fmt}{'' if both else ':half_left'}",
                timer.run(f_lowp, reps=args.reps), pack)

        # ---- existing simdgroup kernel (f32 operands; fast path when 64/64/16-aligned)
        def f_sg():
            C, ran = rt.apple_gpu_mtl4_matmul_sg(A32, B32, np)
            assert ran
            ns, _ = timer.device_ns(); return ns
        row("matmul", shape, "f32->f32", "mtl4_matmul_sg_fast_f32", timer.run(f_sg, reps=args.reps), pack)

        # ---- fused vs decomposed epilogue (bias + gelu)
        def f_epi16():
            C, ran = rt.apple_gpu_mtl4_matmul2d_epilogue(A16, B16, np, bias=bias, act="gelu", dtype="f16")
            assert ran
            ns, _ = timer.device_ns(); return ns
        row("matmul_bias_gelu", shape, "f16->f32", "mtl4_matmul2d_epilogue_f16:fused",
            timer.run(f_epi16, reps=args.reps))

        def f_dec16():
            C, ran = rt.apple_gpu_mtl4_matmul2d_f16(A16, B16, np); assert ran
            n1, _ = timer.device_ns()
            rt.apple_gpu_mtl4_bias_act_f32(C, np, bias=bias, act="gelu")
            n2, _ = timer.device_ns()
            return None if n1 is None or n2 is None else n1 + n2
        row("matmul_bias_gelu", shape, "f16->f32", "mtl4_matmul2d_f16+bias_act:decomposed",
            timer.run(f_dec16, dispatches=2, reps=args.reps))

        Ae = _codes(rng, M, K, "fp8_e4m3"); Be = _codes(rng, K, N, "fp8_e4m3")
        def f_epi8():
            rt.apple_gpu_mtl4_matmul2d_lowp(Ae, Be, np, fmt="fp8_e4m3", M=M, N=N, K=K, bias=bias, act="gelu")
            ns, _ = timer.device_ns(); return ns
        row("matmul_bias_gelu", shape, "fp8_e4m3->f32", "mtl4_matmul2d_lowp_epilogue:e4m3:fused",
            timer.run(f_epi8, reps=args.reps))

        def f_dec8():
            C = rt.apple_gpu_mtl4_matmul2d_lowp(Ae, Be, np, fmt="fp8_e4m3", M=M, N=N, K=K)
            n1, _ = timer.device_ns()
            rt.apple_gpu_mtl4_bias_act_f32(C, np, bias=bias, act="gelu")
            n2, _ = timer.device_ns()
            return None if n1 is None or n2 is None else n1 + n2
        row("matmul_bias_gelu", shape, "fp8_e4m3->f32", "mtl4_matmul2d_lowp:e4m3+bias_act:decomposed",
            timer.run(f_dec8, dispatches=2, reps=args.reps))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "lane": "sdk27_lowp_matmul2d_matched_kernels", "device": device,
        "os": platform.mac_ver()[0], "telemetry_enabled": timer.enabled,
        "timing_sources_seen": {k: v for k, v in sources.items()},
        "note": ("device_ms = Metal 4 counter-heap interval per dispatch (summed for decomposed routes); "
                 "MPS has no device clock here (wall only). Host pack_* columns are numpy/ml_dtypes "
                 "conversion time for BOTH operands at that shape. Correctness proven separately."),
        "rows": rows}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
