"""Paired admission corpus for the compiled Metal 4 matmul2d route.

APPLE-MATMUL2D-1 follow-through. For every (operand pair, shape) row the
CANDIDATE is the compiled route exactly as the compiler emits it -- the Graph
IR matmul lowered by `tessera-opt` through the shared tiling pass and the
Apple matmul2d passes, the value call extracted by the driver, and executed by
the value-lane dispatcher on the projected view ABI -- and the INCUMBENT is
what production dispatches for that dtype today (`runtime.py` lane table):
MPS fp16 GEMM for f16 (the MTL4 f16 GEMV route for M == 1), the MTL4 bf16
route for bf16, and for the low-precision pairs the fp16 MPS GEMM on the SAME
quantized values (there is no production low-precision route to displace).

The two routes run interleaved, order alternating per repetition, from the
same operands; each repetition yields one incumbent/candidate wall-time
ratio. The row reports the median ratio and a 95% bootstrap lower bound on
the median, and is ADMITTED only when that lower bound clears 1.0 + margin
(stability_gates_must_converge: a confidence bound, never a range). Both
outputs are checked against a float64 reference first; a wrong or nonfinite
result fails the row before it is timed. Host packing cost for the
low-precision operands is recorded separately and never hidden in a ratio.

Wall time is host -> ctypes -> GPU -> host, the level at which the routes
compete. Two independent processes make a packet; one run is characterization.

    PYTHONPATH=python:. python3 benchmarks/apple_gpu/benchmark_matmul2d_route_corpus.py --reps 20 --out run_0.json

Informational: exits 0, skips off-Mac or without the Metal 4 stack. No
promotion follows from this script; the plan record decides admission.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PASSES = ("tessera-tiling", "tessera-apple-canonical-gemm-matmul2d",
          "tessera-apple-matmul2d-fuse-epilogue", "tessera-apple-matmul2d-to-call")
MARGIN = 0.02

# (name, M, N, K). Low-precision rows need K and N on Apple's 128-element FP8
# quantum; the ragged row is 16-bit only.
SHAPES = [("square_512", 512, 512, 512), ("square_1024", 1024, 1024, 1024), ("square_2048", 2048, 2048, 2048),
          ("mlp_512x4096x1024", 512, 4096, 1024), ("decode_m1_4096", 1, 4096, 4096),
          ("ragged_1000x1000x1024", 1000, 1000, 1024)]
PAIRS = [("f16", "f16"), ("bf16", "bf16"), ("f8E4M3FN", "f8E4M3FN"), ("f16", "f8E4M3FN")]


def _gemm_module(m, k, n, a, b):
    return (f"module {{\n  func.func @gemm(%a: tensor<{m}x{k}x{a}>, %b: tensor<{k}x{n}x{b}>) -> tensor<{m}x{n}xf32> {{\n"
            f'    %0 = "tessera.matmul"(%a, %b) : (tensor<{m}x{k}x{a}>, tensor<{k}x{n}x{b}>) -> tensor<{m}x{n}xf32>\n'
            f"    return %0 : tensor<{m}x{n}xf32>\n  }}\n}}")


def _find_tessera_opt():
    for cand in (ROOT / "build/tools/tessera-opt/tessera-opt", ROOT / "build/bin/tessera-opt"):
        if cand.exists():
            return cand
    return None


def _bootstrap_lower(ratios, seed=0, resamples=2000, q=0.025):
    rng = np.random.default_rng(seed)
    r = np.asarray(ratios, np.float64)
    meds = np.median(rng.choice(r, size=(resamples, r.size), replace=True), axis=1)
    return float(np.quantile(meds, q))


def _timed(fn):
    t0 = time.perf_counter_ns()
    out = fn()
    return (time.perf_counter_ns() - t0) / 1e6, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default=",".join(s[0] for s in SHAPES))
    args = ap.parse_args(argv)
    if platform.system() != "Darwin":
        print("[matmul2d-corpus] not Darwin -- skipping"); return 0
    from tessera import __version__
    from tessera import runtime as rt
    from tessera.compiler import driver
    if not rt._apple_gpu_mtl4_matmul2d_lane_available():
        print("[matmul2d-corpus] strided-view matmul2d lane unavailable (needs macOS 27 + SDK27 dylib) -- skipping")
        return 0
    opt = _find_tessera_opt()
    if opt is None:
        print("[matmul2d-corpus] no tessera-opt build -- the candidate is the COMPILED route; skipping")
        return 0
    try:
        import ml_dtypes
    except ImportError:
        print("[matmul2d-corpus] ml_dtypes unavailable -- skipping"); return 0
    mps_f16 = rt._apple_gpu_mps_matmul_f16()
    if mps_f16 is None:
        print("[matmul2d-corpus] MPS fp16 GEMM symbol missing -- no incumbent; skipping"); return 0
    import ctypes
    u16 = ctypes.POINTER(ctypes.c_uint16)
    device = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, capture_output=True).stdout.strip()
    wanted = set(args.shapes.split(","))
    rng = np.random.default_rng(20260915)
    rows, sources = [], {}

    def incumbent_f16(A16, B16):
        # The production f16 lane: MTL4 for M == 1 (P7), MPS otherwise.
        routed = rt._mtl4_route_matmul2d_f16(A16, B16, np)
        if routed is not None:
            return routed, "mtl4_route_matmul2d_f16"
        C16 = np.empty((A16.shape[0], B16.shape[1]), np.float16)
        mps_f16(A16.ctypes.data_as(u16), B16.ctypes.data_as(u16), C16.ctypes.data_as(u16),
                ctypes.c_int32(A16.shape[0]), ctypes.c_int32(B16.shape[1]), ctypes.c_int32(A16.shape[1]))
        return C16, "mps_matmul_f16"

    for name, M, N, K in SHAPES:
        if name not in wanted:
            continue
        for a_elem, b_elem in PAIRS:
            lowp = b_elem != "f16" and a_elem != "bf16"
            if lowp and (K % 128 or N % 128):
                continue
            proc = subprocess.run([str(opt), "-", *(f"--{p}" for p in PASSES), "--allow-unregistered-dialect"],
                                  input=_gemm_module(M, K, N, a_elem, b_elem), capture_output=True, text=True)
            if proc.returncode != 0:
                rows.append({"op": "matmul", "shape": [M, N, K], "dtype": f"{a_elem}x{b_elem}", "status": "not_lowered",
                             "reason": proc.stderr.strip()[-300:]})
                print(f"{name:24s} {a_elem}x{b_elem}: not lowered"); continue
            calls = driver.extract_apple_value_calls(proc.stdout)
            assert len(calls) == 1 and driver.apple_value_call_is_executable(calls[0]), calls
            call = calls[0]

            # Operands: 16-bit pairs draw normals; low-precision pairs draw exact
            # codes and the incumbent sees the SAME values as fp16.
            A32 = (rng.standard_normal((M, K)) * 0.25).astype(np.float32)
            B32 = (rng.standard_normal((K, N)) * 0.25).astype(np.float32)
            pack_ms = None
            if a_elem == "bf16":
                A, B = A32.astype(ml_dtypes.bfloat16), B32.astype(ml_dtypes.bfloat16)
                A16, B16 = A, B
            elif lowp:
                t0 = time.perf_counter_ns()
                B = B32.astype(ml_dtypes.float8_e4m3fn)
                A = A32.astype(np.float16) if a_elem == "f16" else A32.astype(ml_dtypes.float8_e4m3fn)
                pack_ms = (time.perf_counter_ns() - t0) / 1e6
                A16, B16 = A.astype(np.float16), B.astype(np.float16)
            else:
                A = A16 = A32.astype(np.float16)
                B = B16 = B32.astype(np.float16)
            ref = A.astype(np.float64) @ B.astype(np.float64)
            scale = np.abs(ref).max() + 1.0

            def candidate():
                return rt._dispatch_gpu_mtl4_matmul2d([A, B], call, np)

            def incumbent():
                if a_elem == "bf16":
                    out = rt._mtl4_route_matmul2d_bf16(A16, B16, np)
                    if out is None:
                        raise RuntimeError("production bf16 route declined")
                    return out, "mtl4_route_matmul2d_bf16"
                return incumbent_f16(A16, B16)

            # Correctness of BOTH routes before any timing.
            cand = candidate()
            inc, inc_route = incumbent()
            err_c = float(np.abs(cand.astype(np.float64) - ref).max() / scale)
            err_i = float(np.abs(inc.astype(np.float64) - ref).max() / scale)
            tol = 3e-2  # f16/bf16 storage rounding and a 16-bit incumbent output
            if not (np.isfinite(cand).all() and np.isfinite(inc).all() and err_c <= tol and err_i <= tol):
                rows.append({"op": "matmul", "shape": [M, N, K], "dtype": f"{a_elem}x{b_elem}", "status": "wrong",
                             "candidate_err": err_c, "incumbent_err": err_i})
                print(f"{name:24s} {a_elem}x{b_elem}: WRONG cand={err_c:.2e} inc={err_i:.2e}"); continue

            for _ in range(args.warmup):
                candidate(); incumbent()
            cand_ms, inc_ms = [], []
            for rep in range(args.reps):
                if rep % 2 == 0:
                    ti, _ = _timed(incumbent); tc, _ = _timed(candidate)
                else:
                    tc, _ = _timed(candidate); ti, _ = _timed(incumbent)
                cand_ms.append(tc); inc_ms.append(ti)
            ratios = [i / c for i, c in zip(inc_ms, cand_ms)]
            med = float(np.median(ratios))
            lower = _bootstrap_lower(ratios)
            admitted = lower >= 1.0 + MARGIN
            lat = float(np.median(cand_ms))
            row = {
                "backend": "apple_gpu", "op": "matmul", "shape": [M, N, K], "dtype": f"{a_elem}x{b_elem}->f32",
                "latency_ms": lat, "tflops": 2.0 * M * N * K / (lat * 1e-3) / 1e12, "memory_bw_gb_s": None,
                "device": device, "tessera_version": __version__, "route": "compiled_mtl4_matmul2d_view",
                "shape_name": name, "status": "measured", "incumbent_route": inc_route,
                "incumbent_latency_ms": float(np.median(inc_ms)), "candidate_ms": cand_ms, "incumbent_ms": inc_ms,
                "speedup_median": med, "speedup_lower_95": lower, "admitted": admitted, "margin": MARGIN,
                "pack_ms_lowp": pack_ms, "candidate_err": err_c, "incumbent_err": err_i,
                "incumbent_output_dtype": str(inc.dtype), "candidate_output_dtype": "float32",
                "pair_code": int(call["tessera_apple.pair"]), "reps": args.reps,
            }
            rows.append(row)
            print(f"{name:24s} {a_elem + 'x' + b_elem:18s} vs {inc_route:26s} cand {lat:8.3f} ms  "
                  f"inc {row['incumbent_latency_ms']:8.3f} ms  x{med:5.2f} (lb {lower:5.2f})  "
                  f"{'ADMIT' if admitted else 'retain incumbent'}")

    summary = {}
    for r in rows:
        if r.get("status") != "measured":
            continue
        summary.setdefault(r["dtype"], {"admitted": [], "retained": []})
        summary[r["dtype"]]["admitted" if r["admitted"] else "retained"].append(r["shape_name"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "lane": "apple_matmul2d_route_admission_corpus", "device": device, "os": platform.mac_ver()[0],
        "passes": PASSES, "margin": MARGIN, "reps": args.reps, "timing": "wall host->ctypes->GPU->host, interleaved, alternating order",
        "admission_rule": "95% bootstrap lower bound of the median incumbent/candidate ratio >= 1 + margin",
        "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
