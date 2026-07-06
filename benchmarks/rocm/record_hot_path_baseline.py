"""E2 (2026-07-06) — record the ROCm gfx1151 hot-path perf-ratchet baseline.

Times the WMMA matmul hot path through the SAME production C-ABI symbol
(`tessera_rocm_wmma_gemm_f16`, RDNA WMMA + f32 accumulate) that
`runtime._execute_rocm_wmma_artifact` dispatches to, then writes
`benchmarks/baselines/rocm_gfx1151_hot_paths.json` with per-row
thresholds = median * margin. The CI ratchet (`perf_gate.evaluate_ratchet`,
locked by `tests/unit/test_rocm_perf_ratchet.py`) re-times and fails on
regressions past the margin.

Host-gated exactly like the Apple lane: with no AMD GPU / GEMM lib the
runtime probe (`_rocm_wmma_runtime_available`) returns False, so this
skip-cleans (prints a note, writes nothing, exit 0) — it NEVER fabricates
GPU numbers (repo Decision #26; benchmarks/rocm/*.py honesty rule).

Run on the machine that hosts CI, once the gfx1151 GPU is live
(`hipGetDeviceCount` sees the device):

    python benchmarks/rocm/record_hot_path_baseline.py [--margin 2.0]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

OUT = ROOT / "benchmarks" / "baselines" / "rocm_gfx1151_hot_paths.json"

# WMMA matmul ladder, f16 storage / f32 accumulate — the executable ROCm hot
# path (compiler-generated matmul, repo Decision #26). Kept small and square so
# a ratchet regression pins to the primitive, not to shape edge cases.
HOT_PATH_SIZES = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]


def _median_ms(fn, reps: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def hot_path_cases(rt):
    """(op, shape, dtype, mode, thunk) per named hot path — module-level so the
    ratchet test re-times the IDENTICAL work through the production symbol.

    Each thunk calls the shipped `tessera_rocm_wmma_gemm_f16` C-ABI symbol
    (host pointers; the lib does H2D/compute/D2H), the same symbol
    `_execute_rocm_wmma_artifact` dispatches to. Raises if the lib/symbol is
    absent — callers gate on `_rocm_wmma_runtime_available()` first."""
    lib = rt._load_rocm_gemm_runtime()
    if lib is None:
        raise RuntimeError("libtessera_rocm_gemm.so not loadable — no ROCm lane")
    sym = rt._ROCM_GEMM_SYMBOLS["float16"]
    fn = getattr(lib, sym, None)
    if fn is None:
        raise RuntimeError(f"libtessera_rocm_gemm.so lacks {sym}")

    rng = np.random.default_rng(0)

    def _make(m, n, k):
        a = rng.standard_normal((m, k)).astype(np.float16)
        b = rng.standard_normal((k, n)).astype(np.float16)
        d = np.zeros((m, n), np.float32)

        def _run():
            rc = fn(a.ctypes.data_as(ctypes.c_void_p),
                    b.ctypes.data_as(ctypes.c_void_p),
                    d.ctypes.data_as(ctypes.c_void_p), m, n, k)
            if rc != 0:
                raise RuntimeError(f"{sym} rc={rc} at {m}x{n}x{k}")
        return _run

    cases = []
    for (m, n, k) in HOT_PATH_SIZES:
        cases.append(("matmul", f"{m}x{n}x{k}", "f16", "wmma", _make(m, n, k)))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", type=float, default=2.0,
                        help="threshold = median * margin (CI noise headroom)")
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()

    from tessera import runtime as rt

    if not rt._rocm_wmma_runtime_available():
        print("rocm WMMA runtime unavailable (no AMD GPU / GEMM lib); "
              "skipping baseline record (no numbers fabricated)")
        return 0

    rows = []
    for op, shape, dtype, mode, thunk in hot_path_cases(rt):
        med = _median_ms(thunk, reps=args.reps)
        rows.append({
            "op": op, "shape": shape, "dtype": dtype, "mode": mode,
            "median_ms": round(med, 4),
            "max_latency_ms": round(med * args.margin, 4),
        })
        print(f"{op:12s} {shape:16s} median {med:8.3f} ms  "
              f"cap {med * args.margin:8.3f} ms")
    OUT.write_text(json.dumps({
        "schema": "tessera.benchmark.ratchet.v1",
        "margin": args.margin,
        "rows": rows,
    }, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
