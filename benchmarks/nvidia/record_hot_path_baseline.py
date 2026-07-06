"""E2 (2026-07-06) — record the NVIDIA sm_120 hot-path perf-ratchet baseline.

Times the mma.sync matmul hot path through the SAME production C-ABI symbol
(`tessera_nvidia_mma_gemm_f16`, warp-level mma.sync + f32 accumulate) that
`runtime._execute_nvidia_mma_artifact` dispatches to, then writes
`benchmarks/baselines/nvidia_sm120_hot_paths.json` with per-row
thresholds = median * margin. The CI ratchet (`perf_gate.evaluate_ratchet`,
locked by `tests/unit/test_nvidia_perf_ratchet.py`) re-times and fails on
regressions past the margin.

Host-gated exactly like the ROCm/Apple lanes: with no NVIDIA GPU / GEMM lib
the runtime probe (`_nvidia_mma_runtime_available`) returns False, so this
skip-cleans (prints a note, writes nothing, exit 0) — it NEVER fabricates
GPU numbers (repo Decision #26).

Run on the machine that hosts CI, once the sm_120 GPU is live:

    python benchmarks/nvidia/record_hot_path_baseline.py [--margin 2.0]
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

OUT = ROOT / "benchmarks" / "baselines" / "nvidia_sm120_hot_paths.json"

# mma.sync matmul ladder, f16 storage / f32 accumulate — the hardware-verified
# NVIDIA sm_120 hot path (consumer Blackwell, repo Decision #26). Small squares
# so a ratchet regression pins to the primitive.
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

    Each thunk calls the shipped `tessera_nvidia_mma_gemm_f16` C-ABI symbol
    (host pointers; the lib does H2D/compute/D2H), the same symbol
    `_execute_nvidia_mma_artifact` dispatches to. Raises if the lib/symbol is
    absent — callers gate on `_nvidia_mma_runtime_available()` first."""
    lib = rt._load_nvidia_gemm_runtime()
    if lib is None:
        raise RuntimeError("libtessera_nvidia_gemm.so not loadable — no NVIDIA lane")
    sym = rt._NVIDIA_GEMM_SYMBOLS["float16"]
    fn = getattr(lib, sym, None)
    if fn is None:
        raise RuntimeError(f"libtessera_nvidia_gemm.so lacks {sym}")

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
        cases.append(("matmul", f"{m}x{n}x{k}", "f16", "mma_sync", _make(m, n, k)))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", type=float, default=2.0,
                        help="threshold = median * margin (CI noise headroom)")
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()

    from tessera import runtime as rt

    if not rt._nvidia_mma_runtime_available():
        print("nvidia mma.sync runtime unavailable (no NVIDIA GPU / GEMM lib); "
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
