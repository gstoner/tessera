"""Record comparative ROCM-7 ratchets on the WSL gfx1151 system.

Each row times the retained pre-redesign kernel and its cooperative candidate
through the same production runtime, then records both a candidate latency cap
and a minimum speedup.  No GPU means a clean skip and no fabricated numbers.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
OUT = ROOT / "benchmarks" / "baselines" / "rocm_gfx1151_sparse_redesign.json"


def _median_ms(fn, *, reps: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def comparative_cases(rt):
    """Yield ``(op, shape, baseline, candidate)`` production-runtime thunks."""
    rng = np.random.default_rng(7007)

    # DK2/MSA selected attention: 128 selected tokens per query.  The scalar
    # baseline recomputes QK once per value lane; the candidate computes every
    # selected score once and shares softmax weights through LDS.
    B, Hq, Hkv, Sq, Sk, D, Dv, block, top_k = 1, 8, 2, 128, 2048, 64, 64, 16, 8
    q = (rng.standard_normal((B, Hq, Sq, D), dtype=np.float32) * 0.1)
    k = (rng.standard_normal((B, Hkv, Sk, D), dtype=np.float32) * 0.1)
    v = (rng.standard_normal((B, Hkv, Sk, Dv), dtype=np.float32) * 0.1)
    selected = np.tile(np.arange(top_k, dtype=np.int64), (B, Hkv, Sq, 1))

    def attention(tiled):
        return lambda: rt._rocm_selected_block_attention_native(
            q, k, v, selected, np, block_size=block, causal=False, tiled=tiled)

    yield ("sparse_attention", "1x8x128x2048x64-k8-b16",
           attention(False), attention(True))

    # The resident-selection crossover on this APU is a small row batch with
    # thousands of candidates.  Keep two block-count rungs so a regression
    # cannot hide behind one favorable point.
    for blocks in (2048, 4096):
        scores = rng.standard_normal((1, 4, 64, blocks), dtype=np.float32)

        def select(cooperative, scores=scores):
            return lambda: rt._rocm_block_sparse_topk_select_native(
                scores, np, top_k=8, block_size=16, causal=False,
                cooperative=cooperative)

        yield ("sparse_topk", f"1x4x64x{blocks}-k8",
               select(False), select(True))


def measure_cases(rt, *, reps: int = 7):
    rows = []
    for op, shape, baseline, candidate in comparative_cases(rt):
        base_ms = _median_ms(baseline, reps=reps)
        cand_ms = _median_ms(candidate, reps=reps)
        rows.append({
            "op": op, "shape": shape, "dtype": "f32",
            "baseline_ms": base_ms, "candidate_ms": cand_ms,
            "speedup": base_ms / cand_ms,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--margin", type=float, default=1.5)
    parser.add_argument("--min-speedup", type=float, default=1.10)
    args = parser.parse_args()

    from tessera import runtime as rt
    if rt._tessera_opt_path() is None or not rt._rocm_wmma_runtime_available():
        print("ROCm compiler/GPU unavailable; no sparse numbers fabricated")
        return 0

    rows = measure_cases(rt, reps=args.reps)
    for row in rows:
        row["baseline_ms"] = round(row["baseline_ms"], 4)
        row["candidate_ms"] = round(row["candidate_ms"], 4)
        row["speedup"] = round(row["speedup"], 4)
        row["max_candidate_ms"] = round(row["candidate_ms"] * args.margin, 4)
        row["min_speedup"] = args.min_speedup
        print(f"{row['op']:18s} {row['shape']:26s} "
              f"{row['speedup']:.3f}x")
    OUT.write_text(json.dumps({
        "schema": "tessera.benchmark.comparative-ratchet.v1",
        "target": f"rocm:{rt._rocm_chip()}",
        "rows": rows,
    }, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
