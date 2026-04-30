#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from benchmarks.common import (  # noqa: E402
    ArtifactLevels,
    BenchmarkOperator,
    BenchmarkRow,
    CompilerPath,
    Profile,
    RuntimeStatus,
    compiler_flash_attention_ir,
    correctness_report,
)


def flash_attn(q, k, v, scale=None, eps=1e-9):
    scale = scale or (1.0 / np.sqrt(q.shape[-1]))
    out = np.zeros_like(q)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            scores = (q[b, h] @ k[b, h].T) * scale
            scores = scores - np.max(scores, axis=1, keepdims=True)
            p = np.exp(scores)
            p = p / (np.sum(p, axis=1, keepdims=True) + eps)
            out[b, h] = p @ v[b, h]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(123)
    q = (rng.standard_normal((args.batch, args.heads, args.seq, args.d)) * 0.01).astype(np.float32)
    k = (rng.standard_normal(q.shape) * 0.01).astype(np.float32)
    v = (rng.standard_normal(q.shape) * 0.01).astype(np.float32)
    info = compiler_flash_attention_ir()

    best_tps = 0.0
    last_ms = 0.0
    out = None
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        out = flash_attn(q, k, v)
        last_ms = (time.perf_counter() - t0) * 1000.0
        best_tps = max(best_tps, args.batch * args.heads * args.seq / max(last_ms * 1e-3, 1e-12))

    ref = flash_attn(q.astype(np.float64), k.astype(np.float64), v.astype(np.float64)).astype(np.float32)
    corr = correctness_report(out, ref, tolerance=1e-5)
    row = BenchmarkRow(
        operator=BenchmarkOperator("flash_attn", "f32", f"{args.batch}x{args.heads}x{args.seq}x{args.d}", "cpu"),
        compiler_path=CompilerPath.GRAPH_IR_ONLY if info.get("available") else CompilerPath.REFERENCE,
        runtime_status=RuntimeStatus.SKIPPED if info.get("available") else RuntimeStatus.EXECUTABLE,
        artifact_levels=ArtifactLevels(graph=bool(info.get("graph_ir")), artifact_hash=info.get("artifact_hash")),
        correctness=corr,
        profile=Profile(cpu_wall_ms=last_ms),
        metrics={
            "throughput_tokens_per_s": best_tps,
            "latency_ms": last_ms,
            "max_abs_err": corr.max_error,
            "compiler_lowering": str(info.get("lowering", "")),
        },
        reason="Tile/Target runtime path for FlashAttention is not executable yet" if info.get("available") else "tessera import failed",
    )
    print(json.dumps(row.flat_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
