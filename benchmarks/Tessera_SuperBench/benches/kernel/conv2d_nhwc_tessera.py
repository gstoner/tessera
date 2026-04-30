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
    compiler_conv2d_ir,
    correctness_report,
)


def conv2d_nhwc(x, w, stride=1, padding=0):
    n, h, width, c = x.shape
    kh, kw, _, oc = w.shape
    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (width + 2 * padding - kw) // stride + 1
    y = np.zeros((n, out_h, out_w, oc), dtype=x.dtype)
    if padding > 0:
        padded = np.zeros((n, h + 2 * padding, width + 2 * padding, c), dtype=x.dtype)
        padded[:, padding:padding + h, padding:padding + width, :] = x
        x = padded
    for batch in range(n):
        for i in range(out_h):
            for j in range(out_w):
                patch = x[batch, i * stride:i * stride + kh, j * stride:j * stride + kw, :]
                y[batch, i, j, :] = np.tensordot(patch, w, axes=([0, 1, 2], [0, 1, 2]))
    return y


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--w", type=int, default=64)
    ap.add_argument("--c", type=int, default=32)
    ap.add_argument("--oc", type=int, default=64)
    ap.add_argument("--kh", type=int, default=3)
    ap.add_argument("--kw", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--pad", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(123)
    x = ((rng.random((args.n, args.h, args.w, args.c), dtype=np.float32) - 0.5) * 0.1).astype(np.float32)
    w = ((rng.random((args.kh, args.kw, args.c, args.oc), dtype=np.float32) - 0.5) * 0.1).astype(np.float32)
    info = compiler_conv2d_ir()

    best_flops = 0.0
    last_ms = 0.0
    y = None
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        y = conv2d_nhwc(x, w, stride=args.stride, padding=args.pad)
        last_ms = (time.perf_counter() - t0) * 1000.0
        flops = args.n * y.shape[1] * y.shape[2] * args.oc * args.kh * args.kw * args.c * 2.0
        best_flops = max(best_flops, flops / max(last_ms * 1e-3, 1e-12))

    ref = conv2d_nhwc(x.astype(np.float64), w.astype(np.float64), stride=args.stride, padding=args.pad).astype(np.float32)
    corr = correctness_report(y, ref, tolerance=1e-5)
    row = BenchmarkRow(
        operator=BenchmarkOperator("conv2d_nhwc", "f32", f"{args.n}x{args.h}x{args.w}x{args.c}->{args.oc}", "cpu"),
        compiler_path=CompilerPath.GRAPH_IR_ONLY if info.get("available") else CompilerPath.REFERENCE,
        runtime_status=RuntimeStatus.SKIPPED if info.get("available") else RuntimeStatus.EXECUTABLE,
        artifact_levels=ArtifactLevels(graph=bool(info.get("graph_ir")), artifact_hash=info.get("artifact_hash")),
        correctness=corr,
        profile=Profile(cpu_wall_ms=last_ms),
        metrics={
            "throughput_flops": best_flops,
            "latency_ms": last_ms,
            "max_abs_err": corr.max_error,
            "compiler_lowering": str(info.get("lowering", "")),
        },
        reason="Tile/Target runtime path for Conv2D is not executable yet" if info.get("available") else "tessera import failed",
    )
    print(json.dumps(row.flat_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
