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
    compiler_matmul_relu,
    correctness_report,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--dtype", type=str, default="f32", choices=["f32", "f16", "bf16"])
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args(argv)

    dtype = np.float32 if args.dtype == "f32" else np.float16
    a = (np.arange(args.m * args.k, dtype=np.float32) % 13 / 13).reshape(args.m, args.k).astype(dtype)
    b = (np.arange(args.k * args.n, dtype=np.float32) % 17 / 17).reshape(args.k, args.n).astype(dtype)
    ref = np.maximum(a.astype(np.float64) @ b.astype(np.float64), 0).astype(dtype)

    best_flops = 0.0
    last_ms = 0.0
    last_out = None
    compiler_path = CompilerPath.REFERENCE
    runtime_status = RuntimeStatus.EXECUTABLE
    artifact = ArtifactLevels()
    lowering = ""
    reason = ""

    for _ in range(args.repeat):
        t0 = time.perf_counter()
        compiler_run = compiler_matmul_relu(a.astype(np.float32), b.astype(np.float32), (128, 128, 32))
        if compiler_run is not None:
            last_out = np.asarray(compiler_run.output).astype(dtype)
            last_ms = compiler_run.latency_ms
            compiler_path = CompilerPath.TESSERA_JIT_CPU if compiler_run.uses_compiled_path else CompilerPath.REFERENCE
            artifact = ArtifactLevels(
                graph=compiler_run.graph_ir is not None,
                schedule=compiler_run.schedule_ir is not None,
                tile=compiler_run.tile_ir is not None,
                target=compiler_run.target_ir is not None,
                artifact_hash=compiler_run.artifact_hash,
            )
            lowering = compiler_run.lowering
        else:
            last_out = np.maximum(a @ b, 0)
            last_ms = (time.perf_counter() - t0) * 1000.0
            reason = "tessera import failed; used NumPy reference"
        flops = 2.0 * args.m * args.n * args.k / max(last_ms * 1e-3, 1e-12)
        best_flops = max(best_flops, flops)

    corr = correctness_report(last_out, ref, tolerance=1e-3 if args.dtype == "f32" else 5e-2)
    row = BenchmarkRow(
        operator=BenchmarkOperator("gemm", args.dtype, f"{args.m}x{args.n}x{args.k}", "cpu"),
        compiler_path=compiler_path,
        runtime_status=runtime_status,
        artifact_levels=artifact,
        correctness=corr,
        profile=Profile(cpu_wall_ms=last_ms),
        metrics={
            "throughput_flops": best_flops,
            "latency_ms": last_ms,
            "max_abs_err": corr.max_error,
            "compiler_lowering": lowering,
        },
        reason=reason,
    )
    print(json.dumps(row.flat_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
