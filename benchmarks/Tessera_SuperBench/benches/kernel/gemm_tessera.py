#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import replace

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
    compiler_matmul_relu_target,
    correctness_report,
    telemetry_for_row,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--dtype", type=str, default="f32", choices=["f32", "f16", "bf16"])
    ap.add_argument("--target", type=str, default="cpu", choices=["cpu", "apple_cpu"])
    ap.add_argument("--autotune", action="store_true", help="Attach current GEMM schedule artifact metadata")
    ap.add_argument("--autotune-dtype", type=str, default=None, choices=["bf16", "f16", "tf32", "int8", "fp8"])
    ap.add_argument("--autotune-cache", type=str, default="/tmp/tessera_superbench_autotune.db")
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

    schedule_artifact = None
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        compiler_run = compiler_matmul_relu_target(
            a.astype(np.float32),
            b.astype(np.float32),
            tile=(128, 128, 32),
            target=args.target,
        )
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
    if args.autotune:
        try:
            import tessera as ts  # noqa: WPS433

            tune_dtype = args.autotune_dtype or ("bf16" if args.dtype == "f32" else args.dtype)
            tune = ts.autotune(
                "matmul",
                shapes=(args.m, args.n, args.k),
                dtype=tune_dtype,
                arch=args.target,
                max_trials=2,
                cache_path=args.autotune_cache,
            )
            schedule_artifact = ts.autotune.schedule_artifact(
                tune,
                op="matmul",
                shapes=(args.m, args.n, args.k),
                dtype=tune_dtype,
                arch=args.target,
            )
        except Exception as exc:  # pragma: no cover - defensive benchmark path
            reason = f"{reason}; autotune unavailable: {exc}" if reason else f"autotune unavailable: {exc}"

    row = BenchmarkRow(
        operator=BenchmarkOperator("gemm", args.dtype, f"{args.m}x{args.n}x{args.k}", args.target),
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
            "schedule_hash": schedule_artifact.get("hash") if schedule_artifact else None,
        },
        telemetry={},
        reason=reason,
    )
    row = replace(
        row,
        telemetry=telemetry_for_row(
            row,
            schedule_hash=schedule_artifact.get("hash") if schedule_artifact else None,
            metadata={"schedule_artifact": schedule_artifact} if schedule_artifact else None,
        ),
    )
    print(json.dumps(row.flat_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
