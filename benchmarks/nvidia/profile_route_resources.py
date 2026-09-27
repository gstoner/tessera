"""Nsight launch target: one production arbiter route, profiled in isolation.

The TEST-5 route-resource manifest (``nvidia_sm120_test5_route_resources.json``)
attests, per candidate route, the Nsight Compute launch facts of the kernels
that route launches: registers per thread, static/dynamic shared memory,
theoretical and achieved occupancy and spill evidence
(``parse_ncu_resources.py``), mapped to the route by
``build_test5_resource_manifest.py``. The original capture
(``profile_test5_routes.py``) ran many routes in one process and mapped kernels
to routes by NAME -- which cannot tell the tf32 / fp8_e4m3 / fp8_e5m2 builds of
one emitted lane apart (every storage compiles a kernel called
``tessera_nvidia_mma_fused_kernel``), nor keep the shipped-GEMM probe's
``gemm`` launch out of a composed route.

So this target profiles ONE route per process, and only that route's launches:
it runs the candidate once untimed (compile, load, runtime probes), then
brackets exactly one device-timer invocation (``warmup=0, reps=1``: the same
entry and launch configuration the device rows time) with
``cuProfilerStart`` / ``cuProfilerStop``. Run it under
``ncu --profile-from-start off --set full``; every kernel in the report belongs
to the named route, and ``build_test5_resource_manifest.py --route NAME=...``
attributes them to it. Sync ``AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27``
(``AUTOTUNE-SM120-ROUTE-RESOURCES``).
"""
from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

#: op -> the workload the route is profiled at (a committed sm_120 row's shape).
WORKLOADS = {
    "fused_region": (128, 512, 256),       # M, N, K
    "attention": (128, 128, 64, 64),       # M, Nk, D, Dv
    "gated_matmul": (128, 512, 512),       # M, H, K
}


def _inputs(op: str, storage: str | None):
    from tessera.compiler.emit.candidate import (
        OP_ATTENTION, OP_FUSED_REGION, OP_GATED_MATMUL)
    from tessera.compiler.fusion import (
        AttentionRegion, FusedRegion, GatedMatmulRegion)

    rng = np.random.default_rng(1205)
    kw = {} if storage is None else {"storage_dtype": storage}
    if op == "fused_region":
        m, n, k = WORKLOADS[op]
        a = (rng.standard_normal((m, k)) * .1).astype(np.float32)
        b = (rng.standard_normal((k, n)) * .1).astype(np.float32)
        bias = (rng.standard_normal(n) * .05).astype(np.float32)
        return OP_FUSED_REGION, FusedRegion(epilogue=("bias", "gelu"), **kw), (a, b, bias)
    if op == "attention":
        m, nk, d, dv = WORKLOADS[op]
        q, k, v = ((rng.standard_normal(s) * .1).astype(np.float32)
                   for s in ((m, d), (nk, d), (nk, dv)))
        return (OP_ATTENTION, AttentionRegion(scale=d ** -.5, causal=True, **kw),
                (q, k, v))
    if op == "gated_matmul":
        m, h, k = WORKLOADS[op]
        a, wg, wu = ((rng.standard_normal(s) * .1).astype(np.float32)
                     for s in ((m, k), (k, h), (k, h)))
        return OP_GATED_MATMUL, GatedMatmulRegion(gate_act="silu", **kw), (a, wg, wu)
    raise ValueError(f"no profiling workload for op {op!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--op", required=True, choices=sorted(WORKLOADS))
    parser.add_argument("--storage", default=None,
                        help="region storage dtype (f32, fp8_e4m3, fp8_e5m2, f16, ...)")
    args = parser.parse_args(argv)

    from tessera import runtime as rt
    from tessera.compiler.emit import nvidia_cuda  # noqa: F401 - registers candidates
    from tessera.compiler.emit.candidate import candidates_for

    if rt._nvidia_device_name() != "sm_120":
        print("exact sm_120 device unavailable; nothing profiled", file=sys.stderr)
        return 2
    op, region, inputs = _inputs(args.op, args.storage)
    found = [c for c in candidates_for("nvidia", op) if c.name == args.candidate]
    if len(found) != 1:
        raise SystemExit(f"candidate {args.candidate!r} is not registered for {op}")
    cand = found[0]
    if not cand.available() or not cand.applies_to(region):
        raise SystemExit(f"{args.candidate} does not run {region!r} on this host")
    # Warm: compile, load, runtime probes -- none of it inside the window.
    if cand.measure_device_latency(region, *inputs, reps=1, warmup=0) is None:
        raise SystemExit(f"{args.candidate} has no device timer; cannot isolate its launch")
    driver = ctypes.CDLL("libcuda.so.1")
    if driver.cuProfilerStart() != 0:
        raise SystemExit("cuProfilerStart failed (no current CUDA context?)")
    latency = cand.measure_device_latency(region, *inputs, reps=1, warmup=0)
    if driver.cuProfilerStop() != 0:
        raise SystemExit("cuProfilerStop failed")
    if latency is None:
        raise SystemExit(f"{args.candidate} failed inside the profiling window")
    print(f"{args.candidate} {op} storage={args.storage}: profiled one timed invocation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
