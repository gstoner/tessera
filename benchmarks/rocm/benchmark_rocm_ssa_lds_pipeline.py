#!/usr/bin/env python3
"""Measure ROCm SSA LDS planning and target-lowering cost.

This is a host-compiler benchmark, not a gfx1151 kernel-performance claim.  It
measures repeated planner and full-lowering invocations while also checking the
structural contract: one SSA LDS allocation per independent stage, explicit
producer/consumer state, token-bearing waits, and no legacy buffer reference in
planner output.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TESSERA_OPT = Path(
    os.environ.get(
        "TESSERA_OPT",
        ROOT / "build-rocm-7.14-llvm23-clean/tools/tessera-opt/tessera-opt",
    )
)

PLAN_PIPELINE = "builtin.module(rocm-wave-lds-pipeline)"
LOWER_PIPELINE = "builtin.module(rocm-wave-lds-pipeline,rocm-wave-lds-legality,lower-tile-to-rocm{arch=gfx1151})"


def _module(stages: int) -> str:
    copies = "\n".join(
        f'    %copy{i} = "tile.async_copy"(%dst{i}, %src, %bytes) : (memref<64xf16>, memref<64xf16>, index) -> i32'
        for i in range(stages)
    )
    waits = "\n".join('    "tile.wait_async"() : () -> ()' for _ in range(stages))
    destinations = ", ".join(f"%dst{i}: memref<64xf16>" for i in range(stages))
    return f"""module {{
  func.func @ssa_lds_{stages}({destinations}, %src: memref<64xf16>,
                              %bytes: index) {{
{copies}
{waits}
    return
  }}
}}
"""


def _run(source: str, pipeline: str) -> tuple[str, float]:
    start = time.perf_counter_ns()
    result = subprocess.run(
        [
            str(TESSERA_OPT),
            "-",
            "--allow-unregistered-dialect",
            f"--pass-pipeline={pipeline}",
        ],
        input=source,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout, elapsed_ms


def _median_mad(samples: list[float]) -> tuple[float, float]:
    median = statistics.median(samples)
    return median, statistics.median(abs(sample - median) for sample in samples)


def _validate_plan(plan: str, stages: int) -> dict[str, int]:
    counts = {
        "allocations": plan.count("tile.alloc "),
        "pipeline_inits": plan.count("tile.pipeline_init "),
        "pipeline_advances": plan.count("tile.pipeline_advance "),
        "async_copies": plan.count("tile.async_copy "),
        "waits": plan.count("tile.wait_async "),
        "deallocations": plan.count("tile.dealloc "),
        "legacy_buffer_refs": plan.count("#tile.buffer_ref"),
    }
    expected = {
        "allocations": stages,
        "pipeline_inits": 2,
        "pipeline_advances": stages * 2,
        "async_copies": stages,
        "waits": stages,
        "deallocations": stages,
        "legacy_buffer_refs": 0,
    }
    if counts != expected:
        raise RuntimeError(f"unexpected SSA planner structure: {counts}, expected {expected}")
    return counts


def _validate_lowered(lowered: str, stages: int) -> dict[str, int]:
    counts = {
        "target_async_copies": lowered.count("tessera_rocm.async_copy "),
        "target_waits": lowered.count("tessera_rocm.wait "),
        "ssa_buffer_identities": lowered.count('buffer_identity = "ssa_allocation"'),
        "portable_allocations": lowered.count("tile.alloc "),
        "portable_pipeline_ops": lowered.count("tile.pipeline_init ")
        + lowered.count("tile.pipeline_advance "),
    }
    expected = {
        "target_async_copies": stages,
        "target_waits": stages,
        "ssa_buffer_identities": stages,
        "portable_allocations": 0,
        "portable_pipeline_ops": 0,
    }
    if counts != expected:
        raise RuntimeError(f"unexpected ROCm target structure: {counts}, expected {expected}")
    return counts


def measure(stages: int, repetitions: int, warmups: int) -> dict[str, Any]:
    source = _module(stages)
    for _ in range(warmups):
        _run(source, PLAN_PIPELINE)
        _run(source, LOWER_PIPELINE)

    plan_samples: list[float] = []
    lower_samples: list[float] = []
    plan = ""
    lowered = ""
    for _ in range(repetitions):
        plan, elapsed = _run(source, PLAN_PIPELINE)
        plan_samples.append(elapsed)
        lowered, elapsed = _run(source, LOWER_PIPELINE)
        lower_samples.append(elapsed)

    plan_median, plan_mad = _median_mad(plan_samples)
    lower_median, lower_mad = _median_mad(lower_samples)
    return {
        "stages": stages,
        "planner_median_ms": round(plan_median, 4),
        "planner_mad_ms": round(plan_mad, 4),
        "lower_median_ms": round(lower_median, 4),
        "lower_mad_ms": round(lower_mad, 4),
        "planner_structure": _validate_plan(plan, stages),
        "target_structure": _validate_lowered(lowered, stages),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmups < 0:
        parser.error("repetitions must be positive and warmups non-negative")
    if any(stages < 1 for stages in args.stages):
        parser.error("every stage count must be positive")

    packet = {
        "schema": "tessera.rocm_ssa_lds_pipeline.compiler_benchmark.v1",
        "sync_key": "ROCM-SSA-LDS-PIPELINE-2026-07-26",
        "arch": "gfx1151",
        "evidence": "host_compiler_only",
        "device_latency_ms": None,
        "repetitions": args.repetitions,
        "warmups": args.warmups,
        "cases": [measure(stages, args.repetitions, args.warmups) for stages in args.stages],
    }
    text = json.dumps(packet, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
