#!/usr/bin/env python3
"""Exact gfx1151 proof for direct shared CORE-GEMM-KLOOP consumption.

Each row starts from Graph IR, runs the shared explicit M/N/K ``scf.for``
tiler, and lets ROCm re-form that verified semantic loop into its one existing
problem-size-generic WMMA kernel. Modules and buffers remain resident while
``perf_counter_ns`` brackets launch plus ``hipDeviceSynchronize``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from tessera import runtime as rt  # noqa: E402

from benchmarks.rocm.benchmark_rocm_gemm_schedule_matrix import (  # noqa: E402
    Case,
    DeviceCase,
    _correct,
    _error_metrics,
    _inputs,
    _reference,
    code_object_resources,
)

CASES = (
    Case("canonical_aligned", 256, 256, 256, "f16"),
    Case("canonical_ragged", 131, 127, 65, "f16"),
    Case("canonical_aligned", 256, 256, 256, "bf16"),
    Case("canonical_ragged", 131, 127, 65, "bf16"),
    Case("canonical_aligned", 256, 256, 256, "int8"),
    Case("canonical_ragged", 131, 127, 65, "int8"),
)
KERNEL_WALL_BASELINES_MS = {
    ("canonical_aligned", "f16"): 0.098115,
    ("canonical_ragged", "f16"): 0.120182,
    ("canonical_aligned", "bf16"): 0.099710,
    ("canonical_ragged", "bf16"): 0.087169,
    ("canonical_aligned", "int8"): 0.092164,
    ("canonical_ragged", "int8"): 0.086102,
}
KERNEL_WALL_MAX_REGRESSION = 0.10


def _resident_wall_samples(device: DeviceCase, hip: Any, *, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        if device.launch() != 0:
            raise RuntimeError("canonical GEMM warmup launch failed")
    if hip.hipDeviceSynchronize() != 0:
        raise RuntimeError("canonical GEMM warmup synchronization failed")
    samples = []
    for _ in range(iterations):
        started_ns = time.perf_counter_ns()
        if device.launch() != 0:
            raise RuntimeError("canonical GEMM timed launch failed")
        if hip.hipDeviceSynchronize() != 0:
            raise RuntimeError("canonical GEMM timed synchronization failed")
        samples.append((time.perf_counter_ns() - started_ns) / 1_000_000.0)
    return samples


def run(*, warmup: int, iterations: int, staging: str = "register") -> dict[str, Any]:
    if staging not in {"register", "lds"}:
        raise ValueError("staging must be register or lds")
    chip = os.environ.get("TESSERA_ROCM_CHIP", "gfx1151")
    if chip != "gfx1151":
        raise RuntimeError(f"exact canonical GEMM packet requires gfx1151, got {chip}")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("a live gfx1151 HIP device is required")
    rows = []
    for case in CASES:
        a, b, bias = _inputs(case)
        reference = _reference(case, a, b, bias)
        assert reference is not None
        compile_started = time.perf_counter_ns()
        hsaco = rt._build_canonical_gemm_hsaco(
            case.m, case.n, case.k, case.dtype, staging=staging
        )
        compile_ms = (time.perf_counter_ns() - compile_started) / 1_000_000.0
        operation_started = time.perf_counter_ns()
        device = DeviceCase(hip, hsaco, case, (1, 1), a, b, bias)
        try:
            actual = device.download()
            operation_total_ms = (time.perf_counter_ns() - operation_started) / 1_000_000.0
            metrics = _error_metrics(actual, reference)
            samples = _resident_wall_samples(device, hip, warmup=warmup, iterations=iterations)
        finally:
            device.close()
        median_ms = statistics.median(samples)
        baseline_ms = KERNEL_WALL_BASELINES_MS[(case.family, case.dtype)]
        limit_ms = baseline_ms * (1.0 + KERNEL_WALL_MAX_REGRESSION)
        selector_eligible = staging == "register"
        rows.append(
            {
                "case": case.key,
                "shape": [case.m, case.n, case.k],
                "storage": case.dtype,
                "accumulate": ("i32" if case.dtype == "int8" else "f32"),
                "semantic_source": "canonical_mnk_scf_for",
                "schedule": f"gfx1151_wmma_16x16x16_{staging}",
                "correct": _correct(case, metrics),
                "numerics": metrics,
                "artifact": {
                    "binary_format": "hsaco",
                    "bytes": len(hsaco),
                    "resources": code_object_resources(hsaco),
                },
                "timing": {
                    "compiler_ms": compile_ms,
                    "operation_total_ms": operation_total_ms,
                    "kernel_wall": {
                        "clock": "time.perf_counter_ns",
                        "launch_api": "hipModuleLaunchKernel",
                        "completion_api": "hipDeviceSynchronize",
                        "resident_module": True,
                        "resident_buffers": True,
                        "warmup_iterations": warmup,
                        "sample_count": iterations,
                        "median_ms": median_ms,
                        "samples_ms": samples,
                        "selector_eligible": selector_eligible,
                    },
                },
                "ratchet": {
                    "baseline_ms": baseline_ms,
                    "max_regression_fraction": KERNEL_WALL_MAX_REGRESSION,
                    "limit_ms": limit_ms,
                    "observed_ratio": median_ms / baseline_ms,
                    "passed": (median_ms <= limit_ms if selector_eligible else None),
                    "disposition": (
                        "production_non_regression"
                        if selector_eligible
                        else "comparison_only_against_register_incumbent"
                    ),
                },
            }
        )
    return {
        "schema": "tessera.rocm.canonical_gemm_kloop.v1",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": chip,
        "toolchain": {
            "rocm_hip": "7.14.60850",
            "compiler": "AMD clang 23",
        },
        "rows": rows,
        "all_correct": all(row["correct"] for row in rows),
        "physical_staging": staging,
        "all_ratchets_pass": all(
            row["ratchet"]["passed"] is not False for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=21)
    parser.add_argument("--staging", choices=("register", "lds"), default="register")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be nonnegative and iterations positive")
    packet = run(warmup=args.warmup, iterations=args.iterations, staging=args.staging)
    text = json.dumps(packet, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    return 0 if packet["all_correct"] and packet["all_ratchets_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
