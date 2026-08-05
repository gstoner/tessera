#!/usr/bin/env python3
"""E2E-REAL-4 exact-gfx1151 scheduled-vs-direct matmul evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from benchmarks.rocm.benchmark_rocm_gemm_pipeline_vs_direct import (  # noqa: E402
    _build as build_direct,
    _find_mlir_opt,
    _load_hip,
)
from benchmarks.rocm.benchmark_rocm_gemm_schedule_matrix import (  # noqa: E402
    Case,
    DeviceCase,
    _correct,
    _error_metrics,
    _inputs,
    _reference,
    code_object_resources,
)
from tessera import runtime as rt  # noqa: E402
from tessera.compiler import rocm_native  # noqa: E402
from tessera.compiler.graph_ir import (  # noqa: E402
    GraphIRFunction,
    GraphIRModule,
    IRArg,
    IROp,
    IRType,
)
from tessera.compiler.scheduled_matmul import lower_scheduled_matmul  # noqa: E402

SCHEMA = "tessera.compiler.e2e_real4.rocm_matmul.v1"
BASELINE_TFLOPS = 8.02
DIRECT_PARITY_FLOOR = 0.85
CORRECTNESS_CASES = ((64, 64, 64), (65, 67, 31))  # M, N, K.


def _gpu_name() -> str:
    try:
        process = subprocess.run(["rocminfo"], capture_output=True, text=True, check=False)
    except OSError:
        return "unknown"
    agent = process.stdout.split("Agent 2", 1)[-1]
    for line in agent.splitlines():
        if "Marketing Name:" in line:
            return line.partition("Marketing Name:")[2].strip()
    return "unknown"


def _tensor(shape: tuple[int, int], *, dtype: str) -> IRType:
    element = "f16" if dtype == "fp16" else "f32"
    return IRType(
        f"tensor<{shape[0]}x{shape[1]}x{element}>",
        tuple(map(str, shape)),
        dtype,
    )


def _module(m: int, n: int, k: int) -> GraphIRModule:
    a, b = _tensor((m, k), dtype="fp16"), _tensor((k, n), dtype="fp16")
    output = _tensor((m, n), dtype="fp32")
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="gemm",
                args=[IRArg("a", a), IRArg("b", b)],
                result_types=[output],
                body=[
                    IROp(
                        result="o",
                        op_name="tessera.matmul",
                        operands=["%a", "%b"],
                        operand_types=[str(a), str(b)],
                        result_type=str(output),
                        kwargs={},
                    )
                ],
                return_values=["%o"],
            )
        ]
    )


def _compile_scheduled(m: int, n: int, k: int) -> tuple[Any, Any, float]:
    artifact = lower_scheduled_matmul(_module(m, n, k), target="rocm_gfx1151")
    started = time.perf_counter_ns()
    package = rocm_native.package_scheduled_matmul(
        artifact,
        pipeline_name="tessera-lower-to-rocm",
    )
    return artifact, package, (time.perf_counter_ns() - started) / 1e6


def _resident_wall_ms(session: DeviceCase, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        if session.launch() != 0:
            raise RuntimeError("gfx1151 warmup launch failed")
    if session.hip.hipDeviceSynchronize() != 0:
        raise RuntimeError("gfx1151 warmup synchronization failed")
    started = time.perf_counter_ns()
    for _ in range(iterations):
        if session.launch() != 0:
            raise RuntimeError("gfx1151 timed launch failed")
    if session.hip.hipDeviceSynchronize() != 0:
        raise RuntimeError("gfx1151 timed synchronization failed")
    return (time.perf_counter_ns() - started) / 1e6 / iterations


def _correctness_row(hip: Any, direct_hsaco: bytes, shape: tuple[int, int, int]) -> dict[str, Any]:
    m, n, k = shape
    case = Case("e2e_real4", m, n, k)
    a, b, _ = _inputs(case)
    expected = _reference(case, a, b, None)
    assert expected is not None
    artifact, package, compile_ms = _compile_scheduled(m, n, k)
    direct = DeviceCase(hip, direct_hsaco, case, (2, 4), a, b, None)
    scheduled = DeviceCase(
        hip,
        package.image.payload,
        case,
        (artifact.macro_tile_m // 16, artifact.macro_tile_n // 16),
        a,
        b,
        None,
    )
    try:
        direct_output = direct.download()
        scheduled_output = scheduled.download()
    finally:
        scheduled.close()
        direct.close()
    direct_error = _error_metrics(direct_output, expected)
    scheduled_error = _error_metrics(scheduled_output, expected)
    parity_error = _error_metrics(scheduled_output, direct_output)
    passed = _correct(case, direct_error) and _correct(case, scheduled_error)
    return {
        "shape_mnk": list(shape),
        "shape_class": "aligned" if all(value % 16 == 0 for value in shape) else "ragged",
        "direct": direct_error,
        "scheduled": scheduled_error,
        "route_parity": parity_error,
        "passed": passed,
        "scheduled_compile_ms": compile_ms,
        "digests": {
            "graph": artifact.graph_digest,
            "schedule": artifact.schedule_ir_digest,
            "decision": artifact.schedule_digest,
            "tile": artifact.tile_digest,
            "target": package.image.target_ir_digest,
            "image": package.image.image_digest,
        },
    }


def decide(*, correct: bool, tflops: float, direct_ratio: float, selector_eligible: bool) -> str:
    if not correct or tflops < BASELINE_TFLOPS:
        return "reject"
    if selector_eligible and direct_ratio >= DIRECT_PARITY_FLOOR:
        return "promote"
    return "retain"


def run(*, size: int, trials: int, iterations: int, warmup: int) -> dict[str, Any]:
    if trials < 3 or iterations < 1 or warmup < 1:
        raise ValueError("E2E-REAL-4 requires >=3 trials, >=1 iteration, and >=1 warmup")
    if not rocm_native.native_packaging_available():
        raise RuntimeError("ROCm native packaging is unavailable")
    mlir_opt = _find_mlir_opt()
    hip = _load_hip()
    if mlir_opt is None or hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("an exact live gfx1151 compiler/device is required")
    architecture = rt._rocm_chip()
    if architecture != "gfx1151":
        raise RuntimeError(f"E2E-REAL-4 requires gfx1151, got {architecture}")

    direct_started = time.perf_counter_ns()
    direct_hsaco = build_direct(mlir_opt, "direct", 2, 4)
    direct_compile_ms = (time.perf_counter_ns() - direct_started) / 1e6
    correctness = [_correctness_row(hip, direct_hsaco, shape) for shape in CORRECTNESS_CASES]

    rocm_native._cache.clear()
    artifact, scheduled_package, scheduled_cold_ms = _compile_scheduled(size, size, size)
    _, scheduled_warm_package, scheduled_warm_ms = _compile_scheduled(size, size, size)
    if scheduled_package.image.image_digest != scheduled_warm_package.image.image_digest:
        raise RuntimeError("scheduled gfx1151 image changed between cold and warm builds")

    perf_case = Case("e2e_real4", size, size, size)
    a, b, _ = _inputs(perf_case)
    direct = DeviceCase(hip, direct_hsaco, perf_case, (2, 4), a, b, None)
    scheduled = DeviceCase(
        hip,
        scheduled_package.image.payload,
        perf_case,
        (artifact.macro_tile_m // 16, artifact.macro_tile_n // 16),
        a,
        b,
        None,
    )
    direct_samples: list[float] = []
    scheduled_samples: list[float] = []
    try:
        for trial in range(trials):
            routes = ((direct, direct_samples), (scheduled, scheduled_samples))
            if trial & 1:
                routes = tuple(reversed(routes))
            for session, samples in routes:
                samples.append(_resident_wall_ms(session, warmup=warmup, iterations=iterations))
    finally:
        scheduled.close()
        direct.close()

    direct_median = statistics.median(direct_samples)
    scheduled_median = statistics.median(scheduled_samples)
    flop = 2.0 * size * size * size
    direct_tflops = flop / (direct_median / 1e3) / 1e12
    scheduled_tflops = flop / (scheduled_median / 1e3) / 1e12
    direct_ratio = scheduled_tflops / direct_tflops
    baseline_ratio = scheduled_tflops / BASELINE_TFLOPS
    # HIP events return zero/invalid data on the WSL /dev/dxg path. Preserve
    # the valid exact-device host-wall comparison, but never promote from it.
    selector_eligible = False
    decision = decide(
        correct=all(row["passed"] for row in correctness),
        tflops=scheduled_tflops,
        direct_ratio=direct_ratio,
        selector_eligible=selector_eligible,
    )
    return {
        "schema": SCHEMA,
        "work_item": "E2E-REAL-4",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": architecture,
        "device": _gpu_name(),
        "host": platform.platform(),
        "python": platform.python_version(),
        "timing_domain": "resident_host_wall_synchronized",
        "timing_source_reason": "WSL /dev/dxg HIP events return invalid zero samples",
        "selector_eligible": selector_eligible,
        "warmup_launches_per_trial": warmup,
        "iterations_per_trial": iterations,
        "trials": trials,
        "correctness": correctness,
        "compile": {
            "direct_ms": direct_compile_ms,
            "scheduled_cold_ms": scheduled_cold_ms,
            "scheduled_warm_ms": scheduled_warm_ms,
            "scheduled_cold_state": scheduled_package.image.compile_state,
            "scheduled_warm_state": scheduled_warm_package.image.compile_state,
            "compiler_fingerprint": scheduled_warm_package.image.compiler_fingerprint,
            "toolchain_fingerprint": scheduled_warm_package.image.toolchain_fingerprint,
            "device_libraries": [item.to_dict() for item in scheduled_warm_package.image.device_libraries],
        },
        "digests": {
            "graph": artifact.graph_digest,
            "schedule": artifact.schedule_ir_digest,
            "decision": artifact.schedule_digest,
            "tile": artifact.tile_digest,
            "target": scheduled_warm_package.image.target_ir_digest,
            "scheduled_image": scheduled_warm_package.image.image_digest,
            "direct_image": hashlib.sha256(direct_hsaco).hexdigest(),
        },
        "resources": {
            "scheduled": code_object_resources(scheduled_warm_package.image.payload),
            "direct": code_object_resources(direct_hsaco),
        },
        "performance": {
            "shape_mnk": [size, size, size],
            "direct_samples_ms": direct_samples,
            "scheduled_samples_ms": scheduled_samples,
            "direct_median_ms": direct_median,
            "scheduled_median_ms": scheduled_median,
            "direct_tflops": direct_tflops,
            "scheduled_tflops": scheduled_tflops,
            "committed_floor_tflops": BASELINE_TFLOPS,
            "scheduled_over_floor": baseline_ratio,
            "scheduled_over_direct": direct_ratio,
            "direct_parity_floor": DIRECT_PARITY_FLOOR,
        },
        "verdict": decision,
        "selector_changed": decision == "promote",
        "next_action": (
            "none"
            if decision == "promote"
            else (
                "reject production promotion; retain only the correctness/debug path and resolve launch tiling plus address sharing before remeasurement"
                if decision == "reject"
                else "retain scheduled candidate; obtain valid device-event timing before promotion"
            )
        ),
        "report_sha256": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        size=args.size,
        trials=args.trials,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["report_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    perf = report["performance"]
    print(
        f"gfx1151 scheduled={perf['scheduled_tflops']:.2f} TFLOP/s "
        f"direct={perf['direct_tflops']:.2f} TFLOP/s "
        f"ratio={perf['scheduled_over_direct']:.3f} verdict={report['verdict']}",
        flush=True,
    )
    return 0 if report["verdict"] != "reject" else 1


if __name__ == "__main__":
    raise SystemExit(main())
