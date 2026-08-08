#!/usr/bin/env python3
"""E2E-REAL-4 exact-host comparison for the scheduled AVX-512 matmul lane."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from tessera import runtime as rt  # noqa: E402
from tessera.compiler.graph_ir import (  # noqa: E402
    GraphIRFunction,
    GraphIRModule,
    IRArg,
    IROp,
    IRType,
)
from tessera.compiler.scheduled_matmul import lower_scheduled_matmul  # noqa: E402
from tessera.compiler.x86_native import (  # noqa: E402
    package_matmul,
    package_scheduled_matmul,
    tools_available,
)

SCHEMA = "tessera.compiler.e2e_real4.x86_matmul.v1"
SHAPES = ((64, 128, 96), (127, 65, 79))  # M, K, N; aligned + ragged.
NON_REGRESSION_LIMIT = 1.10


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or "unknown"


def _tensor(shape: tuple[int, int]) -> IRType:
    return IRType(
        f"tensor<{shape[0]}x{shape[1]}xf32>",
        tuple(map(str, shape)),
        "fp32",
    )


def _module(shape: tuple[int, int, int]) -> GraphIRModule:
    m, k, n = shape
    a, b, output = _tensor((m, k)), _tensor((k, n)), _tensor((m, n))
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="e2e_real4_x86_matmul",
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


def _runtime_artifact(package: Any) -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(
        metadata={"target": "x86", "compiler_path": "e2e_real4"},
        tile_ir=package.tile_ir,
        target_ir=package.target_ir,
        native_image=package.image,
        launch_descriptor=package.descriptor,
    )


def _launch_ms(artifact: rt.RuntimeArtifact, arguments: dict[str, Any]) -> float:
    started = time.perf_counter_ns()
    result = rt.launch(artifact, arguments)
    elapsed = (time.perf_counter_ns() - started) / 1e6
    if not result.get("ok"):
        raise RuntimeError(f"AVX-512 launch failed: {result.get('reason')}")
    return elapsed


def _compile_pair(module: GraphIRModule) -> tuple[Any, Any, dict[str, Any]]:
    started = time.perf_counter_ns()
    production = package_matmul(module, pipeline_name="tessera-lower-to-x86")
    production_ms = (time.perf_counter_ns() - started) / 1e6

    started = time.perf_counter_ns()
    scheduled_artifact = lower_scheduled_matmul(module, target="x86")
    scheduled = package_scheduled_matmul(
        scheduled_artifact,
        pipeline_name="tessera-lower-to-x86",
    )
    scheduled_cold_ms = (time.perf_counter_ns() - started) / 1e6
    started = time.perf_counter_ns()
    warm_artifact = lower_scheduled_matmul(module, target="x86")
    scheduled_warm = package_scheduled_matmul(
        warm_artifact,
        pipeline_name="tessera-lower-to-x86",
    )
    scheduled_warm_ms = (time.perf_counter_ns() - started) / 1e6
    if scheduled.image.image_digest != scheduled_warm.image.image_digest:
        raise RuntimeError("scheduled AVX-512 image changed between cold and warm builds")
    return (
        production,
        scheduled,
        {
            "production_ms": production_ms,
            "scheduled_cold_ms": scheduled_cold_ms,
            "scheduled_warm_ms": scheduled_warm_ms,
            "scheduled_compile_state": scheduled.image.compile_state,
            "compiler_fingerprint": scheduled.image.compiler_fingerprint,
            "toolchain_fingerprint": scheduled.image.toolchain_fingerprint,
            "digests": {
                "graph": scheduled_artifact.graph_digest,
                "schedule": scheduled_artifact.schedule_ir_digest,
                "decision": scheduled_artifact.schedule_digest,
                "tile": scheduled_artifact.tile_digest,
                "target": scheduled.image.target_ir_digest,
                "image": scheduled.image.image_digest,
                "production_image": production.image.image_digest,
            },
        },
    )


def _summarize(production: list[float], scheduled: list[float]) -> dict[str, Any]:
    production_median = statistics.median(production)
    scheduled_median = statistics.median(scheduled)
    return {
        "production_samples_ms": production,
        "scheduled_samples_ms": scheduled,
        "production_median_ms": production_median,
        "scheduled_median_ms": scheduled_median,
        "scheduled_over_production": scheduled_median / production_median,
        "non_regression_10pct": scheduled_median <= production_median * NON_REGRESSION_LIMIT,
    }


def verdict(rows: list[dict[str, Any]]) -> str:
    if not all(row["correctness"]["passed"] for row in rows):
        return "reject"
    return "promote" if all(row["timing"]["non_regression_10pct"] for row in rows) else "retain"


def run(*, trials: int, warmup: int) -> dict[str, Any]:
    if trials < 3 or warmup < 1:
        raise ValueError("E2E-REAL-4 requires at least three trials and one warmup")
    if not tools_available():
        raise RuntimeError("the production x86 compiler/image is unavailable")
    rng = np.random.default_rng(4004)
    rows = []
    for shape in SHAPES:
        m, k, n = shape
        production_package, scheduled_package, compile_record = _compile_pair(_module(shape))
        production = _runtime_artifact(production_package)
        scheduled = _runtime_artifact(scheduled_package)
        a = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=np.float32)
        b = np.ascontiguousarray(rng.standard_normal((k, n)), dtype=np.float32)
        expected = a @ b
        production_output = np.zeros((m, n), dtype=np.float32)
        scheduled_output = np.zeros((m, n), dtype=np.float32)
        production_args = {"a": a, "b": b, "o": production_output, "M": m, "N": n, "K": k}
        scheduled_args = {"a": a, "b": b, "o": scheduled_output, "M": m, "N": n, "K": k}
        for _ in range(warmup):
            _launch_ms(production, production_args)
            _launch_ms(scheduled, scheduled_args)
        production_error = float(np.max(np.abs(production_output - expected)))
        scheduled_error = float(np.max(np.abs(scheduled_output - expected)))
        parity_error = float(np.max(np.abs(scheduled_output - production_output)))
        passed = max(production_error, scheduled_error) <= 3e-5
        production_samples: list[float] = []
        scheduled_samples: list[float] = []
        for trial in range(trials):
            routes = (
                (production, production_args, production_samples),
                (scheduled, scheduled_args, scheduled_samples),
            )
            if trial & 1:
                routes = tuple(reversed(routes))
            for artifact, arguments, samples in routes:
                samples.append(_launch_ms(artifact, arguments))
        row = {
            "shape_mkn": list(shape),
            "shape_class": "aligned" if all(value % 16 == 0 for value in shape) else "ragged",
            "correctness": {
                "production_max_abs": production_error,
                "scheduled_max_abs": scheduled_error,
                "route_parity_max_abs": parity_error,
                "tolerance": 3e-5,
                "passed": passed,
            },
            "compile": compile_record,
            "resources": {
                "scheduled_image_bytes": len(scheduled_package.image.payload),
                "production_image_bytes": len(production_package.image.payload),
                "required_features": ["avx512f", "fma"],
                "threads": "OpenMP runtime owned",
            },
            "timing": _summarize(production_samples, scheduled_samples),
        }
        rows.append(row)
        print(
            f"x86 {shape}: parity={parity_error:.3e} "
            f"scheduled/production={row['timing']['scheduled_over_production']:.3f}x",
            flush=True,
        )
    decision = verdict(rows)
    return {
        "schema": SCHEMA,
        "work_item": "E2E-REAL-4",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "zen5-avx512",
        "device": _cpu_model(),
        "host": platform.platform(),
        "python": platform.python_version(),
        "timing_domain": "host_wall_operation_total",
        "timing_policy": "serial alternating production/scheduled runtime.launch calls",
        "warmup_iterations": warmup,
        "trials": trials,
        "ratchet": {"kind": "production_non_regression", "limit": NON_REGRESSION_LIMIT},
        "rows": rows,
        "verdict": decision,
        # E2E-REAL-3 already made this the canonical x86 package route.  This
        # packet validates that selection; it does not silently mutate it.
        "selector_changed": False,
        "selector_action": (
            "retain existing canonical scheduled selection" if decision == "promote" else "no selector change"
        ),
        "report_sha256": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(trials=args.trials, warmup=args.warmup)
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["report_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["verdict"] != "reject" else 1


if __name__ == "__main__":
    raise SystemExit(main())
