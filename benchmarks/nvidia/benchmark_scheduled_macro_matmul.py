"""Compare the canonical scheduled SM120 producer with its direct fallback.

The default mode performs correctness first and then reports independent
CUDA-event samples.  ``--profile-route`` launches exactly one resident-buffer
kernel after validation, making it suitable for a separate Nsight Compute run.
This benchmark records evidence only; it never changes selector policy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from tessera import runtime as rt  # noqa: E402
from tessera.compiler import nvidia_native, scheduled_matmul  # noqa: E402
from tessera.compiler.graph_ir import (  # noqa: E402
    GraphIRFunction,
    GraphIRModule,
    IRArg,
    IROp,
    IRType,
)

PROOF_CASES = (
    (16, 32, 8),
    (32, 32, 16),
    (48, 64, 24),
    # Exercises arbitrary packed-row alignment through masked scalar staging.
    (257, 513, 257),
    # Exercises the aligned cp.async path with a partial final K panel.
    (257, 520, 257),
)
# Larger aligned cases separate kernel work from launch/event quantization and
# expose the mathematical cost of independently loading each 16x8 output tile.
PERFORMANCE_CASES = (
    (128, 128, 128),
    (128, 256, 64),
    (256, 128, 256),
    (256, 256, 128),
    (256, 256, 256),
    (256, 512, 256),
    (512, 256, 512),
    (512, 512, 512),
)
CASES = PROOF_CASES + PERFORMANCE_CASES
MACRO_CROSSOVER_FLOPS = 67_108_864


def _traffic_model(
    shape: tuple[int, int, int], route: str
) -> dict[str, float | int | str]:
    """Model logical panel loads for the selected physical route.

    This is logical traffic before cache effects. It exposes the algorithmic
    reuse gap without claiming that every repeated load reaches DRAM; Nsight
    supplies that observation separately.
    """
    m, k, n = shape
    macro = route == "scheduled" and scheduled_matmul._uses_sm120_macro_cta(
        m, n, k, "f16", "f32"
    )
    if macro:
        workgroups = ((m + 31) // 32) * ((n + 31) // 32)
        logical_input_bytes = workgroups * (32 * k + 32 * k) * 2
        physical_route = (
            "macro_cta_cp_async_2stage_shared_ab_f16"
            if k % 8 == 0
            else "macro_cta_masked_scalar_shared_ab_f16"
        )
    else:
        workgroups = ((m + 15) // 16) * ((n + 7) // 8)
        logical_input_bytes = workgroups * (16 * k + 8 * k) * 2
        physical_route = "independent_warp_global"
    unique_input_bytes = (m * k + k * n) * 2
    return {
        "physical_route": physical_route,
        "workgroups": workgroups,
        "logical_input_bytes": logical_input_bytes,
        "unique_input_bytes": unique_input_bytes,
        "logical_input_redundancy": logical_input_bytes / unique_input_bytes,
        "output_bytes": m * n * 4,
        "flops": 2 * m * n * k,
    }


def _resources(package) -> dict[str, object] | None:
    record = package.image.resource_record
    return record.to_dict() if record is not None else None


def _decision(rows: list[dict[str, Any]]) -> dict[str, object]:
    eligible = [
        row for row in rows
        if 2 * int(row["shape_mkn"][0]) * int(row["shape_mkn"][1])
        * int(row["shape_mkn"][2]) >= MACRO_CROSSOVER_FLOPS
    ]
    low_variance = all(
        max(float(value) for value in row["sample_cov"].values()) <= 0.03
        for row in eligible
    )
    material = all(
        float(row["scheduled_over_direct"]) <= 0.97 for row in eligible
    )
    return {
        "route": "sm120_scheduled_macro_cta_32x32",
        "minimum_flops": MACRO_CROSSOVER_FLOPS,
        "eligible_rows": len(eligible),
        "all_numerical_rows_green": all(
            max(float(value) for value in row["max_abs_error"].values()) <= 2e-4
            for row in rows
        ),
        "eligible_rows_low_variance": low_variance,
        "eligible_rows_at_least_three_percent_faster": material,
        "scheduled_route_enabled": low_variance and material,
        # target_perf.apply_corpus() intentionally rejects WSL as global
        # selector authority. Keep that distinction machine-readable.
        "global_selector_changed": False,
        "selector_eligibility": "pruning_only_wsl",
    }


def _module(shape: tuple[int, int, int]) -> GraphIRModule:
    m, k, n = shape
    a = IRType(f"tensor<{m}x{k}xf16>", (str(m), str(k)), "fp16")
    b = IRType(f"tensor<{k}x{n}xf16>", (str(k), str(n)), "fp16")
    out = IRType(f"tensor<{m}x{n}xf32>", (str(m), str(n)), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="nvidia_sm120_scheduled_matmul_profile",
        args=[IRArg("a", a), IRArg("b", b)],
        result_types=[out],
        body=[IROp(
            result="o", op_name="tessera.matmul", operands=["%a", "%b"],
            operand_types=[str(a), str(b)], result_type=str(out), kwargs={},
        )],
        return_values=["%o"],
    )])


def _packages(shape: tuple[int, int, int]):
    module = _module(shape)
    scheduled = scheduled_matmul.lower_scheduled_matmul(
        module, target="nvidia_sm120")
    return {
        "scheduled": nvidia_native.package_scheduled_matmul(
            module, scheduled,
            pipeline_name="tessera-lower-to-nvidia-sm120"),
        "direct": nvidia_native.package_f16_matmul(
            module, pipeline_name="tessera-lower-to-nvidia-sm120",
            schedule="direct"),
    }


def _inputs(shape: tuple[int, int, int]):
    m, k, n = shape
    rng = np.random.default_rng(120_016 + m + k + n)
    a = np.ascontiguousarray(
        (rng.standard_normal((m, k)) * 0.25).astype(np.float16))
    b = np.asfortranarray(
        (rng.standard_normal((k, n)) * 0.25).astype(np.float16))
    return a, b, np.zeros((m, n), dtype=np.float32)


def _args(shape: tuple[int, int, int], arrays):
    m, k, n = shape
    a, b, out = arrays
    return {"a": a, "b": b, "o": out, "M": m, "N": n, "K": k}


def _validate(package, shape: tuple[int, int, int], arrays) -> float:
    a, b, out = arrays
    artifact = rt.RuntimeArtifact(
        metadata={"target": "nvidia_sm120"},
        native_image=package.image,
        launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir,
        target_ir=package.target_ir,
    )
    result = rt.launch(artifact, _args(shape, arrays))
    if not result.get("ok") or result.get("execution_kind") != "native_gpu":
        raise RuntimeError(f"native validation failed: {result}")
    reference = a.astype(np.float32) @ b.astype(np.float32)
    error = float(np.max(np.abs(out - reference)))
    if error > 2e-4:
        raise RuntimeError(f"numerical validation failed: max_abs_error={error}")
    return error


def _latency(package, shape, arrays, *, warmup: int, reps: int) -> float:
    return rt._nvidia_native_descriptor_device_latency(
        package.image, package.descriptor, _args(shape, arrays),
        warmup=warmup, reps=reps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=["x".join(map(str, s)) for s in CASES])
    parser.add_argument("--profile-route", choices=("scheduled", "direct"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument(
        "--output", type=Path,
        help="write the complete reproducible JSON packet to this path",
    )
    args = parser.parse_args(argv)
    cases = CASES
    if args.shape:
        cases = (tuple(int(v) for v in args.shape.split("x")),)
    rows = []
    for shape in cases:
        packages = _packages(shape)
        arrays = {route: _inputs(shape) for route in packages}
        errors = {
            route: _validate(package, shape, arrays[route])
            for route, package in packages.items()
        }
        if args.profile_route:
            route = args.profile_route
            latency = _latency(
                packages[route], shape, arrays[route], warmup=0, reps=1)
            print(json.dumps({
                "shape_mkn": list(shape), "route": route,
                "entry": packages[route].descriptor.entry_symbol,
                "max_abs_error": errors[route],
                "single_event_ms": latency,
                "traffic_model": _traffic_model(shape, route),
                "compile_resources": _resources(packages[route]),
            }))
            continue
        timings = {
            route: [
                _latency(package, shape, arrays[route],
                         warmup=args.warmup, reps=args.reps)
                for _ in range(args.samples)
            ]
            for route, package in packages.items()
        }
        medians = {route: statistics.median(values)
                   for route, values in timings.items()}
        cov = {
            route: statistics.stdev(values) / statistics.mean(values)
            for route, values in timings.items()
        }
        rows.append({
            "shape_mkn": list(shape), "max_abs_error": errors,
            "cuda_event_ms": timings, "median_ms": medians,
            "sample_cov": cov,
            "scheduled_over_direct": medians["scheduled"] / medians["direct"],
            "traffic_model": {
                route: _traffic_model(shape, route) for route in packages
            },
            "compile_resources": {
                route: _resources(package)
                for route, package in packages.items()
            },
            "selector_changed": False,
        })
    if not args.profile_route:
        packet = {
            "schema": "tessera.nvidia.scheduled-macro-matmul.v3",
            "device": rt._nvidia_device_name(),
            "host": {
                "node": platform.node(),
                "platform": platform.platform(),
                "wsl": "microsoft" in platform.release().lower()
                or "WSL_INTEROP" in os.environ,
            },
            "method": "correctness_then_resident_buffer_cuda_events",
            "decision": _decision(rows),
            "rows": rows,
        }
        encoded = json.dumps(packet, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
