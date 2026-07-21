#!/usr/bin/env python3
"""Record an exact-host E2E-SPINE-3 packet for portable base x86-64."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]


def _softmax_module(shape: tuple[int, int] = (2, 2)):
    from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType

    tensor = IRType(
        f"tensor<{shape[0]}x{shape[1]}xf32>", tuple(map(str, shape)), "fp32",
    )
    return GraphIRModule(functions=[GraphIRFunction(
        name="fleet_softmax", args=[IRArg("x", tensor)], result_types=[tensor],
        body=[IROp(
            result="o", op_name="tessera.softmax", operands=["%x"],
            operand_types=[str(tensor)], result_type=str(tensor), kwargs={"axis": -1},
        )], return_values=["%o"],
    )])


def _reduction_module(shape: tuple[int, int] = (2, 3)):
    from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType

    source = IRType(
        f"tensor<{shape[0]}x{shape[1]}xf32>", tuple(map(str, shape)), "fp32",
    )
    result = IRType(f"tensor<{shape[0]}xf32>", (str(shape[0]),), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="fleet_reduction", args=[IRArg("x", source)], result_types=[result],
        body=[IROp(
            result="o", op_name="tessera.sum", operands=["%x"],
            operand_types=[str(source)], result_type=str(result),
            kwargs={"axis": -1, "keepdims": False},
        )], return_values=["%o"],
    )])


def _two_run_medians_ns(
    call: Callable[[], object], *, samples: int, iterations: int,
) -> list[float]:
    call()  # First-use loader/compiler/cache effects never enter the timing sample.
    cohorts: tuple[list[float], list[float]] = ([], [])
    for sample in range(samples):
        order = (0, 1) if sample % 2 == 0 else (1, 0)
        for cohort in order:
            started = time.perf_counter_ns()
            for _ in range(iterations):
                call()
            cohorts[cohort].append((time.perf_counter_ns() - started) / iterations)
    return [float(statistics.median(values)) for values in cohorts]


def _stability(run_medians: list[float]) -> float:
    return (max(run_medians) - min(run_medians)) / min(run_medians) * 100.0


def _cpu_identity() -> tuple[str, list[str]]:
    text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    model = next(
        line.split(":", 1)[1].strip()
        for line in text.splitlines() if line.startswith("model name")
    )
    flags_line = next(
        line.split(":", 1)[1].strip()
        for line in text.splitlines() if line.startswith("flags")
    )
    return model, sorted(flags_line.split())


def record(*, samples: int, iterations: int, stability_limit: float) -> tuple[dict, dict]:
    from tessera import runtime as rt
    from tessera.compiler.e2e_fleet import load_fixture_corpus, validate_backend_report
    from tessera.compiler.x86_native import (
        X86_BASE_ARCHITECTURE, package_reduction, package_softmax,
    )

    fixtures = load_fixture_corpus()
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    model, flags = _cpu_identity()
    if "avx512f" in flags:
        raise RuntimeError("base-x86 packet must be recorded on the assigned non-AVX512 host")
    available_cpus = os.sched_getaffinity(0)
    measurement_cpu = min(available_cpus)
    os.sched_setaffinity(0, {measurement_cpu})

    inputs = {
        "softmax": np.array([[0.0, 0.0], [1000.0, 1000.0]], dtype=np.float32),
        "reduction": np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]], dtype=np.float32),
    }
    definitions = (
        ("softmax", "softmax-f32-2x2-extreme-v1", _softmax_module(), package_softmax),
        ("reduction", "reduce-sum-f32-axis1-v1", _reduction_module(), package_reduction),
    )
    fixture_rows, cache_rows, benchmark_rows, resource_rows = [], [], [], []
    toolchains: set[str] = set()
    for family, fixture_id, module, packager in definitions:
        cold = packager(
            module, pipeline_name="tessera-lower-to-x86",
            architecture=X86_BASE_ARCHITECTURE,
        )
        warm = packager(
            module, pipeline_name="tessera-lower-to-x86",
            architecture=X86_BASE_ARCHITECTURE,
        )
        toolchains.add(cold.image.toolchain_fingerprint)
        if (
            cold.image.cache_key != warm.image.cache_key
            or cold.image.image_digest != warm.image.image_digest
            or cold.descriptor.descriptor_digest != warm.descriptor.descriptor_digest
        ):
            raise RuntimeError(f"{family} prepackaged image/descriptor is not reproducible")
        artifact = rt.RuntimeArtifact(
            metadata={"target": "x86", "architecture": X86_BASE_ARCHITECTURE},
            native_image=cold.image, launch_descriptor=cold.descriptor,
            tile_ir=cold.tile_ir, target_ir=cold.target_ir,
        )
        x = inputs[family]
        output = np.zeros_like(x) if family == "softmax" else np.zeros((2,), np.float32)
        bindings = (
            {"x": x, "o": output, "Rows": 2, "K": 2}
            if family == "softmax"
            else {"x": x, "o": output, "Outer": 2, "AxisExtent": 3, "Inner": 1}
        )
        launch_result = rt.launch(artifact, bindings)
        if not launch_result["ok"]:
            raise RuntimeError(str(launch_result.get("reason")))
        expected = np.asarray(fixtures[fixture_id]["oracle"], dtype=np.float32)
        np.testing.assert_allclose(output, expected, rtol=2e-6, atol=2e-6)

        timing_shape = (64, 256)
        timing_module = (
            _softmax_module(timing_shape) if family == "softmax"
            else _reduction_module(timing_shape)
        )
        timing_package = packager(
            timing_module, pipeline_name="tessera-lower-to-x86",
            architecture=X86_BASE_ARCHITECTURE,
        )
        timing_artifact = rt.RuntimeArtifact(
            metadata={"target": "x86", "architecture": X86_BASE_ARCHITECTURE},
            native_image=timing_package.image,
            launch_descriptor=timing_package.descriptor,
            tile_ir=timing_package.tile_ir, target_ir=timing_package.target_ir,
        )
        timing_x = np.linspace(-1.0, 1.0, num=np.prod(timing_shape), dtype=np.float32).reshape(timing_shape)
        timing_output = (
            np.zeros_like(timing_x) if family == "softmax"
            else np.zeros((timing_shape[0],), np.float32)
        )
        timing_bindings = (
            {"x": timing_x, "o": timing_output, "Rows": timing_shape[0], "K": timing_shape[1]}
            if family == "softmax"
            else {
                "x": timing_x, "o": timing_output, "Outer": timing_shape[0],
                "AxisExtent": timing_shape[1], "Inner": 1,
            }
        )
        library = rt._load_x86_native_image(timing_package.image)
        function = getattr(library, timing_package.descriptor.entry_symbol)
        pointer = ctypes.POINTER(ctypes.c_float)
        if family == "softmax":
            function.argtypes = [pointer, ctypes.c_int64, ctypes.c_int64, pointer]
            direct = lambda: function(
                timing_x.ctypes.data_as(pointer), ctypes.c_int64(timing_shape[0]),
                ctypes.c_int64(timing_shape[1]), timing_output.ctypes.data_as(pointer),
            )
        else:
            function.argtypes = [pointer, ctypes.c_int64, ctypes.c_int64, pointer, ctypes.c_int]
            direct = lambda: function(
                timing_x.ctypes.data_as(pointer), ctypes.c_int64(timing_shape[0]),
                ctypes.c_int64(timing_shape[1]), timing_output.ctypes.data_as(pointer),
                ctypes.c_int(0),
            )
        end_to_end = lambda: rt.launch(timing_artifact, timing_bindings)
        run_medians = {
            "kernel_wall": _two_run_medians_ns(
                direct, samples=samples, iterations=iterations,
            ),
            "end_to_end": _two_run_medians_ns(
                end_to_end, samples=samples, iterations=iterations,
            ),
        }
        resource_fingerprint = hashlib.sha256(json.dumps({
            "architecture": X86_BASE_ARCHITECTURE,
            "entry": timing_package.descriptor.entry_symbol,
            "image_digest": timing_package.image.image_digest,
            "timing_shape": list(timing_shape),
            "instruction_envelope": "x86-64; no AVX/AVX2/AVX512",
        }, sort_keys=True).encode()).hexdigest()
        for domain, medians in run_medians.items():
            benchmark_rows.append({
                "family": family, "route": "x86_64_base_c_abi",
                "timing_domain": domain,
                "median_ns": float(statistics.median(medians)),
                "run_medians_ns": medians,
                "stability_limit_pct": stability_limit,
                "stable": _stability(medians) <= stability_limit,
                "selected": True, "repetitions": samples * iterations,
                "warmups": 1, "discard_first": True,
                "resource_fingerprint": resource_fingerprint,
            })
        fixture_rows.append({
            "fixture_id": fixture_id,
            "levels": {"a": "proven", "b": "proven", "c": "proven"},
            "actual": output.tolist(), "image_digest": cold.image.image_digest,
            "descriptor_digest": cold.descriptor.descriptor_digest,
        })
        state = {
            "compile_state": "prepackaged", "cache_key": cold.image.cache_key,
            "image_digest": cold.image.image_digest,
            "descriptor_digest": cold.descriptor.descriptor_digest,
        }
        cache_rows.append({"fixture_id": fixture_id, "cold": state, "warm": dict(state)})
        resource_rows.append({
            "family": family, "resource_fingerprint": resource_fingerprint,
            "entry": timing_package.descriptor.entry_symbol,
            "image_digest": timing_package.image.image_digest,
            "timing_shape": list(timing_shape),
            "instruction_envelope": "x86-64; no AVX/AVX2/AVX512",
        })

    toolchain = hashlib.sha256("\n".join(sorted(toolchains)).encode()).hexdigest()
    report = {
        "schema": "tessera.e2e-backend-report.v1",
        "target": "x86", "architecture": "x86_64_base",
        "device": {"exact": True, "identity": f"{platform.node()} | {model}"},
        "source_commit": source_commit, "toolchain_fingerprint": toolchain,
        "scope": ["softmax", "reduction"],
        "required_timing_domains": ["kernel_wall", "end_to_end"],
        "fixtures": fixture_rows, "cache_proofs": cache_rows,
        "benchmarks": benchmark_rows,
    }
    validate_backend_report(report)
    resources = {
        "schema": "tessera.e2e-x86-resource-record.v1",
        "device": {"model": model, "flags": flags},
        "measurement_cpu": measurement_cpu,
        "compile_flags": ["-march=x86-64", "-mno-avx", "-mno-avx2", "-mno-avx512f"],
        "rows": resource_rows,
    }
    return report, resources


def main() -> int:
    from tessera.compiler.e2e_fleet import seal_packet

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--stability-limit", type=float, default=4.0)
    parser.add_argument("--packet-dir", type=Path)
    args = parser.parse_args()
    if args.samples < 2 or args.iterations < 10:
        parser.error("use at least two samples and ten amortized iterations")
    report, resources = record(
        samples=args.samples, iterations=args.iterations,
        stability_limit=args.stability_limit,
    )
    if args.packet_dir:
        args.packet_dir.mkdir(parents=True, exist_ok=True)
        (args.packet_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        (args.packet_dir / "resources.json").write_text(
            json.dumps(resources, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        seal_packet(args.packet_dir)
        print(f"sealed {args.packet_dir}")
    else:
        print(json.dumps({"report": report, "resources": resources}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
