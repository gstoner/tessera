#!/usr/bin/env python3
"""Record the bounded ROCm E2E-SPINE-3 packet on exact gfx1151.

The packet covers the already-closed ROCM-E2E-1/-2 families: softmax,
reduction, paged-KV read, and MoE dispatch. ``selected`` in the packet denotes
the route used for release evidence; it does not change the production ROCm
selector, whose retained-route dispositions remain recorded by ROCM-E2E-1/-2.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]


def _softmax_module(rows: int, columns: int):
    from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType

    tensor = IRType(f"tensor<{rows}x{columns}xf32>", (str(rows), str(columns)), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="fleet_rocm_softmax", args=[IRArg("x", tensor)], result_types=[tensor],
        body=[IROp(
            result="o", op_name="tessera.softmax", operands=["%x"],
            operand_types=[str(tensor)], result_type=str(tensor), kwargs={"axis": -1},
        )], return_values=["%o"],
    )])


def _reduction_module(rows: int, columns: int):
    from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType

    source = IRType(f"tensor<{rows}x{columns}xf32>", (str(rows), str(columns)), "fp32")
    result = IRType(f"tensor<{rows}xf32>", (str(rows),), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="fleet_rocm_reduction", args=[IRArg("x", source)], result_types=[result],
        body=[IROp(
            result="o", op_name="tessera.sum", operands=["%x"],
            operand_types=[str(source)], result_type=str(result),
            kwargs={"axis": -1, "keepdims": False},
        )], return_values=["%o"],
    )])


def _paged_kv_module(
    pages: int, page_size: int, heads: int, dim: int, start: int, tokens: int,
):
    from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType

    page_type = IRType(
        f"tensor<{pages}x{page_size}x{heads}x{dim}xf32>",
        tuple(map(str, (pages, page_size, heads, dim))), "fp32",
    )
    table = IRType(f"tensor<{pages}xi32>", (str(pages),), "int32")
    result = IRType(
        f"tensor<{tokens}x{heads}x{dim}xf32>",
        tuple(map(str, (tokens, heads, dim))), "fp32",
    )
    return GraphIRModule(functions=[GraphIRFunction(
        name="fleet_rocm_paged_kv", args=[IRArg("pages", page_type), IRArg("page_table", table)],
        result_types=[result], body=[IROp(
            result="slice", op_name="tessera.kv_cache.read",
            operands=["%pages", "%page_table"], operand_types=[str(page_type), str(table)],
            result_type=str(result), kwargs={"start": start, "end": start + tokens},
        )], return_values=["%slice"],
    )])


def _moe_module(tokens: int, slots: int, hidden: int):
    from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType

    source = IRType(f"tensor<{tokens}x{hidden}xf32>", (str(tokens), str(hidden)), "fp32")
    token = IRType(f"tensor<{slots}xi32>", (str(slots),), "int32")
    result = IRType(f"tensor<{slots}x{hidden}xf32>", (str(slots), str(hidden)), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="fleet_rocm_moe", args=[IRArg("x", source), IRArg("token", token)],
        result_types=[result], body=[IROp(
            result="o", op_name="tessera.moe_dispatch", operands=["%x", "%token"],
            operand_types=[str(source), str(token)], result_type=str(result), kwargs={},
        )], return_values=["%o"],
    )])


def _artifact(package):
    from tessera import runtime as rt

    return rt.RuntimeArtifact(
        graph_ir="graph", tile_ir=package.tile_ir, target_ir=package.target_ir,
        metadata={
            "target": "rocm_gfx1151",
            "compiler_path": "rocm_gfx1151_native_descriptor",
        },
        native_image=package.image, launch_descriptor=package.descriptor,
    )


def _memref(pointer: ctypes.c_void_p, size: int) -> list[object]:
    return [
        ctypes.c_void_p(pointer.value), ctypes.c_void_p(pointer.value),
        ctypes.c_int64(0), ctypes.c_int64(size), ctypes.c_int64(1),
    ]


class _ResidentKernel:
    def __init__(
        self, hip: ctypes.CDLL, package, inputs: tuple[np.ndarray, ...],
        output: np.ndarray, dimensions: tuple[int, ...], grid_x: int,
    ) -> None:
        self.hip = hip
        self.module = ctypes.c_void_p()
        self.function = ctypes.c_void_p()
        self.devices: list[ctypes.c_void_p] = []
        self.output = output
        self.grid_x = grid_x
        if hip.hipModuleLoadData(ctypes.byref(self.module), package.image.payload):
            raise RuntimeError("ROCm fleet packet module load failed")
        if hip.hipModuleGetFunction(
            ctypes.byref(self.function), self.module,
            package.descriptor.entry_symbol.encode(),
        ):
            self.close()
            raise RuntimeError("ROCm fleet packet entry symbol is missing")
        try:
            for array in (*inputs, output):
                device = ctypes.c_void_p()
                if hip.hipMalloc(ctypes.byref(device), array.nbytes):
                    raise RuntimeError("ROCm fleet packet hipMalloc failed")
                self.devices.append(device)
            for device, array in zip(self.devices[:-1], inputs, strict=True):
                if hip.hipMemcpy(
                    device, array.ctypes.data_as(ctypes.c_void_p), array.nbytes, 1,
                ):
                    raise RuntimeError("ROCm fleet packet H2D copy failed")
        except Exception:
            self.close()
            raise
        values: list[object] = []
        for device, array in zip(self.devices[:-1], inputs, strict=True):
            values.extend(_memref(device, array.size))
        values.extend(_memref(self.devices[-1], output.size))
        values.extend(ctypes.c_int64(value) for value in dimensions)
        self._values = values
        self.arguments = (ctypes.c_void_p * len(values))()
        for index, value in enumerate(values):
            self.arguments[index] = ctypes.cast(ctypes.byref(value), ctypes.c_void_p)

    def launch(self) -> None:
        rc = self.hip.hipModuleLaunchKernel(
            self.function, self.grid_x, 1, 1, 256, 1, 1, 0, None,
            self.arguments, None,
        )
        if rc:
            raise RuntimeError(f"ROCm fleet packet launch failed rc={rc}")

    def synchronize(self) -> None:
        if self.hip.hipDeviceSynchronize():
            raise RuntimeError("ROCm fleet packet synchronization failed")

    def read(self) -> np.ndarray:
        self.synchronize()
        if self.hip.hipMemcpy(
            self.output.ctypes.data_as(ctypes.c_void_p), self.devices[-1],
            self.output.nbytes, 2,
        ):
            raise RuntimeError("ROCm fleet packet D2H copy failed")
        return self.output

    def close(self) -> None:
        for device in reversed(self.devices):
            if device.value:
                self.hip.hipFree(device)
        self.devices.clear()
        if self.module.value:
            self.hip.hipModuleUnload(self.module)
            self.module = ctypes.c_void_p()


def _two_run_medians_ns(
    call: Callable[[], object], *, samples: int, iterations: int,
) -> list[float]:
    call()
    cohorts: tuple[list[float], list[float]] = ([], [])
    for sample in range(samples):
        for cohort in ((0, 1) if sample % 2 == 0 else (1, 0)):
            started = time.perf_counter_ns()
            for _ in range(iterations):
                call()
            cohorts[cohort].append((time.perf_counter_ns() - started) / iterations)
    return [float(statistics.median(values)) for values in cohorts]


def _resident_two_run_medians_ns(
    session: _ResidentKernel, *, samples: int, iterations: int,
) -> list[float]:
    """Amortize WSL submission jitter without relying on invalid HIP events."""

    session.launch()
    session.synchronize()
    cohorts: tuple[list[float], list[float]] = ([], [])
    for sample in range(samples):
        for cohort in ((0, 1) if sample % 2 == 0 else (1, 0)):
            started = time.perf_counter_ns()
            for _ in range(iterations):
                session.launch()
            session.synchronize()
            cohorts[cohort].append((time.perf_counter_ns() - started) / iterations)
    return [float(statistics.median(values)) for values in cohorts]


def _stability(values: list[float]) -> float:
    return (max(values) - min(values)) / min(values) * 100.0


def _device_identity() -> str:
    result = subprocess.run(["rocminfo"], text=True, capture_output=True, check=True)
    lines = [line.strip() for line in result.stdout.splitlines()]
    gfx = next((line for line in lines if line.startswith("Name:") and "gfx1151" in line), "")
    marketing = next(
        (line for line in reversed(lines) if line.startswith("Marketing Name:") and "Radeon" in line),
        "",
    )
    return " | ".join(part for part in (platform.node(), gfx, marketing) if part)


def record(
    *, samples: int, kernel_iterations: int, e2e_iterations: int,
    stability_limit: float,
) -> tuple[dict, dict]:
    from tessera import runtime as rt
    from tessera.compiler import rocm_native
    from tessera.compiler.e2e_fleet import load_fixture_corpus, validate_backend_report

    if rt._rocm_chip() != "gfx1151":
        raise RuntimeError(f"ROCm fleet packet requires exact gfx1151, got {rt._rocm_chip()}")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):
        raise RuntimeError("ROCm fleet packet requires a usable HIP device")
    if hasattr(os, "sched_getaffinity"):
        cpu = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cpu})
    else:
        cpu = None
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    corpus = load_fixture_corpus()
    definitions = (
        {
            "family": "softmax", "fixture": "softmax-f32-2x2-extreme-v1",
            "module": _softmax_module(2, 2), "packager": rocm_native.package_softmax,
            "inputs": (np.array(corpus["softmax-f32-2x2-extreme-v1"]["inputs"]["x"], np.float32),),
            "output": np.zeros((2, 2), np.float32), "dims": (2, 2), "grid": 2,
            "bindings": lambda values, out: {"x": values[0], "o": out, "Rows": 2, "K": 2},
        },
        {
            "family": "reduction", "fixture": "reduce-sum-f32-axis1-v1",
            "module": _reduction_module(2, 3), "packager": rocm_native.package_reduction,
            "inputs": (np.array(corpus["reduce-sum-f32-axis1-v1"]["inputs"]["x"], np.float32),),
            "output": np.zeros((2,), np.float32), "dims": (2, 3, 1), "grid": 2,
            "bindings": lambda values, out: {
                "x": values[0], "o": out, "Outer": 2, "AxisExtent": 3, "Inner": 1,
            },
        },
        {
            "family": "paged_kv", "fixture": "paged-kv-f32-permuted-2x2-v1",
            "module": _paged_kv_module(2, 2, 1, 2, 1, 2),
            "packager": rocm_native.package_paged_kv_read,
            "inputs": (
                np.array(corpus["paged-kv-f32-permuted-2x2-v1"]["inputs"]["pages"], np.float32),
                np.array(corpus["paged-kv-f32-permuted-2x2-v1"]["inputs"]["page_table"], np.int32),
            ),
            "output": np.zeros((2, 1, 2), np.float32),
            "dims": (2, 2, 2, 1, 2, 1, 2), "grid": 1,
            "bindings": lambda values, out: {
                "pages": values[0], "page_table": values[1], "slice": out,
                "P": 2, "LP": 2, "PageSize": 2, "H": 1, "D": 2,
                "Start": 1, "Tokens": 2,
            },
        },
        {
            "family": "moe", "fixture": "moe-dispatch-f32-permuted-3x2-v1",
            "module": _moe_module(3, 2, 2), "packager": rocm_native.package_moe_dispatch,
            "inputs": (
                np.array(corpus["moe-dispatch-f32-permuted-3x2-v1"]["inputs"]["x"], np.float32),
                np.array(corpus["moe-dispatch-f32-permuted-3x2-v1"]["inputs"]["token"], np.int32),
            ),
            "output": np.zeros((2, 2), np.float32), "dims": (3, 2, 2), "grid": 1,
            "bindings": lambda values, out: {
                "x": values[0], "token": values[1], "o": out, "T": 3, "S": 2, "H": 2,
            },
        },
    )
    fixture_rows: list[dict] = []
    cache_rows: list[dict] = []
    benchmark_rows: list[dict] = []
    resource_rows: list[dict] = []
    toolchains: set[str] = set()
    for definition in definitions:
        family = str(definition["family"])
        fixture_id = str(definition["fixture"])
        rocm_native._cache.clear()
        packager = definition["packager"]
        cold = packager(definition["module"], pipeline_name="tessera-lower-to-rocm")
        warm = packager(definition["module"], pipeline_name="tessera-lower-to-rocm")
        if (
            cold.image.cache_key != warm.image.cache_key
            or cold.image.image_digest != warm.image.image_digest
            or cold.descriptor.descriptor_digest != warm.descriptor.descriptor_digest
        ):
            raise RuntimeError(f"{family} cold/warm ROCm package identity drifted")
        toolchains.add(cold.image.toolchain_fingerprint)
        inputs = tuple(np.ascontiguousarray(value) for value in definition["inputs"])
        output = np.ascontiguousarray(definition["output"])
        bindings = definition["bindings"](inputs, output)
        artifact = _artifact(cold)
        launch_result = rt.launch(artifact, bindings)
        if launch_result.get("ok") is not True:
            raise RuntimeError(f"{family} exact-device launch failed: {launch_result}")
        expected = np.asarray(corpus[fixture_id]["oracle"], dtype=np.float32)
        tolerance = corpus[fixture_id]["tolerance"]
        np.testing.assert_allclose(
            output, expected, atol=float(tolerance["atol"]), rtol=float(tolerance["rtol"]),
        )
        resident_output = np.zeros_like(output)
        resident = _ResidentKernel(
            hip, cold, inputs, resident_output, tuple(definition["dims"]),
            int(definition["grid"]),
        )
        try:
            resident.launch()
            np.testing.assert_allclose(
                resident.read(), expected, atol=float(tolerance["atol"]),
                rtol=float(tolerance["rtol"]),
            )
            end_to_end_call = lambda: rt.launch(artifact, bindings)
            medians = {
                "kernel_wall": _resident_two_run_medians_ns(
                    resident, samples=samples, iterations=kernel_iterations,
                ),
                "end_to_end": _two_run_medians_ns(
                    end_to_end_call, samples=samples, iterations=e2e_iterations,
                ),
            }
        finally:
            resident.close()
        route = str(cold.descriptor.provenance.get("schedule", cold.descriptor.geometry.policy))
        resource_payload = {
            "family": family,
            "entry": cold.descriptor.entry_symbol,
            "route": route,
            "image_digest": cold.image.image_digest,
            "payload_digest": cold.image.payload_digest,
            "descriptor_digest": cold.descriptor.descriptor_digest,
            "resource_record": (
                cold.image.resource_record.to_dict() if cold.image.resource_record else None
            ),
        }
        resource_fingerprint = hashlib.sha256(
            json.dumps(resource_payload, sort_keys=True).encode(),
        ).hexdigest()
        for domain, values in medians.items():
            benchmark_rows.append({
                "family": family,
                "route": route,
                "timing_domain": domain,
                "median_ns": float(statistics.median(values)),
                "run_medians_ns": values,
                "stability_limit_pct": stability_limit,
                "stable": _stability(values) <= stability_limit,
                "selected": True,
                "repetitions": samples * (
                    kernel_iterations if domain == "kernel_wall" else e2e_iterations
                ),
                "warmups": 1,
                "discard_first": True,
                "resource_fingerprint": resource_fingerprint,
            })
        fixture_rows.append({
            "fixture_id": fixture_id,
            "levels": {"a": "proven", "b": "proven", "c": "proven"},
            "actual": output.tolist(),
            "image_digest": cold.image.image_digest,
            "descriptor_digest": cold.descriptor.descriptor_digest,
        })
        cache_rows.append({
            "fixture_id": fixture_id,
            "cold": {
                "compile_state": cold.image.compile_state,
                "cache_key": cold.image.cache_key,
                "image_digest": cold.image.image_digest,
                "descriptor_digest": cold.descriptor.descriptor_digest,
            },
            "warm": {
                "compile_state": warm.image.compile_state,
                "cache_key": warm.image.cache_key,
                "image_digest": warm.image.image_digest,
                "descriptor_digest": warm.descriptor.descriptor_digest,
            },
        })
        resource_rows.append({
            **resource_payload,
            "resource_fingerprint": resource_fingerprint,
            "device_libraries": [row.to_dict() for row in cold.image.device_libraries],
        })

    report = {
        "schema": "tessera.e2e-backend-report.v1",
        "target": "rocm_gfx1151",
        "architecture": "gfx1151",
        "device": {"exact": True, "identity": _device_identity()},
        "source_commit": source_commit,
        "toolchain_fingerprint": hashlib.sha256(
            "\n".join(sorted(toolchains)).encode(),
        ).hexdigest(),
        "scope": ["softmax", "reduction", "paged_kv", "moe"],
        "required_timing_domains": ["kernel_wall", "end_to_end"],
        "fixtures": fixture_rows,
        "cache_proofs": cache_rows,
        "benchmarks": benchmark_rows,
    }
    validate_backend_report(report)
    resources = {
        "schema": "tessera.e2e-rocm-resource-record.v1",
        "device": report["device"],
        "measurement_cpu": cpu,
        "timing_note": (
            "kernel_wall is resident launch-plus-synchronize wall time because WSL HIP "
            "events can return invalid zero intervals; end_to_end uses runtime.launch"
        ),
        "selection_scope": "release evidence only",
        "production_selector_changed": False,
        "production_disposition": (
            "ROCM-E2E-1/-2 retained production routes remain selected; this packet "
            "does not promote the typed descriptor routes"
        ),
        "rows": resource_rows,
    }
    return report, resources


def main() -> int:
    from tessera.compiler.e2e_fleet import seal_packet

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--kernel-iterations", type=int, default=1000)
    parser.add_argument("--e2e-iterations", type=int, default=50)
    parser.add_argument("--stability-limit", type=float, default=5.0)
    parser.add_argument("--packet-dir", type=Path)
    args = parser.parse_args()
    if args.samples < 2 or args.kernel_iterations < 10 or args.e2e_iterations < 2:
        parser.error("use at least two samples, ten kernel iterations, and two E2E iterations")
    report, resources = record(
        samples=args.samples, kernel_iterations=args.kernel_iterations,
        e2e_iterations=args.e2e_iterations, stability_limit=args.stability_limit,
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
