#!/usr/bin/env python3
"""ROCM-E2E-2 serial reduction comparison against ``rocm_reduce_compiled``.

HIP-event measurements keep modules and buffers resident. End-to-end samples
exercise each real runtime route, including allocation, copies, launch, and
synchronization. Candidates alternate A/B then B/A; no selector is changed.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from tessera import runtime as rt  # noqa: E402
from tessera.compiler import rocm_native  # noqa: E402
from tessera.compiler.graph_ir import (  # noqa: E402
    GraphIRFunction,
    GraphIRModule,
    IRArg,
    IROp,
    IRType,
)


CASES: tuple[tuple[tuple[int, ...], int, str], ...] = (
    ((32, 257), 1, "sum"),
    ((8, 33, 16), 1, "mean"),
    ((7, 9, 65), 0, "max"),
)
DTYPES: tuple[tuple[str, float], ...] = (
    ("fp32", 1e-5),
    ("fp16", 3e-2),
    ("bf16", 8e-2),
)
SCHEMA = "tessera.rocm.e2e_reduce_comparison.v1"


def _numpy_dtype(dtype: str) -> Any:
    if dtype == "fp32":
        return np.float32
    if dtype == "fp16":
        return np.float16
    if dtype == "bf16":
        import ml_dtypes

        return ml_dtypes.bfloat16
    raise ValueError(f"unsupported benchmark dtype {dtype!r}")


def _module(dtype: str, shape: tuple[int, ...], axis: int, kind: str) -> GraphIRModule:
    element = {"fp16": "f16", "bf16": "bf16", "fp32": "f32"}[dtype]
    source = IRType(
        f"tensor<{'x'.join(map(str, shape))}x{element}>",
        tuple(map(str, shape)),
        dtype,
    )
    output_shape = shape[:axis] + shape[axis + 1 :]
    output = IRType(
        f"tensor<{'x'.join(map(str, output_shape))}xf32>",
        tuple(map(str, output_shape)),
        "fp32",
    )
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="gfx1151_reduce_benchmark",
                args=[IRArg("x", source)],
                result_types=[output],
                body=[
                    IROp(
                        result="o",
                        op_name={"sum": "tessera.sum", "mean": "tessera.mean", "max": "tessera.max"}[kind],
                        operands=["%x"],
                        operand_types=[str(source)],
                        result_type=str(output),
                        kwargs={"axis": axis, "keepdims": False},
                    )
                ],
                return_values=["%o"],
            )
        ]
    )


def _retained_artifact(axis: int, kind: str) -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(
        metadata={
            "target": "rocm",
            "compiler_path": "rocm_reduce_compiled",
            "executable": True,
            "execution_kind": "native_gpu",
            "arg_names": ["x"],
            "output_name": "o",
            "ops": [
                {
                    "op_name": {"sum": "tessera.sum", "mean": "tessera.mean", "max": "tessera.max"}[kind],
                    "result": "o",
                    "operands": ["x"],
                    "kwargs": {"axis": axis, "keepdims": False},
                }
            ],
        }
    )


def _native_artifact(package: rocm_native.ROCMNativePackage) -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(
        graph_ir="graph",
        tile_ir=package.tile_ir,
        target_ir=package.target_ir,
        metadata={"target": "rocm_gfx1151", "compiler_path": "rocm_gfx1151_native_descriptor"},
        native_image=package.image,
        launch_descriptor=package.descriptor,
    )


def _memref(pointer: ctypes.c_void_p, size: int) -> list[Any]:
    return [
        ctypes.c_void_p(pointer.value),
        ctypes.c_void_p(pointer.value),
        ctypes.c_int64(0),
        ctypes.c_int64(size),
        ctypes.c_int64(1),
    ]


class _ResidentReduce:
    def __init__(
        self,
        hip: ctypes.CDLL,
        hsaco: bytes,
        symbol: str,
        x: np.ndarray,
        output_dtype: Any,
        dimensions: tuple[int, ...],
        grid_x: int,
    ) -> None:
        self.hip = hip
        self.module = ctypes.c_void_p()
        self.function = ctypes.c_void_p()
        self.device_x = ctypes.c_void_p()
        self.device_o = ctypes.c_void_p()
        self.x = np.ascontiguousarray(x)
        self.output = np.zeros(grid_x, dtype=output_dtype)
        self.grid_x = grid_x
        if hip.hipModuleLoadData(ctypes.byref(self.module), hsaco):
            raise RuntimeError(f"module load failed for {symbol}")
        if hip.hipModuleGetFunction(ctypes.byref(self.function), self.module, symbol.encode()):
            self.close()
            raise RuntimeError(f"kernel symbol {symbol!r} is absent")
        try:
            if hip.hipMalloc(ctypes.byref(self.device_x), self.x.nbytes):
                raise RuntimeError("reduction input hipMalloc failed")
            if hip.hipMalloc(ctypes.byref(self.device_o), self.output.nbytes):
                raise RuntimeError("reduction output hipMalloc failed")
            if hip.hipMemcpy(self.device_x, self.x.ctypes.data_as(ctypes.c_void_p), self.x.nbytes, 1):
                raise RuntimeError("reduction resident H2D copy failed")
        except Exception:
            self.close()
            raise
        values = (
            _memref(self.device_x, self.x.size)
            + _memref(self.device_o, self.output.size)
            + [ctypes.c_int64(value) for value in dimensions]
        )
        self._values = values
        self.arguments = (ctypes.c_void_p * len(values))()
        for index, value in enumerate(values):
            self.arguments[index] = ctypes.cast(ctypes.byref(value), ctypes.c_void_p)

    def launch(self) -> None:
        rc = self.hip.hipModuleLaunchKernel(self.function, self.grid_x, 1, 1, 256, 1, 1, 0, None, self.arguments, None)
        if rc:
            raise RuntimeError(f"reduction launch failed rc={rc}")

    def read(self) -> np.ndarray:
        if self.hip.hipDeviceSynchronize():
            raise RuntimeError("reduction synchronization failed")
        if self.hip.hipMemcpy(
            self.output.ctypes.data_as(ctypes.c_void_p),
            self.device_o,
            self.output.nbytes,
            2,
        ):
            raise RuntimeError("reduction resident D2H copy failed")
        return self.output

    def close(self) -> None:
        if self.device_o.value:
            self.hip.hipFree(self.device_o)
            self.device_o = ctypes.c_void_p()
        if self.device_x.value:
            self.hip.hipFree(self.device_x)
            self.device_x = ctypes.c_void_p()
        if self.module.value:
            self.hip.hipModuleUnload(self.module)
            self.module = ctypes.c_void_p()


def _event_ms(hip: ctypes.CDLL, session: _ResidentReduce, iterations: int) -> float:
    start, stop = ctypes.c_void_p(), ctypes.c_void_p()
    if hip.hipEventCreate(ctypes.byref(start)) or hip.hipEventCreate(ctypes.byref(stop)):
        raise RuntimeError("HIP event creation failed")
    try:
        if hip.hipEventRecord(start, None):
            raise RuntimeError("HIP start-event record failed")
        for _ in range(iterations):
            session.launch()
        if hip.hipEventRecord(stop, None) or hip.hipEventSynchronize(stop):
            raise RuntimeError("HIP stop-event failed")
        elapsed = ctypes.c_float()
        if hip.hipEventElapsedTime(ctypes.byref(elapsed), start, stop):
            raise RuntimeError("HIP event elapsed-time query failed")
        value = float(elapsed.value) / iterations
        if value <= 0:
            raise RuntimeError("HIP event timing returned zero")
        return value
    finally:
        hip.hipEventDestroy(stop)
        hip.hipEventDestroy(start)


def _wall_ms(function: Callable[[], Any]) -> float:
    start = time.perf_counter_ns()
    result = function()
    elapsed = (time.perf_counter_ns() - start) / 1e6
    if not isinstance(result, dict) or not result.get("ok"):
        raise RuntimeError(f"runtime launch failed: {result}")
    return elapsed


def _summary(retained: list[float], compiler: list[float]) -> dict[str, Any]:
    if len(retained) != len(compiler) or not retained:
        raise ValueError("paired timing samples must be non-empty and equal length")
    ratios = [old / new for old, new in zip(retained, compiler, strict=True)]
    old_median = statistics.median(retained)
    new_median = statistics.median(compiler)
    return {
        "retained_samples_ms": retained,
        "compiler_samples_ms": compiler,
        "retained_median_ms": old_median,
        "compiler_median_ms": new_median,
        "median_speedup": old_median / new_median,
        "paired_speedup_median": statistics.median(ratios),
        "compiler_win_rate": sum(value >= 1.0 for value in ratios) / len(ratios),
        "non_regression_10pct": new_median <= old_median * 1.10,
    }


def _resource_metadata(hsaco: bytes) -> dict[str, Any]:
    candidates = (
        os.environ.get("TESSERA_LLVM_READOBJ"),
        "/usr/lib/llvm-23/bin/llvm-readobj",
        "/opt/rocm/core/llvm/bin/llvm-readobj",
    )
    tool = next((Path(value) for value in candidates if value and Path(value).is_file()), None)
    result: dict[str, Any] = {
        "hsaco_bytes": len(hsaco),
        "hsaco_sha256": hashlib.sha256(hsaco).hexdigest(),
        "vgpr_count": None,
        "sgpr_count": None,
        "lds_bytes": None,
        "private_segment_bytes": None,
        "vgpr_spill_count": None,
        "sgpr_spill_count": None,
    }
    if tool is None:
        return result
    with tempfile.NamedTemporaryFile(suffix=".hsaco") as stream:
        stream.write(hsaco)
        stream.flush()
        process = subprocess.run([str(tool), "--notes", stream.name], capture_output=True, text=True, check=False)
    patterns = {
        "vgpr_count": r"\.vgpr_count:\s*(\d+)",
        "sgpr_count": r"\.sgpr_count:\s*(\d+)",
        "lds_bytes": r"\.group_segment_fixed_size:\s*(\d+)",
        "private_segment_bytes": r"\.private_segment_fixed_size:\s*(\d+)",
        "vgpr_spill_count": r"\.vgpr_spill_count:\s*(\d+)",
        "sgpr_spill_count": r"\.sgpr_spill_count:\s*(\d+)",
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, process.stdout)
        if match:
            result[name] = int(match.group(1))
    return result


def _compile_pair(dtype: str, shape: tuple[int, ...], axis: int, kind: str) -> dict[str, Any]:
    rocm_native._cache.clear()
    rt._rocm_reduce_hsaco_cache.clear()
    module = _module(dtype, shape, axis, kind)
    start = time.perf_counter_ns()
    cold = rocm_native.package_reduction(module, pipeline_name="tessera-lower-to-rocm")
    compiler_cold_ms = (time.perf_counter_ns() - start) / 1e6
    start = time.perf_counter_ns()
    warm = rocm_native.package_reduction(module, pipeline_name="tessera-lower-to-rocm")
    compiler_warm_ms = (time.perf_counter_ns() - start) / 1e6
    tag = {"fp16": "f16", "bf16": "bf16", "fp32": "f32"}[dtype]
    start = time.perf_counter_ns()
    retained_cold = rt._build_compiled_reduce_hsaco(kind, tag)
    retained_cold_ms = (time.perf_counter_ns() - start) / 1e6
    start = time.perf_counter_ns()
    retained_warm = rt._build_compiled_reduce_hsaco(kind, tag)
    retained_warm_ms = (time.perf_counter_ns() - start) / 1e6
    if cold.image.image_digest != warm.image.image_digest or retained_cold != retained_warm:
        raise RuntimeError("cold/warm reduction image identity drifted")
    return {
        "package": warm,
        "compiler_hsaco": warm.image.payload,
        "retained_hsaco": retained_warm,
        "compile": {
            "compiler": {
                "cold_ms": compiler_cold_ms,
                "warm_ms": compiler_warm_ms,
                "cold_state": cold.image.compile_state,
                "warm_state": warm.image.compile_state,
                "image_digest": warm.image.image_digest,
                "toolchain_fingerprint": warm.image.toolchain_fingerprint,
                "device_libraries": [item.to_dict() for item in warm.image.device_libraries],
            },
            "retained": {
                "cold_ms": retained_cold_ms,
                "warm_ms": retained_warm_ms,
                "cold_state": "cold",
                "warm_state": "warm_cache",
                "hsaco_sha256": hashlib.sha256(retained_warm).hexdigest(),
            },
        },
    }


def _case(
    hip: ctypes.CDLL,
    dtype: str,
    tolerance: float,
    shape: tuple[int, ...],
    axis: int,
    kind: str,
    compiled: dict[str, Any],
    trials: int,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    numpy_dtype = _numpy_dtype(dtype)
    x = np.ascontiguousarray(rng.standard_normal(shape), dtype=numpy_dtype)
    outer = int(np.prod(shape[:axis], dtype=np.int64))
    extent = shape[axis]
    inner = int(np.prod(shape[axis + 1 :], dtype=np.int64))
    output_shape = shape[:axis] + shape[axis + 1 :]
    retained_x = np.ascontiguousarray(np.moveaxis(x, axis, -1))
    package = compiled["package"]
    compiler_session = _ResidentReduce(
        hip,
        compiled["compiler_hsaco"],
        package.descriptor.entry_symbol,
        x,
        np.float32,
        (outer, extent, inner),
        outer * inner,
    )
    retained_session = _ResidentReduce(
        hip,
        compiled["retained_hsaco"],
        "rd",
        retained_x,
        numpy_dtype,
        (outer * inner, extent),
        outer * inner,
    )
    try:
        compiler_session.launch()
        retained_session.launch()
        compiler_output = compiler_session.read().reshape(output_shape).astype(np.float32)
        retained_output = retained_session.read().reshape(output_shape).astype(np.float32)
        expected = getattr(np, kind)(x.astype(np.float32), axis=axis).astype(np.float32)
        compiler_error = float(np.max(np.abs(compiler_output - expected)))
        retained_error = float(np.max(np.abs(retained_output - expected)))
        parity_error = float(np.max(np.abs(compiler_output - retained_output)))
        if compiler_error > tolerance or retained_error > tolerance:
            raise AssertionError(f"reduction oracle mismatch compiler={compiler_error} retained={retained_error}")

        compiler_artifact = _native_artifact(package)
        retained_artifact = _retained_artifact(axis, kind)

        def compiler_e2e() -> dict[str, Any]:
            output = np.zeros(output_shape, dtype=np.float32)
            return rt.launch(
                compiler_artifact,
                {
                    "x": x,
                    "o": output,
                    "Outer": outer,
                    "AxisExtent": extent,
                    "Inner": inner,
                },
            )

        def retained_e2e() -> dict[str, Any]:
            return rt.launch(retained_artifact, (x,))

        for _ in range(3):
            compiler_session.launch()
            retained_session.launch()
        hip.hipDeviceSynchronize()
        compiler_e2e()
        retained_e2e()

        device_retained: list[float] = []
        device_compiler: list[float] = []
        e2e_retained: list[float] = []
        e2e_compiler: list[float] = []
        for trial in range(trials):
            order = (
                (retained_session, device_retained, e2e_retained, retained_e2e),
                (compiler_session, device_compiler, e2e_compiler, compiler_e2e),
            )
            if trial & 1:
                order = tuple(reversed(order))
            for session, device_samples, wall_samples, wall_call in order:
                device_samples.append(_event_ms(hip, session, iterations))
                wall_samples.append(_wall_ms(wall_call))
        device = _summary(device_retained, device_compiler)
        end_to_end = _summary(e2e_retained, e2e_compiler)
        overhead = {
            "retained_median_ms": end_to_end["retained_median_ms"] - device["retained_median_ms"],
            "compiler_median_ms": end_to_end["compiler_median_ms"] - device["compiler_median_ms"],
        }
        overhead["compiler_minus_retained_ms"] = overhead["compiler_median_ms"] - overhead["retained_median_ms"]
        device_comparable = axis == len(shape) - 1
        return {
            "dtype": dtype,
            "shape": list(shape),
            "axis": axis,
            "kind": kind,
            "correctness": {
                "compiler_max_abs": compiler_error,
                "retained_max_abs": retained_error,
                "route_parity_max_abs": parity_error,
                "tolerance": tolerance,
                "passed": True,
            },
            "device": device,
            "end_to_end": end_to_end,
            "host_overhead": overhead,
            "device_comparable": device_comparable,
            "non_regression": bool(
                end_to_end["non_regression_10pct"]
                and (not device_comparable or device["non_regression_10pct"])
            ),
        }
    finally:
        retained_session.close()
        compiler_session.close()


def run(trials: int, iterations: int) -> dict[str, Any]:
    if trials < 3 or iterations < 1:
        raise ValueError("use at least three serial trials and one event iteration")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):
        raise RuntimeError("an exact live ROCm device is required")
    arch = rt._rocm_chip()
    if arch != "gfx1151":
        raise RuntimeError(f"ROCM-E2E-2 requires exact gfx1151, got {arch}")
    rng = np.random.default_rng(2202)
    rows: list[dict[str, Any]] = []
    compile_records: dict[str, Any] = {}
    resources: dict[str, Any] = {}
    for dtype, tolerance in DTYPES:
        for shape, axis, kind in CASES:
            key = f"{dtype}:{kind}"
            compiled = _compile_pair(dtype, shape, axis, kind)
            compile_records[key] = compiled["compile"]
            resources[key] = {
                "compiler": _resource_metadata(compiled["compiler_hsaco"]),
                "retained": _resource_metadata(compiled["retained_hsaco"]),
            }
            row = _case(
                hip,
                dtype,
                tolerance,
                shape,
                axis,
                kind,
                compiled,
                trials,
                iterations,
                rng,
            )
            rows.append(row)
            print(
                f"{dtype:4s} {kind:4s} axis={axis} {shape} "
                f"device={row['device']['median_speedup']:.3f}x "
                f"e2e={row['end_to_end']['median_speedup']:.3f}x "
                f"host_delta={row['host_overhead']['compiler_minus_retained_ms']:+.3f}ms "
                f"gate={'pass' if row['non_regression'] else 'fail'}",
                flush=True,
            )
    return {
        "schema": SCHEMA,
        "work_item": "ROCM-E2E-2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_arch": arch,
        "host": platform.platform(),
        "python": platform.python_version(),
        "trials": trials,
        "event_iterations": iterations,
        "timing_policy": "serial alternating retained/compiler; resident HIP events and allocation/copy-inclusive runtime.launch wall time are separate",
        "non_regression_threshold": "compiler E2E median <= retained * 1.10 for every row; device median <= retained * 1.10 only when both routes consume the same last-axis layout",
        "compile": compile_records,
        "resources": resources,
        "rows": rows,
        "all_correct": all(row["correctness"]["passed"] for row in rows),
        "all_non_regression": all(row["non_regression"] for row in rows),
        "selector_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.trials, args.iterations)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["all_correct"] and result["all_non_regression"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
