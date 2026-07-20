#!/usr/bin/env python3
"""ROCM-E2E-1 serial comparison against ``rocm_softmax_compiled``.

The benchmark keeps two timing domains separate:

* HIP-event device time uses resident modules and buffers;
* end-to-end wall time uses each real ``runtime.launch`` route and includes
  output allocation, module load, device allocation, copies, launch, and sync.

Candidates run serially in alternating A/B then B/A order.  Raw samples,
cold/warm compile state, image identity, code-object resources, correctness,
and environment provenance are retained.  No selector is changed here.
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


CASES: tuple[tuple[int, int], ...] = (
    (32, 17),
    (128, 256),
    (64, 1024),
    (16, 4096),
)
DTYPES: tuple[tuple[str, Any, float], ...] = (
    ("fp32", np.float32, 1e-5),
    ("fp16", np.float16, 3e-3),
)
SCHEMA = "tessera.rocm.e2e_softmax_comparison.v1"


def _module(dtype: str, shape: tuple[int, int]) -> GraphIRModule:
    element = "f16" if dtype == "fp16" else "f32"
    tensor = IRType(
        f"tensor<{shape[0]}x{shape[1]}x{element}>",
        tuple(str(value) for value in shape),
        dtype,
    )
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="gfx1151_softmax_benchmark",
                args=[IRArg("x", tensor)],
                result_types=[tensor],
                body=[
                    IROp(
                        result="o",
                        op_name="tessera.softmax",
                        operands=["%x"],
                        operand_types=[str(tensor)],
                        result_type=str(tensor),
                        kwargs={"axis": -1},
                    )
                ],
                return_values=["%o"],
            )
        ]
    )


def _legacy_artifact() -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(
        metadata={
            "target": "rocm",
            "compiler_path": "rocm_softmax_compiled",
            "executable": True,
            "execution_kind": "native_gpu",
            "arg_names": ["x"],
            "output_name": "o",
            "ops": [
                {
                    "op_name": "tessera.softmax",
                    "result": "o",
                    "operands": ["x"],
                    "kwargs": {"axis": -1},
                }
            ],
        }
    )


def _native_artifact(package: rocm_native.ROCMNativePackage) -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(
        graph_ir="graph",
        tile_ir=package.tile_ir,
        target_ir=package.target_ir,
        metadata={
            "target": "rocm_gfx1151",
            "compiler_path": "rocm_gfx1151_native_descriptor",
        },
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


class _ResidentSoftmax:
    def __init__(
        self,
        hip: ctypes.CDLL,
        hsaco: bytes,
        symbol: str,
        x: np.ndarray,
    ) -> None:
        self.hip = hip
        self.module = ctypes.c_void_p()
        self.function = ctypes.c_void_p()
        if hip.hipModuleLoadData(ctypes.byref(self.module), hsaco):
            raise RuntimeError(f"module load failed for {symbol}")
        if hip.hipModuleGetFunction(
            ctypes.byref(self.function), self.module, symbol.encode()
        ):
            raise RuntimeError(f"kernel symbol {symbol!r} is absent")
        self.x = np.ascontiguousarray(x)
        self.output = np.zeros_like(self.x)
        self.rows, self.columns = self.x.shape
        self.device_x = ctypes.c_void_p()
        self.device_o = ctypes.c_void_p()
        for device in (self.device_x, self.device_o):
            if hip.hipMalloc(ctypes.byref(device), self.x.nbytes):
                raise RuntimeError("softmax resident hipMalloc failed")
        if hip.hipMemcpy(
            self.device_x,
            self.x.ctypes.data_as(ctypes.c_void_p),
            self.x.nbytes,
            1,
        ):
            raise RuntimeError("softmax resident H2D copy failed")
        values = (
            _memref(self.device_x, self.x.size)
            + _memref(self.device_o, self.x.size)
            + [ctypes.c_int64(self.rows), ctypes.c_int64(self.columns)]
        )
        self._values = values
        self.arguments = (ctypes.c_void_p * len(values))()
        for index, value in enumerate(values):
            self.arguments[index] = ctypes.cast(
                ctypes.byref(value), ctypes.c_void_p
            )

    def launch(self) -> None:
        rc = self.hip.hipModuleLaunchKernel(
            self.function,
            self.rows,
            1,
            1,
            256,
            1,
            1,
            0,
            None,
            self.arguments,
            None,
        )
        if rc:
            raise RuntimeError(f"softmax launch failed rc={rc}")

    def read(self) -> np.ndarray:
        if self.hip.hipDeviceSynchronize():
            raise RuntimeError("softmax synchronization failed")
        if self.hip.hipMemcpy(
            self.output.ctypes.data_as(ctypes.c_void_p),
            self.device_o,
            self.output.nbytes,
            2,
        ):
            raise RuntimeError("softmax resident D2H copy failed")
        return self.output

    def close(self) -> None:
        self.hip.hipFree(self.device_o)
        self.hip.hipFree(self.device_x)
        self.hip.hipModuleUnload(self.module)


def _event_ms(hip: ctypes.CDLL, session: _ResidentSoftmax, iterations: int) -> float:
    start, stop = ctypes.c_void_p(), ctypes.c_void_p()
    if hip.hipEventCreate(ctypes.byref(start)) or hip.hipEventCreate(
        ctypes.byref(stop)
    ):
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
            raise RuntimeError("HIP event timing returned zero; device gate is invalid")
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
        process = subprocess.run(
            [str(tool), "--notes", stream.name],
            capture_output=True,
            text=True,
            check=False,
        )
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


def _compile_pair(dtype: str, shape: tuple[int, int]) -> dict[str, Any]:
    rocm_native._cache.clear()
    rt._rocm_softmax_hsaco_cache.clear()
    start = time.perf_counter_ns()
    cold = rocm_native.package_softmax(
        _module(dtype, shape), pipeline_name="tessera-lower-to-rocm"
    )
    compiler_cold_ms = (time.perf_counter_ns() - start) / 1e6
    start = time.perf_counter_ns()
    warm = rocm_native.package_softmax(
        _module(dtype, shape), pipeline_name="tessera-lower-to-rocm"
    )
    compiler_warm_ms = (time.perf_counter_ns() - start) / 1e6
    tag = "f16" if dtype == "fp16" else "f32"
    start = time.perf_counter_ns()
    retained_cold = rt._build_compiled_softmax_hsaco(tag)
    retained_cold_ms = (time.perf_counter_ns() - start) / 1e6
    start = time.perf_counter_ns()
    retained_warm = rt._build_compiled_softmax_hsaco(tag)
    retained_warm_ms = (time.perf_counter_ns() - start) / 1e6
    if cold.image.image_digest != warm.image.image_digest:
        raise RuntimeError("compiler-owned cold/warm image identity drifted")
    if retained_cold != retained_warm:
        raise RuntimeError("retained cold/warm HSACO identity drifted")
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
                "device_libraries": [
                    item.to_dict() for item in warm.image.device_libraries
                ],
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
    numpy_dtype: Any,
    tolerance: float,
    shape: tuple[int, int],
    compiled: dict[str, Any],
    trials: int,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    x = np.ascontiguousarray(rng.standard_normal(shape), dtype=numpy_dtype)
    package = rocm_native.package_softmax(
        _module(dtype, shape), pipeline_name="tessera-lower-to-rocm"
    )
    compiler_session = _ResidentSoftmax(
        hip, compiled["compiler_hsaco"], package.descriptor.entry_symbol, x
    )
    retained_session = _ResidentSoftmax(hip, compiled["retained_hsaco"], "sm", x)
    try:
        compiler_session.launch()
        retained_session.launch()
        compiler_output = compiler_session.read().astype(np.float32)
        retained_output = retained_session.read().astype(np.float32)
        xf = x.astype(np.float32)
        shifted = xf - xf.max(axis=-1, keepdims=True)
        expected = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        compiler_error = float(np.max(np.abs(compiler_output - expected)))
        retained_error = float(np.max(np.abs(retained_output - expected)))
        parity_error = float(np.max(np.abs(compiler_output - retained_output)))
        if compiler_error > tolerance or retained_error > tolerance:
            raise AssertionError(
                f"softmax oracle mismatch compiler={compiler_error} retained={retained_error}"
            )

        compiler_artifact = _native_artifact(package)
        retained_artifact = _legacy_artifact()

        def compiler_e2e() -> dict[str, Any]:
            output = np.zeros_like(x)
            return rt.launch(
                compiler_artifact,
                {
                    "x": x,
                    "o": output,
                    "Rows": shape[0],
                    "K": shape[1],
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
                (retained_session, device_retained, retained_e2e),
                (compiler_session, device_compiler, compiler_e2e),
            )
            if trial & 1:
                order = tuple(reversed(order))
            for session, device_samples, e2e in order:
                device_samples.append(_event_ms(hip, session, iterations))
                e2e_samples = e2e_retained if session is retained_session else e2e_compiler
                e2e_samples.append(_wall_ms(e2e))
        device = _summary(device_retained, device_compiler)
        end_to_end = _summary(e2e_retained, e2e_compiler)
        return {
            "dtype": dtype,
            "shape": list(shape),
            "correctness": {
                "compiler_max_abs": compiler_error,
                "retained_max_abs": retained_error,
                "route_parity_max_abs": parity_error,
                "tolerance": tolerance,
                "passed": True,
            },
            "device": device,
            "end_to_end": end_to_end,
            "non_regression": bool(
                device["non_regression_10pct"]
                and end_to_end["non_regression_10pct"]
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
        raise RuntimeError(f"ROCM-E2E-1 requires exact gfx1151, got {arch}")
    rng = np.random.default_rng(1151)
    rows: list[dict[str, Any]] = []
    compile_records: dict[str, Any] = {}
    resources: dict[str, Any] = {}
    for dtype, numpy_dtype, tolerance in DTYPES:
        compiled = _compile_pair(dtype, CASES[0])
        compile_records[dtype] = compiled["compile"]
        resources[dtype] = {
            "compiler": _resource_metadata(compiled["compiler_hsaco"]),
            "retained": _resource_metadata(compiled["retained_hsaco"]),
        }
        for shape in CASES:
            row = _case(
                hip,
                dtype,
                numpy_dtype,
                tolerance,
                shape,
                compiled,
                trials,
                iterations,
                rng,
            )
            rows.append(row)
            print(
                f"{dtype:4s} {shape[0]:4d}x{shape[1]:4d} "
                f'device={row["device"]["median_speedup"]:.3f}x '
                f'e2e={row["end_to_end"]["median_speedup"]:.3f}x '
                f'gate={"pass" if row["non_regression"] else "fail"}',
                flush=True,
            )
    return {
        "schema": SCHEMA,
        "work_item": "ROCM-E2E-1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_arch": arch,
        "host": platform.platform(),
        "python": platform.python_version(),
        "trials": trials,
        "event_iterations": iterations,
        "timing_policy": "serial alternating retained/compiler; HIP event and allocation/copy-inclusive runtime.launch wall time are separate",
        "non_regression_threshold": "compiler median <= retained median * 1.10 in both timing domains",
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
