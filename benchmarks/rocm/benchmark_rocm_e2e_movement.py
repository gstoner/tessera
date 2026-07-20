#!/usr/bin/env python3
"""ROCM-E2E-2 paired descriptor/retained movement comparison.

The paged-KV and MoE rows use equivalent f32/i32 semantics. HIP events keep
the compiler-owned descriptor buffers resident; end-to-end samples include
allocation, copies, launch, synchronization, and output download for both
routes. No selector is changed by this recorder.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
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
from tessera.compiler.emit.rocm_hip import run_paged_kv_cache_read_f32  # noqa: E402

SCHEMA = "tessera.rocm.e2e_movement_comparison.v1"
def _paged_module(pages: int, page_size: int, heads: int, dim: int, start: int, tokens: int) -> GraphIRModule:
    p = IRType(
        f"tensor<{pages}x{page_size}x{heads}x{dim}xf32>",
        tuple(map(str, (pages, page_size, heads, dim))),
        "fp32",
    )
    table = IRType(f"tensor<{pages}xi32>", (str(pages),), "int32")
    out = IRType(
        f"tensor<{tokens}x{heads}x{dim}xf32>",
        tuple(map(str, (tokens, heads, dim))),
        "fp32",
    )
    return GraphIRModule(functions=[GraphIRFunction(
        name="gfx1151_paged_kv_benchmark",
        args=[IRArg("pages", p), IRArg("page_table", table)],
        result_types=[out],
        body=[IROp(
            result="slice", op_name="tessera.kv_cache.read",
            operands=["%pages", "%page_table"], operand_types=[str(p), str(table)],
            result_type=str(out), kwargs={"start": start, "end": start + tokens},
        )],
        return_values=["%slice"],
    )])


def _moe_module(tokens: int, slots: int, hidden: int) -> GraphIRModule:
    x = IRType(f"tensor<{tokens}x{hidden}xf32>", (str(tokens), str(hidden)), "fp32")
    token = IRType(f"tensor<{slots}xi32>", (str(slots),), "int32")
    out = IRType(f"tensor<{slots}x{hidden}xf32>", (str(slots), str(hidden)), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="gfx1151_moe_dispatch_benchmark",
        args=[IRArg("x", x), IRArg("token", token)],
        result_types=[out],
        body=[IROp(
            result="o", op_name="tessera.moe_dispatch",
            operands=["%x", "%token"], operand_types=[str(x), str(token)],
            result_type=str(out), kwargs={},
        )],
        return_values=["%o"],
    )])


def _artifact(package: rocm_native.ROCMNativePackage) -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(
        graph_ir="graph", tile_ir=package.tile_ir, target_ir=package.target_ir,
        metadata={"target": "rocm_gfx1151", "compiler_path": "rocm_gfx1151_native_descriptor"},
        native_image=package.image, launch_descriptor=package.descriptor,
    )


def _memref(pointer: ctypes.c_void_p, size: int) -> list[Any]:
    return [
        ctypes.c_void_p(pointer.value), ctypes.c_void_p(pointer.value),
        ctypes.c_int64(0), ctypes.c_int64(size), ctypes.c_int64(1),
    ]


class _ResidentDescriptor:
    def __init__(
        self, hip: ctypes.CDLL, package: rocm_native.ROCMNativePackage,
        inputs: tuple[np.ndarray, ...], output: np.ndarray,
        dimensions: tuple[int, ...], work: int,
    ) -> None:
        self.hip = hip
        self.module, self.function = ctypes.c_void_p(), ctypes.c_void_p()
        self.devices: list[ctypes.c_void_p] = []
        self.output = output
        if hip.hipModuleLoadData(ctypes.byref(self.module), package.image.payload):
            raise RuntimeError("descriptor module load failed")
        if hip.hipModuleGetFunction(
            ctypes.byref(self.function), self.module,
            package.descriptor.entry_symbol.encode(),
        ):
            self.close()
            raise RuntimeError("descriptor symbol missing")
        try:
            for array in (*inputs, output):
                device = ctypes.c_void_p()
                if hip.hipMalloc(ctypes.byref(device), array.nbytes):
                    raise RuntimeError("descriptor hipMalloc failed")
                self.devices.append(device)
            for device, array in zip(self.devices[:-1], inputs, strict=True):
                if hip.hipMemcpy(
                    device, array.ctypes.data_as(ctypes.c_void_p), array.nbytes, 1
                ):
                    raise RuntimeError("descriptor resident H2D failed")
        except Exception:
            self.close()
            raise
        values: list[Any] = []
        for device, array in zip(self.devices[:-1], inputs, strict=True):
            values.extend(_memref(device, array.size))
        values.extend(_memref(self.devices[-1], output.size))
        values.extend(ctypes.c_int64(value) for value in dimensions)
        self._values = values
        self.arguments = (ctypes.c_void_p * len(values))()
        for index, value in enumerate(values):
            self.arguments[index] = ctypes.cast(ctypes.byref(value), ctypes.c_void_p)
        self.grid = max((work + 255) // 256, 1)

    def launch(self) -> None:
        rc = self.hip.hipModuleLaunchKernel(
            self.function, self.grid, 1, 1, 256, 1, 1, 0, None,
            self.arguments, None,
        )
        if rc:
            raise RuntimeError(f"descriptor launch failed rc={rc}")

    def read(self) -> np.ndarray:
        if self.hip.hipDeviceSynchronize():
            raise RuntimeError("descriptor synchronize failed")
        if self.hip.hipMemcpy(
            self.output.ctypes.data_as(ctypes.c_void_p), self.devices[-1],
            self.output.nbytes, 2,
        ):
            raise RuntimeError("descriptor D2H failed")
        return self.output

    def close(self) -> None:
        for device in reversed(self.devices):
            if device.value:
                self.hip.hipFree(device)
        self.devices.clear()
        if self.module.value:
            self.hip.hipModuleUnload(self.module)
            self.module = ctypes.c_void_p()


def _event_ms(hip: ctypes.CDLL, session: _ResidentDescriptor, iterations: int) -> float:
    start, stop = ctypes.c_void_p(), ctypes.c_void_p()
    if hip.hipEventCreate(ctypes.byref(start)) or hip.hipEventCreate(ctypes.byref(stop)):
        raise RuntimeError("HIP event creation failed")
    try:
        hip.hipEventRecord(start, None)
        for _ in range(iterations):
            session.launch()
        hip.hipEventRecord(stop, None)
        hip.hipEventSynchronize(stop)
        elapsed = ctypes.c_float()
        if hip.hipEventElapsedTime(ctypes.byref(elapsed), start, stop):
            raise RuntimeError("HIP event timing failed")
        value = float(elapsed.value) / iterations
        if value <= 0:
            raise RuntimeError("HIP event timing returned zero")
        return value
    finally:
        hip.hipEventDestroy(stop)
        hip.hipEventDestroy(start)


def _wall_ms(fn: Callable[[], Any]) -> float:
    start = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns() - start) / 1e6


def _summary(retained: list[float], compiler: list[float]) -> dict[str, Any]:
    if not retained or len(retained) != len(compiler):
        raise ValueError("paired timing samples are required")
    retained_median = statistics.median(retained)
    compiler_median = statistics.median(compiler)
    return {
        "retained_samples_ms": retained,
        "compiler_samples_ms": compiler,
        "retained_median_ms": retained_median,
        "compiler_median_ms": compiler_median,
        "median_speedup": retained_median / compiler_median,
        "non_regression_10pct": compiler_median <= retained_median * 1.10,
    }


def run(trials: int, iterations: int) -> dict[str, Any]:
    if rt._rocm_chip() != "gfx1151":
        raise RuntimeError(f"ROCM-E2E-2 requires exact gfx1151, got {rt._rocm_chip()}")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):
        raise RuntimeError("usable gfx1151 HIP device required")
    rng = np.random.default_rng(2205)
    rows: list[dict[str, Any]] = []

    pages_count, page_size, heads, dim, start, tokens = 4, 16, 3, 8, 7, 41
    pages = np.ascontiguousarray(
        rng.standard_normal((pages_count, page_size, heads, dim)), dtype=np.float32
    )
    table = np.array([2, 0, 3, 1], dtype=np.int32)
    indices = np.arange(start, start + tokens, dtype=np.int64)
    paged_package = rocm_native.package_paged_kv_read(
        _paged_module(pages_count, page_size, heads, dim, start, tokens),
        pipeline_name="tessera-lower-to-rocm",
    )
    paged_artifact = _artifact(paged_package)
    paged_output = np.zeros((tokens, heads, dim), np.float32)
    paged_scalars = {
        "P": pages_count, "LP": pages_count, "PageSize": page_size,
        "H": heads, "D": dim, "Start": start, "Tokens": tokens,
    }
    paged_args = {"pages": pages, "page_table": table, "slice": paged_output, **paged_scalars}
    paged_session = _ResidentDescriptor(
        hip, paged_package, (pages, table), paged_output,
        tuple(paged_scalars[name] for name in ("P", "LP", "PageSize", "H", "D", "Start", "Tokens")),
        tokens * heads * dim,
    )
    try:
        retained, retained_device = run_paged_kv_cache_read_f32(
            pages, table, indices, return_device_ms=True, reps=iterations
        )
        np.testing.assert_array_equal(retained, pages[table].reshape(-1, heads, dim)[start:start + tokens])
        paged_session.launch()
        np.testing.assert_array_equal(paged_session.read(), retained)
        rd, cd, re, ce = [], [], [], [],
        for trial in range(trials):
            order = ("retained", "compiler") if not trial & 1 else ("compiler", "retained")
            for route in order:
                if route == "retained":
                    _, value = run_paged_kv_cache_read_f32(
                        pages, table, indices, return_device_ms=True, reps=iterations
                    )
                    rd.append(value)
                    re.append(_wall_ms(lambda: run_paged_kv_cache_read_f32(pages, table, indices)))
                else:
                    cd.append(_event_ms(hip, paged_session, iterations))
                    ce.append(_wall_ms(lambda: rt.launch(paged_artifact, paged_args)))
        row = {"operation": "paged_kv_read", "shape": list(pages.shape),
               "range": [start, start + tokens], "correct": True,
               "device": _summary(rd, cd), "end_to_end": _summary(re, ce)}
        rows.append(row)
    finally:
        paged_session.close()

    source_tokens, slots, hidden = 17, 5, 257
    x = np.ascontiguousarray(rng.standard_normal((source_tokens, hidden)), dtype=np.float32)
    token = np.array([16, 0, 8, 3, 11], dtype=np.int32)
    moe_package = rocm_native.package_moe_dispatch(
        _moe_module(source_tokens, slots, hidden), pipeline_name="tessera-lower-to-rocm"
    )
    moe_artifact = _artifact(moe_package)
    moe_output = np.zeros((slots, hidden), np.float32)
    moe_scalars = {"T": source_tokens, "S": slots, "H": hidden}
    moe_args = {"x": x, "token": token, "o": moe_output, **moe_scalars}
    moe_session = _ResidentDescriptor(
        hip, moe_package, (x, token), moe_output,
        (source_tokens, slots, hidden), slots * hidden,
    )
    try:
        retained = rt._rocm_gather_rows(x, token.astype(np.int64), np)
        np.testing.assert_array_equal(retained, x[token])
        moe_session.launch()
        np.testing.assert_array_equal(moe_session.read(), retained)
        # The retained row-gather helper owns its resident-event evidence in the
        # established transport corpus; this comparison gates equivalent full
        # calls and records the typed kernel event independently.
        re, ce, cd = [], [], []
        for trial in range(trials):
            order = ("retained", "compiler") if not trial & 1 else ("compiler", "retained")
            for route in order:
                if route == "retained":
                    re.append(_wall_ms(lambda: rt._rocm_gather_rows(x, token.astype(np.int64), np)))
                else:
                    cd.append(_event_ms(hip, moe_session, iterations))
                    ce.append(_wall_ms(lambda: rt.launch(moe_artifact, moe_args)))
        end_to_end = _summary(re, ce)
        rows.append({
            "operation": "moe_dispatch", "shape": [source_tokens, slots, hidden],
            "correct": True, "device": {
                "compiler_samples_ms": cd,
                "compiler_median_ms": statistics.median(cd),
                "comparison": "retained resident-event evidence is owned by rocm.transport_retune.v1",
            },
            "end_to_end": end_to_end,
        })
    finally:
        moe_session.close()

    all_non_regression = all(row["end_to_end"]["non_regression_10pct"] for row in rows)
    all_non_regression = all_non_regression and rows[0]["device"]["non_regression_10pct"]
    return {
        "schema": SCHEMA, "work_item": "ROCM-E2E-2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_arch": "gfx1151", "trials": trials,
        "event_iterations": iterations,
        "policy": "all E2E rows and directly comparable paged-KV device events must be within 10%",
        "rows": rows, "all_correct": True,
        "all_non_regression": all_non_regression,
        "closure_disposition": "retain production routes; typed paged-KV is not E2E promotion-eligible",
        "selector_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.trials, args.iterations)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["rows"]:
        device = row["device"].get("median_speedup", "diagnostic")
        print(f'{row["operation"]}: device={device} e2e={row["end_to_end"]["median_speedup"]:.3f}x')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
