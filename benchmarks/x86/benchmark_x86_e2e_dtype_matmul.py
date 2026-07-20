#!/usr/bin/env python3
"""Exact-host BF16, VNNI U8/S8, and FP64 descriptor/kernel comparison."""

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
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType  # noqa: E402
from tessera.compiler.x86_native import _library_path, package_matmul  # noqa: E402

SCHEMA = "tessera.x86.e2e_dtype_matmul_comparison.v1"
SHAPES = ((16, 16, 32), (31, 29, 37), (96, 96, 96))  # M, N, K
DTYPES = ("bf16", "u8s8", "fp64")


def _module(kind: str, shape: tuple[int, int, int]) -> GraphIRModule:
    m, n, k = shape
    contracts = {
        "bf16": ("bf16", "bf16", "fp32", "bf16", "bf16", "f32"),
        "u8s8": ("uint8", "int8", "int32", "i8", "i8", "i32"),
        "fp64": ("fp64", "fp64", "fp64", "f64", "f64", "f64"),
    }
    ad, bd, od, am, bm, om = contracts[kind]
    a = IRType(f"tensor<{m}x{k}x{am}>", (str(m), str(k)), ad)
    b = IRType(f"tensor<{k}x{n}x{bm}>", (str(k), str(n)), bd)
    o = IRType(f"tensor<{m}x{n}x{om}>", (str(m), str(n)), od)
    return GraphIRModule(functions=[GraphIRFunction(
        name=f"x86_{kind}_benchmark", args=[IRArg("a", a), IRArg("b", b)], result_types=[o],
        body=[IROp(result="o", op_name="tessera.matmul", operands=["%a", "%b"],
                   operand_types=[str(a), str(b)], result_type=str(o), kwargs={})],
        return_values=["%o"],
    )])


def _wall_ms(fn: Callable[[], Any]) -> float:
    start = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns() - start) / 1e6


def _measure(kind: str, shape: tuple[int, int, int], trials: int,
             rng: np.random.Generator, library: ctypes.CDLL, ml) -> dict[str, Any]:
    m, n, k = shape
    package = package_matmul(_module(kind, shape), pipeline_name="tessera-lower-to-x86")
    artifact = rt.RuntimeArtifact(
        metadata={"target": "x86", "compiler_path": "x86_native_descriptor"},
        tile_ir=package.tile_ir, target_ir=package.target_ir,
        native_image=package.image, launch_descriptor=package.descriptor,
    )
    if kind == "bf16":
        a = rng.uniform(-1, 1, (m, k)).astype(ml.bfloat16)
        b = rng.uniform(-1, 1, (k, n)).astype(ml.bfloat16)
        output, reference = np.zeros((m, n), np.float32), np.zeros((m, n), np.float32)
        u16p, f32p = ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float)
        av, bv = np.ascontiguousarray(a).view(np.uint16), np.ascontiguousarray(b).view(np.uint16)
        native, oracle = library.tessera_x86_avx512_gemm_bf16, library.tessera_x86_reference_gemm_bf16
        native.argtypes = oracle.argtypes = [u16p, u16p, f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        native_call = lambda: native(av.ctypes.data_as(u16p), bv.ctypes.data_as(u16p), output.ctypes.data_as(f32p), m, n, k, 0.0)
        oracle_call = lambda: oracle(av.ctypes.data_as(u16p), bv.ctypes.data_as(u16p), reference.ctypes.data_as(f32p), m, n, k, 0.0)
        tolerance = {"rtol": 2e-5, "atol": 2e-5}
    elif kind == "u8s8":
        a = rng.integers(0, 32, (m, k), dtype=np.uint8)
        b = rng.integers(-16, 16, (k, n), dtype=np.int8)
        output, reference = np.zeros((m, n), np.int32), np.zeros((m, n), np.int32)
        u8p, s8p, s32p = ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int32)
        native, oracle = library.tessera_x86_avx512_vnni_gemm_u8s8_s32, library.tessera_x86_reference_gemm_u8s8_s32
        native.argtypes = oracle.argtypes = [u8p, s8p, s32p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        native_call = lambda: native(a.ctypes.data_as(u8p), b.ctypes.data_as(s8p), output.ctypes.data_as(s32p), m, n, k, 0)
        oracle_call = lambda: oracle(a.ctypes.data_as(u8p), b.ctypes.data_as(s8p), reference.ctypes.data_as(s32p), m, n, k, 0)
        tolerance = None
    else:
        a = rng.uniform(-1, 1, (m, k)).astype(np.float64)
        b = rng.uniform(-1, 1, (k, n)).astype(np.float64)
        output, reference = np.zeros((m, n), np.float64), np.zeros((m, n), np.float64)
        f64p = ctypes.POINTER(ctypes.c_double)
        native, oracle = library.tessera_x86_avx512_gemm_f64, library.tessera_x86_reference_gemm_f64
        native.argtypes = oracle.argtypes = [f64p, f64p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, f64p]
        native_call = lambda: native(a.ctypes.data_as(f64p), b.ctypes.data_as(f64p), m, n, k, output.ctypes.data_as(f64p))
        oracle_call = lambda: oracle(a.ctypes.data_as(f64p), b.ctypes.data_as(f64p), m, n, k, reference.ctypes.data_as(f64p))
        tolerance = {"rtol": 1e-13, "atol": 1e-13}
    native_call(); oracle_call()
    if tolerance is None:
        np.testing.assert_array_equal(output, reference)
    else:
        np.testing.assert_allclose(output, reference, **tolerance)
    descriptor_output = np.zeros_like(output)
    descriptor_values = {"a": a, "b": b, "o": descriptor_output, "M": m, "N": n, "K": k}
    result = rt.launch(artifact, descriptor_values)
    if not result["ok"]:
        raise RuntimeError(result.get("reason"))
    if tolerance is None:
        np.testing.assert_array_equal(descriptor_output, reference)
    else:
        np.testing.assert_allclose(descriptor_output, reference, **tolerance)
    native_samples, reference_samples, descriptor_samples = [], [], []
    for trial in range(trials):
        calls = ((oracle_call, reference_samples), (native_call, native_samples),
                 (lambda: rt.launch(artifact, descriptor_values), descriptor_samples))
        if trial & 1:
            calls = tuple(reversed(calls))
        for call, samples in calls:
            samples.append(_wall_ms(call))
    native_median, reference_median = statistics.median(native_samples), statistics.median(reference_samples)
    return {
        "dtype": kind, "shape": list(shape), "correct": True,
        "required_features": package.descriptor.provenance["required_features"],
        "kernel": {
            "native_samples_ms": native_samples, "reference_samples_ms": reference_samples,
            "native_median_ms": native_median, "reference_median_ms": reference_median,
            "median_speedup": reference_median / native_median,
        },
        "end_to_end": {
            "descriptor_samples_ms": descriptor_samples,
            "descriptor_median_ms": statistics.median(descriptor_samples),
        },
    }


def run(trials: int) -> dict[str, Any]:
    import ml_dtypes as ml
    path = _library_path()
    if path is None:
        raise RuntimeError("libtessera_x86_elementwise.so is unavailable")
    library = ctypes.CDLL(str(path))
    rng = np.random.default_rng(8643)
    rows = [_measure(kind, shape, trials, rng, library, ml) for kind in DTYPES for shape in SHAPES]
    return {
        "schema": SCHEMA, "work_item": "X86-E2E-2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_arch": "x86_64_avx512_ryzen_ai_max_395", "trials": trials,
        "timing_policy": "serial alternating reference-kernel/native-kernel/descriptor CPU wall time",
        "rows": rows, "all_correct": True,
        "all_native_faster": all(row["kernel"]["median_speedup"] > 1.0 for row in rows),
        "selector_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=21)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.trials)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["rows"]:
        print(f'{row["dtype"]:5s} {row["shape"]}: kernel {row["kernel"]["median_speedup"]:.3f}x, descriptor {row["end_to_end"]["descriptor_median_ms"]:.4f} ms')
    return 0 if result["all_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
