#!/usr/bin/env python3
"""Exact-host physical math packet for x86/AVX-512 or ROCm/gfx1151."""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import statistics
import time
from typing import Any, Callable

import numpy as np


def _artifact(rt: Any, target: str, family: str, op_name: str, operands, kwargs=None):
    names = [f"a{index}" for index in range(len(operands))]
    return rt.RuntimeArtifact(metadata={
        "target": target,
        "compiler_path": f"{target}_{family}_compiled",
        "executable": True,
        "execution_kind": "native_cpu" if target == "x86" else "native_gpu",
        "arg_names": names,
        "output_name": "o",
        "ops": [{
            "op_name": f"tessera.{op_name}",
            "result": "o",
            "operands": names,
            "kwargs": kwargs or {},
        }],
    })


def _dtype(name: str):
    if name == "f32":
        return np.float32
    if name == "f16":
        return np.float16
    import ml_dtypes

    return ml_dtypes.bfloat16


def _cases(target: str, dtype_name: str):
    rng = np.random.default_rng(20260806)
    dtype = _dtype(dtype_name)
    rows, cols = 256, 1024
    x = rng.uniform(0.125, 2.0, size=(rows, cols)).astype(dtype)
    y = rng.uniform(0.5, 2.0, size=(rows, cols)).astype(dtype)
    return [
        ("unary", "sqrt", (x,), {}, lambda: np.sqrt(x.astype(np.float32))),
        (
            "transcendental" if target == "x86" else "unary",
            "exp", (x,), {}, lambda: np.exp(x.astype(np.float32)),
        ),
        ("binary", "add", (x, y), {}, lambda: x.astype(np.float32) + y.astype(np.float32)),
        ("binary", "div", (x, y), {}, lambda: x.astype(np.float32) / y.astype(np.float32)),
        ("reduce", "sum", (x,), {"axis": -1}, lambda: np.sum(x.astype(np.float32), axis=-1)),
        ("scan", "cumsum", (x,), {"axis": -1}, lambda: np.cumsum(x.astype(np.float32), axis=-1)),
        ("scan", "cummax", (x,), {"axis": -1}, lambda: np.maximum.accumulate(x.astype(np.float32), axis=-1)),
    ]


def _x86_scan_selector_evidence(rt: Any, iterations: int) -> list[dict[str, Any]]:
    lib = rt._load_x86_elementwise()
    if lib is None:
        raise RuntimeError("x86 elementwise package is unavailable")
    pointer = ctypes.POINTER(ctypes.c_float)
    reference = lib.tessera_x86_reference_scan_f32
    selected = lib.tessera_x86_avx512_scan_f32
    for function in (reference, selected):
        function.argtypes = [
            pointer, ctypes.c_int64, ctypes.c_int64, pointer, ctypes.c_int
        ]
        function.restype = None
    rng = np.random.default_rng(20260807)
    rows, cols = 256, 1024
    result = []
    for kind, name, low, high, policy in (
        (0, "cumsum", -1.0, 1.0, "avx512_hillis_steele_16"),
        (1, "cumprod", 0.999, 1.001, "avx512_hillis_steele_16"),
        (2, "cummax", -1.0, 1.0, "scalar_recurrence_retained"),
        (3, "cummin", -1.0, 1.0, "scalar_recurrence_retained"),
    ):
        values = rng.uniform(low, high, size=(rows, cols)).astype(np.float32)
        reference_output = np.empty_like(values)
        selected_output = np.empty_like(values)
        reference_samples = []
        selected_samples = []

        def measure(function, output, samples):
            start = time.perf_counter_ns()
            function(
                values.ctypes.data_as(pointer), rows, cols,
                output.ctypes.data_as(pointer), kind,
            )
            samples.append(time.perf_counter_ns() - start)

        for index in range(iterations):
            ordered = (
                ((reference, reference_output, reference_samples),
                 (selected, selected_output, selected_samples))
                if index % 2 == 0
                else ((selected, selected_output, selected_samples),
                      (reference, reference_output, reference_samples))
            )
            for function, output, samples in ordered:
                measure(function, output, samples)
        reference_ns = statistics.median(reference_samples)
        selected_ns = statistics.median(selected_samples)
        error = np.abs(selected_output - reference_output)
        speedup = reference_ns / selected_ns
        if policy.startswith("avx512") and speedup <= 1.05:
            raise RuntimeError(f"{name} AVX-512 scan failed the 1.05x retain gate")
        result.append({
            "op_name": name,
            "selected_policy": policy,
            "reference_median_ms": reference_ns / 1.0e6,
            "selected_median_ms": selected_ns / 1.0e6,
            "speedup": speedup,
            "max_abs_difference": float(error.max(initial=0.0)),
        })
    return result


def _run(target: str, dtype_name: str, iterations: int) -> dict[str, Any]:
    from tessera import runtime as rt

    result_rows = []
    for family, op_name, operands, kwargs, reference_fn in _cases(target, dtype_name):
        artifact = _artifact(rt, target, family, op_name, operands, kwargs)
        start = time.perf_counter_ns()
        cold = rt.launch(artifact, operands)
        cold_ns = time.perf_counter_ns() - start
        if not cold.get("ok"):
            raise RuntimeError(cold.get("reason", f"{family}/{op_name} failed"))
        samples = []
        output = None
        for _ in range(iterations):
            start = time.perf_counter_ns()
            launched = rt.launch(artifact, operands)
            samples.append(time.perf_counter_ns() - start)
            if not launched.get("ok"):
                raise RuntimeError(launched.get("reason", f"{family}/{op_name} failed"))
            output = np.asarray(launched["output"])
        reference = reference_fn()
        error = np.abs(output.astype(np.float32) - reference.astype(np.float32))
        if dtype_name == "f32":
            error_limit = 5.0e-3
        elif family in {"reduce", "scan"}:
            error_limit = 6.0e-1 if dtype_name == "f16" else 5.0
        else:
            error_limit = 1.0e-2 if dtype_name == "f16" else 5.0e-2
        max_error = float(error.max(initial=0.0))
        if max_error > error_limit:
            raise RuntimeError(
                f"{target}/{dtype_name}/{family}/{op_name} error {max_error} "
                f"exceeds {error_limit}"
            )
        median_ns = statistics.median(samples)
        input_bytes = sum(value.nbytes for value in operands)
        output_bytes = output.nbytes
        result_rows.append({
            "family": family,
            "op_name": op_name,
            "dtype": dtype_name,
            "shape": list(operands[0].shape),
            "cold_ms": cold_ns / 1.0e6,
            "warm_median_ms": median_ns / 1.0e6,
            "warm_mad_ms": statistics.median(abs(value - median_ns) for value in samples) / 1.0e6,
            "effective_gb_s": (input_bytes + output_bytes) / median_ns,
            "max_abs_error": max_error,
            "error_limit": error_limit,
            "rms_error": float(np.sqrt(np.mean(np.square(error, dtype=np.float64)))),
        })
    packet = {
        "schema": "tessera.physical_math_evidence.v1",
        "target": target,
        "architecture": "zen5-avx512" if target == "x86" else "gfx1151",
        "dtype": dtype_name,
        "host": platform.platform(),
        "processor": platform.processor(),
        "timing_domain": "synchronized_host_wall",
        "iterations": iterations,
        "rows": result_rows,
    }
    if target == "x86":
        packet["scan_selector_evidence"] = _x86_scan_selector_evidence(
            rt, max(iterations, 20)
        )
    return packet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("x86", "rocm"), required=True)
    parser.add_argument("--dtype", choices=("f32", "f16", "bf16"), default="f32")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if args.target == "x86" and args.dtype != "f32":
        parser.error("the current x86 math ABI admits f32 only")
    print(json.dumps(_run(args.target, args.dtype, args.iterations), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
