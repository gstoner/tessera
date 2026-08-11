#!/usr/bin/env python3
"""Reproducible physical general-solver and ISTFT-window-JVP evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

import numpy as np

import tessera
from tessera import runtime as rt
from tessera.autodiff.jvp import get_jvp
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.implicit_solver import compile_physical_general_solver_from_graph
from tessera.compiler.scheduled_matmul import find_tessera_opt


@tessera.jit(target="x86", autodiff="jvp", wrt=("spectrum", "window"))
def _x86_istft(spectrum, window):
    return tessera.ops.istft(spectrum, window, hop=4)


@tessera.jit(target="rocm", autodiff="jvp", wrt=("spectrum", "window"))
def _rocm_istft(spectrum, window):
    return tessera.ops.istft(spectrum, window, hop=4)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nonlinear_residual_graph(shape: tuple[int, ...]) -> GraphIRModule:
    ty = IRType("tensor<" + "x".join(map(str, shape)) + "xf32>")
    text = str(ty)
    return GraphIRModule(functions=[GraphIRFunction(
        name="nonlinear_residual",
        args=[IRArg("theta", ty), IRArg("x", ty)], result_types=[ty],
        body=[
            IROp("xx", "tessera.mul", ["%x", "%x"], [text, text], text),
            IROp("sx", "tessera.sin", ["%x"], [text], text),
            IROp("sum", "tessera.add", ["%xx", "%sx"], [text, text], text),
            IROp("residual", "tessera.sub", ["%sum", "%theta"], [text, text], text),
        ], return_values=["%residual"],
    )])


def _reduction_residual_graph(shape: tuple[int, int]) -> GraphIRModule:
    ty = IRType("tensor<" + "x".join(map(str, shape)) + "xf32>")
    reduced = IRType(f"tensor<{shape[0]}x1xf32>")
    text = str(ty)
    return GraphIRModule(functions=[GraphIRFunction(
        name="reduction_residual",
        args=[IRArg("theta", ty), IRArg("x", ty)], result_types=[ty],
        body=[
            IROp("m", "tessera.mean", ["%x"], [text], str(reduced),
                 kwargs={"axis": 1, "keepdims": True}),
            IROp("xm", "tessera.add", ["%x", "%m"],
                 [text, str(reduced)], text),
            IROp("r", "tessera.sub", ["%xm", "%theta"],
                 [text, text], text),
        ], return_values=["%r"],
    )])


def _matmul_residual_graph(n: int) -> GraphIRModule:
    storage = IRType(f"tensor<{n}x{n}xf16>")
    accumulation = IRType(f"tensor<{n}x{n}xf32>")
    return GraphIRModule(functions=[GraphIRFunction(
        name="matmul_residual",
        args=[IRArg("theta", storage), IRArg("x", storage)],
        result_types=[accumulation],
        body=[
            IROp("xw", "tessera.matmul", ["%x", "%theta"],
                 [str(storage), str(storage)], str(accumulation)),
            IROp("r", "tessera.sub", ["%xw", "%x"],
                 [str(accumulation), str(storage)], str(accumulation)),
        ], return_values=["%r"],
    )])


def _dynamic_mixed_residual_graph() -> GraphIRModule:
    theta = IRType("tensor<?x?xf16>")
    value = IRType("tensor<?x?xf32>")
    return GraphIRModule(functions=[GraphIRFunction(
        name="dynamic_mixed_residual",
        args=[IRArg("theta", theta), IRArg("x", value)], result_types=[value],
        body=[IROp("r", "tessera.sub", ["%x", "%theta"],
                   [str(value), str(theta)], str(value))],
        return_values=["%r"],
    )])


def _predicate_residual_graph(kind: str, shape: tuple[int, ...]) -> GraphIRModule:
    ty = IRType("tensor<" + "x".join(map(str, shape)) + "xf32>")
    scalar = IRType("tensor<1xf32>")
    predicate = IRType("tensor<1xi1>")
    text = str(ty)
    means = [
        IROp("mx", "tessera.mean", ["%x"], [text], str(scalar),
             kwargs={"axis": None, "keepdims": True}),
        IROp("mt", "tessera.mean", ["%theta"], [text], str(scalar),
             kwargs={"axis": None, "keepdims": True}),
    ]
    if kind == "if":
        body = means + [
            IROp("pred", "tessera.greater", ["%mx", "%mt"],
                 [str(scalar), str(scalar)], str(predicate)),
            IROp(
                "branch", "tessera.control_if", ["%pred"], [str(predicate)], text,
                kwargs={
                    "_region": "if", "_flag_ssa": "pred",
                    "_then_body": [IROp(
                        "then", "tessera.mul", ["%x", "%x"],
                        [text, text], text,
                    )],
                    "_then_ssa": "then",
                    "_else_body": [IROp(
                        "else", "tessera.add", ["%x", "%theta"],
                        [text, text], text,
                    )],
                    "_else_ssa": "else",
                },
            ),
            IROp("r", "tessera.sub", ["%branch", "%theta"],
                 [text, text], text),
        ]
    elif kind == "while":
        body = [
            IROp(
                "loop", "tessera.control_while", ["%x"], [text], text,
                kwargs={
                    "_region": "while", "_max_iters": 6,
                    "_carry_ssa": "carry",
                    "_cond": [
                        IROp("mc", "tessera.mean", ["%carry"], [text], str(scalar),
                             kwargs={"axis": None, "keepdims": True}),
                        IROp("mt", "tessera.mean", ["%theta"], [text], str(scalar),
                             kwargs={"axis": None, "keepdims": True}),
                        IROp("pred", "tessera.less", ["%mc", "%mt"],
                             [str(scalar), str(scalar)], str(predicate)),
                    ],
                    "_pred_ssa": "pred",
                    "_body": [IROp(
                        "next", "tessera.add", ["%carry", "%x"],
                        [text, text], text,
                    )],
                    "_next_ssa": "next",
                },
            ),
            IROp("r", "tessera.sub", ["%loop", "%theta"],
                 [text, text], text),
        ]
    else:
        raise ValueError(f"unknown predicate residual kind {kind}")
    return GraphIRModule(functions=[GraphIRFunction(
        name=f"predicate_{kind}_residual",
        args=[IRArg("theta", ty), IRArg("x", ty)], result_types=[ty],
        body=body, return_values=["%r"],
    )])


def _samples(call, warmup: int, count: int) -> tuple[Any, int, list[int]]:
    cold_start = time.perf_counter_ns()
    result = call()
    cold_ns = time.perf_counter_ns() - cold_start
    for _ in range(warmup):
        call()
    durations: list[int] = []
    for _ in range(count):
        start = time.perf_counter_ns()
        result = call()
        durations.append(time.perf_counter_ns() - start)
    return result, cold_ns, durations


def run(target: str, warmup: int, samples: int, clean_host: bool) -> dict[str, Any]:
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("production tessera-opt is unavailable")
    rng = np.random.default_rng(20260811)
    shape = (257,)
    package = compile_physical_general_solver_from_graph(
        _nonlinear_residual_graph(shape), target=target, shape=shape,
        product_mode="vjp", tolerance=2.0e-6, max_iterations=32,
        restart=16,
    )
    solver_artifact = rt.RuntimeArtifact(metadata=package.runtime_metadata())
    solution = rng.uniform(0.75, 1.75, size=shape).astype(np.float32)
    parameter = (solution * solution + np.sin(solution)).astype(np.float32)
    cotangent = rng.normal(size=shape).astype(np.float32)
    expected_solver_product = cotangent / (
        2.0 * solution + np.cos(solution)
    )

    def solve():
        result = rt.launch(solver_artifact, (parameter, solution, cotangent))
        if not result["ok"]:
            raise RuntimeError(result["reason"])
        return result["output"]

    solver_output, solver_cold_ns, solver_ns = _samples(solve, warmup, samples)
    solver_error = max(
        float(np.max(np.abs(np.asarray(solver_output[0])))),
        float(np.max(np.abs(
            np.asarray(solver_output[1]) - expected_solver_product
        ))),
        float(np.max(np.abs(
            np.asarray(solver_output[2]) - expected_solver_product
        ))),
    )
    expanded_solver_rows: list[tuple[
        str, str, int, list[int], float, dict[str, Any], str,
    ]] = []

    def measure_general(
        name: str,
        graph: GraphIRModule,
        bounds: tuple[int, ...],
        theta: np.ndarray,
        x: np.ndarray,
        vector: np.ndarray,
        expected: tuple[np.ndarray, np.ndarray, np.ndarray],
        resources: dict[str, Any],
        dtype: str = "f32",
        tolerance: float = 2.0e-6,
    ) -> None:
        candidate = compile_physical_general_solver_from_graph(
            graph, target=target, shape=bounds, product_mode="vjp",
            tolerance=tolerance, max_iterations=32, restart=16,
        )
        artifact = rt.RuntimeArtifact(metadata=candidate.runtime_metadata())

        def invoke():
            result = rt.launch(artifact, (theta, x, vector))
            if not result["ok"]:
                raise RuntimeError(result["reason"])
            return result["output"]

        output, cold_ns, durations = _samples(invoke, warmup, samples)
        error = max(
            float(np.max(np.abs(np.asarray(actual) - np.asarray(reference))))
            for actual, reference in zip(output, expected)
        )
        replay = candidate.contract["children"]["solution_vjp"]["metadata"][
            "solver_residual_program"
        ]["predicate_replay"]
        expanded_solver_rows.append((
            name, candidate.artifact_hash, cold_ns, durations, error,
            {
                **resources,
                "physical_children": 5,
                "children_generated_from_graph": True,
                "matrix_materialized": False,
                "solver": "restarted_gmres",
                "last_solve": artifact.metadata.get("last_solve"),
                "predicate_replay": replay,
            },
            dtype,
        ))

    reduction_shape = (8, 16)
    reduction_x = rng.normal(size=reduction_shape).astype(np.float32)
    reduction_theta = (
        reduction_x + np.mean(reduction_x, axis=1, keepdims=True)
    ).astype(np.float32)
    reduction_vector = rng.normal(size=reduction_shape).astype(np.float32)
    reduction_linear = (
        reduction_vector
        - 0.5 * np.mean(reduction_vector, axis=1, keepdims=True)
    ).astype(np.float32)
    measure_general(
        "generated_reduction_residual_vjp",
        _reduction_residual_graph(reduction_shape), reduction_shape,
        reduction_theta, reduction_x, reduction_vector,
        (np.zeros(reduction_shape, np.float32), reduction_linear, reduction_linear),
        {"family": "sum_mean_broadcast", "shape": list(reduction_shape)},
    )

    matmul_n = 16
    matmul_theta = (2.0 * np.eye(matmul_n)).astype(np.float16)
    # Exactly representable storage/product values keep this packet focused on
    # the generated matmul lineage rather than conflating it with f16 rounding.
    matmul_x = (
        (np.arange(matmul_n * matmul_n) % 17).reshape(matmul_n, matmul_n) / 8.0
    ).astype(np.float16)
    matmul_vector = (
        ((np.arange(matmul_n * matmul_n) % 13) - 6).reshape(matmul_n, matmul_n) / 8.0
    ).astype(np.float32)
    matmul_x_f32 = matmul_x.astype(np.float32)
    measure_general(
        "generated_matmul_residual_vjp",
        _matmul_residual_graph(matmul_n), (matmul_n, matmul_n),
        matmul_theta, matmul_x, matmul_vector,
        (matmul_x_f32, matmul_vector, -(matmul_x_f32.T @ matmul_vector)),
        {
            "family": "rank2_matmul", "shape": [matmul_n, matmul_n],
            "storage": "f16", "accumulation": "f32",
        },
        "f16_storage_f32_accumulation",
        5.0e-4,
    )

    dynamic_shape = (17, 31)
    dynamic_bounds = (32, 64)
    dynamic_theta = rng.uniform(-1.0, 1.0, size=dynamic_shape).astype(np.float16)
    dynamic_x = dynamic_theta.astype(np.float32)
    dynamic_vector = rng.normal(size=dynamic_shape).astype(np.float32)
    measure_general(
        "generated_dynamic_mixed_residual_vjp",
        _dynamic_mixed_residual_graph(), dynamic_bounds,
        dynamic_theta, dynamic_x, dynamic_vector,
        (np.zeros(dynamic_shape, np.float32), dynamic_vector, dynamic_vector),
        {
            "family": "bounded_dynamic_mixed_storage",
            "runtime_shape": list(dynamic_shape),
            "shape_bounds": list(dynamic_bounds),
            "parameter_storage": "f16",
            "accumulation": "f32",
        },
        "f16_parameter_f32_compute",
    )

    predicate_shape = (257,)
    predicate_vector = rng.normal(size=predicate_shape).astype(np.float32)
    if_x = np.full(predicate_shape, 2.0, np.float32)
    if_theta = np.full(predicate_shape, 1.0, np.float32)
    measure_general(
        "generated_control_if_residual_vjp",
        _predicate_residual_graph("if", predicate_shape), predicate_shape,
        if_theta, if_x, predicate_vector,
        (
            np.full(predicate_shape, 3.0, np.float32),
            predicate_vector / 4.0,
            predicate_vector / 4.0,
        ),
        {"family": "data_dependent_if", "predicate": "mean(x)>mean(theta)"},
    )

    while_x = np.ones(predicate_shape, np.float32)
    while_theta = np.full(predicate_shape, 4.5, np.float32)
    measure_general(
        "generated_control_while_residual_vjp",
        _predicate_residual_graph("while", predicate_shape), predicate_shape,
        while_theta, while_x, predicate_vector,
        (
            np.full(predicate_shape, 0.5, np.float32),
            predicate_vector / 5.0,
            predicate_vector / 5.0,
        ),
        {
            "family": "data_dependent_while",
            "predicate": "mean(carry)<mean(theta)",
            "max_iterations": 6,
            "executed_iterations": 4,
        },
    )

    spectrum = (
        rng.normal(size=(16, 5)) + 1j * rng.normal(size=(16, 5))
    ).astype(np.complex64)
    spectrum[:, (0, -1)] = spectrum[:, (0, -1)].real
    window = (np.hanning(8) + 0.2).astype(np.float32)
    dspectrum = (
        rng.normal(size=spectrum.shape) + 1j * rng.normal(size=spectrum.shape)
    ).astype(np.complex64)
    dspectrum[:, (0, -1)] = dspectrum[:, (0, -1)].real
    dwindow = rng.normal(size=window.shape).astype(np.float32)
    compiled = _x86_istft if target == "x86" else _rocm_istft

    def istft_product():
        return compiled.native_jvp(
            spectrum, window, tangents=(dspectrum, dwindow)
        )

    istft_output, istft_cold_ns, istft_ns = _samples(
        istft_product, warmup, samples
    )
    rule = get_jvp("istft")
    assert rule is not None
    istft_reference = rule(
        (spectrum, window), (dspectrum, dwindow), n_fft=8, hop=4
    )
    istft_error = max(
        float(np.max(np.abs(np.asarray(actual) - np.asarray(reference))))
        for actual, reference in zip(istft_output, istft_reference)
    )

    release = platform.release().lower()
    wsl = "microsoft" in release or bool(os.environ.get("WSL_INTEROP"))
    timing_common = {
        "source": "synchronized_host_wall",
        "clock": "time.perf_counter_ns",
        "warmup": warmup,
        "submission_completion": "runtime package returns after target synchronization",
        "device_clock": {"valid": False, "value": None},
    }

    def row(
        name: str, digest: str, cold_ns: int, durations: list[int],
        error: float, resources: dict[str, Any], dtype: str = "f32",
    ) -> dict[str, Any]:
        return {
            "name": name,
            "artifact_digest": digest,
            "dtype": dtype,
            "timing": {
                **timing_common,
                "cold_sample_ns": cold_ns,
                "cache_state": "warmed_after_explicit_cold_sample",
                "samples_ns": durations,
                "median_ns": int(statistics.median(durations)),
                "minimum_ns": min(durations),
            },
            "numerical": {"max_abs_error": error, "passed": error <= 5.0e-4},
            "resources": resources,
        }

    architecture = "zen5_avx512" if target == "x86" else "gfx1151"
    clean = bool(clean_host and not wsl)
    return {
        "schema": "tessera.autodiff_physical_products.evidence.v1",
        "work_items": ["AD-SOLVER-IFT-1", "AD-FWD-NATIVE-1"],
        "target": target,
        "architecture": architecture,
        "device": (
            next((line.split(":", 1)[1].strip()
                  for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.startswith("model name")), "unknown")
            if target == "x86" else rt._rocm_chip()
        ),
        "host": platform.platform(),
        "host_clean": clean,
        "wsl": wsl,
        "toolchain": subprocess.run(
            [str(tool), "--version"], capture_output=True, check=True, text=True
        ).stdout.splitlines()[0],
        "tessera_opt_sha256": _sha256(tool),
        "rows": [
            row(
                "generated_nonlinear_residual_vjp", package.artifact_hash,
                solver_cold_ns, solver_ns, solver_error,
                {
                    "physical_children": 5,
                    "residual_expression": "x*x+sin(x)-theta",
                    "children_generated_from_graph": True,
                    "matrix_materialized": False,
                    "solver": "restarted_gmres",
                    "last_solve": solver_artifact.metadata.get("last_solve"),
                },
            ),
            *(row(*entry) for entry in expanded_solver_rows),
            row("istft_spectrum_window_jvp",
                str(compiled.last_jvp_execution["artifact_hash"]),
                istft_cold_ns, istft_ns, istft_error,
                {
                    "inverse_fft_products": 2,
                    "overlap_add_products": 1,
                    "window_energy_derivative": True,
                    "workspace_policy": (
                        "thread_local_cached" if target == "x86" else
                        "per_call_device_temporary"
                    ),
                }),
        ],
        "promotion": {
            "correctness_eligible": (
                solver_error <= 5.0e-4
                and istft_error <= 5.0e-4
                and all(entry[4] <= 5.0e-4 for entry in expanded_solver_rows)
            ),
            "performance_eligible": False,
            "reason": (
                "clean host wall timing lacks an independent device clock"
                if clean else
                "WSL or unasserted clean-host timing is regression-only"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("x86", "rocm_gfx1151"), required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--clean-host", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = run(args.target, args.warmup, args.samples, args.clean_host)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
