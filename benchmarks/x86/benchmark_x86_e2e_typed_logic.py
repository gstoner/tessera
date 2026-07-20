#!/usr/bin/env python3
"""X86-E2E-2 compare/logical/bitwise descriptor comparisons."""

from __future__ import annotations

import argparse
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
from tessera.compiler.x86_native import package_elementwise  # noqa: E402

SCHEMA = "tessera.x86.e2e_typed_logic_comparison.v1"
SHAPES = ((130,), (32, 257), (32768,), (1024, 1024))
CASES = (
    ("compare", "tessera.lt", "x86_compare_compiled", "fp32", "f32", "bool", "i1"),
    ("logical", "tessera.logical_and", "x86_logical_compiled", "bool", "i1", "bool", "i1"),
    ("bitwise", "tessera.bitwise_xor", "x86_bitwise_compiled", "int32", "i32", "int32", "i32"),
)
PROMOTION_MIN_ELEMENTS = {"compare": 32_768, "logical": 1, "bitwise": 32_768}


def _module(case: tuple[str, ...], shape: tuple[int, ...]) -> GraphIRModule:
    family, op_name, _, input_dtype, input_mlir, output_dtype, output_mlir = case
    del family
    extent = "x".join(map(str, shape))
    source = IRType(f"tensor<{extent}x{input_mlir}>", tuple(map(str, shape)), input_dtype)
    output = IRType(f"tensor<{extent}x{output_mlir}>", tuple(map(str, shape)), output_dtype)
    return GraphIRModule(functions=[GraphIRFunction(
        name="x86_typed_logic_benchmark",
        args=[IRArg("a", source), IRArg("b", source)], result_types=[output],
        body=[IROp(
            result="o", op_name=op_name, operands=["%a", "%b"],
            operand_types=[str(source), str(source)], result_type=str(output), kwargs={},
        )], return_values=["%o"],
    )])


def _typed(package) -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(
        metadata={"target": "x86", "compiler_path": "x86_native_descriptor"},
        tile_ir=package.tile_ir, target_ir=package.target_ir,
        native_image=package.image, launch_descriptor=package.descriptor,
    )


def _retained(op_name: str, compiler_path: str) -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": compiler_path,
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["a", "b"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": ["a", "b"], "kwargs": {}}],
    })


def _wall_ms(fn: Callable[[], Any]) -> float:
    start = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns() - start) / 1e6


def _summary(retained: list[float], compiler: list[float]) -> dict[str, Any]:
    old, new = statistics.median(retained), statistics.median(compiler)
    return {
        "retained_samples_ms": retained, "compiler_samples_ms": compiler,
        "retained_median_ms": old, "compiler_median_ms": new,
        "median_speedup": old / new, "non_regression_10pct": new <= old * 1.10,
    }


def _inputs(family: str, shape: tuple[int, ...], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if family == "compare":
        return (
            np.ascontiguousarray(rng.standard_normal(shape), dtype=np.float32),
            np.ascontiguousarray(rng.standard_normal(shape), dtype=np.float32),
        )
    if family == "logical":
        return np.ascontiguousarray(rng.random(shape) < 0.5), np.ascontiguousarray(rng.random(shape) < 0.5)
    return (
        np.ascontiguousarray(rng.integers(-(1 << 30), 1 << 30, shape, dtype=np.int32)),
        np.ascontiguousarray(rng.integers(-(1 << 30), 1 << 30, shape, dtype=np.int32)),
    )


def _measure(case: tuple[str, ...], shape: tuple[int, ...], trials: int,
             rng: np.random.Generator) -> dict[str, Any]:
    family, op_name, compiler_path, *_ = case
    package = package_elementwise(_module(case, shape), pipeline_name="tessera-lower-to-x86")
    typed, retained = _typed(package), _retained(op_name, compiler_path)
    a, b = _inputs(family, shape, rng)
    output = np.zeros(shape, dtype=np.int32 if family == "bitwise" else np.bool_)
    typed_args = {"a": a, "b": b, "o": output, "N": output.size}
    retained_args = {"a": a, "b": b}
    typed_result = rt.launch(typed, typed_args)
    retained_result = rt.launch(retained, retained_args)
    if not typed_result["ok"] or not retained_result["ok"]:
        raise RuntimeError(f"x86 route failed: {typed_result.get('reason')} / {retained_result.get('reason')}")
    np.testing.assert_array_equal(output, retained_result["output"])
    old: list[float] = []
    new: list[float] = []
    for trial in range(trials):
        routes = ((retained, retained_args, old), (typed, typed_args, new))
        if trial & 1:
            routes = tuple(reversed(routes))
        for artifact, values, samples in routes:
            samples.append(_wall_ms(lambda art=artifact, vals=values: rt.launch(art, vals)))
    return {
        "family": family, "operation": op_name, "shape": list(shape),
        "elements": int(np.prod(shape)), "correct": True,
        "retained_route": compiler_path, "compiler_route": "x86_native_descriptor",
        "selector_eligible": int(np.prod(shape)) >= PROMOTION_MIN_ELEMENTS[family],
        "end_to_end": _summary(old, new),
    }


def run(trials: int) -> dict[str, Any]:
    rng = np.random.default_rng(8632)
    rows = [_measure(case, shape, trials, rng) for case in CASES for shape in SHAPES]
    selector_policy_pass = all(
        row["end_to_end"]["non_regression_10pct"]
        for row in rows if row["selector_eligible"]
    )
    return {
        "schema": SCHEMA, "work_item": "X86-E2E-2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_arch": "x86_64_avx512", "trials": trials,
        "timing_policy": "serial alternating retained/compiler CPU wall time",
        "coverage_policy": "binary representative per stable ABI family across ragged, medium, and large tensors",
        "rows": rows, "all_correct": True,
        "all_non_regression": all(row["end_to_end"]["non_regression_10pct"] for row in rows),
        "selector_policy_pass": selector_policy_pass,
        "selector_changed": selector_policy_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=41)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.trials)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["rows"]:
        print(f'{row["family"]:8s} {row["shape"]}: {row["end_to_end"]["median_speedup"]:.3f}x')
    return 0 if result["all_correct"] and result["selector_policy_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
