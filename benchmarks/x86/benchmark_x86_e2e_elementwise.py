#!/usr/bin/env python3
"""X86-E2E-2 typed elementwise descriptor versus retained-route comparison."""

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

SCHEMA = "tessera.x86.e2e_elementwise_comparison.v1"
SHAPES = ((130,), (32, 257), (1024, 1024))
BINARY_PROMOTION_MIN_ELEMENTS = 16_384
CASES = (
    ("unary", "tessera.absolute", "x86_unary_compiled"),
    ("binary", "tessera.add", "x86_binary_compiled"),
    ("predicate", "tessera.isfinite", "x86_predicate_compiled"),
)


def _module(op_name: str, shape: tuple[int, ...]) -> GraphIRModule:
    extent = "x".join(map(str, shape))
    source = IRType(f"tensor<{extent}xf32>", tuple(map(str, shape)), "fp32")
    predicate = op_name == "tessera.isfinite"
    result = IRType(
        f"tensor<{extent}x{'i1' if predicate else 'f32'}>",
        tuple(map(str, shape)), "bool" if predicate else "fp32",
    )
    binary = op_name == "tessera.add"
    args = [IRArg("a", source)] + ([IRArg("b", source)] if binary else [])
    operands = ["%a"] + (["%b"] if binary else [])
    return GraphIRModule(functions=[GraphIRFunction(
        name="x86_elementwise_benchmark", args=args, result_types=[result],
        body=[IROp(
            result="o", op_name=op_name, operands=operands,
            operand_types=[str(source)] * len(operands), result_type=str(result), kwargs={},
        )], return_values=["%o"],
    )])


def _typed(package) -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(
        metadata={"target": "x86", "compiler_path": "x86_native_descriptor"},
        tile_ir=package.tile_ir, target_ir=package.target_ir,
        native_image=package.image, launch_descriptor=package.descriptor,
    )


def _retained(op_name: str, compiler_path: str, binary: bool) -> rt.RuntimeArtifact:
    operands = ["a", "b"] if binary else ["a"]
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": compiler_path,
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": operands, "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": operands, "kwargs": {}}],
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


def _measure(family: str, op_name: str, compiler_path: str, shape: tuple[int, ...],
             trials: int, rng: np.random.Generator) -> dict[str, Any]:
    package = package_elementwise(_module(op_name, shape), pipeline_name="tessera-lower-to-x86")
    typed = _typed(package)
    binary = family == "binary"
    retained = _retained(op_name, compiler_path, binary)
    a = np.ascontiguousarray(rng.standard_normal(shape), dtype=np.float32)
    inputs = [a]
    if binary:
        inputs.append(np.ascontiguousarray(rng.standard_normal(shape), dtype=np.float32))
    output = np.zeros(shape, dtype=np.bool_ if family == "predicate" else np.float32)
    typed_args = {"a": inputs[0], "o": output, "N": output.size}
    retained_args = {"a": inputs[0]}
    if binary:
        typed_args["b"] = retained_args["b"] = inputs[1]
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
    selector_eligible = family != "binary" or int(np.prod(shape)) >= BINARY_PROMOTION_MIN_ELEMENTS
    return {
        "family": family, "operation": op_name, "shape": list(shape),
        "elements": int(np.prod(shape)), "correct": True,
        "retained_route": compiler_path, "compiler_route": "x86_native_descriptor",
        "selector_eligible": selector_eligible, "end_to_end": _summary(old, new),
    }


def run(trials: int) -> dict[str, Any]:
    rng = np.random.default_rng(8623)
    rows = [
        _measure(family, op_name, path, shape, trials, rng)
        for family, op_name, path in CASES for shape in SHAPES
    ]
    all_non_regression = all(row["end_to_end"]["non_regression_10pct"] for row in rows)
    selector_policy_pass = all(
        row["end_to_end"]["non_regression_10pct"] for row in rows if row["selector_eligible"]
    )
    return {
        "schema": SCHEMA, "work_item": "X86-E2E-2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_arch": "x86_64_avx512", "trials": trials,
        "timing_policy": "serial alternating retained/compiler CPU wall time",
        "coverage_policy": "one representative per stable ABI family across ragged, medium, and large tensors",
        "rows": rows, "all_correct": True, "all_non_regression": all_non_regression,
        "selector_policy_pass": selector_policy_pass,
        "selector_changed": selector_policy_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=21)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.trials)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["rows"]:
        timing = row["end_to_end"]
        print(f'{row["family"]:9s} {row["shape"]}: {timing["median_speedup"]:.3f}x')
    return 0 if result["all_correct"] and result["selector_policy_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
