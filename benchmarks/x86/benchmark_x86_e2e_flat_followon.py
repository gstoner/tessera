#!/usr/bin/env python3
"""X86-E2E-2 where/transcendental/binary-math descriptor comparison."""

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

SCHEMA = "tessera.x86.e2e_flat_followon_comparison.v1"
SHAPES = ((130,), (32, 257), (1024, 1024))
CASES = (
    ("where", "tessera.where", "x86_where_compiled"),
    ("transcendental", "tessera.exp", "x86_transcendental_compiled"),
    ("pow", "tessera.pow", "x86_binary_math_compiled"),
    ("silu_mul", "tessera.silu_mul", "x86_binary_math_compiled"),
)
PROMOTION_MIN_ELEMENTS = {"where": 1_048_576, "pow": 8_224, "silu_mul": 8_224, "transcendental": 1}


def _module(op_name: str, shape: tuple[int, ...]) -> GraphIRModule:
    extent = "x".join(map(str, shape))
    f32 = IRType(f"tensor<{extent}xf32>", tuple(map(str, shape)), "fp32")
    boolean = IRType(f"tensor<{extent}xi1>", tuple(map(str, shape)), "bool")
    if op_name == "tessera.where":
        args, operands = [IRArg("c", boolean), IRArg("a", f32), IRArg("b", f32)], ["%c", "%a", "%b"]
    elif op_name in {"tessera.pow", "tessera.silu_mul"}:
        args, operands = [IRArg("a", f32), IRArg("b", f32)], ["%a", "%b"]
    else:
        args, operands = [IRArg("x", f32)], ["%x"]
    return GraphIRModule(functions=[GraphIRFunction(
        name="x86_flat_followon_benchmark", args=args, result_types=[f32],
        body=[IROp(result="o", op_name=op_name, operands=operands,
                   operand_types=[str(arg.ir_type) for arg in args],
                   result_type=str(f32), kwargs={})], return_values=["%o"],
    )])


def _typed(package) -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(
        metadata={"target": "x86", "compiler_path": "x86_native_descriptor"},
        tile_ir=package.tile_ir, target_ir=package.target_ir,
        native_image=package.image, launch_descriptor=package.descriptor,
    )


def _retained(op_name: str, path: str, names: list[str]) -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": path, "executable": True,
        "execution_kind": "native_cpu", "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": names, "kwargs": {}}],
    })


def _wall_ms(fn: Callable[[], Any]) -> float:
    start = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns() - start) / 1e6


def _summary(old: list[float], new: list[float]) -> dict[str, Any]:
    retained, compiler = statistics.median(old), statistics.median(new)
    return {
        "retained_samples_ms": old, "compiler_samples_ms": new,
        "retained_median_ms": retained, "compiler_median_ms": compiler,
        "median_speedup": retained / compiler,
        "non_regression_10pct": compiler <= retained * 1.10,
    }


def _measure(family: str, op_name: str, path: str, shape: tuple[int, ...],
             trials: int, rng: np.random.Generator) -> dict[str, Any]:
    package = package_elementwise(_module(op_name, shape), pipeline_name="tessera-lower-to-x86")
    typed = _typed(package)
    if family == "where":
        values = {
            "c": np.ascontiguousarray(rng.random(shape) > 0.5),
            "a": np.ascontiguousarray(rng.standard_normal(shape), dtype=np.float32),
            "b": np.ascontiguousarray(rng.standard_normal(shape), dtype=np.float32),
        }
    elif family in {"pow", "silu_mul"}:
        values = {
            "a": np.ascontiguousarray(rng.uniform(0.1, 2.0, shape), dtype=np.float32),
            "b": np.ascontiguousarray(rng.uniform(-2.0, 2.0, shape), dtype=np.float32),
        }
    else:
        values = {"x": np.ascontiguousarray(rng.uniform(-2.0, 2.0, shape), dtype=np.float32)}
    output = np.zeros(shape, dtype=np.float32)
    typed_values = {**values, "o": output, "N": output.size}
    names = list(values)
    retained = _retained(op_name, path, names)
    typed_result, retained_result = rt.launch(typed, typed_values), rt.launch(retained, values)
    if not typed_result["ok"] or not retained_result["ok"]:
        raise RuntimeError(f"x86 route failed: {typed_result.get('reason')} / {retained_result.get('reason')}")
    np.testing.assert_allclose(output, retained_result["output"], rtol=2e-5, atol=2e-5)
    old: list[float] = []
    new: list[float] = []
    for trial in range(trials):
        routes = ((retained, values, old), (typed, typed_values, new))
        if trial & 1:
            routes = tuple(reversed(routes))
        for artifact, arguments, samples in routes:
            samples.append(_wall_ms(lambda a=artifact, v=arguments: rt.launch(a, v)))
    return {
        "family": family, "operation": op_name, "shape": list(shape),
        "elements": int(np.prod(shape)), "correct": True,
        "retained_route": path, "compiler_route": "x86_native_descriptor",
        "selector_eligible": int(np.prod(shape)) >= PROMOTION_MIN_ELEMENTS[family],
        "end_to_end": _summary(old, new),
    }


def run(trials: int) -> dict[str, Any]:
    rng = np.random.default_rng(8642)
    rows = [_measure(*case, shape, trials, rng) for case in CASES for shape in SHAPES]
    selector_policy_pass = all(
        row["end_to_end"]["non_regression_10pct"]
        for row in rows if row["selector_eligible"]
    )
    return {
        "schema": SCHEMA, "work_item": "X86-E2E-2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_arch": "x86_64_avx512", "trials": trials,
        "timing_policy": "serial alternating retained/compiler CPU wall time",
        "rows": rows, "all_correct": True,
        "all_non_regression": all(row["end_to_end"]["non_regression_10pct"] for row in rows),
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
        print(f'{row["family"]:14s} {row["shape"]}: {row["end_to_end"]["median_speedup"]:.3f}x')
    return 0 if result["all_correct"] and result["selector_policy_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
