#!/usr/bin/env python3
"""X86-E2E-1 paired typed-descriptor versus retained-executor comparison."""

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
from tessera.compiler import x86_native  # noqa: E402
from tessera.compiler.graph_ir import (  # noqa: E402
    GraphIRFunction, GraphIRModule, IRArg, IROp, IRType,
)

SCHEMA = "tessera.x86.e2e_comparison.v1"
SOFTMAX_SHAPES = ((3, 17), (4, 256), (64, 1024))
REDUCTIONS = (("sum", (32, 257)), ("mean", (8, 33)), ("max", (7, 65)))


def _module(shape: tuple[int, ...], op: str) -> GraphIRModule:
    source_text = "x".join(map(str, shape))
    source = IRType(f"tensor<{source_text}xf32>", tuple(map(str, shape)), "fp32")
    if op == "softmax":
        output = source
        op_name = "tessera.softmax"
    else:
        output_shape = shape[:-1]
        output_text = "x".join(map(str, output_shape))
        output = IRType(f"tensor<{output_text}xf32>", tuple(map(str, output_shape)), "fp32")
        op_name = f"tessera.{op}"
    return GraphIRModule(functions=[GraphIRFunction(
        name=f"x86_{op}_benchmark", args=[IRArg("x", source)], result_types=[output],
        body=[IROp(
            result="o", op_name=op_name, operands=["%x"], operand_types=[str(source)],
            result_type=str(output), kwargs={"axis": -1, "keepdims": False},
        )], return_values=["%o"],
    )])


def _typed(package: x86_native.X86NativePackage) -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(
        metadata={"target": "x86", "compiler_path": "x86_native_descriptor"},
        tile_ir=package.tile_ir, target_ir=package.target_ir,
        native_image=package.image, launch_descriptor=package.descriptor,
    )


def _retained(op: str) -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_softmax_compiled" if op == "softmax" else "x86_reduce_compiled",
        "executable": True, "execution_kind": "native_cpu", "arg_names": ["x"],
        "output_name": "o", "ops": [{
            "op_name": "tessera.softmax" if op == "softmax" else f"tessera.{op}",
            "result": "o", "operands": ["x"], "kwargs": {"axis": -1, "keepdims": False},
        }],
    })


def _wall_ms(fn: Callable[[], Any]) -> float:
    start = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns() - start) / 1e6


def _summary(retained: list[float], compiler: list[float]) -> dict[str, Any]:
    if not retained or len(retained) != len(compiler):
        raise ValueError("paired timing samples are required")
    old, new = statistics.median(retained), statistics.median(compiler)
    return {
        "retained_samples_ms": retained, "compiler_samples_ms": compiler,
        "retained_median_ms": old, "compiler_median_ms": new,
        "median_speedup": old / new, "non_regression_10pct": new <= old * 1.10,
    }


def _row(op: str, shape: tuple[int, ...], trials: int, rng: np.random.Generator) -> dict[str, Any]:
    module = _module(shape, op)
    package = (
        x86_native.package_softmax(module, pipeline_name="tessera-lower-to-x86")
        if op == "softmax"
        else x86_native.package_reduction(module, pipeline_name="tessera-lower-to-x86")
    )
    typed, retained = _typed(package), _retained(op)
    x = np.ascontiguousarray(rng.standard_normal(shape), dtype=np.float32)
    output_shape = shape if op == "softmax" else shape[:-1]
    output = np.zeros(output_shape, np.float32)
    args = (
        {"x": x, "o": output, "Rows": int(np.prod(shape[:-1])), "K": shape[-1]}
        if op == "softmax"
        else {"x": x, "o": output, "Outer": int(np.prod(shape[:-1])), "AxisExtent": shape[-1], "Inner": 1}
    )
    typed_result = rt.launch(typed, args)
    retained_result = rt.launch(retained, {"x": x})
    if not typed_result["ok"] or not retained_result["ok"]:
        raise RuntimeError(f"x86 route failed: {typed_result.get('reason')} / {retained_result.get('reason')}")
    np.testing.assert_allclose(output, retained_result["output"], rtol=2e-5, atol=2e-5)
    old, new = [], []
    for trial in range(trials):
        routes = ((retained, {"x": x}, old), (typed, args, new))
        if trial & 1:
            routes = tuple(reversed(routes))
        for artifact, values, samples in routes:
            samples.append(_wall_ms(lambda a=artifact, v=values: rt.launch(a, v)))
    return {"operation": op, "shape": list(shape), "correct": True, "end_to_end": _summary(old, new)}


def run(trials: int) -> dict[str, Any]:
    rng = np.random.default_rng(8603)
    rows = [_row("softmax", shape, trials, rng) for shape in SOFTMAX_SHAPES]
    rows.extend(_row(op, shape, trials, rng) for op, shape in REDUCTIONS)
    return {
        "schema": SCHEMA, "work_item": "X86-E2E-1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_arch": "x86_64_avx512", "trials": trials,
        "timing_policy": "serial alternating retained/compiler CPU wall time",
        "rows": rows, "all_correct": True,
        "all_non_regression": all(row["end_to_end"]["non_regression_10pct"] for row in rows),
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
        print(f'{row["operation"]:7s} {row["shape"]}: {row["end_to_end"]["median_speedup"]:.3f}x')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
