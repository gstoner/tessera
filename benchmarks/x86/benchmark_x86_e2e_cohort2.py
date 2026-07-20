#!/usr/bin/env python3
"""X86-E2E-2 cohort-2 descriptor versus retained-route comparison."""

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
from tessera.compiler.x86_native import package_cohort2  # noqa: E402

SCHEMA = "tessera.x86.e2e_cohort2_comparison.v1"
CASES = (
    ("argreduce", "tessera.argmax", "x86_argreduce_compiled", (256, 1024)),
    ("scan", "tessera.cumsum", "x86_scan_compiled", (256, 1024)),
    ("rmsnorm", "tessera.rmsnorm", "x86_norm_compiled", (256, 1024)),
    ("layernorm", "tessera.layer_norm", "x86_norm_compiled", (256, 1024)),
    ("rope", "tessera.rope", "x86_rope_compiled", (256, 1024)),
    ("alibi", "tessera.alibi", "x86_alibi_compiled", (32, 256)),
)


def _type(shape: tuple[int, ...], dtype: str = "fp32") -> IRType:
    suffix = "f32" if dtype == "fp32" else "i32"
    extents = "x".join(map(str, shape))
    return IRType(f"tensor<{extents + 'x' if extents else ''}{suffix}>", tuple(map(str, shape)), dtype)


def _module(op_name: str, shape: tuple[int, ...]) -> GraphIRModule:
    if op_name == "tessera.alibi":
        h, s = shape
        source, result = _type((h,)), _type((h, s, s))
        args, operands, kwargs = [IRArg("slopes", source)], ["%slopes"], {"num_heads": h, "seq_len": s}
    elif op_name == "tessera.rope":
        source = result = _type(shape)
        args, operands, kwargs = [IRArg("x", source), IRArg("theta", source)], ["%x", "%theta"], {}
    else:
        source = _type(shape)
        result = _type(shape[:-1], "int32") if op_name == "tessera.argmax" else source
        args, operands = [IRArg("x", source)], ["%x"]
        kwargs = {"axis": -1, "keepdims": False} if op_name == "tessera.argmax" else ({"axis": -1} if op_name == "tessera.cumsum" else {"eps": 1e-5})
    return GraphIRModule(functions=[GraphIRFunction(
        name="x86_cohort2_benchmark", args=args, result_types=[result],
        body=[IROp(result="o", op_name=op_name, operands=operands,
                   operand_types=[str(arg.ir_type) for arg in args],
                   result_type=str(result), kwargs=kwargs)], return_values=["%o"],
    )])


def _wall_ms(fn: Callable[[], Any]) -> float:
    start = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns() - start) / 1e6


def _summary(old: list[float], new: list[float]) -> dict[str, Any]:
    retained, compiler = statistics.median(old), statistics.median(new)
    return {"retained_samples_ms": old, "compiler_samples_ms": new,
            "retained_median_ms": retained, "compiler_median_ms": compiler,
            "median_speedup": retained / compiler,
            "non_regression_10pct": compiler <= retained * 1.10}


def _measure(family: str, op_name: str, path: str, shape: tuple[int, ...], trials: int,
             rng: np.random.Generator) -> dict[str, Any]:
    package = package_cohort2(_module(op_name, shape), pipeline_name="tessera-lower-to-x86")
    typed = rt.RuntimeArtifact(metadata={"target": "x86"}, tile_ir=package.tile_ir,
                               target_ir=package.target_ir, native_image=package.image,
                               launch_descriptor=package.descriptor)
    if family == "alibi":
        inputs = {"slopes": np.ascontiguousarray(np.linspace(0.01, 0.2, shape[0]), dtype=np.float32)}
        kwargs = {"num_heads": shape[0], "seq_len": shape[1]}
    elif family == "rope":
        inputs = {"x": np.ascontiguousarray(rng.normal(size=shape), dtype=np.float32),
                  "theta": np.ascontiguousarray(rng.normal(size=shape), dtype=np.float32)}
        kwargs = {}
    else:
        inputs = {"x": np.ascontiguousarray(rng.normal(size=shape), dtype=np.float32)}
        kwargs = {"axis": -1, "keepdims": False} if family == "argreduce" else ({"axis": -1} if family == "scan" else {"eps": 1e-5})
    output_shape = tuple(package.descriptor.provenance["output_shape"])
    output = np.zeros(output_shape, np.int32 if family == "argreduce" else np.float32)
    typed_values = {**inputs, "o": output}
    if family == "alibi":
        typed_values.update({"H": shape[0], "S": shape[1]})
    else:
        typed_values.update({"Rows": package.descriptor.provenance["rows"], "Cols": package.descriptor.provenance["cols"]})
    if family in {"rmsnorm", "layernorm"}:
        typed_values["Epsilon"] = 1e-5
    names = list(inputs)
    retained = rt.RuntimeArtifact(metadata={"target": "x86", "compiler_path": path,
        "executable": True, "execution_kind": "native_cpu", "arg_names": names,
        "output_name": "o", "ops": [{"op_name": op_name, "result": "o", "operands": names, "kwargs": kwargs}]})
    old_result, new_result = rt.launch(retained, inputs), rt.launch(typed, typed_values)
    if not old_result["ok"] or not new_result["ok"]:
        raise RuntimeError(f"route failure: {old_result.get('reason')} / {new_result.get('reason')}")
    np.testing.assert_allclose(output, old_result["output"], rtol=3e-5, atol=3e-5)
    old: list[float] = []
    new: list[float] = []
    for trial in range(trials):
        routes = ((retained, inputs, old), (typed, typed_values, new))
        if trial & 1:
            routes = tuple(reversed(routes))
        for artifact, arguments, samples in routes:
            samples.append(_wall_ms(lambda a=artifact, v=arguments: rt.launch(a, v)))
    return {"family": family, "operation": op_name, "shape": list(shape),
            "correct": True, "retained_route": path,
            "compiler_route": "x86_native_descriptor", "end_to_end": _summary(old, new)}


def run(trials: int) -> dict[str, Any]:
    rng = np.random.default_rng(86422)
    rows = [_measure(*case, trials, rng) for case in CASES]
    return {"schema": SCHEMA, "work_item": "X86-E2E-2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "evidence_arch": "x86_64_avx512", "trials": trials,
            "timing_policy": "serial alternating retained/compiler CPU wall time",
            "rows": rows, "all_correct": True,
            "all_non_regression": all(row["end_to_end"]["non_regression_10pct"] for row in rows),
            "selector_changed": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=21)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.trials)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["rows"]:
        print(f'{row["family"]:12s} {row["end_to_end"]["median_speedup"]:.3f}x')
    return 0 if result["all_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
