#!/usr/bin/env python3
"""X86-E2E-1 matmul/basic-attention/extended-attention retained comparisons."""

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
from tessera.compiler.x86_native import package_attention, package_matmul  # noqa: E402

SCHEMA = "tessera.x86.e2e_breadth_comparison.v1"
MATMUL_SHAPES = ((64, 128, 96), (127, 65, 79))
ATTENTION_CASES = ((False, (1, 4, 32, 32, 32, 32)), (True, (1, 4, 32, 40, 32, 24)))


def _tensor(shape: tuple[int, ...]) -> IRType:
    text = "x".join(map(str, shape))
    return IRType(f"tensor<{text}xf32>", tuple(map(str, shape)), "fp32")


def _matmul_module(shape: tuple[int, int, int]) -> GraphIRModule:
    m, k, n = shape
    a, b, output = _tensor((m, k)), _tensor((k, n)), _tensor((m, n))
    return GraphIRModule(functions=[GraphIRFunction(
        name="x86_matmul_benchmark", args=[IRArg("a", a), IRArg("b", b)], result_types=[output],
        body=[IROp(result="o", op_name="tessera.matmul", operands=["%a", "%b"],
                   operand_types=[str(a), str(b)], result_type=str(output), kwargs={})],
        return_values=["%o"],
    )])


def _attention_module(extended: bool, dims: tuple[int, ...]) -> GraphIRModule:
    b, h, sq, sk, d, dv = dims
    q, key, value = _tensor((b, h, sq, d)), _tensor((b, h, sk, d)), _tensor((b, h, sk, dv))
    output = _tensor((b, h, sq, dv))
    args = [IRArg("q", q), IRArg("k", key), IRArg("v", value)]
    operands, types = ["%q", "%k", "%v"], [str(q), str(key), str(value)]
    kwargs: dict[str, Any] = {"scale": d ** -0.5, "causal": False}
    if extended:
        bias = _tensor((b, h, sq, sk))
        args.append(IRArg("bias", bias))
        operands.append("%bias")
        types.append(str(bias))
        kwargs.update({"window": 17, "softcap": 6.0})
    return GraphIRModule(functions=[GraphIRFunction(
        name="x86_attention_benchmark", args=args, result_types=[output],
        body=[IROp(result="o", op_name="tessera.flash_attn", operands=operands,
                   operand_types=types, result_type=str(output), kwargs=kwargs)],
        return_values=["%o"],
    )])


def _typed(package) -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(
        metadata={"target": "x86", "compiler_path": "x86_native_descriptor"},
        tile_ir=package.tile_ir, target_ir=package.target_ir,
        native_image=package.image, launch_descriptor=package.descriptor,
    )


def _retained(op: str, operands: list[str], kwargs: dict[str, Any]) -> rt.RuntimeArtifact:
    path = "x86_matmul_family_compiled" if op == "tessera.matmul" else "x86_flash_attn_compiled"
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": path, "executable": True,
        "execution_kind": "native_cpu", "arg_names": operands,
        "output_name": "o", "ops": [{"op_name": op, "result": "o", "operands": operands, "kwargs": kwargs}],
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


def _measure(typed, typed_args, retained, retained_args, output, trials: int) -> dict[str, Any]:
    typed_result = rt.launch(typed, typed_args)
    retained_result = rt.launch(retained, retained_args)
    if not typed_result["ok"] or not retained_result["ok"]:
        raise RuntimeError(f"x86 route failed: {typed_result.get('reason')} / {retained_result.get('reason')}")
    np.testing.assert_allclose(output, retained_result["output"], rtol=5e-5, atol=5e-5)
    old, new = [], []
    for trial in range(trials):
        routes = ((retained, retained_args, old), (typed, typed_args, new))
        if trial & 1:
            routes = tuple(reversed(routes))
        for artifact, values, samples in routes:
            samples.append(_wall_ms(lambda a=artifact, v=values: rt.launch(a, v)))
    return _summary(old, new)


def run(trials: int) -> dict[str, Any]:
    rng = np.random.default_rng(8606)
    rows = []
    for m, k, n in MATMUL_SHAPES:
        package = package_matmul(_matmul_module((m, k, n)), pipeline_name="tessera-lower-to-x86")
        a = np.ascontiguousarray(rng.standard_normal((m, k)), np.float32)
        b = np.ascontiguousarray(rng.standard_normal((k, n)), np.float32)
        output = np.zeros((m, n), np.float32)
        timing = _measure(
            _typed(package), {"a": a, "b": b, "o": output, "M": m, "N": n, "K": k},
            _retained("tessera.matmul", ["a", "b"], {}), {"a": a, "b": b}, output, trials,
        )
        rows.append({"operation": "matmul", "shape": [m, k, n], "correct": True, "end_to_end": timing})
    for extended, dims in ATTENTION_CASES:
        module = _attention_module(extended, dims)
        package = package_attention(module, pipeline_name="tessera-lower-to-x86")
        b, h, sq, sk, d, dv = dims
        q = np.ascontiguousarray(rng.standard_normal((b, h, sq, d)), np.float32)
        key = np.ascontiguousarray(rng.standard_normal((b, h, sk, d)), np.float32)
        value = np.ascontiguousarray(rng.standard_normal((b, h, sk, dv)), np.float32)
        output = np.zeros((b, h, sq, dv), np.float32)
        scalar_args = {"B": b, "Hq": h, "Hkv": h, "Sq": sq, "Sk": sk, "D": d, "Dv": dv}
        typed_args = {"q": q, "k": key, "v": value, "o": output, **scalar_args}
        retained_args = {"q": q, "k": key, "v": value}
        operands, kwargs = ["q", "k", "v"], {"scale": d ** -0.5, "causal": False}
        if extended:
            bias = np.ascontiguousarray(rng.standard_normal((b, h, sq, sk)) * 0.1, np.float32)
            typed_args["bias"] = bias
            retained_args["bias"] = bias
            operands.append("bias")
            kwargs.update({"window": 17, "logit_softcap": 6.0})
        timing = _measure(
            _typed(package), typed_args,
            _retained("tessera.flash_attn", operands, kwargs), retained_args, output, trials,
        )
        rows.append({
            "operation": "attention_ext" if extended else "attention",
            "shape": list(dims), "correct": True, "end_to_end": timing,
        })
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
        print(f'{row["operation"]:14s} {row["shape"]}: {row["end_to_end"]["median_speedup"]:.3f}x')
    return 0 if result["all_correct"] and result["all_non_regression"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
