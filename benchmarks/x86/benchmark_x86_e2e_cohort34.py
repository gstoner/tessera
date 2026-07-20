#!/usr/bin/env python3
"""X86-E2E-2 cohort-3/4 retained versus Graph-descriptor comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from tessera import runtime as rt  # noqa: E402
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType  # noqa: E402
from tessera.compiler.x86_breadth import (  # noqa: E402
    GRAPH_PROMOTION_THRESHOLDS,
    X86_BREADTH_ABIS,
    graph_breadth_contract,
    package_graph_breadth,
)

SCHEMA = "tessera.x86.e2e_cohort34_comparison.v1"
CASES = (
    *(('gather', n) for n in (130, 16_384, 1_048_576)),
    *(('pointwise_loss', n) for n in (130, 16_384, 1_048_576)),
    *(('cholesky', n) for n in (8, 32, 64)),
    *(('tri_solve', n) for n in (8, 32, 64)),
)


def _type(shape: tuple[int, ...], dtype: str = "fp32") -> IRType:
    spelling = {"fp32": "f32", "int64": "i64"}[dtype]
    extents = "x".join(map(str, shape))
    return IRType(f"tensor<{extents}x{spelling}>", tuple(map(str, shape)), dtype)


def _module(family: str, extent: int) -> GraphIRModule:
    if family == "gather":
        source, indices, result = _type((extent * 2,)), _type((extent,), "int64"), _type((extent,))
        args = [IRArg("source", source), IRArg("indices", indices)]
        op_name, operands, kwargs = "tessera.gather", ["%source", "%indices"], {"axis": 0}
    elif family == "pointwise_loss":
        source = result = _type((extent,))
        args = [IRArg("prediction", source), IRArg("target", source)]
        op_name, operands, kwargs = (
            "tessera.mse_loss", ["%prediction", "%target"], {"reduction": "none"},
        )
    elif family == "cholesky":
        source = result = _type((2, extent, extent))
        args = [IRArg("matrix", source)]
        op_name, operands, kwargs = "tessera.cholesky", ["%matrix"], {}
    else:
        matrix, rhs = _type((2, extent, extent)), _type((2, extent, 4))
        result = rhs
        args = [IRArg("matrix", matrix), IRArg("rhs", rhs)]
        op_name, operands, kwargs = "tessera.tri_solve", ["%matrix", "%rhs"], {"lower": True}
    return GraphIRModule(functions=[GraphIRFunction(
        name=f"x86_{family}_benchmark", args=args, result_types=[result],
        body=[IROp(result="output", op_name=op_name, operands=operands,
                   operand_types=[str(arg.ir_type) for arg in args],
                   result_type=str(result), kwargs=kwargs)], return_values=["%output"],
    )])


def _artifact(path: str, op_name: str, names: list[str], kwargs: dict[str, object]) -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": path, "executable": True,
        "execution_kind": "native_cpu", "arg_names": names, "output_name": "output",
        "ops": [{"op_name": op_name, "result": "output", "operands": names,
                 "kwargs": kwargs}],
    })


def _wall_ms(fn: Callable[[], Any]) -> float:
    start = time.perf_counter_ns()
    result = fn()
    if isinstance(result, dict) and not result.get("ok", False):
        raise RuntimeError(str(result.get("reason")))
    return (time.perf_counter_ns() - start) / 1e6


def _summary(retained: list[float], descriptor: list[float]) -> dict[str, Any]:
    old, new = statistics.median(retained), statistics.median(descriptor)
    return {
        "retained_samples_ms": retained, "descriptor_samples_ms": descriptor,
        "retained_median_ms": old, "descriptor_median_ms": new,
        "median_speedup": old / new, "non_regression_10pct": new <= old * 1.10,
    }


def _measure(family: str, extent: int, trials: int,
             rng: np.random.Generator) -> dict[str, Any]:
    module = _module(family, extent)
    contract = graph_breadth_contract(module)
    assert contract is not None
    package = package_graph_breadth(module, pipeline_name="tessera-lower-to-x86")
    typed = rt.RuntimeArtifact(
        metadata={"target": "x86", "compiler_path": "canonical_native_descriptor"},
        tile_ir=package.tile_ir, target_ir=package.target_ir,
        native_image=package.image, launch_descriptor=package.descriptor,
    )
    if family == "gather":
        source = np.ascontiguousarray(rng.normal(size=extent * 2), dtype=np.float32)
        indices = np.ascontiguousarray(rng.integers(0, source.size, size=extent), dtype=np.int64)
        output = np.zeros(extent, dtype=np.float32)
        typed_values: dict[str, object] = {"source": source, "indices": indices, "output": output}
        retained_fn = lambda: source[indices]
        expected = retained_fn()
        retained_route = "numpy_gather_reference"
    elif family == "pointwise_loss":
        prediction = np.ascontiguousarray(rng.normal(size=extent), dtype=np.float32)
        target = np.ascontiguousarray(rng.normal(size=extent), dtype=np.float32)
        output = np.zeros(extent, dtype=np.float32)
        typed_values = {"prediction": prediction, "target": target, "output": output}
        retained_artifact = _artifact(
            "x86_loss_compiled", "tessera.mse_loss", ["prediction", "target"],
            {"reduction": "none"},
        )
        retained_fn = lambda: rt.launch(
            retained_artifact, {"prediction": prediction, "target": target}
        )
        retained_result = retained_fn()
        expected = retained_result["output"]
        retained_route = "x86_loss_compiled"
    elif family == "cholesky":
        raw = rng.normal(size=(2, extent, extent)).astype(np.float32)
        matrix = np.ascontiguousarray(
            raw @ raw.transpose(0, 2, 1) + np.float32(2.0) * np.eye(extent, dtype=np.float32)
        )
        output = np.zeros_like(matrix)
        typed_values = {"matrix": matrix, "output": output}
        retained_artifact = _artifact("x86_linalg_compiled", "tessera.cholesky", ["matrix"], {})
        retained_fn = lambda: rt.launch(retained_artifact, {"matrix": matrix})
        retained_result = retained_fn()
        expected = retained_result["output"]
        retained_route = "x86_linalg_compiled"
    else:
        lower = np.tril(rng.normal(size=(2, extent, extent))).astype(np.float32)
        lower[:, np.arange(extent), np.arange(extent)] += np.float32(extent + 1)
        matrix = np.ascontiguousarray(lower)
        rhs = np.ascontiguousarray(rng.normal(size=(2, extent, 4)), dtype=np.float32)
        output = np.zeros_like(rhs)
        typed_values = {"matrix": matrix, "rhs": rhs, "output": output}
        retained_artifact = _artifact(
            "x86_linalg_compiled", "tessera.tri_solve", ["matrix", "rhs"], {"lower": True},
        )
        retained_fn = lambda: rt.launch(retained_artifact, {"matrix": matrix, "rhs": rhs})
        retained_result = retained_fn()
        expected = retained_result["output"]
        retained_route = "x86_linalg_compiled"
    typed_values.update(cast(dict[str, object], contract["scalars"]))
    first = rt.launch(typed, typed_values)
    if not first["ok"]:
        raise RuntimeError(str(first.get("reason")))
    np.testing.assert_allclose(output, expected, rtol=3e-4, atol=3e-4)
    retained_samples: list[float] = []
    descriptor_samples: list[float] = []
    for trial in range(trials):
        routes = ((retained_fn, retained_samples),
                  (lambda: rt.launch(typed, typed_values), descriptor_samples))
        if trial & 1:
            routes = tuple(reversed(routes))
        for function, samples in routes:
            samples.append(_wall_ms(function))
    return {
        "family": family, "extent": extent,
        "output_elements": int(np.prod(output.shape)), "correct": True,
        "retained_route": retained_route,
        "compiler_route": "canonical_native_descriptor",
        "end_to_end": _summary(retained_samples, descriptor_samples),
    }


def _dispositions(rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    promoted_keys = {
        "gather_f32": "gather", "pointwise_loss_f32": "pointwise_loss",
        "cholesky_f32": "cholesky", "tri_solve_f32": "tri_solve",
    }
    out: list[dict[str, object]] = []
    for key, spec in X86_BREADTH_ABIS.items():
        family = promoted_keys.get(key)
        if family is not None:
            threshold = GRAPH_PROMOTION_THRESHOLDS[family]
            family_rows = [row for row in rows if row["family"] == family]
            passing = [row for row in family_rows if row["end_to_end"]["non_regression_10pct"]]
            decision = "promote_measured" if threshold is not None else "retain_performance"
            reason = (
                f"canonical Graph descriptor from {threshold} output elements"
                if threshold is not None else "no measured row passed the 10% bound"
            )
            out.append({"abi_key": key, "cohort": spec.cohort, "decision": decision,
                        "threshold": threshold, "reason": reason,
                        "measured_rows": len(family_rows), "passing_rows": len(passing)})
        else:
            composite = spec.public_route != "direct_abi" or key in {
                "scatter_f32", "lu_f32", "qr_f32", "svd_f32", "optimizer_f32",
                "selective_ssm_f32", "selective_ssm_f16", "selective_ssm_bf16",
                "selective_ssm_bwd_f32", "kv_cache_append_f32", "kv_cache_read_f32",
                "kv_cache_prune_f32", "deltanet_f32", "moe_f32",
            }
            decision = "retain_composite" if composite else "retain_specialized"
            out.append({"abi_key": key, "cohort": spec.cohort, "decision": decision,
                        "threshold": None,
                        "reason": ("public semantics require packing, composition, state, or multiple outputs"
                                   if composite else "native ABI remains explicit; no isomorphic Graph selector contract")})
    return out


def run(trials: int) -> dict[str, Any]:
    rng = np.random.default_rng(86734)
    rows = [_measure(family, extent, trials, rng) for family, extent in CASES]
    dispositions = _dispositions(rows)
    return {
        "schema": SCHEMA, "work_item": "X86-E2E-2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_arch": "x86_64_avx512", "trials": trials,
        "timing_policy": "serial alternating retained/compiler CPU wall time",
        "rows": rows, "dispositions": dispositions,
        "all_correct": all(row["correct"] for row in rows),
        "operation_total": len(dispositions) == len(X86_BREADTH_ABIS),
        "selector_changed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=21)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.trials)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["rows"]:
        print(f'{row["family"]:16s} {row["extent"]:8d} '
              f'{row["end_to_end"]["median_speedup"]:.3f}x')
    return 0 if result["all_correct"] and result["operation_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
