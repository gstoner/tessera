"""Measure APPLE-NATIVE-E2E-2 low-precision GELU descriptor launches.

The benchmark uses the compiler-produced native image and launch descriptor,
not the convenience Python GPU wrapper.  Every row is oracle checked before
timing and reports host-observed synchronous launch latency.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import statistics
import time

import numpy as np

from tessera.compiler.driver import compile_graph_module
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.runtime import RuntimeArtifact, launch


def _shape(value: str) -> tuple[int, int]:
    parts = tuple(int(part) for part in value.lower().split("x"))
    if len(parts) != 2 or any(extent <= 0 for extent in parts):
        raise argparse.ArgumentTypeError(f"shape must be positive MxN, got {value!r}")
    return parts


def _module(storage: str, shape: tuple[int, int] | None) -> GraphIRModule:
    spelling = {"fp16": "f16", "bf16": "bf16"}[storage]
    dimensions = ("?", "?") if shape is None else tuple(map(str, shape))
    ir_type = IRType(
        "tensor<" + "x".join(dimensions) + f"x{spelling}>", dimensions, storage,
    )
    return GraphIRModule(functions=[GraphIRFunction(
        name="gelu", args=[IRArg("x", ir_type)], result_types=[ir_type],
        body=[IROp(
            result="out", op_name="tessera.gelu", operands=["%x"],
            operand_types=[str(ir_type)], result_type=str(ir_type),
        )],
        return_values=["%out"],
    )])


def _artifact(storage: str, shape: tuple[int, int] | None) -> RuntimeArtifact:
    bundle = compile_graph_module(
        _module(storage, shape), source_origin="apple-e2e2-gelu-benchmark",
        target="apple_gpu", options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    return RuntimeArtifact(
        metadata={"target": "apple_gpu", "compiler_path": "apple_native_descriptor"},
        target_ir=bundle.target_ir.text, native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
    )


def _run_case(
    storage: str, shape: tuple[int, int], dynamic: bool, reps: int, warmup: int,
) -> dict[str, object]:
    dtype = np.float16
    if storage == "bf16":
        import ml_dtypes
        dtype = ml_dtypes.bfloat16
    artifact = _artifact(storage, None if dynamic else shape)
    source = np.random.default_rng(sum(shape)).standard_normal(shape).astype(np.float32)
    x = source.astype(dtype)
    x32 = x.astype(np.float32)
    expected = (
        .5 * x32 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x32 + .044715 * x32**3)))
    ).astype(dtype).astype(np.float32)

    def invoke():
        buffers = {"x": x, "out": np.empty_like(x)}
        arguments = ({"buffers": buffers, "scalars": {"Elements": x.size}}
                     if dynamic else buffers)
        result = launch(artifact, arguments)
        if not result["ok"]:
            raise RuntimeError(result["reason"])
        return result["output"]

    observed = invoke()
    tolerance = 4e-3 if storage == "fp16" else 2e-2
    np.testing.assert_allclose(
        observed.astype(np.float32), expected, rtol=tolerance, atol=tolerance,
    )
    for _ in range(warmup):
        invoke()
    samples = []
    for _ in range(reps):
        start = time.perf_counter_ns()
        invoke()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    median_ms = statistics.median(samples)
    logical_bytes = 2 * x.size * np.dtype(dtype).itemsize
    return {
        "work_item": "APPLE-NATIVE-E2E-2",
        "op": "tessera.gelu",
        "storage": storage,
        "shape": "x".join(map(str, shape)),
        "contract": "dynamic_elements" if dynamic else "static",
        "abi_id": artifact.launch_descriptor.abi_id,
        "reps": reps,
        "warmup": warmup,
        "latency_ms_median": median_ms,
        "latency_ms_min": min(samples),
        "latency_ms_stdev": statistics.stdev(samples) if reps > 1 else 0.0,
        "logical_bandwidth_gb_s": logical_bytes / (median_ms * 1e6),
        "numerically_validated": True,
        "execution_kind": "native_gpu",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", type=_shape, default=[(64, 256), (256, 1024)])
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.reps <= 0 or args.warmup < 0:
        parser.error("reps must be positive and warmup must be non-negative")

    rows = [
        _run_case(storage, shape, dynamic, args.reps, args.warmup)
        for storage in ("fp16", "bf16")
        for dynamic in (False, True)
        for shape in args.shapes
    ]
    report = {
        "schema_version": 1,
        "backend": "apple_gpu",
        "device": platform.machine(),
        "macos": platform.mac_ver()[0],
        "timing_scope": "synchronous_descriptor_launch_host_interval",
        "rows": rows,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
