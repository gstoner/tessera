#!/usr/bin/env python3
"""Measure compiler cost and IR size for native Graph adjoints.

This is intentionally a compiler-domain benchmark: it times the real
``tessera-opt --tessera-autodiff-paired`` transform and records emitted Graph
IR size/operation count. It does not claim target execution or device timing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time


OPS = ("relu", "rmsnorm", "rmsnorm_affine", "layer_norm",
       "layer_norm_affine", "softmax")


def render_module(op: str, shape: str) -> str:
    graph_op = op.removesuffix("_affine")
    attrs = {
        "relu": "",
        "rmsnorm": " {eps = 1.0e-5 : f64}",
        "rmsnorm_affine": " {eps = 1.0e-5 : f64}",
        "layer_norm": " {eps = 1.0e-5 : f64}",
        "layer_norm_affine": " {eps = 1.0e-5 : f64}",
        "softmax": " {axis = -1 : i64}",
    }[op]
    tensor = f"tensor<{shape}xf32>"
    channel = shape.split("x")[-1]
    if op == "rmsnorm_affine":
        arguments = f"%x: {tensor}, %gamma: tensor<{channel}xf32>"
        operands = "%x, %gamma"
        operand_types = f"{tensor}, tensor<{channel}xf32>"
    elif op == "layer_norm_affine":
        arguments = (f"%x: {tensor}, %gamma: tensor<{channel}xf32>, "
                     f"%beta: tensor<{channel}xf32>")
        operands = "%x, %gamma, %beta"
        operand_types = (f"{tensor}, tensor<{channel}xf32>, "
                         f"tensor<{channel}xf32>")
    else:
        arguments, operands, operand_types = f"%x: {tensor}", "%x", tensor
    return f"""module {{
  func.func @bench({arguments}) -> {tensor}
      attributes {{tessera.autodiff = "reverse"}} {{
    %y = "tessera.{graph_op}"({operands}){attrs} : ({operand_types}) -> {tensor}
    return %y : {tensor}
  }}
}}
"""


def benchmark(opt: Path, op: str, shape: str, warmup: int, reps: int) -> dict:
    module = render_module(op, shape)
    command = [str(opt), "--tessera-autodiff-paired", "/dev/stdin"]
    samples: list[float] = []
    output = ""
    for iteration in range(warmup + reps):
        start = time.perf_counter_ns()
        result = subprocess.run(
            command, input=module, capture_output=True, text=True, check=False)
        elapsed_ms = (time.perf_counter_ns() - start) / 1.0e6
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        output = result.stdout
        if iteration >= warmup:
            samples.append(elapsed_ms)
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered))))
    backward = output.split("@bench__bwd", 1)[-1]
    return {
        "op": op,
        "shape": shape,
        "timing_domain": "host_wall_compiler_transform",
        "reps": reps,
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[p95_index],
        "emitted_bytes": len(output.encode("utf-8")),
        "backward_op_count": sum(
            1 for line in backward.splitlines() if " = " in line),
        "has_custom_adjoint_call": "tessera.custom_adjoint_call" in backward,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tessera-opt", type=Path,
        default=Path("build/tools/tessera-opt/tessera-opt"))
    parser.add_argument("--ops", nargs="+", choices=OPS, default=list(OPS))
    parser.add_argument("--shapes", nargs="+", default=["32x128", "?x128"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [
        benchmark(args.tessera_opt, op, shape, args.warmup, args.reps)
        for op in args.ops for shape in args.shapes
    ]
    payload = {
        "schema": "tessera.compiler_autodiff_graph_benchmark.v1",
        "tool": str(args.tessera_opt),
        "rows": rows,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
