"""Operation-total timing for compiler-owned factored Adafactor on gfx1151."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from tessera import runtime as rt  # noqa: E402


def artifact() -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(metadata={
        "target": "rocm",
        "compiler_path": "rocm_adafactor_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": ["parameter", "gradient", "row", "col"],
        "output_name": "output",
        "ops": [{
            "op_name": "tessera.adafactor",
            "result": "output",
            "operands": ["parameter", "gradient", "row", "col"],
            "kwargs": {"lr": 1e-2, "beta2": 0.9, "eps": 1e-6, "step": 8},
        }],
    })


def backward_artifact() -> rt.RuntimeArtifact:
    return rt.RuntimeArtifact(metadata={
        "target": "rocm",
        "compiler_path": "rocm_adafactor_bwd_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": ["parameter", "gradient", "row", "col", "dy"],
        "output_name": "gradients",
        "ops": [{
            "op_name": "tessera.adafactor",
            "result": "gradients",
            "operands": ["parameter", "gradient", "row", "col"],
            "out_cotangent": "dy",
            "kwargs": {"lr": 1e-2, "beta2": 0.9, "eps": 1e-6, "step": 8},
        }],
    })


def record(*, shape=(257, 255), warmup=5, iterations=30, backward=False):
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations positive")
    rng = np.random.default_rng(20260727)
    parameter = rng.normal(size=shape).astype(np.float32)
    gradient = rng.normal(scale=0.2, size=shape).astype(np.float32)
    row = np.zeros(shape[:-1], np.float32)
    col = np.zeros(shape[-1], np.float32)
    compiled = backward_artifact() if backward else artifact()
    dy = rng.normal(size=shape).astype(np.float32)

    def launch():
        operands = (
            (parameter, gradient, row, col, dy)
            if backward
            else (parameter, gradient, row, col)
        )
        result = rt.launch(compiled, operands)
        if not result["ok"]:
            raise RuntimeError(result.get("reason", "Adafactor launch failed"))

    for _ in range(warmup):
        launch()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        launch()
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
    ordered = sorted(samples)
    return {
        "schema": "tessera.rocm.adafactor.benchmark.v2",
        "device": "gfx1151",
        "direction": "backward" if backward else "forward",
        "timing_scope": "operation_total_ms",
        "scope_detail": (
            "allocation+copies+cache_lookup+"
            + ("seven_launches" if backward else "four_launches")
            + "+synchronize+"
            "result_copies"
        ),
        "selector_eligible": False,
        "shape": list(shape),
        "dtype": "f32",
        "warmup": warmup,
        "iterations": iterations,
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "samples_ms": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    payload = record(
        warmup=args.warmup,
        iterations=args.iterations,
        backward=args.backward,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
