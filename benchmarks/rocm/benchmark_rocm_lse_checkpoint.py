"""Paired gfx1151 saved-LSE versus backward-recompute benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from benchmark_rocm_attention_backward_program import _module, _reference  # noqa: E402
from tessera.compiler.rocm_native import package_attention_backward  # noqa: E402
from tessera.runtime import (  # noqa: E402
    _submit_rocm_gfx1151_attention_backward_program,
)


def _run_case(
    *,
    sq: int,
    sk: int,
    dtype: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    b, hq, hkv, d = 1, 4, 2, 64
    shape = (b, hq, hkv, sq, sk, d)
    rng = np.random.default_rng(20260727 + sq * 17 + sk)
    if dtype == "bf16":
        import ml_dtypes

        storage = ml_dtypes.bfloat16
    else:
        storage = np.float16
    q = rng.normal(0.0, 0.25, (b, hq, sq, d)).astype(storage)
    key = rng.normal(0.0, 0.25, (b, hkv, sk, d)).astype(storage)
    value = rng.normal(0.0, 0.25, key.shape).astype(storage)
    do = rng.normal(0.0, 0.25, q.shape).astype(storage)
    bias = rng.normal(0.0, 0.1, (b, hq, sq, sk)).astype(np.float32)
    reference = _reference(do, q, key, value, bias)
    modes: dict[str, Any] = {}
    for checkpoint in ("recompute", "saved"):
        outputs = (
            np.empty(q.shape, dtype=np.float32),
            np.empty(key.shape, dtype=np.float32),
            np.empty(value.shape, dtype=np.float32),
        )
        program = package_attention_backward(
            _module(*shape, dtype=dtype, lse_checkpoint=checkpoint),
            pipeline_name="tessera-lower-to-rocm",
        )
        result = _submit_rocm_gfx1151_attention_backward_program(
            program,
            {
                "do": do,
                "q": q,
                "key": key,
                "v": value,
                "bias": bias,
                "dq": outputs[0],
                "dk": outputs[1],
                "dv": outputs[2],
            },
            warmup=warmup,
            iterations=iterations,
        )
        modes[checkpoint] = {
            "median_ms": statistics.median(result["kernel_wall_samples_ms"]),
            "samples_ms": result["kernel_wall_samples_ms"],
            "image_bytes": len(program.image.payload),
            "workspace_bytes": result["workspace_bytes"],
            "max_abs_error": {
                name: float(np.max(np.abs(actual - expected)))
                for name, actual, expected in zip(("dq", "dk", "dv"), result["outputs"], reference, strict=True)
            },
        }
    recompute_ms = float(modes["recompute"]["median_ms"])
    saved_ms = float(modes["saved"]["median_ms"])
    return {
        "shape": list(shape),
        "storage": dtype,
        "modes": modes,
        "saved_speedup": recompute_ms / saved_ms,
        "saved_delta_percent": (saved_ms / recompute_ms - 1.0) * 100.0,
        "winner": "saved" if saved_ms < recompute_ms else "recompute",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--lengths", type=int, nargs="+", default=(17, 64, 128, 256))
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    if any(length <= 0 for length in args.lengths):
        parser.error("all lengths must be positive")

    cases = [
        _run_case(
            sq=length,
            sk=length + 2 if length == 17 else length,
            dtype=args.dtype,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        for length in args.lengths
    ]
    record = {
        "schema": "tessera.rocm.lse_checkpoint.benchmark.v1",
        "device": "gfx1151",
        "clock": "time.perf_counter_ns",
        "completion_api": "hipDeviceSynchronize",
        "resident_module": True,
        "resident_buffers": True,
        "cases": cases,
    }
    text = json.dumps(record, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    max_error = max(
        value for case in cases for mode in case["modes"].values() for value in mode["max_abs_error"].values()
    )
    return 0 if max_error < 2.0e-2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
