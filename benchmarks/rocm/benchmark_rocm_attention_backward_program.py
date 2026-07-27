"""Resident gfx1151 multi-entry attention-backward benchmark.

The compiler packages forward recompute, prepass, split dK/dV, reduction, and
dQ into one HSACO. The runtime loads that image and allocates/copies user
buffers once, warms up the ordered program, then records synchronized host-wall
samples around the five-kernel launch sequence.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from tessera.compiler.graph_ir import (  # noqa: E402
    GraphIRFunction,
    GraphIRModule,
    IRArg,
    IROp,
    IRType,
)
from tessera.compiler.attention_contract import (  # noqa: E402
    reference_attention_backward_split_reduced,
)
from tessera.compiler.rocm_native import package_attention_backward  # noqa: E402
from tessera.runtime import (  # noqa: E402
    _submit_rocm_gfx1151_attention_backward_program,
)

PROGRAM_WALL_BASELINE_MS = 0.368203
PROGRAM_WALL_BASELINE_MS_BY_DTYPE = {
    "fp16": PROGRAM_WALL_BASELINE_MS,
    "bf16": 0.362481,
}
PROGRAM_WALL_MAX_REGRESSION = 0.10


def _module(
    b: int,
    hq: int,
    hkv: int,
    sq: int,
    sk: int,
    d: int,
    *,
    dtype: str = "fp16",
    dropout_p: float = 0.0,
    dropout_seed: int = 37,
    lse_checkpoint: str = "auto",
) -> GraphIRModule:
    def tensor(shape: tuple[int, ...], dtype: str) -> IRType:
        element = {"fp16": "f16", "bf16": "bf16", "fp32": "f32"}[dtype]
        return IRType(
            f"tensor<{'x'.join(map(str, shape))}x{element}>",
            tuple(map(str, shape)),
            dtype,
        )

    q = tensor((b, hq, sq, d), dtype)
    key = tensor((b, hkv, sk, d), dtype)
    value = tensor((b, hkv, sk, d), dtype)
    do = tensor((b, hq, sq, d), dtype)
    bias = tensor((b, hq, sq, sk), "fp32")
    dq = tensor((b, hq, sq, d), "fp32")
    dk = tensor((b, hkv, sk, d), "fp32")
    dv = tensor((b, hkv, sk, d), "fp32")
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="gfx1151_attention_backward_program",
                args=[
                    IRArg("do", do),
                    IRArg("q", q),
                    IRArg("key", key),
                    IRArg("v", value),
                    IRArg("bias", bias),
                ],
                result_types=[dq, dk, dv],
                body=[
                    IROp(
                        result="dq,dk,dv",
                        op_name="tessera.flash_attn_bwd",
                        operands=["%do", "%q", "%key", "%v", "%bias"],
                        operand_types=[
                            str(do),
                            str(q),
                            str(key),
                            str(value),
                            str(bias),
                        ],
                        kwargs={
                            "scale": d**-0.5,
                            "causal": True,
                            "window": (8, 0),
                            "softcap": 8.0,
                            "dropout_p": dropout_p,
                            "dropout_seed": dropout_seed,
                            "lse_checkpoint": lse_checkpoint,
                            "route": "deterministic_direct",
                            "deterministic": True,
                        },
                    )
                ],
                return_values=["%dq", "%dk", "%dv"],
            )
        ]
    )


def _reference(
    do: np.ndarray,
    q: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    bias: np.ndarray,
    *,
    dropout_p: float = 0.0,
    dropout_seed: int = 37,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return reference_attention_backward_split_reduced(
        do,
        q,
        key,
        value,
        split_count=2,
        scale=q.shape[-1] ** -0.5,
        bias=bias,
        causal=True,
        window_left=8,
        window_right=0,
        softcap=8.0,
        dropout_p=dropout_p,
        dropout_seed=dropout_seed,
    )


def _record(
    *,
    package_ms: float,
    operation_total_ms: float,
    result: dict[str, Any],
    max_abs_error: dict[str, float],
    image_bytes: int,
    dtype: str,
    dropout_p: float,
    dropout_seed: int,
    lse_checkpoint: str = "auto",
) -> dict[str, Any]:
    samples = list(result["kernel_wall_samples_ms"])
    median_ms = statistics.median(samples)
    baseline_ms = PROGRAM_WALL_BASELINE_MS_BY_DTYPE[dtype]
    limit_ms = baseline_ms * (1.0 + PROGRAM_WALL_MAX_REGRESSION)
    return {
        "schema": "tessera.rocm.attention_backward_program.benchmark.v1",
        "device": os.environ.get("TESSERA_ROCM_CHIP", "gfx1151"),
        "storage": dtype,
        "image_bytes": image_bytes,
        "entry_symbols": list(result["entry_symbols"]),
        "workspace_bytes": int(result["workspace_bytes"]),
        "lse_checkpoint": lse_checkpoint,
        "dropout": {
            "probability": dropout_p,
            "seed": dropout_seed,
            "counter": "lcg32_counter_v1",
            "replay": "forward_backward_identical",
        },
        "max_abs_error": max_abs_error,
        "timing": {
            "compiler_package_ms": package_ms,
            "operation_total_ms": operation_total_ms,
            "program_wall": {
                "clock": "time.perf_counter_ns",
                "median_ms": median_ms,
                "samples_ms": samples,
                "resident_module": True,
                "resident_buffers": True,
                "launch_count_per_sample": 5,
                "completion_api": "hipDeviceSynchronize",
                "selector_eligible": False,
                "baseline_ms": baseline_ms,
                "max_regression": PROGRAM_WALL_MAX_REGRESSION,
                "limit_ms": limit_ms,
                "passes_ratchet": median_ms <= limit_ms,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--output")
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--dropout-seed", type=int, default=37)
    parser.add_argument(
        "--lse-checkpoint",
        choices=("auto", "saved", "recompute"),
        default="auto",
    )
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if not 0.0 <= args.dropout_p < 1.0:
        parser.error("--dropout-p must satisfy 0 <= p < 1")

    shape = (1, 4, 2, 17, 19, 64)
    rng = np.random.default_rng(20260726)
    if args.dtype == "bf16":
        import ml_dtypes

        storage_dtype = ml_dtypes.bfloat16
    else:
        storage_dtype = np.float16
    q = rng.normal(0.0, 0.25, (shape[0], shape[1], shape[3], shape[5])).astype(storage_dtype)
    key = rng.normal(0.0, 0.25, (shape[0], shape[2], shape[4], shape[5])).astype(storage_dtype)
    value = rng.normal(0.0, 0.25, key.shape).astype(storage_dtype)
    do = rng.normal(0.0, 0.25, q.shape).astype(storage_dtype)
    bias = rng.normal(0.0, 0.1, (shape[0], shape[1], shape[3], shape[4])).astype(np.float32)
    dq = np.empty(q.shape, dtype=np.float32)
    dk = np.empty(key.shape, dtype=np.float32)
    dv = np.empty(value.shape, dtype=np.float32)

    started = time.perf_counter()
    program = package_attention_backward(
        _module(
            *shape,
            dtype=args.dtype,
            dropout_p=args.dropout_p,
            dropout_seed=args.dropout_seed,
            lse_checkpoint=args.lse_checkpoint,
        ),
        pipeline_name="tessera-lower-to-rocm",
    )
    package_ms = (time.perf_counter() - started) * 1_000.0
    started = time.perf_counter()
    result = _submit_rocm_gfx1151_attention_backward_program(
        program,
        {
            "do": do,
            "q": q,
            "key": key,
            "v": value,
            "bias": bias,
            "dq": dq,
            "dk": dk,
            "dv": dv,
        },
        warmup=args.warmup,
        iterations=args.iterations,
    )
    operation_total_ms = (time.perf_counter() - started) * 1_000.0
    reference = _reference(
        do,
        q,
        key,
        value,
        bias,
        dropout_p=args.dropout_p,
        dropout_seed=args.dropout_seed,
    )
    errors = {
        name: float(np.max(np.abs(actual - expected)))
        for name, actual, expected in zip(("dq", "dk", "dv"), result["outputs"], reference, strict=True)
    }
    record = _record(
        package_ms=package_ms,
        operation_total_ms=operation_total_ms,
        result=result,
        max_abs_error=errors,
        image_bytes=len(program.image.payload),
        dtype=args.dtype,
        dropout_p=args.dropout_p,
        dropout_seed=args.dropout_seed,
        lse_checkpoint=args.lse_checkpoint,
    )
    text = json.dumps(record, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    passes_ratchet = bool(record["timing"]["program_wall"]["passes_ratchet"])
    return 0 if max(errors.values()) < 2.0e-2 and passes_ratchet else 1


if __name__ == "__main__":
    raise SystemExit(main())
