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
from tessera.compiler.rocm_native import package_attention_backward  # noqa: E402
from tessera.runtime import (  # noqa: E402
    _submit_rocm_gfx1151_attention_backward_program,
)

PROGRAM_WALL_BASELINE_MS = 0.368203
PROGRAM_WALL_MAX_REGRESSION = 0.10


def _module(
    b: int, hq: int, hkv: int, sq: int, sk: int, d: int
) -> GraphIRModule:
    def tensor(shape: tuple[int, ...], dtype: str) -> IRType:
        element = {"fp16": "f16", "fp32": "f32"}[dtype]
        return IRType(
            f"tensor<{'x'.join(map(str, shape))}x{element}>",
            tuple(map(str, shape)),
            dtype,
        )

    q = tensor((b, hq, sq, d), "fp16")
    key = tensor((b, hkv, sk, d), "fp16")
    value = tensor((b, hkv, sk, d), "fp16")
    do = tensor((b, hq, sq, d), "fp16")
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
                            "dropout_p": 0.0,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b, hq, sq, d = q.shape
    hkv, sk = key.shape[1:3]
    scale = d**-0.5
    dq = np.zeros(q.shape, dtype=np.float32)
    dk = np.zeros(key.shape, dtype=np.float32)
    dv = np.zeros(value.shape, dtype=np.float32)
    for batch in range(b):
        for head in range(hq):
            kv_head = head // (hq // hkv)
            qf = q[batch, head].astype(np.float32)
            kf = key[batch, kv_head].astype(np.float32)
            vf = value[batch, kv_head].astype(np.float32)
            dof = do[batch, head].astype(np.float32)
            raw = scale * (qf @ kf.T) + bias[batch, head]
            tanh = np.tanh(raw / 8.0)
            scores = 8.0 * tanh
            qpos = np.arange(sq)[:, None]
            kpos = np.arange(sk)[None, :]
            valid = (kpos <= qpos) & ((qpos - kpos) <= 8)
            scores = np.where(valid, scores, -np.inf)
            row_max = np.max(scores, axis=1, keepdims=True)
            weights = np.exp(scores - row_max)
            weights = np.where(valid, weights, 0.0)
            weights /= np.sum(weights, axis=1, keepdims=True)
            output = weights @ vf
            dp = dof @ vf.T
            delta = np.sum(output * dof, axis=1, keepdims=True)
            ds = weights * (dp - delta) * (1.0 - tanh * tanh)
            dq[batch, head] = scale * (ds @ kf)
            dk[batch, kv_head] += scale * (ds.T @ qf)
            dv[batch, kv_head] += weights.T @ dof
    return dq, dk, dv


def _record(
    *,
    package_ms: float,
    operation_total_ms: float,
    result: dict[str, Any],
    max_abs_error: dict[str, float],
    image_bytes: int,
) -> dict[str, Any]:
    samples = list(result["kernel_wall_samples_ms"])
    median_ms = statistics.median(samples)
    limit_ms = PROGRAM_WALL_BASELINE_MS * (1.0 + PROGRAM_WALL_MAX_REGRESSION)
    return {
        "schema": "tessera.rocm.attention_backward_program.benchmark.v1",
        "device": os.environ.get("TESSERA_ROCM_CHIP", "gfx1151"),
        "image_bytes": image_bytes,
        "entry_symbols": list(result["entry_symbols"]),
        "workspace_bytes": int(result["workspace_bytes"]),
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
                "baseline_ms": PROGRAM_WALL_BASELINE_MS,
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
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")

    shape = (1, 4, 2, 17, 19, 64)
    rng = np.random.default_rng(20260726)
    q = rng.normal(0.0, 0.25, (shape[0], shape[1], shape[3], shape[5])).astype(np.float16)
    key = rng.normal(0.0, 0.25, (shape[0], shape[2], shape[4], shape[5])).astype(np.float16)
    value = rng.normal(0.0, 0.25, key.shape).astype(np.float16)
    do = rng.normal(0.0, 0.25, q.shape).astype(np.float16)
    bias = rng.normal(0.0, 0.1, (shape[0], shape[1], shape[3], shape[4])).astype(np.float32)
    dq = np.empty(q.shape, dtype=np.float32)
    dk = np.empty(key.shape, dtype=np.float32)
    dv = np.empty(value.shape, dtype=np.float32)

    started = time.perf_counter()
    program = package_attention_backward(
        _module(*shape), pipeline_name="tessera-lower-to-rocm"
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
    reference = _reference(do, q, key, value, bias)
    errors = {
        name: float(np.max(np.abs(actual - expected)))
        for name, actual, expected in zip(
            ("dq", "dk", "dv"), result["outputs"], reference, strict=True
        )
    }
    record = _record(
        package_ms=package_ms,
        operation_total_ms=operation_total_ms,
        result=result,
        max_abs_error=errors,
        image_bytes=len(program.image.payload),
    )
    text = json.dumps(record, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    passes_ratchet = bool(record["timing"]["program_wall"]["passes_ratchet"])
    return 0 if max(errors.values()) < 2.0e-2 and passes_ratchet else 1


if __name__ == "__main__":
    raise SystemExit(main())
