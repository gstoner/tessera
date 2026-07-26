"""Exact-device ROCm attention-carrier correctness and operation-total timing.

This benchmark enters through ``tile.attention_kernel`` and the native package
path, not through a handwritten directive. It intentionally uses host-wall
operation-total timing because WSL HIP events may report zero-duration samples.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

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
from tessera.compiler.rocm_native import package_attention  # noqa: E402
from tessera.runtime import _submit_rocm_gfx1151_native  # noqa: E402


def _module(b: int, hq: int, hkv: int, sq: int, sk: int, d: int) -> GraphIRModule:
    q = IRType(f"tensor<{b}x{hq}x{sq}x{d}xf16>", tuple(map(str, (b, hq, sq, d))), "fp16")
    k = IRType(f"tensor<{b}x{hkv}x{sk}x{d}xf16>", tuple(map(str, (b, hkv, sk, d))), "fp16")
    v = IRType(f"tensor<{b}x{hkv}x{sk}x{d}xf16>", tuple(map(str, (b, hkv, sk, d))), "fp16")
    o = IRType(f"tensor<{b}x{hq}x{sq}x{d}xf32>", tuple(map(str, (b, hq, sq, d))), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="gfx1151_attention_carrier",
        args=[IRArg("q", q), IRArg("k", k), IRArg("v", v)],
        result_types=[o],
        body=[IROp(
            result="o",
            op_name="tessera.flash_attn",
            operands=["%q", "%k", "%v"],
            operand_types=[str(q), str(k), str(v)],
            result_type=str(o),
            kwargs={
                "scale": d ** -0.5,
                "causal": True,
                "window": (8, 0),
                "softcap": 8.0,
                "dropout_p": 0.0,
            },
        )],
        return_values=["%o"],
    )])


def _reference(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    b, hq, sq, d = q.shape
    hkv, sk = k.shape[1:3]
    result = np.empty((b, hq, sq, d), dtype=np.float32)
    scale = d ** -0.5
    for batch in range(b):
        for head in range(hq):
            kv_head = head // (hq // hkv)
            scores = (
                q[batch, head].astype(np.float32)
                @ k[batch, kv_head].astype(np.float32).T
            ) * scale
            scores = 8.0 * np.tanh(scores / 8.0)
            qpos = np.arange(sq)[:, None]
            kpos = np.arange(sk)[None, :]
            valid = (kpos <= qpos) & ((qpos - kpos) <= 8)
            scores = np.where(valid, scores, -np.inf)
            row_max = np.max(scores, axis=1, keepdims=True)
            weights = np.exp(scores - row_max)
            weights = np.where(valid, weights, 0.0)
            weights /= np.sum(weights, axis=1, keepdims=True)
            result[batch, head] = weights @ v[batch, kv_head].astype(np.float32)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--output")
    args = parser.parse_args()
    shape = (1, 4, 2, 17, 19, 64)
    rng = np.random.default_rng(20260726)
    q = rng.normal(0.0, 0.25, (shape[0], shape[1], shape[3], shape[5])).astype(np.float16)
    k = rng.normal(0.0, 0.25, (shape[0], shape[2], shape[4], shape[5])).astype(np.float16)
    v = rng.normal(0.0, 0.25, (shape[0], shape[2], shape[4], shape[5])).astype(np.float16)
    output = np.empty((shape[0], shape[1], shape[3], shape[5]), dtype=np.float32)

    started = time.perf_counter()
    package = package_attention(
        _module(*shape), pipeline_name="tessera-lower-to-rocm"
    )
    compile_ms = (time.perf_counter() - started) * 1000.0
    buffers = {"q": q, "k": k, "v": v, "o": output}
    scalars = {
        "Sq": shape[3],
        "Sk": shape[4],
        "Scale": shape[5] ** -0.5,
        "Causal": 1,
        "Hq": shape[1],
        "KvRatio": shape[1] // shape[2],
        "Window": 8,
        "Softcap": 8.0,
    }
    started = time.perf_counter()
    _submit_rocm_gfx1151_native(
        package.image, package.descriptor, buffers, scalars, None
    )
    cold_operation_total_ms = (time.perf_counter() - started) * 1000.0
    samples = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        _submit_rocm_gfx1151_native(
            package.image, package.descriptor, buffers, scalars, None
        )
        samples.append((time.perf_counter() - started) * 1000.0)
    reference = _reference(q, k, v)
    error = float(np.max(np.abs(output - reference)))
    record = {
        "schema": "tessera.rocm.attention_carrier.benchmark.v1",
        "device": os.environ.get("TESSERA_ROCM_CHIP", "gfx1151"),
        "shape": {
            "B": shape[0], "Hq": shape[1], "Hkv": shape[2],
            "Sq": shape[3], "Sk": shape[4], "D": shape[5],
        },
        "carrier": "tile.attention_kernel",
        "schedule": package.descriptor.provenance["schedule"],
        "compile_ms": compile_ms,
        "cold_operation_total_ms": cold_operation_total_ms,
        "operation_total_median_ms": statistics.median(samples),
        "operation_total_samples_ms": samples,
        "max_abs_error": error,
        "tolerance": 0.035,
        "passed": error <= 0.035,
        "hsaco_bytes": len(package.image.payload),
        "image_digest": package.image.image_digest,
    }
    text = json.dumps(record, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
