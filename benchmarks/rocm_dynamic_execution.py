#!/usr/bin/env python3
"""gfx1151 dynamic execution packet for reduction, softmax, attention, and KV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from tessera import runtime as rt


def _artifact(path: str, names: list[str], op: str, kwargs: dict):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": path, "executable": True,
        "execution_kind": "native_gpu", "arg_names": names,
        "output_name": "output",
        "ops": [{
            "op_name": op, "result": "output", "operands": names,
            "kwargs": kwargs,
        }],
    })


def _norm_consumer_artifact(norm: str, consumer: str):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm",
        "compiler_path": "rocm_norm_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": ["x"],
        "output_name": "output",
        "ops": [
            {
                "op_name": norm,
                "result": "normalized",
                "operands": ["x"],
                "kwargs": {"eps": 1.0e-5},
            },
            {
                "op_name": consumer,
                "result": "output",
                "operands": ["normalized"],
                "kwargs": {},
            },
        ],
    })


def _measure(call, warmup: int, reps: int) -> tuple[float, float]:
    samples = []
    for iteration in range(warmup + reps):
        start = time.perf_counter_ns()
        result = call()
        if not result.get("ok"):
            raise RuntimeError(str(result))
        elapsed = (time.perf_counter_ns() - start) / 1.0e6
        if iteration >= warmup:
            samples.append(elapsed)
    ordered = sorted(samples)
    return (
        statistics.median(samples),
        ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    )


def _row(name, path, shapes, policy, identity, call, warmup, reps):
    median, p95 = _measure(call, warmup, reps)
    return {
        "family": name,
        "compiler_path": path,
        "runtime_shapes": shapes,
        "timing_domain": "host_wall_operation_total",
        "median_ms": median,
        "p95_ms": p95,
        "cache_identity": identity,
        "specialization_policy": policy,
        "guard_contract": "shared_executable_layout_prelaunch",
    }


def record(warmup: int = 5, reps: int = 30) -> dict:
    rng = np.random.default_rng(20260723)
    reduce_x = rng.standard_normal((129, 511)).astype(np.float32)
    reduce_art = _artifact(
        "rocm_reduce_compiled", ["x"], "tessera.sum", {"axis": -1}
    )
    softmax_x = rng.standard_normal((129, 511)).astype(np.float32)
    softmax_art = _artifact(
        "rocm_softmax_compiled", ["x"], "tessera.softmax", {"axis": -1}
    )
    q = (0.2 * rng.standard_normal((1, 2, 33, 64))).astype(np.float16)
    k = (0.2 * rng.standard_normal((1, 2, 79, 64))).astype(np.float16)
    v = (0.2 * rng.standard_normal((1, 2, 79, 64))).astype(np.float16)
    attention_art = _artifact(
        "rocm_flash_attn_compiled", ["q", "k", "v"],
        "tessera.flash_attn",
        {"causal": False, "scale": 0.125},
    )
    cache = np.zeros((256, 2, 64), np.float32)
    rows = rng.standard_normal((17, 2, 64)).astype(np.float32)
    kv_art = _artifact(
        "rocm_kv_cache_compiled", ["cache", "rows"],
        "tessera.kv_cache.append", {"start": 113},
    )
    norm_x = rng.standard_normal((7, 300)).astype(np.float32)
    rms_relu_art = _norm_consumer_artifact("tessera.rmsnorm", "tessera.relu")
    layer_silu_art = _norm_consumer_artifact(
        "tessera.layer_norm", "tessera.silu"
    )

    calls = [
        (
            "last_axis_reduction", "rocm_reduce_compiled",
            {"input": list(reduce_x.shape)},
            "fully_dynamic", "chip,kind,dtype",
            lambda: rt.launch(reduce_art, (reduce_x,)),
        ),
        (
            "softmax", "rocm_softmax_compiled",
            {"input": list(softmax_x.shape)},
            "fully_dynamic", "chip,dtype",
            lambda: rt.launch(softmax_art, (softmax_x,)),
        ),
        (
            "attention", "rocm_flash_attn_compiled",
            {"q": list(q.shape), "k": list(k.shape), "v": list(v.shape)},
            "dynamic_sequence_bucket_head_dim_dtype_features",
            "chip,head_dim,dtype,gqa,window,softcap,bias,two_wave",
            lambda: rt.launch(attention_art, (q, k, v)),
        ),
        (
            "growing_kv_cache", "rocm_kv_cache_compiled",
            {"cache": list(cache.shape), "rows": list(rows.shape), "start": 113},
            "fully_dynamic_capacity_and_logical_length",
            "component_gather_scatter_chip_dtype",
            lambda: rt.launch(kv_art, (cache, rows)),
        ),
        (
            "rmsnorm_relu_consumer_fusion", "rocm_norm_compiled",
            {"input": list(norm_x.shape)},
            "fully_dynamic_single_kernel", "chip,kind,dtype,epilogue",
            lambda: rt.launch(rms_relu_art, (norm_x,)),
        ),
        (
            "layer_norm_silu_consumer_fusion", "rocm_norm_compiled",
            {"input": list(norm_x.shape)},
            "fully_dynamic_single_kernel", "chip,kind,dtype,epilogue",
            lambda: rt.launch(layer_silu_art, (norm_x,)),
        ),
    ]
    return {
        "schema": "tessera.rocm_dynamic_execution.v1",
        "target": "gfx1151",
        "rows": [
            _row(*entry, warmup, reps) for entry in calls
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = record(args.warmup, args.reps)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
