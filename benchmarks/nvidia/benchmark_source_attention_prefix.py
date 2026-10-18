"""Correctness-gated native RMSNorm -> saved-LSE attention dispatch windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import ml_dtypes
import numpy as np
import tessera as ts

from tessera.compiler.native_source_program import outline_source_program


def source(q, k, v):
    normalized = ts.ops.rmsnorm(q, eps=1e-5, numeric_policy={"accum": "fp32"})
    return ts.ops.flash_attn(normalized, k, v, causal=False, lse_checkpoint="saved")


def run(storage, shape, samples):
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    b, hq, hkv, sq, sk, d, dv = shape
    rng = np.random.default_rng(3318)
    roots = tuple(rng.normal(size=s).astype(dtype) for s in
                  ((b, hq, sq, d), (b, hkv, sk, d), (b, hkv, sk, dv)))
    q, k, v = roots
    qf = q.astype(np.float32)
    normalized = (qf / np.sqrt(np.mean(qf * qf, axis=-1, keepdims=True)
                              + np.float32(1e-5))).astype(dtype)
    scores = normalized.astype(np.float64) @ np.repeat(
        k.astype(np.float64), hq // hkv, axis=1).swapaxes(-1, -2)
    scores *= d ** -0.5
    maximum = scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores - maximum)
    total = weights.sum(axis=-1, keepdims=True)
    expected_o = weights / total @ np.repeat(v.astype(np.float64), hq // hkv, axis=1)
    expected_lse = maximum[..., 0] + np.log(total[..., 0])
    started = time.perf_counter()
    fn = ts.jit(target="nvidia_sm120")(source)
    graph = fn._traced_autodiff_module(roots, {})
    graph.module_attrs.update({"tessera.target": '"nvidia_sm120"', "tessera.arch": '"sm_120"'})
    program = outline_source_program(graph.to_mlir(target="nvidia_sm120", canonical=True))
    outlined_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    frame = program.prepare()
    package_ms = (time.perf_counter() - started) * 1000
    with frame:
        (o, lse), cold = frame(*roots)
        np.testing.assert_allclose(o.astype(np.float32), expected_o,
                                   rtol=0.02 if storage == "bf16" else 0.003,
                                   atol=0.01 if storage == "bf16" else 0.001)
        np.testing.assert_allclose(lse, expected_lse, rtol=2e-5, atol=2e-5)
        for _ in range(5):
            frame(*roots)
        windows = {key: [] for key in
                   ("norm_device_ms", "attention_device_ms", "program_device_ms", "completed_host_ms")}
        for _ in range(samples):
            (_, _), receipt = frame(*roots)
            for key in windows:
                windows[key].append(receipt[key])
        (last_o, last_lse), _ = frame(*roots)
        np.testing.assert_array_equal(last_o, o)
        np.testing.assert_array_equal(last_lse, lse)
        public = ts.jit(target="nvidia_sm120")(source)
        try:
            public_o, public_lse = public(*roots)
            if public.execution_kind != "native_gpu":
                raise RuntimeError("public source benchmark did not select native execution")
            np.testing.assert_array_equal(public_o, o)
            np.testing.assert_array_equal(public_lse, lse)
            for _ in range(5):
                public(*roots)
            public_samples = []
            for _ in range(samples):
                start = time.perf_counter()
                public(*roots)
                public_samples.append((time.perf_counter() - start) * 1000)
            public_images = public._native_source_attention_last_receipt["member_images"]
        finally:
            public.close_native_storage()
        return {
            "public_completed_host_samples_ms": public_samples,
            "public_completed_host_median_ms": statistics.median(public_samples),
            "public_member_images": public_images,
            "storage": storage, "shape": list(shape),
            "route": "frontend->Graph->native SSA members->Schedule->Tile->NVIDIA->NVVM->PTX",
            "correctness": "passed_before_and_after_timing",
            "o_max_abs_error": float(np.max(np.abs(o.astype(np.float64) - expected_o))),
            "lse_max_abs_error": float(np.max(np.abs(lse - expected_lse))),
            "source_hash": program.manifest["source_hash"],
            "member_images": cold["member_images"], "cooperative_norm": frame.cooperative,
            "frontend_outline_ms": outlined_ms, "member_packaging_ms": package_ms,
            "cold_completed_host_ms": cold["completed_host_ms"],
            "samples": windows,
            "medians_ms": {key: statistics.median(values) for key, values in windows.items()},
            "private_intermediate": cold["private_intermediate"],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 3:
        parser.error("at least three timing samples are required")
    device = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
         "--format=csv,noheader"], text=True).strip().splitlines()[0]
    if device.split(",")[1].strip() != "12.0":
        raise RuntimeError("requires owning SM120 GPU")
    files = (
        "python/tessera/compiler/native_source_program.py",
        "python/tessera/compiler/prepared_source_attention.py",
        "python/tessera/compiler/portable_source_attention.py",
        "python/tessera/compiler/source_attention_runtime.py",
        "python/tessera/compiler/jit.py",
        "python/tessera/runtime.py",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp",
        "benchmarks/nvidia/benchmark_source_attention_prefix.py",
    )
    record = {
        "device": device,
        "timing_scope": "native event dispatch windows; enqueue gaps possible; host staging excluded",
        "output_transport": "independent completed host O and LSE; normalized Q stays device-private",
        "source_hashes": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                          for path in files},
        "tool_hashes": {key: hashlib.sha256(Path(os.environ[key]).read_bytes()).hexdigest()
                        for key in ("TESSERA_OPT", "TESSERA_NVIDIA_PTX_LAUNCH_LIB")},
        "rows": [run(storage, shape, args.samples) for storage in ("f16", "bf16")
                 for shape in ((1, 2, 1, 3, 5, 4, 3), (1, 2, 1, 17, 19, 257, 13))],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    for row in record["rows"]:
        print(row["storage"], row["shape"], row["medians_ms"], flush=True)


if __name__ == "__main__":
    main()
