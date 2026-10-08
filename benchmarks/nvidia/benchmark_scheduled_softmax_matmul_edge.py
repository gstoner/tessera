"""Correctness-gated SM120 scheduled softmax -> matmul resident-edge benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from tessera import runtime as rt
from tessera.compiler import nvidia_native
from tessera.compiler.from_text import from_text


def _median(xs):
    return statistics.median(xs)


def _cv(xs):
    mean = statistics.fmean(xs)
    return statistics.pstdev(xs) / mean if mean else 0.0


def _timed(package, args, stream, *, warmup, reps):
    return rt._nvidia_native_descriptor_resident_device_latency(
        package.image, package.descriptor, args,
        stream=stream, warmup=warmup, reps=reps,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--dynamic-k-bound", type=int)
    parser.add_argument("--active-k", type=int)
    args = parser.parse_args()
    if args.active_k is not None and args.dynamic_k_bound is None:
        parser.error("--active-k requires --dynamic-k-bound")
    if args.dynamic_k_bound is not None and args.dynamic_k_bound <= 0:
        parser.error("--dynamic-k-bound must be positive")
    trace_k = args.dynamic_k_bound or 64
    active_k = args.active_k or trace_k
    if active_k <= 0 or active_k > trace_k:
        parser.error("--active-k must be in [1, --dynamic-k-bound]")

    if rt._nvidia_device_name() != "sm_120":
        raise SystemExit("requires exact NVIDIA sm_120 device")
    producer_jit = from_text("""
        def softmax_frontend(x):
            return ts.ops.softmax(x, axis=-1)
    """)
    consumer_jit = from_text("""
        def matmul_frontend(edge, weights):
            return ts.ops.matmul(edge, weights, output_dtype="fp32")
    """)
    rng = np.random.default_rng(0x5A17)
    trace_source = np.zeros((16, trace_k), dtype=np.float16)
    trace_weights = np.asfortranarray(np.zeros((trace_k, 8), dtype=np.float16))
    source = np.ascontiguousarray(rng.normal(0, 0.25, (16, active_k)).astype(np.float16))
    weights = np.asfortranarray(rng.normal(0, 0.25, (active_k, 8)).astype(np.float16))
    producer_jit(trace_source)
    consumer_jit(trace_source, trace_weights)
    package_options = {"pipeline_name": "tessera-lower-to-nvidia-sm120"}
    if args.dynamic_k_bound is not None:
        package_options["dynamic_k_bound"] = args.dynamic_k_bound
    program = nvidia_native.package_scheduled_tensor_matmul(
        producer_jit.graph_ir, consumer_jit.graph_ir, **package_options,
    )
    program.validate()
    start = time.perf_counter()
    resident = program.execute_resident(source, weights)
    resident_execute_ms = (time.perf_counter() - start) * 1e3
    try:
        edge = resident.intermediate.numpy()
        output = resident.output.numpy()
        x32 = source.astype(np.float32)
        ex = np.exp(x32 - np.max(x32, axis=-1, keepdims=True))
        expected_edge = (ex / np.sum(ex, axis=-1, keepdims=True)).astype(np.float16)
        expected_output = expected_edge.astype(np.float32) @ weights.astype(np.float32)
        edge_error = float(np.max(np.abs(edge.astype(np.float32) - expected_edge.astype(np.float32))))
        output_error = float(np.max(np.abs(output - expected_output)))
        if edge_error > 2e-3 or output_error > 2e-3:
            raise RuntimeError(f"correctness failed: softmax={edge_error}, matmul={output_error}")
        if resident.producer_receipt.get("execution_kind") != "native_gpu" or resident.consumer_receipt.get("execution_kind") != "native_gpu":
            raise RuntimeError("both packages must report native_gpu execution")
        if resident.intermediate.ptr in {resident.output.ptr, resident.device_session._buffers[0].ptr, resident.device_session._buffers[1].ptr}:
            raise RuntimeError("resident edge allocation aliases a live input/output buffer")

        producer_values = {"Rows": 16, "K": active_k, "Columns": active_k}
        producer_args = {
            program.producer_input_name: resident.device_session._buffers[0],
            program.intermediate_name: resident.intermediate,
        }
        for scalar in program.producer.descriptor.scalars:
            producer_args[scalar.name] = producer_values[scalar.name]
        consumer_values = {"M": 16, "N": 8, "K": active_k,
                           "LDA": active_k, "LDB": active_k, "LDD": 8}
        consumer_edge = (
            resident.intermediate.view(
                0, (16, active_k), resident.intermediate.dtype, layout="strided"
            )
            if program.dynamic_k else resident.intermediate
        )
        consumer_args = {
            program.consumer_input_name: consumer_edge,
            program.consumer_rhs_name: resident.device_session._buffers[1],
            program.output_name: resident.output,
        }
        for scalar in program.consumer.descriptor.scalars:
            consumer_args[scalar.name] = consumer_values[scalar.name]

        producer_ms = [_timed(program.producer, producer_args, resident.device_session.stream,
                              warmup=args.warmup, reps=args.reps) for _ in range(args.samples)]
        consumer_ms = [_timed(program.consumer, consumer_args, resident.device_session.stream,
                              warmup=args.warmup, reps=args.reps) for _ in range(args.samples)]
        packet = {
            "schema": "tessera.nvidia.scheduled-tensor-matmul-edge.v1",
            "owner": "W1.1",
            "target": "nvidia_sm120",
            "architecture": program.consumer.image.architecture,
            "device": subprocess.run(
                ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True, check=True,
            ).stdout.strip(),
            "host": {"node": platform.node(), "platform": platform.platform(),
                     "wsl": "microsoft" in platform.release().lower()},
            "source_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                text=True, check=True,
            ).stdout.strip(),
            "worktree_dirty": bool(subprocess.run(
                ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                text=True, check=True,
            ).stdout.strip()),
            "edge": {"producer": "tessera.softmax", "consumer": "tessera.matmul",
                     "shape_mkn": [16, 8, active_k], "storage": "fp16",
                     "dynamic_k_bound": args.dynamic_k_bound,
                     "same_stream": True, "same_resident_intermediate": True},
            "packages": {"producer_image": program.producer.image.image_digest,
                         "consumer_image": program.consumer.image.image_digest,
                         "producer_abi": program.producer.descriptor.abi_id,
                         "consumer_abi": program.consumer.descriptor.abi_id,
                         "producer_schedule": program.producer.descriptor.provenance["schedule_digest"],
                         "consumer_schedule": program.consumer.descriptor.provenance["schedule_digest"],
                         "consumer_tile": program.consumer.descriptor.provenance["tile_ir_digest"]},
            "correctness": {"producer_max_abs_error": edge_error,
                            "consumer_max_abs_error": output_error,
                            "producer_execution_kind": resident.producer_receipt["execution_kind"],
                            "consumer_execution_kind": resident.consumer_receipt["execution_kind"],
                            "same_intermediate_allocation": True},
            "timing": {"domain": "resident CUDA events; producer and consumer independently timed",
                       "resident_execute_ms_including_upload_launch_sync": resident_execute_ms,
                       "samples": args.samples, "warmup": args.warmup, "reps": args.reps,
                       "producer_ms": producer_ms, "producer_median_ms": _median(producer_ms),
                       "producer_cv": _cv(producer_ms),
                       "consumer_ms": consumer_ms, "consumer_median_ms": _median(consumer_ms),
                       "consumer_cv": _cv(consumer_ms)},
            "promotion": "none; one named fp16 SM120 producer envelope",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
        print(json.dumps(packet, indent=2, sort_keys=True))
    finally:
        resident.close()


if __name__ == "__main__":
    main()
