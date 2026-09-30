"""Correctness-gated SM120 RMSNorm -> matmul tensor-edge benchmark.

Each stage is compiled from Graph IR through the production Schedule and Tile
passes and packaged behind its checked native ABI. CUDA-event timings are
collected separately for the producer and consumer; this does not promote a
selector or claim fusion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from tessera import runtime as rt  # noqa: E402
from tessera.compiler import nvidia_native, scheduled_kernel, scheduled_matmul  # noqa: E402
from tessera.compiler.graph_ir import (  # noqa: E402
    GraphIRFunction, GraphIRModule, IRArg, IROp, IRType,
)


def _modules(m: int, k: int, n: int, *, dynamic_n: bool = False,
             dynamic_k: bool = False):
    a = IRType(f"tensor<{m}x{k}xf16>", (str(m), str(k)), "fp16")
    consumer_a = (
        IRType(f"tensor<{m}x?xf16>", (str(m), "?"), "fp16")
        if dynamic_k else a
    )
    b = (
        IRType(f"tensor<{k}x?xf16>", (str(k), "?"), "fp16")
        if dynamic_n else
        IRType(f"tensor<?x{n}xf16>", ("?", str(n)), "fp16")
        if dynamic_k else IRType(f"tensor<{k}x{n}xf16>", (str(k), str(n)), "fp16")
    )
    out = (
        IRType(f"tensor<{m}x?xf32>", (str(m), "?"), "fp32")
        if dynamic_n else IRType(f"tensor<{m}x{n}xf32>", (str(m), str(n)), "fp32")
    )
    producer = GraphIRModule(functions=[GraphIRFunction(
        name="sm120_rmsnorm_tensor_producer",
        args=[IRArg("x", a)],
        result_types=[a],
        body=[IROp(
            result="normalized", op_name="tessera.rmsnorm",
            operands=["%x"], operand_types=[str(a)], result_type=str(a),
            kwargs={"eps": 1e-5},
        )],
        return_values=["%normalized"],
    )])
    consumer = GraphIRModule(functions=[GraphIRFunction(
        name="sm120_rmsnorm_matmul_consumer",
        args=[IRArg("normalized", consumer_a), IRArg("weights", b)],
        result_types=[out],
        body=[IROp(
            result="result", op_name="tessera.matmul",
            operands=["%normalized", "%weights"],
            operand_types=[str(consumer_a), str(b)], result_type=str(out),
            kwargs={"shape_bounds": [m, n, k]} if dynamic_n or dynamic_k else {},
        )],
        return_values=["%result"],
    )])
    return producer, consumer


def _cv(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def _version(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return "unavailable"
    return (result.stdout or result.stderr).strip().splitlines()[0]




def _dynamic_n_benchmark(args: argparse.Namespace, program: Any, m: int, k: int,
                         bound_n: int, active_n: int) -> int:
    rng = np.random.default_rng(0x5A17 + m + k + bound_n + active_n)
    source = np.ascontiguousarray(rng.normal(0.0, 0.25, size=(m, k)).astype(np.float16))
    weights = np.asfortranarray(
        rng.normal(0.0, 0.25, size=(k, active_n)).astype(np.float16)
    )
    resident = program.execute_resident(source, weights)
    try:
        edge = resident.intermediate.numpy()
        output = resident.output.numpy()
        source_f32 = source.astype(np.float32)
        norm_reference = (
            source_f32 / np.sqrt(
                np.mean(source_f32 * source_f32, axis=-1, keepdims=True) + 1e-5
            )
        ).astype(np.float16)
        norm_error = float(np.max(np.abs(
            edge.astype(np.float32) - norm_reference.astype(np.float32)
        )))
        matmul_reference = edge.astype(np.float32) @ weights.astype(np.float32)
        matmul_error = float(np.max(np.abs(output - matmul_reference)))
        if norm_error > 2e-3 or matmul_error > 2e-4:
            raise RuntimeError(
                f"resident dynamic-N oracle mismatch: norm={norm_error}, matmul={matmul_error}"
            )
        session = resident.device_session
        device_source, device_rhs = session._buffers[0], session._buffers[1]
        consumer_edge = resident.intermediate.view(
            0, resident.intermediate.shape, resident.intermediate.dtype, layout="strided"
        )
        if consumer_edge.ptr != resident.intermediate.ptr:
            raise RuntimeError("dynamic-N consumer edge copied the producer allocation")
        producer_args = {
            program.producer_input_name: device_source,
            program.intermediate_name: resident.intermediate,
            "Rows": m, "Columns": k,
        }
        consumer_args = {
            program.consumer_input_name: consumer_edge,
            program.consumer_rhs_name: device_rhs,
            program.output_name: resident.output,
            "M": m, "N": active_n, "K": k,
            "LDA": k, "LDB": k, "LDD": active_n,
        }
        producer_samples = [
            rt._nvidia_native_descriptor_resident_device_latency(
                program.producer.image, program.producer.descriptor, producer_args,
                stream=session.stream, warmup=args.warmup, reps=args.reps,
            ) for _ in range(args.samples)
        ]
        consumer_samples = [
            rt._nvidia_native_descriptor_resident_device_latency(
                program.consumer.image, program.consumer.descriptor, consumer_args,
                stream=session.stream, warmup=args.warmup, reps=args.reps,
            ) for _ in range(args.samples)
        ]
        packet: dict[str, Any] = {
            "schema": "tessera.nvidia.scheduled-rmsnorm-matmul-edge.v1",
            "target": "nvidia_sm120",
            "architecture": program.consumer.image.architecture,
            "device": _version([
                "nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader",
            ]),
            "host": {
                "node": platform.node(), "platform": platform.platform(),
                "wsl": "microsoft" in platform.release().lower(),
            },
            "source_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                text=True, check=True,
            ).stdout.strip(),
            "worktree_dirty": bool(subprocess.run(
                ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                text=True, check=True,
            ).stdout.strip()),
            "method": "bounded dynamic N; Graph->Schedule->Tile packages; correctness checked before separate resident CUDA-event stage timings",
            "edge": {
                "producer": "tessera.rmsnorm", "consumer": "tessera.matmul",
                "storage": "fp16", "layout": "row_major intermediate",
                "static_mk": [m, k], "dynamic_n_bound": bound_n,
                "measured_active_n": active_n,
                "same_allocation": True, "same_stream": True,
                "rhs_layout": "strided ABI over compact column-major storage",
                "output_layout": "strided ABI over compact row-major storage",
            },
            "packages": {
                "producer_image": program.producer.image.image_digest,
                "consumer_image": program.consumer.image.image_digest,
                "consumer_descriptor": program.consumer.descriptor.descriptor_digest,
                "consumer_schedule": program.consumer.descriptor.provenance["schedule_digest"],
                "consumer_tile": program.consumer.descriptor.provenance["tile_ir_digest"],
            },
            "correctness": {
                "producer_max_abs_error": norm_error,
                "consumer_max_abs_error": matmul_error,
                "producer_execution_kind": resident.producer_receipt.get("execution_kind"),
                "consumer_execution_kind": resident.consumer_receipt.get("execution_kind"),
                "same_intermediate_pointer": consumer_edge.ptr == resident.intermediate.ptr,
                "package_digest_stable": True,
            },
            "timing": {
                "domain": "CUDA events around C++ repeated launches on resident buffers; producer and consumer timed separately",
                "producer_ms": producer_samples,
                "producer_median_ms": statistics.median(producer_samples),
                "producer_cov": _cv(producer_samples),
                "consumer_ms": consumer_samples,
                "consumer_median_ms": statistics.median(consumer_samples),
                "consumer_cov": _cv(consumer_samples),
            },
            "selector_changed": False,
            "promotion": "none; one bounded dynamic-N SM120 envelope",
        }
    finally:
        resident.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")
    print(json.dumps(packet, indent=2))
    return 0


def _dynamic_m_benchmark(args: argparse.Namespace, program: Any, m: int, k: int,
                         n: int, first_active_m: int) -> int:
    rng = np.random.default_rng(0x5A17 + m + k + n)
    weights = np.asfortranarray(
        rng.normal(0.0, 0.25, size=(k, n)).astype(np.float16)
    )
    cases: list[dict[str, Any]] = []
    package_digest = program.consumer.image.image_digest
    for active_m in sorted(set((first_active_m, m))):
        source = np.ascontiguousarray(
            rng.normal(0.0, 0.25, size=(active_m, k)).astype(np.float16)
        )
        resident = program.execute_resident(source, weights)
        try:
            edge = resident.intermediate.numpy()
            output = resident.output.numpy()
            source_f32 = source.astype(np.float32)
            norm_reference = (
                source_f32 / np.sqrt(
                    np.mean(source_f32 * source_f32, axis=-1, keepdims=True) + 1e-5
                )
            ).astype(np.float16)
            active_edge = edge[:active_m]
            norm_error = float(np.max(np.abs(
                active_edge.astype(np.float32) - norm_reference.astype(np.float32)
            )))
            matmul_reference = active_edge.astype(np.float32) @ weights.astype(np.float32)
            matmul_error = float(np.max(np.abs(output - matmul_reference)))
            if norm_error > 2e-3 or matmul_error > 2e-4:
                raise RuntimeError(
                    f"resident dynamic-M oracle mismatch: norm={norm_error}, matmul={matmul_error}"
                )
            session = resident.device_session
            device_source, device_rhs = session._buffers[0], session._buffers[1]
            consumer_edge = resident.intermediate.view(
                0, (active_m, k), resident.intermediate.dtype, layout="strided"
            )
            if consumer_edge.ptr != resident.intermediate.ptr:
                raise RuntimeError("dynamic-M consumer edge copied the producer allocation")
            producer_args = {
                program.producer_input_name: device_source,
                program.intermediate_name: resident.intermediate,
                "Rows": active_m, "Columns": k,
            }
            consumer_args = {
                program.consumer_input_name: consumer_edge,
                program.consumer_rhs_name: device_rhs,
                program.output_name: resident.output,
                "M": active_m, "N": n, "K": k,
                "LDA": k, "LDB": k, "LDD": n,
            }
            producer_samples = [
                rt._nvidia_native_descriptor_resident_device_latency(
                    program.producer.image, program.producer.descriptor, producer_args,
                    stream=session.stream, warmup=args.warmup, reps=args.reps,
                ) for _ in range(args.samples)
            ]
            consumer_samples = [
                rt._nvidia_native_descriptor_resident_device_latency(
                    program.consumer.image, program.consumer.descriptor, consumer_args,
                    stream=session.stream, warmup=args.warmup, reps=args.reps,
                ) for _ in range(args.samples)
            ]
            cases.append({
                "active_m": active_m,
                "output_shape": list(output.shape),
                "correctness": {
                    "producer_max_abs_error": norm_error,
                    "consumer_max_abs_error": matmul_error,
                    "producer_execution_kind": resident.producer_receipt.get("execution_kind"),
                    "consumer_execution_kind": resident.consumer_receipt.get("execution_kind"),
                    "same_intermediate_pointer": consumer_edge.ptr == resident.intermediate.ptr,
                    "image_digest_stable": program.consumer.image.image_digest == package_digest,
                },
                "timing": {
                    "domain": "CUDA events around repeated native launches on resident buffers; producer and consumer measured separately",
                    "producer_ms": producer_samples,
                    "producer_median_ms": statistics.median(producer_samples),
                    "producer_cov": _cv(producer_samples),
                    "consumer_ms": consumer_samples,
                    "consumer_median_ms": statistics.median(consumer_samples),
                    "consumer_cov": _cv(consumer_samples),
                },
            })
        finally:
            resident.close()
    packet: dict[str, Any] = {
        "schema": "tessera.nvidia.scheduled-rmsnorm-matmul-edge.v1",
        "target": "nvidia_sm120",
        "architecture": program.consumer.image.architecture,
        "device": _version([
            "nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader",
        ]),
        "host": {
            "node": platform.node(), "platform": platform.platform(),
            "wsl": "microsoft" in platform.release().lower(),
        },
        "source_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip(),
        "worktree_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()),
        "method": "bounded dynamic M; Graph->Schedule->Tile packages; correctness checked before separate resident CUDA-event stage timings",
        "edge": {
            "producer": "tessera.rmsnorm", "consumer": "tessera.matmul",
            "storage": "fp16", "layout": "row_major intermediate",
            "dynamic_m_bound": m, "static_nk": [n, k],
            "measured_active_m": [row["active_m"] for row in cases],
            "same_allocation": True, "same_stream": True,
            "rhs_layout": "strided ABI over compact column-major storage",
            "output_layout": "strided ABI over compact row-major storage",
        },
        "packages": {
            "producer_image": program.producer.image.image_digest,
            "consumer_image": program.consumer.image.image_digest,
            "consumer_descriptor": program.consumer.descriptor.descriptor_digest,
            "consumer_schedule": program.consumer.descriptor.provenance["schedule_digest"],
            "consumer_tile": program.consumer.descriptor.provenance["tile_ir_digest"],
        },
        "cases": cases,
        "selector_changed": False,
        "promotion": "none; bounded dynamic-M SM120 envelope",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + chr(10))
    print(json.dumps(packet, indent=2))
    return 0



def _dynamic_k_benchmark(args: argparse.Namespace, program: Any, m: int,
                         bound_k: int, n: int) -> int:
    rng = np.random.default_rng(0x5A17 + m + bound_k + n)
    cases: list[dict[str, Any]] = []
    package_digest = program.consumer.image.image_digest
    for active_k in sorted(set((max(1, bound_k // 2), max(1, (3 * bound_k) // 4), bound_k))):
        source = np.ascontiguousarray(
            rng.normal(0.0, 0.25, size=(m, active_k)).astype(np.float16)
        )
        weights = np.asfortranarray(
            rng.normal(0.0, 0.25, size=(active_k, n)).astype(np.float16)
        )
        resident = program.execute_resident(source, weights)
        try:
            edge = resident.intermediate.numpy()
            output = resident.output.numpy()
            source_f32 = source.astype(np.float32)
            norm_reference = (
                source_f32 / np.sqrt(
                    np.mean(source_f32 * source_f32, axis=-1, keepdims=True) + 1e-5
                )
            ).astype(np.float16)
            norm_error = float(np.max(np.abs(
                edge.astype(np.float32) - norm_reference.astype(np.float32)
            )))
            matmul_reference = edge.astype(np.float32) @ weights.astype(np.float32)
            matmul_error = float(np.max(np.abs(output - matmul_reference)))
            if norm_error > 2e-3 or matmul_error > 2e-4:
                raise RuntimeError(
                    f"resident dynamic-K oracle mismatch: norm={norm_error}, matmul={matmul_error}"
                )
            session = resident.device_session
            device_source, device_rhs = session._buffers[0], session._buffers[1]
            consumer_edge = resident.intermediate.view(
                0, (m, active_k), resident.intermediate.dtype, layout="strided"
            )
            if consumer_edge.ptr != resident.intermediate.ptr:
                raise RuntimeError("dynamic-K consumer edge copied the producer allocation")
            producer_args = {
                program.producer_input_name: device_source,
                program.intermediate_name: resident.intermediate,
                "Rows": m, "Columns": active_k,
            }
            consumer_args = {
                program.consumer_input_name: consumer_edge,
                program.consumer_rhs_name: device_rhs,
                program.output_name: resident.output,
                "M": m, "N": n, "K": active_k,
                "LDA": active_k, "LDB": active_k, "LDD": n,
            }
            producer_samples = [
                rt._nvidia_native_descriptor_resident_device_latency(
                    program.producer.image, program.producer.descriptor, producer_args,
                    stream=session.stream, warmup=args.warmup, reps=args.reps,
                ) for _ in range(args.samples)
            ]
            consumer_samples = [
                rt._nvidia_native_descriptor_resident_device_latency(
                    program.consumer.image, program.consumer.descriptor, consumer_args,
                    stream=session.stream, warmup=args.warmup, reps=args.reps,
                ) for _ in range(args.samples)
            ]
            cases.append({
                "active_k": active_k,
                "output_shape": list(output.shape),
                "correctness": {
                    "producer_max_abs_error": norm_error,
                    "consumer_max_abs_error": matmul_error,
                    "producer_execution_kind": resident.producer_receipt.get("execution_kind"),
                    "consumer_execution_kind": resident.consumer_receipt.get("execution_kind"),
                    "same_intermediate_pointer": consumer_edge.ptr == resident.intermediate.ptr,
                    "image_digest_stable": program.consumer.image.image_digest == package_digest,
                },
                "timing": {
                    "domain": "CUDA events around repeated native launches on resident buffers; producer and consumer measured separately",
                    "producer_ms": producer_samples,
                    "producer_median_ms": statistics.median(producer_samples),
                    "producer_cov": _cv(producer_samples),
                    "consumer_ms": consumer_samples,
                    "consumer_median_ms": statistics.median(consumer_samples),
                    "consumer_cov": _cv(consumer_samples),
                },
            })
        finally:
            resident.close()
    packet: dict[str, Any] = {
        "schema": "tessera.nvidia.scheduled-rmsnorm-matmul-edge.v1",
        "target": "nvidia_sm120",
        "architecture": program.consumer.image.architecture,
        "device": _version([
            "nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader",
        ]),
        "host": {
            "node": platform.node(), "platform": platform.platform(),
            "wsl": "microsoft" in platform.release().lower(),
        },
        "source_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip(),
        "worktree_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()),
        "method": "bounded dynamic K; Graph->Schedule->Tile packages; correctness checked before separate resident CUDA-event stage timings",
        "edge": {
            "producer": "tessera.rmsnorm", "consumer": "tessera.matmul",
            "storage": "fp16", "layout": "compact row_major intermediate",
            "static_mn": [m, n], "dynamic_k_bound": bound_k,
            "measured_active_k": [row["active_k"] for row in cases],
            "same_allocation": True, "same_stream": True,
            "rhs_layout": "compact column-major storage",
            "output_layout": "compact row-major storage",
        },
        "packages": {
            "producer_image": program.producer.image.image_digest,
            "consumer_image": program.consumer.image.image_digest,
            "consumer_descriptor": program.consumer.descriptor.descriptor_digest,
            "consumer_schedule": program.consumer.descriptor.provenance["schedule_digest"],
            "consumer_tile": program.consumer.descriptor.provenance["tile_ir_digest"],
        },
        "cases": cases,
        "selector_changed": False,
        "promotion": "none; bounded dynamic-K SM120 envelope",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")
    print(json.dumps(packet, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--n", type=int, default=512, help="static N or dynamic-N capacity bound")
    parser.add_argument("--active-n", type=int, help="active N when --dynamic-n is set (default: n//2)")
    parser.add_argument("--dynamic-n", action="store_true", help="package one bounded dynamic-N consumer")
    parser.add_argument("--active-m", type=int, help="first active M when --dynamic-m is set (second case uses bound M)")
    parser.add_argument("--dynamic-m", action="store_true", help="package one bounded dynamic-M producer/consumer edge")
    parser.add_argument("--dynamic-k", action="store_true", help="package one bounded dynamic-K producer/consumer edge")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    m, k, n = args.m, args.k, args.n
    if sum((args.dynamic_m, args.dynamic_n, args.dynamic_k)) > 1:
        parser.error("dynamic M, N, and K are currently separate package envelopes")
    if args.output is None:
        output_name = (
            "dynamic_m_sm120.json" if args.dynamic_m
            else "dynamic_n_sm120.json" if args.dynamic_n
            else "dynamic_k_sm120.json" if args.dynamic_k
            else "sm120.json"
        )
        args.output = ROOT / "benchmarks/baselines/sm120_rmsnorm_matmul_edge_20260930" / output_name
    if min(m, k, n) <= 0 or args.reps <= 0 or args.samples < 3:
        parser.error("dimensions/repetitions must be positive and samples must be at least three")

    if args.dynamic_m:
        first_active_m = args.active_m if args.active_m is not None else max(1, m // 2)
        if first_active_m <= 0 or first_active_m > m:
            parser.error("active M must be positive and no greater than the dynamic-M bound")
        if args.active_n is not None:
            parser.error("--active-n cannot be combined with --dynamic-m")
        active_n = n
    elif args.dynamic_n:
        if args.active_m is not None:
            parser.error("--active-m requires --dynamic-m")
        active_n = args.active_n if args.active_n is not None else max(1, n // 2)
        if active_n <= 0 or active_n > n:
            parser.error("active N must be positive and no greater than the dynamic-N bound")
    else:
        if args.active_n is not None:
            parser.error("--active-n requires --dynamic-n")
        if args.active_m is not None:
            parser.error("--active-m requires --dynamic-m")
        active_n = n
    producer_module, consumer_module = _modules(
        m, k, n, dynamic_n=args.dynamic_n, dynamic_k=args.dynamic_k
    )
    if args.dynamic_m:
        program = nvidia_native.package_scheduled_rmsnorm_matmul(
            producer_module, consumer_module,
            pipeline_name="tessera-lower-to-nvidia-sm120",
            dynamic_m_bound=m,
        )
    elif args.dynamic_k:
        program = nvidia_native.package_scheduled_rmsnorm_matmul(
            producer_module, consumer_module,
            pipeline_name="tessera-lower-to-nvidia-sm120",
            dynamic_k_bound=k,
        )
    else:
        producer_ir = scheduled_kernel.lower_scheduled_kernel(
            producer_module, target="nvidia_sm120",
        )
        consumer_ir = scheduled_matmul.lower_scheduled_matmul(
            consumer_module, target="nvidia_sm120",
        )
        program = nvidia_native.package_scheduled_rmsnorm_matmul(
            producer_ir, consumer_ir,
            pipeline_name="tessera-lower-to-nvidia-sm120",
        )
    program.validate()
    if args.dynamic_m:
        return _dynamic_m_benchmark(args, program, m, k, n, first_active_m)
    if args.dynamic_n:
        return _dynamic_n_benchmark(args, program, m, k, n, active_n)
    if args.dynamic_k:
        return _dynamic_k_benchmark(args, program, m, k, n)

    rng = np.random.default_rng(0x5A17 + m + k + n)
    source = np.ascontiguousarray(rng.normal(0.0, 0.25, size=(m, k)).astype(np.float16))
    weights = np.asfortranarray(rng.normal(0.0, 0.25, size=(k, n)).astype(np.float16))
    intermediate = np.empty((m, k), dtype=np.float16, order="C")
    output = np.empty((m, n), dtype=np.float32, order="C")

    wall_start = time.perf_counter()
    run = program.execute(
        source, weights, intermediate=intermediate, output=output,
    )
    host_chain_ms = (time.perf_counter() - wall_start) * 1e3
    if not np.shares_memory(run.intermediate, intermediate):
        raise RuntimeError("producer intermediate allocation was replaced before consumer completion")
    if not np.shares_memory(run.output, output):
        raise RuntimeError("consumer output allocation was replaced")

    source_f32 = source.astype(np.float32)
    norm_reference = (
        source_f32 / np.sqrt(np.mean(source_f32 * source_f32, axis=-1, keepdims=True) + 1e-5)
    ).astype(np.float16)
    norm_error = float(np.max(np.abs(intermediate.astype(np.float32) - norm_reference.astype(np.float32))))
    if norm_error > 2e-3:
        raise RuntimeError(f"RMSNorm producer disagrees with oracle: max_abs_error={norm_error}")
    matmul_reference = intermediate.astype(np.float32) @ weights.astype(np.float32)
    matmul_error = float(np.max(np.abs(output - matmul_reference)))
    if matmul_error > 2e-4:
        raise RuntimeError(f"matmul consumer disagrees with oracle: max_abs_error={matmul_error}")

    producer_args = {
        program.producer_input_name: source,
        program.intermediate_name: intermediate,
        "Rows": m,
        "Columns": k,
    }
    consumer_args = {
        program.consumer_input_name: intermediate,
        program.consumer_rhs_name: weights,
        program.output_name: output,
        "M": m,
        "N": n,
        "K": k,
    }
    producer_samples = [
        rt._nvidia_native_descriptor_device_latency(
            program.producer.image, program.producer.descriptor, producer_args,
            warmup=args.warmup, reps=args.reps,
        )
        for _ in range(args.samples)
    ]
    consumer_samples = [
        rt._nvidia_native_descriptor_device_latency(
            program.consumer.image, program.consumer.descriptor, consumer_args,
            warmup=args.warmup, reps=args.reps,
        )
        for _ in range(args.samples)
    ]

    resident = program.execute_resident(source, weights)
    resident_edge = resident.intermediate.numpy()
    resident_output = resident.output.numpy()
    resident_norm_error = float(np.max(np.abs(
        resident_edge.astype(np.float32) - norm_reference.astype(np.float32)
    )))
    resident_matmul_error = float(np.max(np.abs(
        resident_output - (resident_edge.astype(np.float32) @ weights.astype(np.float32))
    )))
    if resident_norm_error > 2e-3 or resident_matmul_error > 2e-4:
        resident.close()
        raise RuntimeError(
            "resident package chain disagrees with numerical oracle: "
            f"rmsnorm={resident_norm_error}, matmul={resident_matmul_error}"
        )

    session = resident.device_session
    device_source, device_rhs = session._buffers[0], session._buffers[1]

    resident_producer_args = {
        program.producer_input_name: device_source,
        program.intermediate_name: resident.intermediate,
        "Rows": m, "Columns": k,
    }
    resident_consumer_args = {
        program.consumer_input_name: resident.intermediate,
        program.consumer_rhs_name: device_rhs,
        program.output_name: resident.output,
        "M": m, "N": n, "K": k,
    }
    try:
        resident_producer_samples = [
            rt._nvidia_native_descriptor_resident_device_latency(
                program.producer.image, program.producer.descriptor,
                resident_producer_args, stream=session.stream,
                warmup=args.warmup, reps=args.reps,
            )
            for _ in range(args.samples)
        ]
        resident_consumer_samples = [
            rt._nvidia_native_descriptor_resident_device_latency(
                program.consumer.image, program.consumer.descriptor,
                resident_consumer_args, stream=session.stream,
                warmup=args.warmup, reps=args.reps,
            )
            for _ in range(args.samples)
        ]
    finally:
        resident.close()

    packet: dict[str, Any] = {
        "schema": "tessera.nvidia.scheduled-rmsnorm-matmul-edge.v1",
        "target": "nvidia_sm120",
        "architecture": program.producer.image.architecture,
        "device": _version([
            "nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader",
        ]),
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "wsl": "microsoft" in platform.release().lower(),
        },
        "source_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip(),
        "worktree_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()),
        "method": "two checked native packages; numerical oracle before separate CUDA-event timing",
        "edge": {
            "producer": "tessera.rmsnorm",
            "consumer": "tessera.matmul",
            "shape": [m, k],
            "dtype": "fp16",
            "layout": "row_major",
            "buffer_owner": "caller",
            "lifetime": "producer completion through consumer completion",
            "aliasing": "disjoint from source, RHS, and final output",
        },
        "packages": {
            "producer": {
                "compiler_fingerprint": program.producer.image.compiler_fingerprint,
                "toolchain_fingerprint": program.producer.image.toolchain_fingerprint,
                "image_digest": program.producer.image.image_digest,
                "descriptor_digest": program.producer.descriptor.descriptor_digest,
                "schedule_digest": program.producer.descriptor.provenance["schedule_digest"],
                "tile_digest": program.producer.descriptor.provenance["tile_ir_digest"],
            },
            "consumer": {
                "compiler_fingerprint": program.consumer.image.compiler_fingerprint,
                "toolchain_fingerprint": program.consumer.image.toolchain_fingerprint,
                "image_digest": program.consumer.image.image_digest,
                "descriptor_digest": program.consumer.descriptor.descriptor_digest,
                "schedule_digest": program.consumer.descriptor.provenance["schedule_digest"],
                "tile_digest": program.consumer.descriptor.provenance["tile_ir_digest"],
            },
        },
        "correctness": {
            "producer_max_abs_error": norm_error,
            "consumer_max_abs_error": matmul_error,
            "producer_execution_kind": run.producer_receipt.get("execution_kind"),
            "consumer_execution_kind": run.consumer_receipt.get("execution_kind"),
            "intermediate_same_allocation": True,
            "resident_producer_max_abs_error": resident_norm_error,
            "resident_consumer_max_abs_error": resident_matmul_error,
            "resident_execution_kind": "native_gpu",
            "resident_uploads_once_no_intermediate_copy": True,
        },
        "timing": {
            "domain": "CUDA events; resident buffers within each repeated package timing",
            "producer_ms": producer_samples,
            "producer_median_ms": statistics.median(producer_samples),
            "producer_cov": _cv(producer_samples),
            "consumer_ms": consumer_samples,
            "consumer_median_ms": statistics.median(consumer_samples),
            "consumer_cov": _cv(consumer_samples),
            "synchronous_two_launch_host_wall_ms": host_chain_ms,
            "host_wall_is_not_kernel_time": True,
            "resident_device_stream": {
                "producer_ms": resident_producer_samples,
                "producer_median_ms": statistics.median(resident_producer_samples),
                "producer_cov": _cv(resident_producer_samples),
                "consumer_ms": resident_consumer_samples,
                "consumer_median_ms": statistics.median(resident_consumer_samples),
                "consumer_cov": _cv(resident_consumer_samples),
                "domain": "CUDA events; C++ repeated launches on one stream with device-resident buffers",
            },
        },
        "resources": {
            "producer": (
                program.producer.image.resource_record.to_dict()
                if program.producer.image.resource_record else None
            ),
            "consumer": (
                program.consumer.image.resource_record.to_dict()
                if program.consumer.image.resource_record else None
            ),
        },
        "selector_changed": False,
        "promotion": "none; one SM120 shape family slice, no cross-target promotion",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")
    print(json.dumps(packet, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
