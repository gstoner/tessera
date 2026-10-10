#!/usr/bin/env python3
"""Correctness-gated exact gfx1151/gfx1201 scheduled image and module reuse."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time

import numpy as np
from tessera import runtime as rt
from tessera.compiler import rocm_native, scheduled_matmul
from tests.unit.test_scheduled_matmul_consumers import _module, _dynamic_module

ROOT = Path(__file__).resolve().parents[2]
SHAPES = ((16, 16, 16), (32, 32, 32), (48, 16, 32), (128, 128, 128), (256, 256, 256), (512, 512, 512))


def _expected_cache_states(num_shapes: int) -> list[str]:
    if num_shapes < 1:
        raise ValueError("at least one shape is required")
    return ["cold"] + ["warm_cache"] * (num_shapes - 1)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _measure(shape: tuple[int, int, int], *, samples: int, warmup: int, rng, capture_events, dynamic: bool = False, fused: bool = False, architecture: str = "gfx1201", k_unroll: int = 1, staging: str = "register", lds_waves: tuple[int,int] = (2,2), separate_timing: bool = False) -> dict[str, object]:
    m, k, n = shape
    module = (_dynamic_module(target="rocm", bounds=(m,n,k), dtype="fp16",
                              bias=fused, activation="relu" if fused else "none")
              if dynamic else _module(target="rocm", shape=shape, dtype="fp16",
                                       bias=fused, activation="relu" if fused else "none"))
    module.functions[0].name = f"{architecture}_matmul_{m}_{n}_{k}"
    start = time.perf_counter_ns()
    scheduled = scheduled_matmul.lower_scheduled_matmul(
        module, target=f"rocm_{architecture}"
    )
    schedule_ms = (time.perf_counter_ns() - start) / 1e6
    start = time.perf_counter_ns()
    package = rocm_native.package_scheduled_matmul(
        scheduled, pipeline_name="tessera-lower-to-rocm", k_unroll=k_unroll, staging=staging, lds_waves=lds_waves
    )
    package_ms = (time.perf_counter_ns() - start) / 1e6

    a = (rng.standard_normal((m, k)) * 0.25).astype(np.float16)
    b = (rng.standard_normal((k, n)) * 0.25).astype(np.float16)
    output = np.zeros((m, n), np.float32)
    expected = a.astype(np.float32) @ b.astype(np.float32)
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image,
        launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir,
        target_ir=package.target_ir,
    )
    arguments = {
        "buffers": {"a": a, "b": b, "o": output},
        "scalars": {"M": m, "N": n, "K": k},
    }
    if fused:
        bias = (rng.standard_normal((n,)) * .1).astype(np.float32)
        arguments["buffers"]["bias"] = bias
        expected = np.maximum(expected + bias, 0)
    e2e_samples_ms: list[float] = []
    capture_events[0] = False
    for _ in range(warmup):
        output.fill(0)
        result = rt.launch(artifact, arguments)
        if not result.get("ok") or result.get("execution_kind") != "native_gpu":
            raise RuntimeError(f"gfx1201 native matmul warmup failed: {result}")
        np.testing.assert_allclose(output, expected, rtol=0, atol=2e-4)
    if separate_timing:
        capture_events[0] = True
        for _ in range(samples):
            result = rt.launch(artifact, arguments)
            if not result.get("ok") or result.get("execution_kind") != "native_gpu":
                raise RuntimeError(f"native device-event sample failed: {result}")
            np.testing.assert_allclose(output, expected, rtol=0, atol=2e-4)
    capture_events[0] = not separate_timing
    for _ in range(samples):
        output.fill(0)
        start = time.perf_counter_ns()
        result = rt.launch(artifact, arguments)
        e2e_samples_ms.append((time.perf_counter_ns() - start) / 1e6)
        if not result.get("ok") or result.get("execution_kind") != "native_gpu":
            raise RuntimeError(f"gfx1201 native matmul failed: {result}")
        np.testing.assert_allclose(output, expected, rtol=0, atol=2e-4)
    return {
        "shape_mnk": [m, n, k],
        "schedule_digest": scheduled.schedule_digest,
        "tile_ir_digest": scheduled.tile_digest,
        "image_digest": package.image.image_digest,
        "entry_symbol": package.descriptor.entry_symbol,
        "split_k": package.descriptor.provenance["split_k"],
        "workspace_bytes": package.descriptor.workspace.bytes,
        "end_to_end_event_instrumented": not separate_timing,
        "compile_state": package.image.compile_state,
        "schedule_ms": schedule_ms,
        "package_ms": package_ms,
        "max_abs_error": float(np.max(np.abs(output - expected))),
        "correctness_atol": 2e-4,
        "correctness_rtol": 0,
        "correctness": "passed_before_timing_during_warmup_and_after_each_e2e_sample",
        "warmup_samples": warmup,
        "end_to_end_ms_samples": e2e_samples_ms,
        "end_to_end_ms_median": statistics.median(e2e_samples_ms),
        "shape_guards": [
            {"binding": guard.binding, "dimension": guard.dimension,
             "predicate": guard.predicate, "value": guard.value}
            for guard in package.descriptor.shape_guards
        ],
    }


def record(samples: int = 7, *, dynamic: bool = False, fused: bool = False, architecture: str = "gfx1201", k_unroll: int = 1, staging: str = "register", lds_waves: tuple[int,int] = (2,2), split_partition: bool = False) -> dict[str, object]:
    if samples < 3:
        raise ValueError("at least three end-to-end samples are required")
    if architecture not in {"gfx1151", "gfx1201"}:
        raise ValueError("requires an audited gfx1151/gfx1201 architecture")
    if split_partition and (architecture != "gfx1201" or staging != "register" or dynamic):
        raise ValueError("split partition recorder requires static gfx1201 register matmul")
    shapes = ((16,2048,256),(15,2048,200)) if split_partition else (
        ((16,64,16),(32,96,32),(48,67,32),(128,128,128),(256,256,256),(512,512,512))
        if architecture == "gfx1201" and not dynamic else SHAPES)
    live, configured = rt._rocm_live_arch(), rt._rocm_chip()
    if live != configured or live != architecture:
        raise RuntimeError(f"requires exact live/configured {architecture}, got {live}/{configured}")
    tool = rocm_native._tessera_opt()
    if tool is None:
        raise RuntimeError("a rebuilt ROCm tessera-opt is required")
    tool = Path(tool).resolve()
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("HIP initialization failed")
    ordinal = ctypes.c_int()
    name = ctypes.create_string_buffer(256)
    if hip.hipGetDevice(ctypes.byref(ordinal)) != 0 or hip.hipDeviceGetName(name, len(name), ordinal.value) != 0:
        raise RuntimeError("HIP device identification failed")
    start_event, stop_event = ctypes.c_void_p(), ctypes.c_void_p()
    if hip.hipEventCreate(ctypes.byref(start_event)) != 0:
        raise RuntimeError("HIP start event creation failed")
    if hip.hipEventCreate(ctypes.byref(stop_event)) != 0:
        hip.hipEventDestroy(start_event)
        raise RuntimeError("HIP stop event creation failed")
    cache_before = rt._rocm_native_image_cache_stats()
    if cache_before is not None:
        rt._clear_rocm_native_image_cache()
        cache_before = rt._rocm_native_image_cache_stats()
    host_module_loads = [0]
    original_module_load = hip.hipModuleLoadData

    def counted_load(*args):
        host_module_loads[0] += 1
        return original_module_load(*args)

    hip.hipModuleLoadData = counted_load
    event_samples_ms: list[float] = []
    capture_events = [False]
    original_launch = hip.hipModuleLaunchKernel

    def timed_launch(*args):
        if not capture_events[0]:
            return original_launch(*args)
        if hip.hipEventRecord(start_event, None) != 0:
            raise RuntimeError("HIP start event record failed")
        rc = original_launch(*args)
        if rc != 0:
            return rc
        if hip.hipEventRecord(stop_event, None) != 0:
            raise RuntimeError("HIP stop event record failed")
        if hip.hipEventSynchronize(stop_event) != 0:
            raise RuntimeError("HIP event synchronization failed")
        elapsed = ctypes.c_float()
        if hip.hipEventElapsedTime(ctypes.byref(elapsed), start_event, stop_event) != 0:
            raise RuntimeError("HIP elapsed time query failed")
        if capture_events[0]:
            event_samples_ms.append(float(elapsed.value))
        return rc

    hip.hipModuleLaunchKernel = timed_launch
    try:
        rocm_native._cache.clear()
        rocm_native._shape_free_targets.clear()
        rng = np.random.default_rng(1201_892)
        rows = [_measure(shape, samples=samples, warmup=5, rng=rng,
                         capture_events=capture_events, dynamic=dynamic, fused=fused, architecture=architecture, k_unroll=k_unroll, staging=staging, lds_waves=lds_waves, separate_timing=split_partition) for shape in shapes]
    finally:
        hip.hipModuleLaunchKernel = original_launch
        hip.hipModuleLoadData = original_module_load
        hip.hipEventDestroy(stop_event)
        hip.hipEventDestroy(start_event)

    cache_after = rt._rocm_native_image_cache_stats()
    cache_delta = ({key: cache_after[key] - cache_before[key] for key in cache_before}
                   if cache_before is not None and cache_after is not None else None)
    expected_launches = len(shapes) * (samples * (2 if split_partition else 1) + 5)
    if cache_delta is not None:
        if cache_delta != {"loads": 1, "hits": expected_launches - 1, "function_lookups": 2 if split_partition else 1, "unloads": 0}:
            raise RuntimeError(f"native module reuse did not match launches: {cache_delta}")
    elif host_module_loads[0] != expected_launches:
        raise RuntimeError(f"uncached host module count disagrees: {host_module_loads[0]}/{expected_launches}")
    digests = {row["image_digest"] for row in rows}
    states = [row["compile_state"] for row in rows]
    entries = {row["entry_symbol"] for row in rows}
    expected_states = _expected_cache_states(len(shapes))
    if len(digests) != 1 or states != expected_states or len(entries) != 1:
        raise RuntimeError(
            "shapes did not reuse one native image and entry: "
            f"states={states}, digests={digests}, entries={entries}"
        )
    phases = 2 if split_partition else 1
    if len(event_samples_ms) != len(shapes) * samples * phases:
        raise RuntimeError(f"expected one HIP event per kernel; got {len(event_samples_ms)}")
    for index, row in enumerate(rows):
        per_shape = event_samples_ms[index * samples * phases:(index + 1) * samples * phases]
        if split_partition:
            partial, reduction = per_shape[::2], per_shape[1::2]
            row["partial_kernel_event_ms_samples"] = partial
            row["partial_kernel_event_ms_median"] = statistics.median(partial)
            row["reduction_kernel_event_ms_samples"] = reduction
            row["reduction_kernel_event_ms_median"] = statistics.median(reduction)
            totals = [a+b for a,b in zip(partial,reduction,strict=True)]
            row["program_kernel_event_ms_samples"] = totals
            row["program_kernel_event_ms_median"] = statistics.median(totals)
        else:
            row["kernel_event_ms_samples"] = per_shape
            row["kernel_event_ms_median"] = statistics.median(per_shape)
    return {
        "schema": f"tessera.rocm_{architecture}_scheduled_matmul_shape_key.v2",
        "work_item": "E2E-REAL-6-ROCM-MATMUL-CACHE",
        "physical_envelope": f"{staging}_k_unroll_{k_unroll}_split_k_{8 if split_partition else 1}",
        "split_partition": split_partition,
        "k_unroll": k_unroll,
        "staging": staging,
        "lds_waves": list(lds_waves),
        "dynamic_mnk": dynamic,
        "bias_relu_epilogue": fused,
        "kernel_identity_projection": "native_mlir_v2",
        "native_module_cache": cache_delta is not None,
        "native_module_cache_delta": cache_delta,
        "host_uncached_module_load_calls": host_module_loads[0],
        "native_module_library_sha256": (
            hashlib.sha256(Path(rt._load_rocm_native_image_runtime()._name).read_bytes()).hexdigest()
            if cache_delta is not None else None),

        "host": socket.gethostname(),
        "architecture": live,
        "device": name.value.decode(),
        "device_ordinal": ordinal.value,
        "configured_architecture": configured,
        "source_commit": _git("rev-parse", "HEAD"),
        "source_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "compiler_path": str(tool),
        "compiler_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "src/compiler/ir/TileOps.cpp",
                "src/compiler/ir/include/Tessera/Dialect/Tile/TileOps.td",
                "python/tessera/compiler/rocm_native.py",
                "python/tessera/compiler/rocm_pipeline.py",
                "python/tessera/compiler/scheduled_matmul.py",
                "python/tessera/compiler/native_artifact.py",
                "python/tessera/runtime.py",
                "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_image_cache.cpp",
                "src/compiler/codegen/Tessera_ROCM_Backend/include/TesseraROCM/IR/TesseraROCMOps.td",
                "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/Passes.cpp",
                "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/ROCMKernelIdentity.cpp",
                "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp",
                "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateWMMAGemmKernel.cpp",
                "benchmarks/rocm/record_gfx1201_matmul_shape_key.py",
            )
        },
        "shapes_share_one_image": True,
        "num_shapes": len(rows),
        "per_kernel_launch_hip_event_median_ms": statistics.median(event_samples_ms),
        "per_kernel_launch_hip_event_samples_ms": event_samples_ms,
        "timing_note": (
            "Each shape receives five correctness-checked warmup launches before sampling. "
            "HIP events bracket only the native matmul kernel launch on the "
            "default stream. End-to-end samples use runtime.launch and include "
            "cold module acquisition or native cached lease acquisition, host/device allocation, transfers, synchronization, "
            "descriptor validation, and Python dispatch. The end_to_end_event_instrumented flag identifies synchronous per-kernel HIP events; split-partition end-to-end samples disable them. Schedule lowering and "
            "package construction are timed separately."
        ),
        "end_to_end_event_instrumented": not split_partition,
        "promotion_eligible": False,
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--k-unroll",type=int,choices=range(1,9),default=1)
    parser.add_argument("--staging", choices=("register","lds"), default="register")
    parser.add_argument("--lds-waves", type=int, nargs=2, default=(2,2))
    parser.add_argument("--split-partition", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--architecture", choices=("gfx1151", "gfx1201"), default="gfx1201")
    args = parser.parse_args()
    args.output.write_text(json.dumps(record(args.samples, dynamic=args.dynamic, fused=args.fused, architecture=args.architecture, k_unroll=args.k_unroll, staging=args.staging, lds_waves=tuple(args.lds_waves), split_partition=args.split_partition), indent=2, sort_keys=True) + "\n")
