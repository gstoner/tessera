#!/usr/bin/env python3
"""Correctness-gated gfx1201 NVFP4 ingest and scheduled-launch timing."""
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

import ml_dtypes
import numpy as np

from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler import rocm_nvfp4_ingest as ingest
from tessera.compiler.rocm_mxfp4_native import package_scaled_wmma_target_ir


ROOT = Path(__file__).resolve().parents[2]


def _compile_package(m: int, n: int, k: int):
    fixture = ROOT / "tests/tessera-ir/phase2/e2e_mxfp4_ingest_rocm.mlir"
    source = fixture.read_text()
    replacements = {
        "tensor<17x64xui8>": f"tensor<{m}x{k}xui8>",
        "tensor<32x19xui8>": f"tensor<{k // 2}x{n}xui8>",
        "tensor<17xf32>": f"tensor<{m}xf32>",
        "tensor<2x19xui8>": f"tensor<{k // 32}x{n}xui8>",
        "tensor<17x19xbf16>": f"tensor<{m}x{n}xbf16>",
    }
    for old_type, new_type in replacements.items():
        if old_type not in source:
            raise RuntimeError(f"dynamic NVFP4 fixture is missing {old_type}")
        source = source.replace(old_type, new_type)
    opt = os.environ["TESSERA_OPT"]
    schedule = subprocess.run(
        [opt, "--tessera-graph-to-schedule", "-"], input=source, check=True,
        capture_output=True, text=True,
    ).stdout
    tile = subprocess.run(
        [opt, "--tessera-schedule-to-tile", "-"], input=schedule, check=True,
        capture_output=True, text=True,
    ).stdout
    common = [opt, "--tessera-graph-to-schedule", "--tessera-schedule-to-tile"]
    target = subprocess.run(
        [*common, "--lower-tile-to-rocm=arch=gfx1201", "-"],
        input=source, check=True, capture_output=True, text=True,
    ).stdout
    return package_scaled_wmma_target_ir(tile, target), schedule, tile, target


def _inputs(m: int = 17, n: int = 19, k: int = 64):
    if m <= 0 or n < 2 or k <= 0 or k % 32:
        raise ValueError("NVFP4 benchmark requires positive M/N and K divisible by 32")
    seed = 1201_873 if (m, n, k) == (17, 19, 64) else 1201_873 + m * 1_000_003 + n * 1009 + k
    rng = np.random.default_rng(seed)
    projections = []
    gate_rows = n // 2
    for name, rows, scale_values, global_scale in (
        ("gate", gate_rows, (0.5, 1.0, 0.75, 1.5), 0.5),
        ("up", n - gate_rows, (2.0, 1.0, 1.5, 0.5), 2.0),
    ):
        codes = rng.integers(0, 16, size=(rows, k), dtype=np.uint8)
        scales = np.asarray(
            np.tile(np.resize(np.asarray(scale_values, np.float32), k // 16),
                    (rows, 1)),
            dtype=ml_dtypes.float8_e4m3fn,
        )
        projections.append(ingest.NVFP4Projection(
            name, mx.pack_e2m1_codes(codes), scales, global_scale,
        ))
    ingest_start_ns = time.perf_counter_ns()
    weights = ingest.ingest_nvfp4_projections(projections)
    ingest_ms = (time.perf_counter_ns() - ingest_start_ns) / 1e6
    preserve_code_baseline = []
    for projection in projections:
        rows, proj_k, packed_source, source_scales, _ = ingest._validate_projection(projection)
        source_codes = mx.unpack_e2m1_codes(packed_source)
        source_values = ingest._E2M1[source_codes]
        baseline_exponents = np.empty((proj_k // 32, rows), dtype=np.uint8)
        for row in range(rows):
            for group in range(proj_k // 32):
                start = group * 32
                pair = source_scales[row, group * 2:group * 2 + 2]
                exponent = ingest._choose_e8m0_exponent(
                    source_values[row, start:start + 32], pair
                )
                baseline_exponents[group, row] = (
                    0 if not np.any(pair) else exponent + 127
                )
        source_weights = source_values * source_scales.repeat(16, axis=1)
        baseline_weights = mx.exact_weights(source_codes, baseline_exponents)
        signal = float(np.square(source_weights.astype(np.float64)).sum())
        error = float(np.square(
            source_weights.astype(np.float64) - baseline_weights.astype(np.float64)
        ).sum())
        preserve_code_baseline.append({
            "projection": projection.name,
            "relative_rms_error": float(np.sqrt(error / signal)) if signal else 0.0,
            "sqnr_db": 10.0 * np.log10(signal / error) if signal and error else None,
            "method": "source_e2m1_codes_with_best_code_preserving_e8m0_scale",
        })
    a_values = rng.integers(-4, 5, size=(m, k)).astype(np.float32)
    a_f8 = a_values.astype(ml_dtypes.float8_e4m3fn)
    a_scale = np.exp2(rng.integers(-1, 2, size=m)).astype(np.float32)
    a = np.ascontiguousarray(a_f8.view(np.uint8))
    out = np.zeros((m, n), dtype=ml_dtypes.bfloat16)
    buffers = {
        "a": a,
        "b_packed": np.ascontiguousarray(weights.packed_codes.T),
        "a_scale": a_scale,
        "b_scale": weights.scale_exponents,
        "output": out,
    }
    b = mx.exact_weights(
        mx.unpack_e2m1_codes(weights.packed_codes), weights.scale_exponents,
    )
    expected = ((a_f8.astype(np.float32) * a_scale[:, None]) @ b.T).astype(
        ml_dtypes.bfloat16
    )
    return (m, n, k), weights, buffers, expected, ingest_ms, preserve_code_baseline


def _device_resident_run(package, buffers, shape, *, expected, repeats=7, iterations=50):
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("HIP device initialization failed")
    image = package.image
    descriptor = package.descriptor
    module = ctypes.c_void_p()
    payload = ctypes.create_string_buffer(image.payload)
    if hip.hipModuleLoadData(ctypes.byref(module), payload) != 0:
        raise RuntimeError("HIP module load failed")
    allocations = []
    events = []
    try:
        function = ctypes.c_void_p()
        if hip.hipModuleGetFunction(
            ctypes.byref(function), module, descriptor.entry_symbol.encode(),
        ) != 0:
            raise RuntimeError(f"missing package entry {descriptor.entry_symbol}")
        device_ptrs = [ctypes.c_void_p() for _ in range(5)]
        arrays = [buffers[name] for name in
                  ("a", "b_packed", "a_scale", "b_scale", "output")]
        for ptr, array in zip(device_ptrs, arrays, strict=True):
            if hip.hipMalloc(ctypes.byref(ptr), int(array.nbytes)) != 0:
                raise RuntimeError("HIP allocation failed")
            allocations.append(ptr)
        for ptr, array in zip(device_ptrs[:4], arrays[:4], strict=True):
            if hip.hipMemcpy(
                ptr, array.ctypes.data_as(ctypes.c_void_p), int(array.nbytes), 1,
            ) != 0:
                raise RuntimeError("HIP input upload failed")
        m, n, k = shape
        values = [*(ctypes.c_void_p(ptr.value) for ptr in device_ptrs),
                  ctypes.c_int64(m), ctypes.c_int64(n), ctypes.c_int64(k)]
        argv = (ctypes.c_void_p * len(values))(
            *[ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values]
        )
        grid, block = descriptor.geometry.grid, descriptor.geometry.workgroup
        if grid is None or block is None:
            raise RuntimeError("package has no fixed launch geometry")

        def launch():
            rc = hip.hipModuleLaunchKernel(
                function, *grid, *block, descriptor.dynamic_local_memory_bytes,
                None, argv, None,
            )
            if rc != 0:
                raise RuntimeError(f"HIP kernel launch failed rc={rc}")

        launch()
        if hip.hipDeviceSynchronize() != 0:
            raise RuntimeError("HIP synchronization failed")
        if hip.hipMemcpy(
            arrays[4].ctypes.data_as(ctypes.c_void_p), device_ptrs[4],
            int(arrays[4].nbytes), 2,
        ) != 0:
            raise RuntimeError("HIP output download failed")
        low_level = arrays[4].copy()
        np.testing.assert_allclose(low_level, expected, rtol=2e-2, atol=2e-2)

        start, stop = ctypes.c_void_p(), ctypes.c_void_p()
        for event in (start, stop):
            if hip.hipEventCreate(ctypes.byref(event)) != 0:
                raise RuntimeError("HIP event creation failed")
            events.append(event)
        for _ in range(20):
            launch()
        if hip.hipDeviceSynchronize() != 0:
            raise RuntimeError("HIP warmup synchronization failed")
        samples_us = []
        for _ in range(repeats):
            if hip.hipEventRecord(start, None) != 0:
                raise RuntimeError("HIP start event failed")
            for _ in range(iterations):
                launch()
            if hip.hipEventRecord(stop, None) != 0 or hip.hipEventSynchronize(stop) != 0:
                raise RuntimeError("HIP stop event failed")
            elapsed_ms = ctypes.c_float()
            if hip.hipEventElapsedTime(ctypes.byref(elapsed_ms), start, stop) != 0:
                raise RuntimeError("HIP elapsed-time query failed")
            samples_us.append(elapsed_ms.value * 1000.0 / iterations)
        return low_level, samples_us, grid, block
    finally:
        for event in events:
            hip.hipEventDestroy(event)
        for ptr in reversed(allocations):
            hip.hipFree(ptr)
        if module.value:
            hip.hipModuleUnload(module)


def measure(m: int = 17, n: int = 19, k: int = 64):
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("benchmark must run on the owning gfx1201 device")
    shape, weights, buffers, expected, ingest_ms, preserve_code_baseline = _inputs(m, n, k)
    compile_start_ns = time.perf_counter_ns()
    package, schedule_ir, tile_ir, target_ir = _compile_package(m, n, k)
    compiler_packaging_ms = (time.perf_counter_ns() - compile_start_ns) / 1e6
    np.testing.assert_array_equal(weights.projection_names, ("gate", "up"))
    resident, kernel_samples, grid, block = _device_resident_run(
        package, buffers, shape, expected=expected,
    )
    np.testing.assert_array_equal(resident, expected)
    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target},
        native_image=package.image,
        launch_descriptor=package.descriptor,
        tile_ir=tile_ir,
        target_ir=target_ir,
    )
    e2e_samples_ms = []
    for _ in range(7):
        buffers["output"].fill(0)
        start = time.perf_counter_ns()
        result = rt.launch(
            artifact, {"buffers": buffers, "scalars": dict(zip(("M", "N", "K"), shape))},
        )
        e2e_samples_ms.append((time.perf_counter_ns() - start) / 1e6)
        if not result.get("ok") or result.get("execution_kind") != "native_gpu":
            raise RuntimeError(f"packaged launch failed: {result}")
        np.testing.assert_array_equal(buffers["output"], expected)
    return {
        "work_item": "ROCM-NVFP4-INGEST-1",
        "sync_key": "ROCM-NVFP4-INGEST-1-2026-10-01",
        "revision": os.environ.get("TESSERA_SOURCE_REVISION", "source_snapshot"),
        "source_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "python/tessera/compiler/rocm_nvfp4_ingest.py",
                "benchmarks/rocm/benchmark_rocm_nvfp4_ingest_schedule.py",
            )
        },
        "compiler_sha256": hashlib.sha256(
            Path(os.environ["TESSERA_OPT"]).read_bytes()
        ).hexdigest(),
        "host": socket.gethostname(),
        "target": "rocm_gfx1201",
        "device_arch": rt._rocm_live_arch(),
        "shape_mnk": list(shape),
        "ingest_ms": ingest_ms,
        "preserve_code_baseline_projection_error": preserve_code_baseline,
        "compiler_packaging_ms": compiler_packaging_ms,
        "source_projection_names": list(weights.projection_names),
        "source_row_offsets": list(weights.row_offsets),
        "projection_error": [item.as_dict() for item in weights.metadata],
        "package_entry": package.descriptor.entry_symbol,
        "package_abi": package.descriptor.abi_id,
        "image_sha256": hashlib.sha256(package.image.payload).hexdigest(),
        "schedule_digest": hashlib.sha256(schedule_ir.encode()).hexdigest(),
        "tile_ir_digest": hashlib.sha256(tile_ir.encode()).hexdigest(),
        "target_ir_digest": hashlib.sha256(target_ir.encode()).hexdigest(),
        "grid": list(grid),
        "workgroup": list(block),
        "correctness": "passed_before_timing_and_after_each_end_to_end_sample",
        "kernel_event_us_samples": kernel_samples,
        "kernel_event_us_median": statistics.median(kernel_samples),
        "end_to_end_ms_samples": e2e_samples_ms,
        "end_to_end_ms_median": statistics.median(e2e_samples_ms),
        "timing_note": (
            "compiler_packaging_ms covers Graph/Schedule/Tile lowering, target "
            "lowering, and native image materialization once. Kernel timing uses "
            "persistent device allocations and HIP events around "
            "the package launch only. End-to-end timing calls runtime.launch per "
            "sample and includes descriptor validation, module load, allocations, "
            "input/output transfers, synchronization, and Python dispatch. Graph/"
            "Schedule/Tile compilation occurs once before either timing loop."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=17)
    parser.add_argument("--n", type=int, default=19)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    packet = json.dumps(measure(args.m, args.n, args.k), indent=2, sort_keys=True)
    if args.output is None:
        print(packet)
    else:
        args.output.write_text(packet + "\n")
