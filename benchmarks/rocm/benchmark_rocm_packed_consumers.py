#!/usr/bin/env python3
"""Exact gfx1151 packet for distinct physical signed-INT4 consumers.

Both rows consume the compiler-generated terminal packer's nibble layout
directly.  The dequant-GEMM row is deliberately separate from IU4 WMMA: it
loads signed nibbles, applies per-group scales, and accumulates f32 without
ever materializing or host-unpacking the logical int8 code tensor.
"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import numpy as np

from tessera import runtime as rt
from tessera.stdlib.quant import PackedQuantTensor, QuantScheme

_consumer_hsaco_cache: dict[str, bytes] = {}


def _build_consumer_hsaco(kind: str) -> bytes:
    cached = _consumer_hsaco_cache.get(kind)
    if cached is not None:
        return cached
    directive = (
        'module {\n  "tessera_rocm.int4_pack"() '
        f'{{name = "int4_{kind}", kind = "{kind}"}} : () -> ()\n}}\n'
    )
    hsaco = rt._build_rocm_elementwise_hsaco(
        "generate-rocm-int4-pack-kernel",
        directive,
        {},
        (rt._rocm_chip(), kind),
    )
    _consumer_hsaco_cache[kind] = hsaco
    return hsaco


def _memref(pointer: ctypes.c_void_p, size: int) -> list[Any]:
    return [
        ctypes.c_void_p(pointer.value), ctypes.c_void_p(pointer.value),
        ctypes.c_int64(0), ctypes.c_int64(size), ctypes.c_int64(1),
    ]


def _launch_consumer(
    kind: str, arrays: list[np.ndarray], scalar_args: list[int],
    output_index: int, work_items: int,
) -> np.ndarray:
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("live ROCm device required for packed consumer")
    hsaco = _build_consumer_hsaco(kind)
    module = ctypes.c_void_p()
    function = ctypes.c_void_p()
    device: list[ctypes.c_void_p] = []
    try:
        if hip.hipModuleLoadData(ctypes.byref(module), hsaco) != 0:
            raise RuntimeError(f"could not load packed {kind} HSACO")
        if hip.hipModuleGetFunction(
            ctypes.byref(function), module, f"int4_{kind}".encode()
        ) != 0:
            raise RuntimeError(f"packed {kind} symbol missing")
        for array in arrays:
            pointer = ctypes.c_void_p()
            if hip.hipMalloc(ctypes.byref(pointer), max(array.nbytes, 1)) != 0:
                raise RuntimeError(f"packed {kind} allocation failed")
            if array.nbytes and hip.hipMemcpy(
                pointer, array.ctypes.data_as(ctypes.c_void_p), array.nbytes, 1
            ) != 0:
                raise RuntimeError(f"packed {kind} input copy failed")
            device.append(pointer)
        launch_args: list[Any] = []
        for pointer, array in zip(device, arrays):
            launch_args.extend(_memref(pointer, int(array.size)))
        launch_args.extend(ctypes.c_int64(value) for value in scalar_args)
        packed_args = (ctypes.c_void_p * len(launch_args))()
        for index, value in enumerate(launch_args):
            packed_args[index] = ctypes.cast(
                ctypes.byref(value), ctypes.c_void_p
            )
        grid = max((work_items + 255) // 256, 1)
        rc = hip.hipModuleLaunchKernel(
            function, grid, 1, 1, 256, 1, 1, 0, None, packed_args, None
        )
        if rc != 0 or hip.hipDeviceSynchronize() != 0:
            raise RuntimeError(f"packed {kind} launch failed rc={rc}")
        output = np.empty_like(arrays[output_index])
        if output.nbytes and hip.hipMemcpy(
            output.ctypes.data_as(ctypes.c_void_p),
            device[output_index], output.nbytes, 2,
        ) != 0:
            raise RuntimeError(f"packed {kind} result copy failed")
        return output
    finally:
        for pointer in device:
            if pointer.value:
                hip.hipFree(pointer)
        if module.value:
            hip.hipModuleUnload(module)


def _packed_relu(codes: np.ndarray) -> np.ndarray:
    packed = rt._rocm_int4_storage_convert(codes, codes.size, "pack", np)
    output = np.zeros_like(packed)
    return _launch_consumer(
        "relu", [packed, output], [int(codes.size)], 1, output.size
    )


def _packed_sparse_gather(
    packed: np.ndarray, indices: np.ndarray, logical_elements: int
) -> np.ndarray:
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    output = np.zeros(indices.size, dtype=np.int8)
    return _launch_consumer(
        "sparse_gather", [packed, indices, output],
        [logical_elements, int(indices.size)], 2, indices.size,
    )


def _packed_cache_append(
    update: np.ndarray, cache: np.ndarray, byte_offset: int
) -> np.ndarray:
    update = np.ascontiguousarray(update, dtype=np.int8)
    cache = np.ascontiguousarray(cache, dtype=np.int8)
    return _launch_consumer(
        "cache_append", [update, cache],
        [byte_offset, int(update.size)], 1, update.size,
    )


def _measure(call: Callable[[], Any], warmup: int, reps: int) -> tuple[float, float, Any]:
    samples: list[float] = []
    result: Any = None
    for iteration in range(warmup + reps):
        start = time.perf_counter_ns()
        result = call()
        elapsed = (time.perf_counter_ns() - start) / 1.0e6
        if iteration >= warmup:
            samples.append(elapsed)
    ordered = sorted(samples)
    return (
        statistics.median(samples),
        ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        result,
    )


def _compiled_packed_weight(
    logical_codes: np.ndarray, scales: np.ndarray, group_size: int
) -> PackedQuantTensor:
    k, n = logical_codes.shape
    # The physical weight ABI is expert/output-major: [N, ceil(K/2)].
    transposed = np.ascontiguousarray(logical_codes.T, dtype=np.int8)
    packed = rt._rocm_int4_storage_convert(
        transposed, int(transposed.size), "pack", np
    ).view(np.uint8).reshape(n, (k + 1) // 2)
    return PackedQuantTensor(
        codes=packed,
        scales=np.ascontiguousarray(scales, dtype=np.float32),
        scheme=QuantScheme("int4", group_size=group_size),
        shape=(k, n),
    )


def record(warmup: int = 5, reps: int = 30) -> dict[str, object]:
    rng = np.random.default_rng(20260724)
    m, k, n, group_size = 33, 64, 29, 8
    x = rng.standard_normal((m, k)).astype(np.float32)
    codes = rng.integers(-8, 8, size=(k, n), dtype=np.int8)
    scales = rng.uniform(0.01, 0.2, size=(k // group_size, n)).astype(np.float32)
    packed = _compiled_packed_weight(codes, scales, group_size)
    artifact = rt.RuntimeArtifact(metadata={
        "target": "rocm",
        "compiler_path": "rocm_dequant_gemm_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": ["x", "packed_w"],
        "ops": [{"op_name": "tessera.dequant_matmul"}],
    })

    median, p95, result = _measure(
        lambda: rt.launch(artifact, (x, packed)), warmup, reps
    )
    if not result.get("ok") or result.get("execution_kind") != "native_gpu":
        raise RuntimeError(f"packed dequant-GEMM did not execute on gfx1151: {result}")
    expected_weight = np.empty((k, n), dtype=np.float32)
    for group in range(k // group_size):
        sl = slice(group * group_size, (group + 1) * group_size)
        expected_weight[sl] = (
            codes[sl].astype(np.float32) * scales[group][None, :]
        )
    reference = x @ expected_weight
    np.testing.assert_allclose(result["output"], reference, rtol=1e-5, atol=1e-5)
    elementwise_codes = rng.integers(-8, 8, size=65_537, dtype=np.int8)
    relu_median, relu_p95, relu_packed = _measure(
        lambda: _packed_relu(elementwise_codes), warmup, reps
    )
    relu_logical = rt._rocm_int4_storage_convert(
        relu_packed, elementwise_codes.size, "unpack", np
    )
    np.testing.assert_array_equal(
        relu_logical, np.maximum(elementwise_codes, 0)
    )

    sparse_codes = rng.integers(-8, 8, size=131_071, dtype=np.int8)
    sparse_packed = rt._rocm_int4_storage_convert(
        sparse_codes, sparse_codes.size, "pack", np
    )
    indices = rng.integers(
        0, sparse_codes.size, size=16_384, dtype=np.int64
    )
    gather_median, gather_p95, gathered = _measure(
        lambda: _packed_sparse_gather(
            sparse_packed, indices, sparse_codes.size
        ),
        warmup, reps,
    )
    np.testing.assert_array_equal(gathered, sparse_codes[indices])

    cache_codes = rng.integers(-8, 8, size=262_144, dtype=np.int8)
    update_codes = rng.integers(-8, 8, size=16_384, dtype=np.int8)
    packed_cache = rt._rocm_int4_storage_convert(
        cache_codes, cache_codes.size, "pack", np
    )
    packed_update = rt._rocm_int4_storage_convert(
        update_codes, update_codes.size, "pack", np
    )
    byte_offset = 37_000
    cache_median, cache_p95, updated_cache = _measure(
        lambda: _packed_cache_append(
            packed_update, packed_cache, byte_offset
        ),
        warmup, reps,
    )
    expected_cache = packed_cache.copy()
    expected_cache[byte_offset:byte_offset + packed_update.size] = packed_update
    np.testing.assert_array_equal(updated_cache, expected_cache)
    return {
        "schema": "tessera.rocm.packed-consumers.v1",
        "target": "gfx1151",
        "packing": "signed_twos_complement_low_nibble_first",
        "rows": [{
            "producer_route": "compiled_gfx1151_int4_pack",
            "consumer_route": "compiled_rocdl_int4_dequant_gemm",
            "consumer_family": "dequant_matmul",
            "shape": [m, k, n],
            "logical_code_bytes": int(codes.nbytes),
            "packed_code_bytes": int(packed.codes.nbytes),
            "host_unpack_or_repack": False,
            "execution_kind": result["execution_kind"],
            "numerical_oracle": "groupwise_dequant_then_f32_matmul",
            "timing_domain": "host_wall_operation_total_packed_consumer",
            "median_ms": median,
            "p95_ms": p95,
        }, {
            "producer_route": "compiled_gfx1151_int4_pack",
            "consumer_route": "compiled_rocdl_packed_int4_relu",
            "consumer_family": "elementwise",
            "logical_elements": int(elementwise_codes.size),
            "host_unpack_or_repack": False,
            "numerical_oracle": "signed_nibblewise_max_zero",
            "timing_domain": "host_wall_operation_total_packed_consumer",
            "median_ms": relu_median,
            "p95_ms": relu_p95,
        }, {
            "producer_route": "compiled_gfx1151_int4_pack",
            "consumer_route": "compiled_rocdl_packed_int4_sparse_gather",
            "consumer_family": "sparse",
            "logical_elements": int(sparse_codes.size),
            "selected_elements": int(indices.size),
            "host_unpack_or_repack": False,
            "numerical_oracle": "logical_index_signed_nibble_gather",
            "timing_domain": "host_wall_operation_total_packed_consumer",
            "median_ms": gather_median,
            "p95_ms": gather_p95,
        }, {
            "producer_route": "compiled_gfx1151_int4_pack",
            "consumer_route": "compiled_rocdl_packed_int4_cache_append",
            "consumer_family": "cache",
            "packed_cache_bytes": int(packed_cache.size),
            "packed_update_bytes": int(packed_update.size),
            "byte_offset": byte_offset,
            "host_unpack_or_repack": False,
            "numerical_oracle": "packed_byte_range_replace",
            "timing_domain": "host_wall_operation_total_packed_consumer",
            "median_ms": cache_median,
            "p95_ms": cache_p95,
        }],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = record(args.warmup, args.reps)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
