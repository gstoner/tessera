"""Exact-device canonical streaming-attention correctness and timing.

The default case enters through rank-4 ``tessera.flash_attn``, lowers through
the shared KV-block recurrence, and reaches ROCm without rebuilding semantics
in ``tile.attention_kernel``. It intentionally uses host-wall operation-total
timing because WSL HIP events may report zero-duration samples.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

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
from tessera.runtime import (  # noqa: E402
    _load_hip_for_launch,
    _submit_rocm_gfx1151_native,
)

KERNEL_WALL_BASELINE_MS = 0.097763
KERNEL_WALL_MAX_REGRESSION = 0.10


def _module(b: int, hq: int, hkv: int, sq: int, sk: int, d: int) -> GraphIRModule:
    q = IRType(f"tensor<{b}x{hq}x{sq}x{d}xf16>", tuple(map(str, (b, hq, sq, d))), "fp16")
    k = IRType(f"tensor<{b}x{hkv}x{sk}x{d}xf16>", tuple(map(str, (b, hkv, sk, d))), "fp16")
    v = IRType(f"tensor<{b}x{hkv}x{sk}x{d}xf16>", tuple(map(str, (b, hkv, sk, d))), "fp16")
    o = IRType(f"tensor<{b}x{hq}x{sq}x{d}xf32>", tuple(map(str, (b, hq, sq, d))), "fp32")
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="gfx1151_attention_carrier",
                args=[IRArg("q", q), IRArg("k", k), IRArg("v", v)],
                result_types=[o],
                body=[
                    IROp(
                        result="o",
                        op_name="tessera.flash_attn",
                        operands=["%q", "%k", "%v"],
                        operand_types=[str(q), str(k), str(v)],
                        result_type=str(o),
                        kwargs={
                            "scale": d**-0.5,
                            "causal": True,
                            "window": (8, 0),
                            "softcap": 0.0,
                            "dropout_p": 0.0,
                        },
                    )
                ],
                return_values=["%o"],
            )
        ]
    )


def _reference(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    b, hq, sq, d = q.shape
    hkv, sk = k.shape[1:3]
    result = np.empty((b, hq, sq, d), dtype=np.float32)
    scale = d**-0.5
    for batch in range(b):
        for head in range(hq):
            kv_head = head // (hq // hkv)
            scores = (q[batch, head].astype(np.float32) @ k[batch, kv_head].astype(np.float32).T) * scale
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


def _memref_arguments(pointer: ctypes.c_void_p, size: int) -> list[Any]:
    return [
        ctypes.c_void_p(pointer.value),
        ctypes.c_void_p(pointer.value),
        ctypes.c_int64(0),
        ctypes.c_int64(size),
        ctypes.c_int64(1),
    ]


@contextmanager
def _resident_attention_launch(
    package: Any,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    output: np.ndarray,
) -> Iterator[tuple[Any, ctypes.CDLL]]:
    """Keep the module and buffers resident around synchronized launch timing."""
    hip = _load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("libamdhip64.so or a usable gfx1151 device is unavailable")

    module = ctypes.c_void_p()
    image_storage = ctypes.create_string_buffer(package.image.payload)
    if hip.hipModuleLoadData(ctypes.byref(module), ctypes.cast(image_storage, ctypes.c_void_p)) != 0:
        raise RuntimeError("gfx1151 attention HSACO module load failed")

    function = ctypes.c_void_p()
    device_buffers: list[ctypes.c_void_p] = []
    try:
        if (
            hip.hipModuleGetFunction(
                ctypes.byref(function),
                module,
                package.descriptor.entry_symbol.encode(),
            )
            != 0
        ):
            raise RuntimeError(f"gfx1151 attention symbol {package.descriptor.entry_symbol!r} not found")

        device_q, device_k, device_v, device_o = (ctypes.c_void_p() for _ in range(4))
        arrays = (q, k, v, output)
        for device, array in zip(
            (device_q, device_k, device_v, device_o),
            arrays,
            strict=True,
        ):
            if hip.hipMalloc(ctypes.byref(device), int(array.nbytes)) != 0:
                raise RuntimeError("gfx1151 resident attention hipMalloc failed")
            device_buffers.append(device)
        for device, array in zip((device_q, device_k, device_v), (q, k, v), strict=True):
            if (
                hip.hipMemcpy(
                    device,
                    array.ctypes.data_as(ctypes.c_void_p),
                    int(array.nbytes),
                    1,
                )
                != 0
            ):
                raise RuntimeError("gfx1151 resident attention host-to-device copy failed")

        b, hq, hkv, sq, sk, _d, _dv = (int(value) for value in package.descriptor.provenance["shape"])
        arguments: list[Any] = []
        for device, array in zip((device_q, device_k, device_v), (q, k, v), strict=True):
            arguments.extend(_memref_arguments(device, int(array.size)))
        arguments.extend(_memref_arguments(device_o, int(output.size)))
        arguments.extend(
            (
                ctypes.c_int64(sq),
                ctypes.c_int64(sk),
                ctypes.c_float(float(package.descriptor.provenance["scale"])),
                ctypes.c_int64(int(bool(package.descriptor.provenance["causal"]))),
            )
        )
        if hq != hkv:
            arguments.extend(
                (
                    ctypes.c_int64(hq),
                    ctypes.c_int64(hq // hkv),
                )
            )
        window_left = int(package.descriptor.provenance["window_left"])
        if window_left >= 0:
            arguments.append(ctypes.c_int64(window_left + 1))
        softcap = float(package.descriptor.provenance["softcap"])
        if softcap > 0.0:
            arguments.append(ctypes.c_float(softcap))

        argument_array = (ctypes.c_void_p * len(arguments))()
        for index, value in enumerate(arguments):
            argument_array[index] = ctypes.cast(ctypes.byref(value), ctypes.c_void_p)
        grid_x = (sq + 15) // 16
        grid_y = b * hq

        def launch_and_synchronize() -> None:
            rc = hip.hipModuleLaunchKernel(
                function,
                grid_x,
                grid_y,
                1,
                32,
                1,
                1,
                0,
                None,
                argument_array,
                None,
            )
            if rc != 0:
                raise RuntimeError(f"gfx1151 resident attention launch failed rc={rc}")
            if hip.hipDeviceSynchronize() != 0:
                raise RuntimeError("gfx1151 resident attention synchronization failed")

        # Keep argument objects, the serialized image, and device buffers alive
        # until every timed launch and the final result copy have completed.
        yield launch_and_synchronize, hip
        if (
            hip.hipMemcpy(
                output.ctypes.data_as(ctypes.c_void_p),
                device_o,
                int(output.nbytes),
                2,
            )
            != 0
        ):
            raise RuntimeError("gfx1151 resident attention device-to-host copy failed")
    finally:
        for device in reversed(device_buffers):
            hip.hipFree(device)
        unload = getattr(hip, "hipModuleUnload", None)
        if unload is not None and module.value:
            unload(module)


def _kernel_wall_samples(
    package: Any,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    output: np.ndarray,
    *,
    warmup: int,
    iterations: int,
) -> list[float]:
    samples: list[float] = []
    with _resident_attention_launch(package, q, k, v, output) as (launch_and_synchronize, _hip):
        for _ in range(warmup):
            launch_and_synchronize()
        for _ in range(iterations):
            started_ns = time.perf_counter_ns()
            launch_and_synchronize()
            samples.append((time.perf_counter_ns() - started_ns) / 1_000_000.0)
    return samples


def _timing_domains(
    *,
    cold_operation_total_ms: float,
    operation_total_samples_ms: list[float],
    kernel_wall_samples_ms: list[float],
    warmup: int,
) -> dict[str, Any]:
    return {
        "operation_total": {
            "clock": "time.perf_counter",
            "cold_ms": cold_operation_total_ms,
            "median_ms": statistics.median(operation_total_samples_ms),
            "samples_ms": operation_total_samples_ms,
            "includes": [
                "module_load",
                "device_allocation",
                "host_to_device",
                "hipModuleLaunchKernel",
                "hipDeviceSynchronize",
                "device_to_host",
                "cleanup",
            ],
        },
        "kernel_wall": {
            "clock": "time.perf_counter_ns",
            "median_ms": statistics.median(kernel_wall_samples_ms),
            "samples_ms": kernel_wall_samples_ms,
            "warmup_iterations": warmup,
            "launch_api": "hipModuleLaunchKernel",
            "completion_api": "hipDeviceSynchronize",
            "resident_module": True,
            "resident_buffers": True,
            "includes": [
                "host_launch",
                "device_execution",
                "device_completion_synchronization",
            ],
            "selector_eligible": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    shape = (1, 4, 2, 17, 19, 64)
    rng = np.random.default_rng(20260726)
    q = rng.normal(0.0, 0.25, (shape[0], shape[1], shape[3], shape[5])).astype(np.float16)
    k = rng.normal(0.0, 0.25, (shape[0], shape[2], shape[4], shape[5])).astype(np.float16)
    v = rng.normal(0.0, 0.25, (shape[0], shape[2], shape[4], shape[5])).astype(np.float16)
    output = np.empty((shape[0], shape[1], shape[3], shape[5]), dtype=np.float32)

    started = time.perf_counter()
    package = package_attention(_module(*shape), pipeline_name="tessera-lower-to-rocm")
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
    }
    started = time.perf_counter()
    _submit_rocm_gfx1151_native(package.image, package.descriptor, buffers, scalars, None)
    cold_operation_total_ms = (time.perf_counter() - started) * 1000.0
    samples = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        _submit_rocm_gfx1151_native(package.image, package.descriptor, buffers, scalars, None)
        samples.append((time.perf_counter() - started) * 1000.0)
    kernel_wall_samples = _kernel_wall_samples(
        package,
        q,
        k,
        v,
        output,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    reference = _reference(q, k, v)
    error = float(np.max(np.abs(output - reference)))
    timing_domains = _timing_domains(
        cold_operation_total_ms=cold_operation_total_ms,
        operation_total_samples_ms=samples,
        kernel_wall_samples_ms=kernel_wall_samples,
        warmup=args.warmup,
    )
    kernel_wall_median_ms = statistics.median(kernel_wall_samples)
    kernel_wall_limit_ms = (
        KERNEL_WALL_BASELINE_MS * (1.0 + KERNEL_WALL_MAX_REGRESSION)
    )
    record = {
        "schema": "tessera.rocm.canonical_streaming_attention.benchmark.v1",
        "device": os.environ.get("TESSERA_ROCM_CHIP", "gfx1151"),
        "shape": {
            "B": shape[0],
            "Hq": shape[1],
            "Hkv": shape[2],
            "Sq": shape[3],
            "Sk": shape[4],
            "D": shape[5],
        },
        "semantic_route": package.descriptor.provenance["semantic_route"],
        "schedule": package.descriptor.provenance["schedule"],
        "compile_ms": compile_ms,
        "cold_operation_total_ms": cold_operation_total_ms,
        "operation_total_median_ms": statistics.median(samples),
        "operation_total_samples_ms": samples,
        "kernel_wall_median_ms": kernel_wall_median_ms,
        "kernel_wall_samples_ms": kernel_wall_samples,
        "kernel_wall_ratchet": {
            "baseline_ms": KERNEL_WALL_BASELINE_MS,
            "max_regression_fraction": KERNEL_WALL_MAX_REGRESSION,
            "limit_ms": kernel_wall_limit_ms,
            "observed_ratio": kernel_wall_median_ms / KERNEL_WALL_BASELINE_MS,
            "passed": kernel_wall_median_ms <= kernel_wall_limit_ms,
        },
        "timing_domains": timing_domains,
        "max_abs_error": error,
        "tolerance": 0.035,
        "passed": (
            error <= 0.035 and kernel_wall_median_ms <= kernel_wall_limit_ms
        ),
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
