"""Record canonical NVIDIA-E2E-2 matmul timing and resource evidence.

Each dtype/shape row is compiled through the Graph -> Tile -> NVIDIA -> PTX
image seam.  Two independent runs retain repeated medians for CUDA-event
kernel time and allocation/copy-inclusive descriptor launch time.  This is an
evidence recorder only; it never changes a production selector.
"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
OUT = ROOT / "benchmarks/baselines/nvidia_sm120_e2e_spine_dtype_matrix.json"
SHAPES = ((256, 256, 256), (127, 259, 63))
STORAGES = (
    "fp64",
    "fp16",
    "bf16",
    "tf32",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp6_e2m3",
    "fp6_e3m2",
    "fp4_e2m1",
    "int8",
)


def _module(storage: str, shape: tuple[int, int, int]):
    from tessera.compiler.graph_ir import (
        GraphIRFunction,
        GraphIRModule,
        IRArg,
        IROp,
        IRType,
    )
    from tessera.compiler.primitive_coverage import NumericPolicy

    m, n, k = shape
    graph_dtype, ir_dtype = {
        "fp64": ("fp64", "f64"),
        "fp16": ("fp16", "f16"),
        "bf16": ("bf16", "bf16"),
        "tf32": ("fp32", "f32"),
        "fp8_e4m3": ("fp8_e4m3", "f8E4M3FN"),
        "fp8_e5m2": ("fp8_e5m2", "f8E5M2"),
        "fp6_e2m3": ("fp6_e2m3", "!tessera.fp6_e2m3"),
        "fp6_e3m2": ("fp6_e3m2", "!tessera.fp6_e3m2"),
        "fp4_e2m1": ("fp4_e2m1", "!tessera.fp4_e2m1"),
        "int8": ("int8", "i8"),
    }[storage]
    result_dtype, result_ir = {
        "fp64": ("fp64", "f64"),
        "int8": ("int32", "i32"),
    }.get(storage, ("fp32", "f32"))
    a = IRType(f"tensor<{m}x{k}x{ir_dtype}>", (str(m), str(k)), graph_dtype)
    b = IRType(f"tensor<{k}x{n}x{ir_dtype}>", (str(k), str(n)), graph_dtype)
    c = IRType(
        f"tensor<{m}x{n}x{result_ir}>",
        (str(m), str(n)),
        result_dtype,
    )
    block_scaled = storage in {"fp6_e2m3", "fp6_e3m2", "fp4_e2m1"}
    scale_k = (k + 31) // 32
    sa = IRType(f"tensor<{m}x{scale_k}xi8>", (str(m), str(scale_k)), "uint8")
    sb = IRType(f"tensor<{scale_k}x{n}xi8>", (str(scale_k), str(n)), "uint8")
    op = IROp(
        result="c",
        op_name="tessera.matmul",
        operands=["%a", "%b"],
        operand_types=[str(a), str(b)],
        result_type=str(c),
        numeric_policy=(NumericPolicy(storage="fp32", accum="fp32", math_mode="tf32") if storage == "tf32" else None),
        kwargs=({"scale_a": "%scale_a", "scale_b": "%scale_b"} if block_scaled else {}),
    )
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name=f"e2e_spine_{storage}",
                args=(
                    [IRArg("a", a), IRArg("b", b), IRArg("scale_a", sa), IRArg("scale_b", sb)]
                    if block_scaled
                    else [IRArg("a", a), IRArg("b", b)]
                ),
                result_types=[c],
                body=[op],
                return_values=["%c"],
            )
        ]
    )


def _pack_nibbles(codes: np.ndarray, axis: int) -> np.ndarray:
    if codes.shape[axis] % 2:
        padding = [(0, 0)] * codes.ndim
        padding[axis] = (0, 1)
        codes = np.pad(codes, padding)
    lo = np.take(codes, np.arange(0, codes.shape[axis], 2), axis=axis)
    hi = np.take(codes, np.arange(1, codes.shape[axis], 2), axis=axis)
    return np.ascontiguousarray(lo | (hi << np.uint8(4)))


def _arrays(storage: str, shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    import ml_dtypes

    m, n, k = shape
    rng = np.random.default_rng(120_202 + m + n + k)
    if storage in {"fp6_e2m3", "fp6_e3m2", "fp4_e2m1"}:
        dtype = {
            "fp6_e2m3": ml_dtypes.float6_e2m3fn,
            "fp6_e3m2": ml_dtypes.float6_e3m2fn,
            "fp4_e2m1": ml_dtypes.float4_e2m1fn,
        }[storage]
        a_codes = (rng.standard_normal((m, k)) * 1.25).astype(dtype).view(np.uint8)
        b_codes = (rng.standard_normal((k, n)) * 1.25).astype(dtype).view(np.uint8)
        if storage == "fp4_e2m1":
            a = _pack_nibbles(a_codes, 1)
            b = _pack_nibbles(b_codes, 0)
        else:
            a, b = np.ascontiguousarray(a_codes), np.ascontiguousarray(b_codes)
        scale_k = (k + 31) // 32
        scale_codes = np.asarray([126, 127, 128], np.uint8)
        scale_a = np.ascontiguousarray(scale_codes[(np.arange(m)[:, None] + np.arange(scale_k)[None, :]) % 3])
        scale_b = np.ascontiguousarray(scale_codes[(2 * np.arange(scale_k)[:, None] + np.arange(n)[None, :]) % 3])
        return {
            "a": a,
            "b": b,
            "scale_a": scale_a,
            "scale_b": scale_b,
            "c": np.zeros((m, n), dtype=np.float32),
        }
    if storage == "int8":
        a = rng.integers(-7, 8, size=(m, k), dtype=np.int8)
        b = rng.integers(-7, 8, size=(k, n), dtype=np.int8)
        output = np.zeros((m, n), dtype=np.int32)
    else:
        dtype = {
            "fp64": np.float64,
            "fp16": np.float16,
            "bf16": ml_dtypes.bfloat16,
            "tf32": np.float32,
            "fp8_e4m3": ml_dtypes.float8_e4m3fn,
            "fp8_e5m2": ml_dtypes.float8_e5m2,
        }[storage]
        scale = 0.125 if storage != "fp8_e5m2" else 0.25
        a = (rng.standard_normal((m, k)) * scale).astype(dtype)
        b = (rng.standard_normal((k, n)) * scale).astype(dtype)
        output = np.zeros(
            (m, n),
            dtype=np.float64 if storage == "fp64" else np.float32,
        )
    return {"a": np.ascontiguousarray(a), "b": np.asfortranarray(b), "c": output}


def _median_delta(first: float, second: float) -> float:
    return abs(first - second) / min(first, second) if min(first, second) else 0.0


def record(
    *,
    samples: int,
    device_reps: int,
    e2e_reps: int,
    warmup: int,
) -> dict[str, Any]:
    from benchmarks.nvidia._clock_conditioning import condition_sm120
    from tessera.compiler.canonical_compile import compile_result_from_bundle
    from tessera.compiler.driver import compile_graph_module
    from tessera.compiler import nvidia_native
    from tessera import runtime as rt

    device = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    rows: list[dict[str, Any]] = []
    lib = rt._load_nvidia_ptx_launch()
    if lib is None:
        raise RuntimeError("libtessera_nvidia_ptx_launch.so is unavailable")

    for storage in STORAGES:
        for shape in SHAPES:
            module = _module(storage, shape)
            nvidia_native._cache.clear()
            cold_start = time.perf_counter()
            bundle = compile_graph_module(
                module,
                source_origin="NVIDIA-E2E-2",
                target="nvidia_sm120",
                options={"package_native": True},
                enable_tool_validation=False,
            )
            cold_ms = (time.perf_counter() - cold_start) * 1e3
            warm_start = time.perf_counter()
            warm = compile_graph_module(
                module,
                source_origin="NVIDIA-E2E-2",
                target="nvidia_sm120",
                options={"package_native": True},
                enable_tool_validation=False,
            )
            warm_ms = (time.perf_counter() - warm_start) * 1e3
            assert bundle.native_image and bundle.launch_descriptor
            assert warm.native_image and warm.launch_descriptor
            assert bundle.native_image.image_digest == warm.native_image.image_digest
            artifact = compile_result_from_bundle(
                bundle,
                module=module,
            ).to_runtime_artifact()
            arrays = _arrays(storage, shape)
            m, n, k = shape
            bindings = {
                **arrays,
                "M": m,
                "N": n,
                "K": k,
            }
            smoke = rt.launch(artifact, bindings)
            if not smoke["ok"]:
                raise RuntimeError(str(smoke.get("reason")))

            entry = bundle.launch_descriptor.entry_symbol.encode()
            ordered = sorted(bundle.launch_descriptor.buffers, key=lambda item: item.ordinal)
            raw = (ctypes.c_void_p * len(ordered))(*(int(arrays[item.name].ctypes.data) for item in ordered))
            dims = (ctypes.c_int64 * 3)(m, n, k)
            disjoint: list[dict[str, list[float]]] = [
                {"device": [], "end_to_end": []},
                {"device": [], "end_to_end": []},
            ]
            # Alternate disjoint cohorts sample-by-sample so clock/thermal drift
            # cannot alias an entire retained run while observations remain
            # independent between run 0 and run 1.
            for _sample in range(samples):
                cohort_order = (0, 1) if _sample % 2 == 0 else (1, 0)
                for run_index in cohort_order:
                    condition_sm120(reps=20)
                    latency = ctypes.c_float()
                    rc = lib.tessera_nvidia_ptx_benchmark(
                        entry,
                        raw,
                        len(ordered),
                        dims,
                        3,
                        warmup,
                        device_reps,
                        ctypes.byref(latency),
                    )
                    if rc:
                        raise RuntimeError(f"device benchmark returned rc={rc}")
                    disjoint[run_index]["device"].append(float(latency.value))
                    # The first allocation/copy-inclusive launch is a lifecycle
                    # probe, not steady-state evidence. Discard it, then
                    # amortize the retained sample across the requested batch.
                    discarded = rt.launch(artifact, bindings)
                    if not discarded["ok"]:
                        raise RuntimeError(str(discarded.get("reason")))
                    start = time.perf_counter()
                    for _ in range(e2e_reps):
                        result = rt.launch(artifact, bindings)
                        if not result["ok"]:
                            raise RuntimeError(str(result.get("reason")))
                    disjoint[run_index]["end_to_end"].append((time.perf_counter() - start) * 1e3 / e2e_reps)
            runs: list[dict[str, Any]] = [
                {
                    "device_event_ms": statistics.median(run["device"]),
                    "end_to_end_ms": statistics.median(run["end_to_end"]),
                    "device_samples_ms": run["device"],
                    "end_to_end_samples_ms": run["end_to_end"],
                }
                for run in disjoint
            ]
            resources = bundle.native_image.resource_record
            rows.append(
                {
                    "storage": storage,
                    "shape": list(shape),
                    "entry": bundle.launch_descriptor.entry_symbol,
                    "abi_id": bundle.launch_descriptor.abi_id,
                    "image_digest": bundle.native_image.image_digest,
                    "compile": {
                        "cold_ms": cold_ms,
                        "warm_ms": warm_ms,
                        "cold_state": bundle.native_image.compile_state,
                        "warm_state": warm.native_image.compile_state,
                    },
                    "resources": resources.to_dict() if resources else None,
                    "runs": runs,
                    "stability": {
                        "device_fraction": _median_delta(
                            float(runs[0]["device_event_ms"]),
                            float(runs[1]["device_event_ms"]),
                        ),
                        "end_to_end_fraction": _median_delta(
                            float(runs[0]["end_to_end_ms"]),
                            float(runs[1]["end_to_end_ms"]),
                        ),
                        "policy_fraction": 0.04,
                    },
                    "selector_changed": False,
                }
            )
    return {
        "schema": "tessera.nvidia.e2e-spine-dtype-matrix.v1",
        "work_item": "NVIDIA-E2E-2",
        "device": device,
        "method": {
            "runs": 2,
            "samples_per_run": samples,
            "device_repetitions_per_sample": device_reps,
            "warmup": warmup,
            "end_to_end_repetitions_per_sample": e2e_reps,
            "discarded_end_to_end_launches_per_sample": 1,
            "amortized_launches_per_end_to_end_sample": e2e_reps,
            "sampling": "sample_interleaved_disjoint_cohorts",
            "clock_conditioning": "resident_tf32_gemm_before_each_disjoint_cohort_sample",
            "timing_domains": ["device_event", "end_to_end"],
        },
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--device-reps", type=int, default=100)
    parser.add_argument("--e2e-reps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    payload = record(
        samples=args.samples,
        device_reps=args.device_reps,
        e2e_reps=args.e2e_reps,
        warmup=args.warmup,
    )
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
