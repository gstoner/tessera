"""Record strict-v2 source reports for the Apple low-precision Tile route.

This is the owner of the ``tile_simdgroup_gemm`` comparison: it pairs the
source-backed f16/bf16 simdgroup ABI against the same-dtype MPS incumbent.
Run it twice in fresh processes, then seal the reports with
``seal_strict_route_ledger.py``.  Its JSON is deliberately the shared route
report schema rather than a Tile-private promotion result, so production
selection can verify context, paired trials, and resource provenance.
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

from tessera._apple_gpu_dispatch import apple_gpu_runtime, bind_registered
from tessera.compiler.apple_route_selector import (
    ROUTE_REPORT_SCHEMA_VERSION,
    live_apple_route_context,
)
from tessera.compiler.apple_target import AppleGPUArch, AppleGPUTargetProfile
from tessera.compiler.msl_gemm_emit import (
    dispatch_apple_simdgroup_tile_f16,
    materialize_apple_simdgroup_tile_msl,
)


DEFAULT_SHAPES = ("8x8x8", "32x16x32", "127x63x129", "256x256x256")


def _shape(spec: str) -> tuple[int, int, int]:
    try:
        values = tuple(int(part) for part in spec.lower().split("x"))
    except ValueError as exc:
        raise ValueError(f"shape must be MxKxN: {spec!r}") from exc
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError(f"shape must be positive MxKxN: {spec!r}")
    return values


def _to_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _from_bf16(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _mps_call(dtype: str, a: np.ndarray, b: np.ndarray) -> Callable[[], tuple[np.ndarray, bool, dict[str, Any]]]:
    symbol = f"tessera_apple_gpu_mps_matmul_{'f16' if dtype == 'f16' else 'bf16'}_status"
    dispatch = bind_registered(symbol)
    if dispatch is None:
        raise RuntimeError(f"required MPS status ABI unavailable: {symbol}")
    runtime = apple_gpu_runtime()
    timing = getattr(runtime, "tessera_apple_gpu_tile_last_device_time_ns", None)
    supported = getattr(runtime, "tessera_apple_gpu_tile_counter_sampling_supported", None)
    if timing is not None:
        timing.restype = ctypes.c_int64
    if supported is not None:
        supported.restype = ctypes.c_int32
    m, k = a.shape
    n = b.shape[1]

    def call() -> tuple[np.ndarray, bool, dict[str, Any]]:
        out = np.empty((m, n), dtype=np.uint16)
        ptr = ctypes.POINTER(ctypes.c_uint16)
        native = bool(dispatch(a.ctypes.data_as(ptr), b.ctypes.data_as(ptr),
                               out.ctypes.data_as(ptr), m, n, k))
        elapsed = None if timing is None else int(timing())
        capable = None if supported is None else bool(supported())
        value = out.view(np.float16).astype(np.float32) if dtype == "f16" else _from_bf16(out)
        return value, native, {
            "device_time_ns": elapsed if elapsed is not None and elapsed > 0 else None,
            "counter_sampling_supported": capable,
            "counter_timestamp_delta": None,
            "resources": {"api": "MPSMatrixMultiplication", "threadgroup_resources": None},
        }
    return call


def _tile_call(dtype: str, a: np.ndarray, b: np.ndarray) -> Callable[[], tuple[np.ndarray | None, bool, Any]]:
    artifact = materialize_apple_simdgroup_tile_msl(
        AppleGPUTargetProfile(AppleGPUArch.APPLE7), dtype, 32, 32, 16)
    return lambda: dispatch_apple_simdgroup_tile_f16(artifact, a, b, return_provenance=True)


def _record_value(record: Any, name: str) -> Any:
    return record[name] if isinstance(record, dict) else getattr(record, name)


def _median(call: Callable[[], tuple[np.ndarray | None, bool, Any]], reps: int) -> tuple[np.ndarray | None, bool, dict[str, Any]]:
    samples: list[int] = []
    device: list[int] = []
    counters: list[int] = []
    native = True
    output = None
    resources: list[Any] = []
    capabilities: set[bool | None] = set()
    for _ in range(reps):
        started = time.perf_counter_ns()
        output, ran, record = call()
        samples.append(time.perf_counter_ns() - started)
        native &= ran
        elapsed = _record_value(record, "device_time_ns")
        if elapsed is not None and elapsed > 0:
            device.append(int(elapsed))
        delta = _record_value(record, "counter_timestamp_delta")
        if delta is not None and delta > 0:
            counters.append(int(delta))
        resources.append(_record_value(record, "resources"))
        capabilities.add(_record_value(record, "counter_sampling_supported"))
    return output, native, {
        "end_to_end_median_ns": int(statistics.median(samples)),
        "device_time_median_ns": int(statistics.median(device)) if len(device) == reps else None,
        "device_time_samples": len(device),
        "device_time_coverage": len(device) / reps,
        "timing_source": "owned_command_buffer",
        "counter_sampling_supported": capabilities.pop() if len(capabilities) == 1 else None,
        "counter_timestamp_delta_median": int(statistics.median(counters)) if len(counters) == reps else None,
        "resources": resources[0] if resources and all(item == resources[0] for item in resources) else None,
    }


def _paired(incumbent: Callable[[], tuple[np.ndarray | None, bool, Any]], candidate: Callable[[], tuple[np.ndarray | None, bool, Any]], *, reps: int, trials: int) -> tuple[tuple[np.ndarray | None, bool, dict[str, Any]], tuple[np.ndarray | None, bool, dict[str, Any]]]:
    incumbent()
    candidate()
    results: dict[str, list[tuple[np.ndarray | None, bool, dict[str, Any]]]] = {"mps": [], "simdgroup_matrix": []}
    for trial in range(trials):
        order = (("mps", incumbent), ("simdgroup_matrix", candidate))
        if trial % 2:
            order = tuple(reversed(order))
        for route, call in order:
            results[route].append(_median(call, reps))

    def combine(values: list[tuple[np.ndarray | None, bool, dict[str, Any]]]):
        telemetry = [item[2] for item in values]
        return values[-1][0], all(item[1] for item in values), {
            "end_to_end_median_ns": int(statistics.median(item["end_to_end_median_ns"] for item in telemetry)),
            "device_time_median_ns": (int(statistics.median(item["device_time_median_ns"] for item in telemetry)) if all(item["device_time_median_ns"] is not None for item in telemetry) else None),
            "device_time_samples": sum(item["device_time_samples"] for item in telemetry),
            "device_time_coverage": sum(item["device_time_samples"] for item in telemetry) / (reps * trials),
            "timing_source": "owned_command_buffer",
            "counter_sampling_supported": telemetry[0]["counter_sampling_supported"] if all(item["counter_sampling_supported"] == telemetry[0]["counter_sampling_supported"] for item in telemetry) else None,
            "counter_timestamp_delta_median": (int(statistics.median(item["counter_timestamp_delta_median"] for item in telemetry)) if all(item["counter_timestamp_delta_median"] is not None for item in telemetry) else None),
            "resources": telemetry[0]["resources"] if all(item["resources"] == telemetry[0]["resources"] for item in telemetry) else None,
            "paired_trial_end_to_end_medians_ns": [item["end_to_end_median_ns"] for item in telemetry],
            "paired_trial_device_medians_ns": [item["device_time_median_ns"] for item in telemetry],
        }
    return combine(results["mps"]), combine(results["simdgroup_matrix"])


def characterize(shapes: tuple[tuple[int, int, int], ...], *, reps: int, trials: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for dtype in ("f16", "bf16"):
        for m, k, n in shapes:
            rng = np.random.default_rng(m * 1000003 + k * 1009 + n)
            source_a = rng.standard_normal((m, k), dtype=np.float32) * 0.1
            source_b = rng.standard_normal((k, n), dtype=np.float32) * 0.1
            a, b = (source_a.astype(np.float16), source_b.astype(np.float16)) if dtype == "f16" else (_to_bf16(source_a), _to_bf16(source_b))
            reference = a.astype(np.float32) @ b.astype(np.float32) if dtype == "f16" else _from_bf16(a) @ _from_bf16(b)
            mps, tile = _paired(_mps_call(dtype, a, b), _tile_call(dtype, a, b), reps=reps, trials=trials)
            for route, result in (("mps", mps), ("simdgroup_matrix", tile)):
                output, native, telemetry = result
                rows.append({
                    "backend": "apple_gpu", "op": "matmul", "shape": f"{m}x{k}x{n}",
                    "dtype": dtype, "device": "apple7", "route": route, "reps": reps * trials,
                    "trials": trials, "native_dispatched": native,
                    "numerically_validated": output is not None and bool(np.allclose(output, reference, rtol=1e-2, atol=1e-2)),
                    "telemetry": telemetry,
                })
    return {"schema_version": ROUTE_REPORT_SCHEMA_VERSION, "context": live_apple_route_context().as_mapping(), "runs": rows, "paired_trials": trials}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=list(DEFAULT_SHAPES))
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.reps < 2 or args.trials < 3:
        parser.error("--reps must be at least 2 and --trials must be at least 3")
    args.output.write_text(json.dumps(characterize(tuple(_shape(s) for s in args.shapes), reps=args.reps, trials=args.trials), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
