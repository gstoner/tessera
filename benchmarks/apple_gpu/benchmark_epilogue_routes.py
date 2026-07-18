"""Paired native Apple GPU characterization for fused matmul epilogues.

The candidate is one synthesized Metal dispatch.  The incumbent is the same
operation composed from an MPS matmul and MPSGraph pointwise dispatches.  Each
row retains end-to-end and summed command-buffer device time, numerical proof,
and every per-dispatch runtime resource record.  Reports use the common Apple
route schema and are intended to be aggregated from two fresh processes.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import numpy as np

from tessera._apple_gpu_dispatch import (
    bind_registered,
    clear_dispatch_telemetry,
    read_dispatch_telemetry,
    read_profiling_capabilities,
    set_dispatch_telemetry_enabled,
)
from tessera.compiler.apple_route_selector import ROUTE_REPORT_SCHEMA_VERSION
from tessera.compiler.fusion import FusedRegion, run_fused_region


FUSED_ROUTE = "synthesized_fused"
UNFUSED_ROUTE = "mps_mpsgraph_unfused"


@dataclass(frozen=True)
class _RouteResult:
    output: np.ndarray
    native_dispatched: bool
    device_time_ns: int | None
    timing_sources: tuple[str, ...]
    resources: dict[str, Any]


def _shape(spec: str) -> tuple[int, int, int]:
    try:
        dims = tuple(int(part) for part in spec.lower().split("x"))
    except ValueError as exc:
        raise ValueError(f"shape must be MxKxN: {spec!r}") from exc
    if len(dims) != 3 or any(dim <= 0 for dim in dims):
        raise ValueError(f"shape must be MxKxN with positive dimensions: {spec!r}")
    return dims  # type: ignore[return-value]


def _dispatch(call: Callable[[], np.ndarray | tuple[np.ndarray, bool]]
              ) -> tuple[np.ndarray, dict[str, Any], bool]:
    clear_dispatch_telemetry()
    value = call()
    record = read_dispatch_telemetry()
    if isinstance(value, tuple):
        output, native = value
    else:
        output = value
        native = isinstance(record.get("device_time_ns"), int)
    return np.asarray(output), record, native


def _combine_dispatches(output: np.ndarray, records: list[dict[str, Any]], *,
                        route: str, native_flags: list[bool] | None = None
                        ) -> _RouteResult:
    device_times = [record.get("device_time_ns") for record in records]
    complete = bool(records) and all(
        isinstance(value, int) and value > 0 for value in device_times)
    sources = tuple(str(record["timing_source"]) for record in records
                    if record.get("timing_source") is not None)
    return _RouteResult(
        output=np.asarray(output),
        native_dispatched=(all(native_flags) if native_flags is not None else complete),
        device_time_ns=(sum(int(value) for value in device_times) if complete else None),
        timing_sources=sources,
        resources={
            "route": route,
            "dispatch_count": len(records),
            "dispatches": [
                {
                    "index": index,
                    "timing_source": record.get("timing_source"),
                    "pipeline": record.get("resources"),
                }
                for index, record in enumerate(records)
            ],
        },
    )


def _mps_matmul(rt: Any, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m, k = a.shape
    n = b.shape[1]
    if a.dtype == np.float32:
        out = np.empty((m, n), dtype=np.float32)
        ptr = ctypes.POINTER(ctypes.c_float)
        rt._apple_gpu_mps_matmul_f32()(
            a.ctypes.data_as(ptr), b.ctypes.data_as(ptr), out.ctypes.data_as(ptr),
            m, n, k)
        return out
    if a.dtype == np.float16:
        symbol = rt._apple_gpu_mps_matmul_f16()
        if symbol is None:
            raise RuntimeError("MPS fp16 matmul symbol unavailable")
        out = np.empty((m, n), dtype=np.float16)
        ptr = ctypes.POINTER(ctypes.c_uint16)
        symbol(a.view(np.uint16).ctypes.data_as(ptr),
               b.view(np.uint16).ctypes.data_as(ptr),
               out.view(np.uint16).ctypes.data_as(ptr), m, n, k)
        return out
    raise TypeError(f"unfused native comparison has no exact {a.dtype} MPS lane")


def _mpsgraph_unary(op: str, x: np.ndarray) -> tuple[np.ndarray, bool]:
    opcode = {"relu": 0, "silu": 4}[op]
    out = np.empty_like(x)
    n = int(x.size)
    if x.dtype == np.float32:
        symbol = bind_registered("tessera_apple_gpu_mpsgraph_unary_f32_status")
        ptr = ctypes.POINTER(ctypes.c_float)
        native = bool(symbol and symbol(
            opcode, x.ctypes.data_as(ptr), out.ctypes.data_as(ptr), n))
    elif x.dtype == np.float16:
        symbol = bind_registered("tessera_apple_gpu_mpsgraph_unary_f16_status")
        ptr = ctypes.POINTER(ctypes.c_uint16)
        native = bool(symbol and symbol(
            opcode, x.view(np.uint16).ctypes.data_as(ptr),
            out.view(np.uint16).ctypes.data_as(ptr), n))
    else:
        native = False
    return out, native


def _mpsgraph_add(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, bool]:
    if a.dtype != np.float32 or b.dtype != np.float32:
        raise TypeError("exact-dtype MPSGraph binary comparison is f32-only")
    out = np.empty_like(a)
    symbol = bind_registered("tessera_apple_gpu_mpsgraph_binary_f32_status")
    ptr = ctypes.POINTER(ctypes.c_float)
    native = bool(symbol and symbol(
        0, a.ctypes.data_as(ptr), b.ctypes.data_as(ptr),
        out.ctypes.data_as(ptr), int(a.size)))
    return out, native


def _run_fused(region: FusedRegion, a: np.ndarray, b: np.ndarray,
               bias: np.ndarray | None) -> _RouteResult:
    clear_dispatch_telemetry()
    output, execution = run_fused_region(region, a, b, bias)
    record = read_dispatch_telemetry()
    result = _combine_dispatches(output, [record], route=FUSED_ROUTE)
    return _RouteResult(result.output,
                        execution == "metal_runtime" and result.native_dispatched,
                        result.device_time_ns, result.timing_sources, result.resources)


def _run_unfused(rt: Any, region: FusedRegion, a: np.ndarray, b: np.ndarray,
                 bias: np.ndarray | None) -> _RouteResult:
    records: list[dict[str, Any]] = []
    native_flags: list[bool] = []
    output, record, native = _dispatch(lambda: _mps_matmul(rt, a, b))
    records.append(record)
    native_flags.append(native)
    for op in region.epilogue:
        if op == "bias":
            if bias is None:
                raise ValueError("bias epilogue requires bias")
            # The MPSGraph binary ABI is f32-only.  Do not silently compare an
            # fp16 candidate against a host-converted or mixed-dtype incumbent.
            if output.dtype != np.float32:
                raise TypeError("exact-dtype unfused bias comparison is f32-only")
            full_bias = np.broadcast_to(bias, output.shape)
            output, record, native = _dispatch(
                lambda output=output, full_bias=np.ascontiguousarray(full_bias):
                    _mpsgraph_add(output, full_bias))
        else:
            output, record, native = _dispatch(
                lambda output=output, op=op: _mpsgraph_unary(op, output))
        records.append(record)
        native_flags.append(native)
    return _combine_dispatches(output, records, route=UNFUSED_ROUTE,
                               native_flags=native_flags)


def _trial(call: Callable[[], _RouteResult], reps: int
           ) -> tuple[_RouteResult, int, int]:
    e2e: list[int] = []
    device: list[int] = []
    last: _RouteResult | None = None
    for _ in range(reps):
        started = time.perf_counter_ns()
        last = call()
        e2e.append(time.perf_counter_ns() - started)
        if last.device_time_ns is not None:
            device.append(last.device_time_ns)
    assert last is not None
    return last, int(statistics.median(e2e)), (
        int(statistics.median(device)) if len(device) == reps else 0)


def _paired_rows(*, op: str, shape: str, dtype: str, device: str,
                 reference: np.ndarray, incumbent: Callable[[], _RouteResult],
                 candidate: Callable[[], _RouteResult], reps: int, trials: int
                 ) -> list[dict[str, Any]]:
    incumbent()
    candidate()
    collected: dict[str, list[tuple[_RouteResult, int, int]]] = {
        UNFUSED_ROUTE: [], FUSED_ROUTE: []}
    for trial in range(trials):
        order = ((UNFUSED_ROUTE, incumbent), (FUSED_ROUTE, candidate))
        if trial % 2:
            order = tuple(reversed(order))
        for route, call in order:
            collected[route].append(_trial(call, reps))

    rows: list[dict[str, Any]] = []
    for route in (UNFUSED_ROUTE, FUSED_ROUTE):
        values = collected[route]
        result = values[-1][0]
        e2e = [value[1] for value in values]
        device_times = [value[2] for value in values]
        valid = bool(np.allclose(
            result.output.astype(np.float32), reference.astype(np.float32),
            rtol=3e-2 if dtype == "f16" else 3e-3,
            atol=3e-2 if dtype == "f16" else 3e-4))
        rows.append({
            "backend": "apple_gpu",
            "op": op,
            "shape": shape,
            "dtype": dtype,
            "device": device,
            "route": route,
            "latency_ms": statistics.median(e2e) / 1e6,
            "stdev_ms": statistics.stdev(e2e) / 1e6,
            "reps": reps * trials,
            "trials": trials,
            "native_dispatched": result.native_dispatched,
            "numerically_validated": valid,
            "telemetry": {
                "end_to_end_median_ns": int(statistics.median(e2e)),
                "device_time_median_ns": (
                    int(statistics.median(device_times)) if all(device_times) else None),
                "device_time_samples": reps * trials if all(device_times) else 0,
                "device_time_coverage": 1.0 if all(device_times) else 0.0,
                "timing_source": "+".join(result.timing_sources),
                "counter_sampling_supported": False,
                "counter_timestamp_delta_median": None,
                "paired_trial_end_to_end_medians_ns": e2e,
                "paired_trial_device_medians_ns": device_times,
                "resources": result.resources,
            },
        })
    return rows


def characterize(*, shapes: list[tuple[int, int, int]], reps: int,
                 trials: int) -> dict[str, Any]:
    from tessera import runtime as rt
    from tessera.compiler.apple_target import probe_apple_runtime_limits

    if not rt.DeviceTensor.is_metal():
        return {"schema_version": ROUTE_REPORT_SCHEMA_VERSION, "runs": [],
                "skipped_apple_gpu": "Apple Metal device unavailable"}
    limits = probe_apple_runtime_limits()
    family = limits.apple_gpu_family if limits is not None else -1
    device = (f"apple{family - 1000}" if 1001 <= family <= 1099
              else "apple_silicon_metal_unknown_family")
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(1707)
    set_dispatch_telemetry_enabled(True)
    try:
        for m, k, n in shapes:
            shape = f"{m}x{k}x{n}"
            for np_dtype, dtype in ((np.float32, "f32"), (np.float16, "f16")):
                a = np.ascontiguousarray(
                    (rng.standard_normal((m, k)) * 0.15).astype(np_dtype))
                b = np.ascontiguousarray(
                    (rng.standard_normal((k, n)) * 0.15).astype(np_dtype))
                # GELU synthesis remains covered by native correctness tests,
                # but there is no standalone MPSGraph GELU opcode to form a
                # like-for-like native incumbent. ReLU and SiLU do have one.
                chains = [("relu",)]
                if dtype == "f32":
                    chains.append(("bias", "silu"))
                for chain in chains:
                    region = FusedRegion(chain)
                    bias = (np.ascontiguousarray(
                        (rng.standard_normal(n) * 0.05).astype(np_dtype))
                        if "bias" in chain else None)
                    reference = region.reference(a, b, bias)
                    op = "matmul_" + "_".join(chain)
                    rows.extend(_paired_rows(
                        op=op, shape=shape, dtype=dtype, device=device,
                        reference=reference,
                        incumbent=lambda region=region, a=a, b=b, bias=bias:
                            _run_unfused(rt, region, a, b, bias),
                        candidate=lambda region=region, a=a, b=b, bias=bias:
                            _run_fused(region, a, b, bias),
                        reps=reps, trials=trials))
    finally:
        set_dispatch_telemetry_enabled(False)
    return {
        "schema_version": ROUTE_REPORT_SCHEMA_VERSION,
        "runs": rows,
        "device": device,
        "paired_trials": trials,
        "profiling_capabilities": read_profiling_capabilities(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=["64x64x64", "65x63x67",
                                                         "256x256x256"])
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.reps < 2:
        parser.error("--reps must be at least 2")
    if args.trials < 3:
        parser.error("--trials must be at least 3")
    payload = characterize(shapes=[_shape(spec) for spec in args.shapes],
                           reps=args.reps, trials=args.trials)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True),
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
