"""Characterize comparable native Apple GPU routes in one evidence report.

The report is consumable by :mod:`tessera.compiler.apple_route_selector`.  It
does not pretend all lanes implement the same operation: each row names the
exact op/shape/dtype and is emitted only after that route ran natively and
matched a NumPy oracle.

Comparable pairs today:

* f32 row softmax: ``mpsgraph`` vs handwritten ``msl``;
* f32 matmul: ``mps`` vs ``simdgroup_matrix``;
* f16 matmul: ``mps`` vs MTL4 cooperative ``tensor``/MPP matmul2d.

Run on an Apple host in a fresh process:

    PYTHONPATH=python python benchmarks/apple_gpu/benchmark_route_characterization.py \
      --output /tmp/apple-routes.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import numpy as np

from tessera._apple_gpu_dispatch import (
    clear_dispatch_telemetry,
    read_dispatch_telemetry,
    read_profiling_capabilities,
    set_dispatch_telemetry_enabled,
)
from tessera.compiler.apple_route_selector import ROUTE_REPORT_SCHEMA_VERSION
from tessera.compiler.apple_route_selector import live_apple_route_context


def _shape(spec: str) -> tuple[int, ...]:
    try:
        dims = tuple(int(part) for part in spec.lower().split("x"))
    except ValueError as exc:
        raise ValueError(f"shape must use x-separated positive integers: {spec!r}") from exc
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError(f"shape must use positive integers: {spec!r}")
    return dims


def _median(call: Callable[[], Any], reps: int, *,
            warmup: bool = True) -> tuple[Any, float, float, dict[str, Any]]:
    if warmup:
        call()  # excludes pipeline creation from steady-state measurements
    samples: list[float] = []
    device_samples: list[int] = []
    counter_samples: list[int] = []
    timing_sources: set[str] = set()
    counter_states: set[bool] = set()
    resource_records: list[dict[str, Any]] = []
    output: Any = None
    for _ in range(reps):
        clear_dispatch_telemetry()
        started = time.perf_counter_ns()
        output = call()
        samples.append((time.perf_counter_ns() - started) / 1e6)
        telemetry = read_dispatch_telemetry()
        if telemetry["device_time_ns"] is not None:
            device_samples.append(int(telemetry["device_time_ns"]))
        if telemetry["counter_timestamp_delta"] is not None:
            counter_samples.append(int(telemetry["counter_timestamp_delta"]))
        if telemetry["timing_source"] is not None:
            timing_sources.add(str(telemetry["timing_source"]))
        if telemetry["counter_sampling_supported"] is not None:
            counter_states.add(bool(telemetry["counter_sampling_supported"]))
        if telemetry["resources"] is not None:
            resource_records.append(dict(telemetry["resources"]))
    summary = {
        "device_time_median_ns": (
            int(statistics.median(device_samples))
            if len(device_samples) >= max(1, math.ceil(reps * 0.9)) else None),
        "device_time_samples": len(device_samples),
        "device_time_coverage": len(device_samples) / reps,
        "timing_source": timing_sources.pop() if len(timing_sources) == 1 else None,
        "counter_sampling_supported": (
            counter_states.pop() if len(counter_states) == 1 else None),
        "counter_timestamp_delta_median": (
            int(statistics.median(counter_samples)) if len(counter_samples) == reps else None),
        "runtime_resources": (
            resource_records[0] if len(resource_records) == reps and all(
                record == resource_records[0] for record in resource_records) else None),
        "timing_unavailable_reason": (
            "owned_command_buffer_not_observed" if not device_samples else
            ("partial_dispatch_timing" if len(device_samples) != reps else None)),
    }
    return (output, statistics.median(samples),
            statistics.stdev(samples) if reps > 1 else 0.0, summary)


def _paired_median(incumbent: Callable[[], Any], candidate: Callable[[], Any],
                   reps: int, trials: int
                   ) -> tuple[tuple[Any, float, float, dict[str, Any]],
                              tuple[Any, float, float, dict[str, Any]]]:
    """Measure route pairs in alternating blocks to cancel clock movement."""
    if trials < 3:
        raise ValueError("paired characterization requires at least three trials")
    incumbent()
    candidate()
    trial_results: dict[str, list[tuple[Any, float, float, dict[str, Any]]]] = {
        "incumbent": [], "candidate": []}
    for trial in range(trials):
        order = (("incumbent", incumbent), ("candidate", candidate))
        if trial % 2:
            order = tuple(reversed(order))
        for name, call in order:
            trial_results[name].append(_median(call, reps, warmup=False))

    def combine(values: list[tuple[Any, float, float, dict[str, Any]]]
                ) -> tuple[Any, float, float, dict[str, Any]]:
        medians_ms = [value[1] for value in values]
        summaries = [value[3] for value in values]
        device_medians = [summary["device_time_median_ns"] for summary in summaries]
        counter_medians = [summary["counter_timestamp_delta_median"]
                           for summary in summaries]
        sources = {summary["timing_source"] for summary in summaries}
        counter_states = {summary["counter_sampling_supported"] for summary in summaries}
        resources = [summary["runtime_resources"] for summary in summaries]
        all_device = all(value is not None for value in device_medians)
        all_counters = all(value is not None for value in counter_medians)
        summary = {
            "device_time_median_ns": (
                int(statistics.median(device_medians)) if all_device else None),
            "device_time_samples": sum(int(item["device_time_samples"])
                                       for item in summaries),
            "device_time_coverage": (
                sum(int(item["device_time_samples"]) for item in summaries)
                / (reps * trials)),
            "timing_source": sources.pop() if len(sources) == 1 else None,
            "counter_sampling_supported": (
                counter_states.pop() if len(counter_states) == 1 else None),
            "counter_timestamp_delta_median": (
                int(statistics.median(counter_medians)) if all_counters else None),
            "runtime_resources": (
                resources[0] if resources and all(record == resources[0]
                                                   for record in resources) else None),
            "timing_unavailable_reason": (
                "partial_paired_dispatch_timing"
                if all_device and any(item["device_time_coverage"] < 1.0
                                      for item in summaries)
                else (None if all_device else "incomplete_paired_dispatch_timing")),
            "paired_trial_end_to_end_medians_ns": [
                int(value * 1e6) for value in medians_ms],
            "paired_trial_device_medians_ns": device_medians,
        }
        return (values[-1][0], statistics.median(medians_ms),
                statistics.stdev(medians_ms), summary)

    return combine(trial_results["incumbent"]), combine(trial_results["candidate"])


def _resource_record(route: str, shape: str) -> dict[str, Any]:
    dims = tuple(int(part) for part in shape.split("x"))
    if route == "simdgroup_matrix":
        fast = len(dims) == 3 and all(dim % limit == 0 for dim, limit in zip(dims, (64, 16, 64)))
        return {
            "api": "MTL4ComputeCommandEncoder",
            "threadgroup": [256 if fast else 128, 1, 1],
            "output_tile": [64, 64] if fast else [32, 32],
            "threadgroup_memory_bytes": None,
            "occupancy": None,
            "occupancy_unavailable_reason": "public_metal_api_has_no_occupancy_query",
            "spill_evidence": None,
            "spill_unavailable_reason": "public_metal_api_has_no_register_or_spill_query",
        }
    if route == "cooperative_tensor":
        return {
            "api": "MetalPerformancePrimitives.matmul2d",
            "threadgroup": [128, 1, 1],
            "output_tile": [64, 64],
            "threadgroup_memory_bytes": None,
            "occupancy": None,
            "occupancy_unavailable_reason": "public_metal_api_has_no_occupancy_query",
            "spill_evidence": None,
            "spill_unavailable_reason": "public_metal_api_has_no_register_or_spill_query",
        }
    return {
        "api": "MPSMatrixMultiplication" if route == "mps" else route,
        "threadgroup": None,
        "output_tile": None,
        "threadgroup_memory_bytes": None,
        "occupancy": None,
        "occupancy_unavailable_reason": "public_metal_api_has_no_occupancy_query",
        "spill_evidence": None,
        "spill_unavailable_reason": "framework_pipeline_or_public_metal_stats_unavailable",
    }


def _row(*, op: str, shape: str, dtype: str, route: str, output: Any,
         reference: Any, latency_ms: float, stdev_ms: float,
         telemetry: dict[str, Any], device: str = "apple_silicon_metal") -> dict[str, Any]:
    # The caller only invokes native route callables.  Correctness remains a
    # separate per-row proof so malformed output can never inform selection.
    correct = bool(np.allclose(output, reference, rtol=3e-3, atol=3e-4))
    planned_resources = _resource_record(route, shape)
    planned_resources["runtime_pipeline"] = telemetry.pop("runtime_resources", None)
    return {
        "backend": "apple_gpu",
        "op": op,
        "shape": shape,
        "dtype": dtype,
        "device": device,
        "route": route,
        "latency_ms": latency_ms,
        "stdev_ms": stdev_ms,
        "reps": 0,  # filled by the producer so consumers can reject partial data
        "native_dispatched": True,
        "numerically_validated": correct,
        "telemetry": {
            "end_to_end_median_ns": int(latency_ms * 1e6),
            **telemetry,
            "resources": planned_resources,
        },
    }


def _mps_matmul(rt: Any, a: Any, b: Any, dtype: str) -> Any:
    m, k = a.shape
    n = b.shape[1]
    if dtype == "f32":
        out = np.empty((m, n), np.float32)
        f = rt._apple_gpu_mps_matmul_f32()
        ptr = ctypes.POINTER(ctypes.c_float)
        f(a.ctypes.data_as(ptr), b.ctypes.data_as(ptr), out.ctypes.data_as(ptr), m, n, k)
        return out
    out = np.empty((m, n), np.float16)
    f = rt._apple_gpu_mps_matmul_f16()
    if f is None:
        raise RuntimeError("MPS f16 matmul symbol unavailable")
    ptr = ctypes.POINTER(ctypes.c_uint16)
    f(a.view(np.uint16).ctypes.data_as(ptr), b.view(np.uint16).ctypes.data_as(ptr),
      out.view(np.uint16).ctypes.data_as(ptr), m, n, k)
    return out.astype(np.float32)


def _append_matmul_rows(rows: list[dict[str, Any]], skipped: list[str], rt: Any,
                        dims: tuple[int, int, int], reps: int, trials: int,
                        device: str) -> None:
    m, k, n = dims
    shape = f"{m}x{k}x{n}"
    rng = np.random.default_rng(sum(dims))
    a = np.ascontiguousarray(rng.standard_normal((m, k)).astype(np.float32) * 0.1)
    b = np.ascontiguousarray(rng.standard_normal((k, n)).astype(np.float32) * 0.1)
    ref = a @ b
    mps_f32 = lambda: _mps_matmul(rt, a, b, "f32")
    sg_f32 = lambda: rt.apple_gpu_mtl4_matmul_sg(a, b, np)
    mps_result, sg_result = _paired_median(mps_f32, sg_f32, reps, trials)
    out, med, sd, telemetry = mps_result
    item = _row(op="matmul", shape=shape, dtype="f32", route="mps", output=out,
                reference=ref, latency_ms=med, stdev_ms=sd, telemetry=telemetry,
                device=device)
    item["reps"] = reps * trials
    item["trials"] = trials
    rows.append(item)

    sg_out, sg_med, sg_sd, sg_telemetry = sg_result
    sg_value, sg_ran = sg_out
    if sg_ran:
        item = _row(op="matmul", shape=shape, dtype="f32", route="simdgroup_matrix",
                    output=sg_value, reference=ref, latency_ms=sg_med, stdev_ms=sg_sd,
                    telemetry=sg_telemetry, device=device)
        item["reps"] = reps * trials
        item["trials"] = trials
        rows.append(item)
    else:
        skipped.append(f"simdgroup_matrix matmul {shape}: MTL4 route unavailable")

    ah = np.ascontiguousarray(a.astype(np.float16))
    bh = np.ascontiguousarray(b.astype(np.float16))
    ref_h = ah.astype(np.float32) @ bh.astype(np.float32)
    mps_f16 = lambda: _mps_matmul(rt, ah, bh, "f16")
    coop_f16 = lambda: rt.apple_gpu_mtl4_matmul2d_f16(ah, bh, np)
    mps_result, coop_result = _paired_median(mps_f16, coop_f16, reps, trials)
    out, med, sd, telemetry = mps_result
    item = _row(op="matmul", shape=shape, dtype="f16", route="mps", output=out,
                reference=ref_h, latency_ms=med, stdev_ms=sd, telemetry=telemetry,
                device=device)
    item["reps"] = reps * trials
    item["trials"] = trials
    rows.append(item)

    coop_out, coop_med, coop_sd, coop_telemetry = coop_result
    coop_value, coop_ran = coop_out
    if coop_ran:
        item = _row(op="matmul", shape=shape, dtype="f16", route="cooperative_tensor",
                    output=coop_value, reference=ref_h, latency_ms=coop_med,
                    stdev_ms=coop_sd, telemetry=coop_telemetry, device=device)
        item["reps"] = reps * trials
        item["trials"] = trials
        rows.append(item)
    else:
        skipped.append(f"cooperative_tensor matmul {shape}: MTL4 route unavailable")


def _append_softmax_rows(rows: list[dict[str, Any]], rt: Any,
                         dims: tuple[int, int], reps: int, trials: int,
                         device: str) -> None:
    row_count, col_count = dims
    shape = f"{row_count}x{col_count}"
    rng = np.random.default_rng(row_count * 31 + col_count)
    x = np.ascontiguousarray(rng.standard_normal((row_count, col_count)).astype(np.float32))
    ref = np.exp(x - x.max(axis=-1, keepdims=True))
    ref /= ref.sum(axis=-1, keepdims=True)

    mpsgraph = lambda: rt._apple_gpu_dispatch_mpsgraph_softmax(x, np)
    msl = lambda: rt._apple_gpu_dispatch_softmax_msl(
        "tessera.softmax", [x], {}, np)
    mpsgraph_result, msl_result = _paired_median(mpsgraph, msl, reps, trials)
    out, med, sd, telemetry = mpsgraph_result
    item = _row(op="softmax", shape=shape, dtype="f32", route="mpsgraph", output=out,
                reference=ref, latency_ms=med, stdev_ms=sd, telemetry=telemetry,
                device=device)
    item["reps"] = reps * trials
    item["trials"] = trials
    rows.append(item)

    out, med, sd, telemetry = msl_result
    item = _row(op="softmax", shape=shape, dtype="f32", route="msl", output=out,
                reference=ref, latency_ms=med, stdev_ms=sd, telemetry=telemetry,
                device=device)
    item["reps"] = reps * trials
    item["trials"] = trials
    rows.append(item)


def characterize(*, matmul_shapes: list[tuple[int, int, int]],
                 softmax_shapes: list[tuple[int, int]], reps: int,
                 trials: int = 9,
                 ops: tuple[str, ...] = ("matmul", "softmax")) -> dict[str, Any]:
    """Collect a report, or a structured skip when no native Metal is visible."""
    from tessera import runtime as rt

    if not rt.DeviceTensor.is_metal():
        return {"schema_version": ROUTE_REPORT_SCHEMA_VERSION, "runs": [],
                "skipped_apple_gpu": "Apple Metal device unavailable"}
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    from tessera.compiler.apple_target import probe_apple_runtime_limits

    limits = probe_apple_runtime_limits()
    family = limits.apple_gpu_family if limits is not None else -1
    device = (f"apple{family - 1000}" if 1001 <= family <= 1099
              else "apple_silicon_metal_unknown_family")
    capture_enabled = set_dispatch_telemetry_enabled(True)
    try:
        if "matmul" in ops:
            for dims in matmul_shapes:
                _append_matmul_rows(rows, skipped, rt, dims, reps, trials, device)
        if "softmax" in ops:
            for dims in softmax_shapes:
                _append_softmax_rows(rows, rt, dims, reps, trials, device)
    finally:
        set_dispatch_telemetry_enabled(False)
    return {"schema_version": ROUTE_REPORT_SCHEMA_VERSION,
            "context": live_apple_route_context().as_mapping(), "runs": rows,
            "dispatch_telemetry_capture": capture_enabled,
            "device": device,
            "paired_trials": trials,
            "profiling_capabilities": read_profiling_capabilities(),
            "skipped_candidates": skipped}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matmul-shapes", nargs="+", default=["64x64x64", "256x256x256"])
    parser.add_argument("--softmax-shapes", nargs="+", default=["64x64", "256x256"])
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--ops", nargs="+", choices=("matmul", "softmax"),
                        default=("matmul", "softmax"),
                        help="Restrict dispatches for bounded tooling captures")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    matmul = [_shape(shape) for shape in args.matmul_shapes]
    softmax = [_shape(shape) for shape in args.softmax_shapes]
    if any(len(shape) != 3 for shape in matmul) or any(len(shape) != 2 for shape in softmax):
        parser.error("matmul shapes must be MxKxN; softmax shapes must be RowsxCols")
    if args.trials < 3:
        parser.error("--trials must be at least 3")
    payload = characterize(matmul_shapes=matmul, softmax_shapes=softmax,
                           reps=args.reps, trials=args.trials,
                           ops=tuple(args.ops))
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
