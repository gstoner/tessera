"""TILE-1 exact-route characterization with two warm runs per timing domain.

The report is evidence, not an autotuner input by itself.  It compares the
source-backed simdgroup ABI with MPS for fp16 and bf16, aligned and ragged
small/medium/throughput shapes.  A route is recorded only when its ABI reports
native Metal execution; unavailable counter samples are represented as such,
never as estimated occupancy or spills.
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
from tessera.compiler.apple_target import AppleGPUArch, AppleGPUTargetProfile
from tessera.compiler.apple_fragment import (
    AppleTilePromotionEvidence,
    select_apple_tile_promotion,
)
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


def _median_ns(call: Callable[[], tuple[np.ndarray | None, bool, Any]], reps: int):
    call()  # independent warm-up for this run; excludes pipeline setup
    end_to_end: list[int] = []
    kernel: list[int] = []
    output = None
    native = True
    record = None
    for _ in range(reps):
        started = time.perf_counter_ns()
        output, ran, record = call()
        end_to_end.append(time.perf_counter_ns() - started)
        native &= ran
        elapsed = record["device_time_ns"] if isinstance(record, dict) else record.device_time_ns
        if elapsed is not None and elapsed > 0:
            kernel.append(elapsed)
    return output, native, record, int(statistics.median(end_to_end)), (
        int(statistics.median(kernel)) if len(kernel) == reps else None)


def _mps_call(dtype: str, a: np.ndarray, b: np.ndarray):
    symbol = f"tessera_apple_gpu_mps_matmul_{'f16' if dtype == 'fp16' else 'bf16'}_status"
    dispatch = bind_registered(symbol)
    if dispatch is None:
        raise RuntimeError(f"required MPS status ABI unavailable: {symbol}")
    runtime = apple_gpu_runtime()
    timing = getattr(runtime, "tessera_apple_gpu_tile_last_device_time_ns", None)
    supported = getattr(runtime, "tessera_apple_gpu_tile_counter_sampling_supported", None)
    m, k = a.shape
    n = b.shape[1]

    def call():
        out = np.empty((m, n), dtype=np.uint16)
        ptr = ctypes.POINTER(ctypes.c_uint16)
        native = bool(dispatch(a.ctypes.data_as(ptr), b.ctypes.data_as(ptr),
                               out.ctypes.data_as(ptr), m, n, k))
        if timing is not None:
            timing.restype = ctypes.c_int64
            elapsed = int(timing())
        else:
            elapsed = -1
        counter_capable = None
        if supported is not None:
            supported.restype = ctypes.c_int32
            counter_capable = bool(supported())
        value = out.view(np.float16).astype(np.float32) if dtype == "fp16" else _from_bf16(out)
        return value, native, {
            "device_time_ns": elapsed if elapsed >= 0 else None,
            "counter_sampling_supported": counter_capable,
            "counter_timestamp_delta": None,
            "resources": {"api": "MPSMatrixMultiplication", "threadgroup_resources": None},
        }
    return call


def _tile_call(dtype: str, a: np.ndarray, b: np.ndarray):
    artifact = materialize_apple_simdgroup_tile_msl(
        AppleGPUTargetProfile(AppleGPUArch.APPLE7), dtype, 32, 32, 16)
    return lambda: dispatch_apple_simdgroup_tile_f16(
        artifact, a, b, return_provenance=True)


def _record_value(record: Any, name: str) -> Any:
    return record[name] if isinstance(record, dict) else getattr(record, name)


def _route_runs(call, reference: np.ndarray, reps: int) -> tuple[dict[str, Any], dict[str, Any]]:
    records = []
    for _ in range(2):
        output, native, record, end_to_end_ns, kernel_ns = _median_ns(call, reps)
        records.append({
            "native_gpu": native,
            "numerically_validated": output is not None and bool(np.allclose(
                output, reference, rtol=1e-2, atol=1e-2)),
            "placement_validated": native,
            "end_to_end_median_ns": end_to_end_ns,
            "kernel_median_ns": kernel_ns,
            "resources": _record_value(record, "resources"),
            "counter_sampling_supported": _record_value(record, "counter_sampling_supported"),
            "counter_timestamp_delta": _record_value(record, "counter_timestamp_delta"),
        })
    return tuple(records)


def _promotion_evidence(route: str, dtype: str, shape: tuple[int, int, int],
                        timing_domain: str, runs: tuple[dict[str, Any], dict[str, Any]]):
    counter_capabilities = {run["counter_sampling_supported"] for run in runs}
    return AppleTilePromotionEvidence(
        route=route, dtype=dtype, shape=shape, timing_domain=timing_domain,
        native_gpu=all(run["native_gpu"] for run in runs),
        numerically_validated=all(run["numerically_validated"] for run in runs),
        placement_validated=all(run["placement_validated"] for run in runs),
        run_medians_ns=tuple(
            int(run[f"{timing_domain}_median_ns"] or -1) for run in runs),
        resource_record=runs[0]["resources"] if all(
            run["resources"] == runs[0]["resources"] for run in runs) else {},
        counter_sampling_supported=(counter_capabilities.pop()
                                    if len(counter_capabilities) == 1 else None),
        counter_timestamp_deltas=tuple(run["counter_timestamp_delta"] for run in runs),
    )


def _promotion_decisions(dtype: str, shape: tuple[int, int, int],
                         mps_runs: tuple[dict[str, Any], dict[str, Any]],
                         simdgroup_runs: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, str]:
    return {
        domain: select_apple_tile_promotion(
            _promotion_evidence("mps", dtype, shape, domain, mps_runs),
            _promotion_evidence("simdgroup_matrix", dtype, shape, domain, simdgroup_runs),
        )
        for domain in ("end_to_end", "kernel")
    }


def characterize(shapes: tuple[tuple[int, int, int], ...], reps: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for dtype in ("fp16", "bf16"):
        for m, k, n in shapes:
            rng = np.random.default_rng(m * 1000003 + k * 1009 + n)
            source_a = rng.standard_normal((m, k), dtype=np.float32) * 0.1
            source_b = rng.standard_normal((k, n), dtype=np.float32) * 0.1
            if dtype == "fp16":
                a, b = source_a.astype(np.float16), source_b.astype(np.float16)
                reference = a.astype(np.float32) @ b.astype(np.float32)
            else:
                a, b = _to_bf16(source_a), _to_bf16(source_b)
                reference = _from_bf16(a) @ _from_bf16(b)
            mps_runs = _route_runs(_mps_call(dtype, a, b), reference, reps)
            simdgroup_runs = _route_runs(_tile_call(dtype, a, b), reference, reps)
            rows.append({
                "op": "tile_gemm",
                "shape": f"{m}x{k}x{n}",
                "dtype": dtype,
                "tile_class": "ragged" if any(x % 8 for x in (m, k, n)) else "aligned",
                "mps": mps_runs,
                "simdgroup_matrix": simdgroup_runs,
                "promotion_decision": _promotion_decisions(
                    dtype, (m, k, n), mps_runs, simdgroup_runs),
            })
            rows[-1]["production_timing_domain"] = "end_to_end"
            rows[-1]["production_route"] = rows[-1]["promotion_decision"]["end_to_end"]
    return {
        "schema_version": 1,
        "warm_runs": 2,
        "reps": reps,
        "promotion_policy": {
            "minimum_win_fraction": 0.05,
            "production_timing_domain": "end_to_end",
            "requires": ["native_gpu", "numerical", "placement", "resources", "counter_record"],
        },
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=list(DEFAULT_SHAPES))
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.reps < 2:
        parser.error("--reps must be at least 2")
    payload = characterize(tuple(_shape(spec) for spec in args.shapes), args.reps)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
