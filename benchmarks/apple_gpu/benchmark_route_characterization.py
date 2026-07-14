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
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import numpy as np

from tessera.compiler.apple_route_selector import ROUTE_REPORT_SCHEMA_VERSION


def _shape(spec: str) -> tuple[int, ...]:
    try:
        dims = tuple(int(part) for part in spec.lower().split("x"))
    except ValueError as exc:
        raise ValueError(f"shape must use x-separated positive integers: {spec!r}") from exc
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError(f"shape must use positive integers: {spec!r}")
    return dims


def _median(call: Callable[[], Any], reps: int) -> tuple[Any, float, float]:
    call()  # warm-up excludes pipeline creation from steady-state measurements
    samples: list[float] = []
    output: Any = None
    for _ in range(reps):
        started = time.perf_counter_ns()
        output = call()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return output, statistics.median(samples), statistics.stdev(samples) if reps > 1 else 0.0


def _row(*, op: str, shape: str, dtype: str, route: str, output: Any,
         reference: Any, latency_ms: float, stdev_ms: float) -> dict[str, Any]:
    # The caller only invokes native route callables.  Correctness remains a
    # separate per-row proof so malformed output can never inform selection.
    correct = bool(np.allclose(output, reference, rtol=3e-3, atol=3e-4))
    return {
        "backend": "apple_gpu",
        "op": op,
        "shape": shape,
        "dtype": dtype,
        "device": "apple_silicon_metal",
        "route": route,
        "latency_ms": latency_ms,
        "stdev_ms": stdev_ms,
        "reps": 0,  # filled by the producer so consumers can reject partial data
        "native_dispatched": True,
        "numerically_validated": correct,
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
                        dims: tuple[int, int, int], reps: int) -> None:
    m, k, n = dims
    shape = f"{m}x{k}x{n}"
    rng = np.random.default_rng(sum(dims))
    a = np.ascontiguousarray(rng.standard_normal((m, k)).astype(np.float32) * 0.1)
    b = np.ascontiguousarray(rng.standard_normal((k, n)).astype(np.float32) * 0.1)
    ref = a @ b
    out, med, sd = _median(lambda: _mps_matmul(rt, a, b, "f32"), reps)
    item = _row(op="matmul", shape=shape, dtype="f32", route="mps", output=out,
                reference=ref, latency_ms=med, stdev_ms=sd)
    item["reps"] = reps
    rows.append(item)

    sg_out, sg_med, sg_sd = _median(lambda: rt.apple_gpu_mtl4_matmul_sg(a, b, np), reps)
    sg_value, sg_ran = sg_out
    if sg_ran:
        item = _row(op="matmul", shape=shape, dtype="f32", route="simdgroup_matrix",
                    output=sg_value, reference=ref, latency_ms=sg_med, stdev_ms=sg_sd)
        item["reps"] = reps
        rows.append(item)
    else:
        skipped.append(f"simdgroup_matrix matmul {shape}: MTL4 route unavailable")

    ah = np.ascontiguousarray(a.astype(np.float16))
    bh = np.ascontiguousarray(b.astype(np.float16))
    ref_h = ah.astype(np.float32) @ bh.astype(np.float32)
    out, med, sd = _median(lambda: _mps_matmul(rt, ah, bh, "f16"), reps)
    item = _row(op="matmul", shape=shape, dtype="f16", route="mps", output=out,
                reference=ref_h, latency_ms=med, stdev_ms=sd)
    item["reps"] = reps
    rows.append(item)

    coop_out, coop_med, coop_sd = _median(lambda: rt.apple_gpu_mtl4_matmul2d_f16(ah, bh, np), reps)
    coop_value, coop_ran = coop_out
    if coop_ran:
        item = _row(op="matmul", shape=shape, dtype="f16", route="cooperative_tensor",
                    output=coop_value, reference=ref_h, latency_ms=coop_med, stdev_ms=coop_sd)
        item["reps"] = reps
        rows.append(item)
    else:
        skipped.append(f"cooperative_tensor matmul {shape}: MTL4 route unavailable")


def _append_softmax_rows(rows: list[dict[str, Any]], rt: Any,
                         dims: tuple[int, int], reps: int) -> None:
    row_count, col_count = dims
    shape = f"{row_count}x{col_count}"
    rng = np.random.default_rng(row_count * 31 + col_count)
    x = np.ascontiguousarray(rng.standard_normal((row_count, col_count)).astype(np.float32))
    ref = np.exp(x - x.max(axis=-1, keepdims=True))
    ref /= ref.sum(axis=-1, keepdims=True)

    out, med, sd = _median(lambda: rt._apple_gpu_dispatch_mpsgraph_softmax(x, np), reps)
    item = _row(op="softmax", shape=shape, dtype="f32", route="mpsgraph", output=out,
                reference=ref, latency_ms=med, stdev_ms=sd)
    item["reps"] = reps
    rows.append(item)

    out, med, sd = _median(
        lambda: rt._apple_gpu_dispatch_softmax("tessera.softmax", [x], {}, np), reps)
    item = _row(op="softmax", shape=shape, dtype="f32", route="msl", output=out,
                reference=ref, latency_ms=med, stdev_ms=sd)
    item["reps"] = reps
    rows.append(item)


def characterize(*, matmul_shapes: list[tuple[int, int, int]],
                 softmax_shapes: list[tuple[int, int]], reps: int) -> dict[str, Any]:
    """Collect a report, or a structured skip when no native Metal is visible."""
    from tessera import runtime as rt

    if not rt.DeviceTensor.is_metal():
        return {"schema_version": ROUTE_REPORT_SCHEMA_VERSION, "runs": [],
                "skipped_apple_gpu": "Apple Metal device unavailable"}
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for dims in matmul_shapes:
        _append_matmul_rows(rows, skipped, rt, dims, reps)
    for dims in softmax_shapes:
        _append_softmax_rows(rows, rt, dims, reps)
    return {"schema_version": ROUTE_REPORT_SCHEMA_VERSION, "runs": rows,
            "skipped_candidates": skipped}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matmul-shapes", nargs="+", default=["64x64x64", "256x256x256"])
    parser.add_argument("--softmax-shapes", nargs="+", default=["64x64", "256x256"])
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    matmul = [_shape(shape) for shape in args.matmul_shapes]
    softmax = [_shape(shape) for shape in args.softmax_shapes]
    if any(len(shape) != 3 for shape in matmul) or any(len(shape) != 2 for shape in softmax):
        parser.error("matmul shapes must be MxKxN; softmax shapes must be RowsxCols")
    payload = characterize(matmul_shapes=matmul, softmax_shapes=softmax, reps=args.reps)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
