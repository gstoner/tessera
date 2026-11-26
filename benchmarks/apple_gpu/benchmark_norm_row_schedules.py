#!/usr/bin/env python3
"""APPLE-NTBI-2 slice 11 -- paired serial vs simdgroup_32 norm row schedules.

Measures the two compiler-synthesized norm row kernels on the live Apple GPU,
paired and interleaved (ABBA order per round), with the command-buffer GPU
interval (``GPUStartTime``/``GPUEndTime``) as the clock, admitted only after
the dispatch layer's host-witness bound accepts it. A sample whose clock is not
that interval is dropped, never substituted with host wall time. Each shape also checks both schedules against an f32
reference and against each other before any timing is believed.

The output is the evidence the Schedule's column threshold is set from; the
threshold is a performance key (Decision #21a), so the record names the host,
the clock and the per-shape medians, and a reader can re-derive the cut.

Run on an idle Mac (a Spotlight index or a build will move the numbers):

    PYTHONPATH=python python3 benchmarks/apple_gpu/benchmark_norm_row_schedules.py \
        --output benchmarks/baselines/apple7_norm_row_schedules.json
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import statistics
import struct
import subprocess
import sys
import time
from typing import Any

import numpy as np

from tessera import _apple_gpu_dispatch as dispatch
from tessera.compiler.emit.apple_msl import norm_entry, synthesize_norm_msl

EPS_BITS = struct.unpack("<I", struct.pack("<f", 1e-5))[0]
ROWS = (1, 8, 64, 512, 4096, 32768)
COLUMNS = (1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096)
FORMS = (("rmsnorm", "none"), ("layernorm", "none"),
         ("rmsnorm", "gamma"), ("layernorm", "gamma_beta"))
MAX_ELEMENTS = 1 << 24


def _storage_dtype(storage: str):
    if storage == "f16":
        return np.float16
    import ml_dtypes
    return ml_dtypes.bfloat16


def _reference(kind: str, affine: str, x, gamma, beta):
    xf = x.astype(np.float32)
    if kind == "rmsnorm":
        y = xf / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + np.float32(1e-5))
    else:
        mean = np.mean(xf, axis=-1, keepdims=True)
        var = np.mean((xf - mean) ** 2, axis=-1, keepdims=True)
        y = (xf - mean) / np.sqrt(var + np.float32(1e-5))
    if affine != "none":
        y = y * gamma.astype(np.float32)
    if affine == "gamma_beta":
        y = y + beta.astype(np.float32)
    return y


def _ptr(a) -> ctypes.c_void_p:
    return ctypes.c_void_p(a.ctypes.data) if a is not None else ctypes.c_void_p(None)


def _run(fn, src: bytes, entry: bytes, x, gamma, beta, out, rows, cols) -> int | None:
    dispatch.clear_dispatch_telemetry()
    status = fn(src, entry, _ptr(x), _ptr(gamma), _ptr(beta), _ptr(out),
                ctypes.c_int32(rows), ctypes.c_int32(cols))
    if status != 1:
        raise RuntimeError(f"Apple norm row dispatch returned {status}")
    record = dispatch.read_dispatch_telemetry()
    # Only the command-buffer device interval, and only once the dispatch
    # layer's host-witness bound has accepted it.
    if (record.get("timing_source") != "metal_command_buffer_interval"
            or not record.get("device_time_accepted_ns")):
        return None
    return int(record["device_time_accepted_ns"])


def _serial_call(affine_fn, rows_fn):
    def call(src, entry, x, gamma, beta, out, rows, cols):
        if gamma.value is None:
            return rows_fn(src, entry, x, out, rows, cols)
        return affine_fn(src, entry, x, gamma, beta, out, rows, cols)
    return call


def _source_commit() -> dict[str, Any]:
    """The checkout that produced the kernels: HEAD and whether it was clean."""
    def git(*argv: str) -> str:
        return subprocess.run(["git", *argv], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    return {"head": git("rev-parse", "HEAD"),
            "dirty_paths": [line[3:] for line in git("status", "--porcelain").splitlines()]}


def _load_average() -> list[float] | None:
    try:
        return [round(v, 2) for v in os.getloadavg()]
    except OSError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--storage", choices=("f16", "bf16"), default="f16")
    parser.add_argument("--rows", type=int, nargs="+", default=list(ROWS))
    parser.add_argument("--columns", type=int, nargs="+", default=list(COLUMNS))
    parser.add_argument("--output")
    args = parser.parse_args()

    if not dispatch.apple_gpu_available():
        print(f"Apple GPU unavailable: {dispatch.apple_gpu_skip_reason()}", file=sys.stderr)
        return 2
    rows_fn = dispatch.bind_registered("tessera_apple_gpu_synth_rows_f16")
    affine_fn = dispatch.bind_registered("tessera_apple_gpu_synth_affine_rows_f16")
    coop_fn = dispatch.bind_registered("tessera_apple_gpu_synth_simdgroup_rows_f16")
    if rows_fn is None or affine_fn is None or coop_fn is None:
        print("Apple norm row symbols are not exported by the loaded runtime", file=sys.stderr)
        return 2
    serial_fn = _serial_call(affine_fn, rows_fn)
    dtype = _storage_dtype(args.storage)
    step = float(np.finfo(np.float16).eps if args.storage == "f16" else 2.0 ** -7)
    rng = np.random.default_rng(1729)
    rows_out: list[dict[str, Any]] = []
    load_before = _load_average()
    started = time.time()
    with dispatch.dispatch_telemetry_capture():
        for kind, affine in FORMS:
            sources = {}
            for schedule in ("serial", "simdgroup_32"):
                sources[schedule] = (
                    synthesize_norm_msl(kind, args.storage, epsilon_bits=EPS_BITS,
                                        affine=affine, schedule=schedule).encode(),
                    norm_entry(kind, args.storage, affine, schedule).encode())
            for rows in args.rows:
                for cols in args.columns:
                    if rows * cols > MAX_ELEMENTS:
                        continue
                    x = (rng.standard_normal((rows, cols)) * 2 + 0.5).astype(dtype)
                    gamma = (rng.standard_normal(cols) * 0.5 + 1).astype(dtype) if affine != "none" else None
                    beta = rng.standard_normal(cols).astype(dtype) if affine == "gamma_beta" else None
                    outs = {s: np.empty((rows, cols), dtype=dtype) for s in sources}
                    arms = {"serial": serial_fn, "simdgroup_32": coop_fn}
                    for s, fn in arms.items():
                        _run(fn, *sources[s], x, gamma, beta, outs[s], rows, cols)
                    ref = _reference(kind, affine, x, gamma, beta)
                    scale = np.maximum(np.abs(ref), 1.0)
                    err = {s: float(np.max(np.abs(outs[s].astype(np.float32) - ref) / scale))
                           for s in outs}
                    cross = float(np.max(np.abs(outs["serial"].astype(np.float32)
                                                - outs["simdgroup_32"].astype(np.float32)) / scale))
                    if max(err.values()) > 2 * step or cross > 2 * step:
                        raise RuntimeError(
                            f"{kind}/{affine} {rows}x{cols}: error {err} cross {cross} "
                            f"exceeds two {args.storage} steps")
                    for _ in range(args.warmup):
                        for s, fn in arms.items():
                            _run(fn, *sources[s], x, gamma, beta, outs[s], rows, cols)
                    samples: dict[str, list[int]] = {s: [] for s in arms}
                    order = ("serial", "simdgroup_32")
                    for r in range(args.rounds):
                        seq = order if r % 2 == 0 else order[::-1]
                        pair = {}
                        for s in seq:
                            pair[s] = _run(arms[s], *sources[s], x, gamma, beta,
                                           outs[s], rows, cols)
                        if all(v is not None for v in pair.values()):
                            for s, v in pair.items():
                                samples[s].append(v)
                    if len(samples["serial"]) < max(3, args.rounds // 2):
                        raise RuntimeError(f"{kind}/{affine} {rows}x{cols}: too few device-clock pairs")
                    med = {s: statistics.median(v) for s, v in samples.items()}
                    ratios = [a / b for a, b in zip(samples["serial"], samples["simdgroup_32"])]
                    row = {
                        "kind": kind, "affine": affine, "rows": rows, "columns": cols,
                        "pairs": len(ratios),
                        "serial_median_ns": med["serial"],
                        "simdgroup_32_median_ns": med["simdgroup_32"],
                        "speedup_median": round(statistics.median(ratios), 4),
                        "speedup_min": round(min(ratios), 4),
                        "speedup_max": round(max(ratios), 4),
                        "simdgroup_wins_pairs": sum(r > 1.0 for r in ratios),
                        "max_rel_error": {s: round(v, 6) for s, v in err.items()},
                        "max_rel_cross": round(cross, 6),
                    }
                    rows_out.append(row)
                    print(f"{kind:9s} {affine:10s} {rows:5d}x{cols:<5d} serial {med['serial']/1e3:9.1f}us "
                          f"simd {med['simdgroup_32']/1e3:9.1f}us  x{row['speedup_median']:.2f} "
                          f"({row['simdgroup_wins_pairs']}/{len(ratios)})", flush=True)
    record = {
        "schema": "tessera.apple.norm_row_schedules.v1",
        "sync_key": "APPLE-NTBI-2-cooperative-norms-2026-11-26",
        "source": _source_commit(),
        "host": platform.node(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True).stdout.strip(),
        "storage": args.storage,
        "clock": "MTLCommandBuffer GPUStartTime/GPUEndTime, host-witness accepted (device_time_accepted_ns)",
        "pairing": "ABBA interleaved per round; a round counts only when both arms read the device clock",
        "rounds": args.rounds, "warmup": args.warmup,
        "row_counts": args.rows, "column_counts": args.columns,
        "load_average_before": load_before, "load_average_after": _load_average(),
        "elapsed_s": round(time.time() - started, 1),
        "rows": rows_out,
    }
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(record, fh, indent=1)
            fh.write("\n")
    print(f"load before {load_before} after {record['load_average_after']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
