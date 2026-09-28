#!/usr/bin/env python3
"""X86-GEMM-ALIGN-1 path-selection probe: several builds of
``tessera_x86_avx512_gemm_f32`` in one process, same buffers, round-robin
interleaved windows (the order rotates every sample), each window timed with
the x86 TSC witness (``profiler_x86_clock``: separate pinned calibration,
unpinned region, TSC vs CLOCK_MONOTONIC_RAW within the fleet band, every
witness re-verified). A and C are 64-byte aligned; B sits ``--offset-b`` bytes
past a 64-byte boundary. Outputs of every build are compared bit for bit with
the first. Prints one JSON line.

usage: probe_gemm_paths.py --lib NAME=PATH [--lib ...] -M -N -K --offset-b
(PYTHONPATH must reach a tessera checkout's ``python/``)."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import statistics
from pathlib import Path

import numpy as np


def placed(src: np.ndarray, offset: int) -> np.ndarray:
    raw = np.empty(src.nbytes + 128, dtype=np.uint8)
    start = (-raw.ctypes.data) % 64 + offset
    view = raw[start:start + src.nbytes].view(src.dtype).reshape(src.shape)
    view[...] = src
    assert view.ctypes.data % 64 == offset
    return view


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", action="append", required=True)
    ap.add_argument("-M", type=int, required=True)
    ap.add_argument("-N", type=int, required=True)
    ap.add_argument("-K", type=int, required=True)
    ap.add_argument("--offset-b", type=int, required=True)
    ap.add_argument("--samples", type=int, default=7)
    ap.add_argument("--window-ms", type=float, default=20.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from tessera.compiler import profiler_x86_clock as clock
    from tessera.compiler.e2e_fleet import X86_AVX512_AGREEMENT_BAND

    fp = ctypes.POINTER(ctypes.c_float)
    i64 = ctypes.c_int64
    names, fns, digests = [], {}, {}
    for spec in args.lib:
        name, path = spec.split("=", 1)
        fn = ctypes.CDLL(str(Path(path).resolve())).tessera_x86_avx512_gemm_f32
        fn.restype = None
        fn.argtypes = [fp, fp, i64, i64, i64, fp]
        names.append(name)
        fns[name] = fn
        digests[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if len({ctypes.cast(f, ctypes.c_void_p).value for f in fns.values()}) != len(fns):
        raise SystemExit("two builds resolved to the same loaded image")

    M, N, K = args.M, args.N, args.K
    rng = np.random.default_rng(20260927 + M * 7 + N * 11 + K * 13)
    a = placed(rng.standard_normal((M, K)).astype(np.float32), 0)
    b = placed(rng.standard_normal((K, N)).astype(np.float32), args.offset_b)
    outs = {n: placed(np.zeros((M, N), np.float32), 0) for n in names}
    calls = {n: (lambda f=fns[n], o=outs[n]: f(a.ctypes.data_as(fp), b.ctypes.data_as(fp),
                                                 i64(M), i64(N), i64(K), o.ctypes.data_as(fp)))
             for n in names}
    for c in calls.values():
        c()
    ref = outs[names[0]].view(np.uint32)
    bitwise = {n: bool(np.array_equal(outs[n].view(np.uint32), ref)) for n in names}

    import time
    t0 = time.perf_counter()
    for c in calls.values():
        c()
    per_call = max((time.perf_counter() - t0) / len(calls), 1e-7)
    iterations = max(1, int(args.window_ms / 1e3 / per_call))
    calibration = clock.calibrate()
    hz = float(calibration["frequency_hz"])
    env = "wsl2" if "microsoft" in platform.release().lower() else "bare_metal"
    per: dict[str, list[float]] = {n: [] for n in names}
    worst = 0.0
    for s in range(args.samples):
        order = names[s % len(names):] + names[:s % len(names)]
        for n in order:
            call = calls[n]

            def region(call=call) -> None:
                for _ in range(iterations):
                    call()
            window = clock.measure(region, calibration)
            witness = clock.witness_sample(calibration, window, {"image": digests[n]},
                                           execution_environment=env)
            if witness["clocks"]["tsc_cycles"].get("eligible_for_promotion") is not True:
                raise SystemExit("TSC witness refused")
            reason = clock.verify_witness_sample(witness)
            if reason is not None:
                raise SystemExit(f"witness does not re-verify: {reason}")
            raw_ns = float(window["raw_end_ns"] - window["raw_start_ns"])
            tsc_ns = float(window["tsc_end"] - window["tsc_start"]) * 1e9 / hz
            err = abs(tsc_ns - raw_ns) / raw_ns
            if err > X86_AVX512_AGREEMENT_BAND:
                raise SystemExit(f"TSC disagrees with CLOCK_MONOTONIC_RAW by {err:.2%}")
            worst = max(worst, err)
            per[n].append(tsc_ns / iterations / 1e3)
    print(json.dumps({
        "tag": args.tag, "host": os.uname().nodename, "M": M, "N": N, "K": K,
        "b_mod64": b.ctypes.data % 64, "iterations_per_window": iterations,
        "timing_source": "tsc_witness", "max_tsc_raw_disagreement": worst,
        "median_us": {n: statistics.median(v) for n, v in per.items()},
        "windows_us": per, "bitwise_eq_first": bitwise, "lib_sha256": digests,
    }))


if __name__ == "__main__":
    main()
