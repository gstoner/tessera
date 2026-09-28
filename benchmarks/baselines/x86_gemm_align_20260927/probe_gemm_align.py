#!/usr/bin/env python3
"""X86-GEMM-ALIGN-1 probe: before/after, one fresh process per (shape, B offset).

Loads two builds of ``libtessera_x86_elementwise.so`` into one process --
``--before`` (the kernel as merged before the fix) and ``--after`` (the fix) --
and times ``tessera_x86_avx512_gemm_f32`` from each on the SAME A, B and C
buffers, as paired interleaved windows (the order alternates every sample).
A and C are 64-byte aligned; B sits ``--offset-b`` bytes past a 64-byte
boundary. Each window is timed with the x86 TSC witness
(``profiler_x86_clock``: TSC calibrated over separate pinned intervals, region
unpinned, TSC converted with that frequency and checked against
CLOCK_MONOTONIC_RAW within the fleet agreement band, and every witness
re-verified); the reported latency is the TSC one.

The "after" library is also packaged the production way
(``x86_native.package_matmul`` -> ``runtime.launch``) when that tree's
``tessera-opt`` is present, and the probe asserts the packaged image's payload
is byte-identical to the library it times -- so the timed symbol is the
production image. Numerics: the before and after outputs for the same inputs
are compared bit for bit, and both against float64 numpy.

Prints one JSON line.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import statistics
import sys
from pathlib import Path

import numpy as np

LIB = "src/compiler/codegen/tessera_x86_backend/libtessera_x86_elementwise.so"


def placed(src: np.ndarray, offset: int) -> np.ndarray:
    raw = np.empty(src.nbytes + 128, dtype=np.uint8)
    start = (-raw.ctypes.data) % 64 + offset
    view = raw[start:start + src.nbytes].view(src.dtype).reshape(src.shape)
    view[...] = src
    assert view.ctypes.data % 64 == offset
    return view


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="tree root of the pre-fix build")
    ap.add_argument("--after", required=True, help="tree root of the fixed build")
    ap.add_argument("-M", type=int, required=True)
    ap.add_argument("-N", type=int, required=True)
    ap.add_argument("-K", type=int, required=True)
    ap.add_argument("--offset-b", type=int, required=True)
    ap.add_argument("--samples", type=int, default=7)
    ap.add_argument("--window-ms", type=float, default=20.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    after_root = Path(args.after).resolve()
    sys.path[:0] = [str(after_root / "python"), str(after_root / "benchmarks" / "e2e_spine")]
    from tessera.compiler import profiler_x86_clock as clock
    from tessera.compiler.e2e_fleet import X86_AVX512_AGREEMENT_BAND

    libs = {name: Path(root).resolve() / "build" / LIB
            for name, root in (("before", args.before), ("after", args.after))}
    digests = {name: sha256(path) for name, path in libs.items()}
    fp = ctypes.POINTER(ctypes.c_float)
    i64 = ctypes.c_int64
    fns = {}
    for name, path in libs.items():
        fn = ctypes.CDLL(str(path)).tessera_x86_avx512_gemm_f32
        fn.restype = None
        fn.argtypes = [fp, fp, i64, i64, i64, fp]
        fns[name] = fn
    after_path = None
    query = getattr(ctypes.CDLL(str(libs["after"])),
                    "tessera_x86_avx512_gemm_f32_uses_packed_path", None)
    addresses = {ctypes.cast(fn, ctypes.c_void_p).value for fn in fns.values()}
    if len(addresses) != 2:
        raise SystemExit("before and after resolved to the same loaded image")

    M, N, K = args.M, args.N, args.K
    if query is not None:
        query.argtypes = [i64, i64, i64]
        query.restype = ctypes.c_int
        after_path = "packed" if query(M, N, K) else "direct"
    rng = np.random.default_rng(20260927 + M * 7 + N * 11 + K * 13)
    a = placed(rng.standard_normal((M, K)).astype(np.float32), 0)
    b = placed(rng.standard_normal((K, N)).astype(np.float32), args.offset_b)
    outs = {name: placed(np.zeros((M, N), np.float32), 0) for name in fns}

    packaged = None
    if (after_root / "build/tools/tessera-opt/tessera-opt").is_file():
        from tessera import runtime as rt
        from tessera.compiler.x86_native import package_matmul
        import record_x86_avx512_packet as rec
        pkg = package_matmul(rec._matmul_module(M, K, N), pipeline_name=rec.PIPELINE)
        if hashlib.sha256(pkg.image.payload).hexdigest() != digests["after"]:
            raise SystemExit("packaged production image is not the timed 'after' library")
        art = rt.RuntimeArtifact(metadata={"target": "x86", "architecture": "x86_64_avx512"},
                                 native_image=pkg.image, launch_descriptor=pkg.descriptor,
                                 tile_ir=pkg.tile_ir, target_ir=pkg.target_ir)
        lo = placed(np.zeros((M, N), np.float32), 0)
        res = rt.launch(art, {"a": a, "b": b, "o": lo, "M": M, "N": N, "K": K})
        if not (res.get("ok") and res.get("execution_kind") == "native_cpu"):
            raise SystemExit(f"runtime.launch failed: {res.get('reason')}")
        packaged = {"entry": pkg.descriptor.entry_symbol, "launch_ok": True, "launch_out": lo}

    calls = {name: (lambda f=fn, o=outs[name]: f(a.ctypes.data_as(fp), b.ctypes.data_as(fp),
                                                  i64(M), i64(N), i64(K), o.ctypes.data_as(fp)))
             for name, fn in fns.items()}
    for call in calls.values():
        call()
    ref = a.astype(np.float64) @ b.astype(np.float64)
    bitwise = bool(np.array_equal(outs["before"].view(np.uint32), outs["after"].view(np.uint32)))
    launch_bitwise = (bool(np.array_equal(packaged["launch_out"].view(np.uint32),
                                          outs["after"].view(np.uint32)))
                      if packaged else None)
    max_err = {name: float(np.max(np.abs(o.astype(np.float64) - ref))) if o.size else 0.0
               for name, o in outs.items()}

    # iterations per window from a rough single-call estimate (the slower build)
    import time
    t0 = time.perf_counter()
    calls["before"]()
    per_call = max(time.perf_counter() - t0, 1e-7)
    iterations = max(1, int(args.window_ms / 1e3 / per_call))

    calibration = clock.calibrate()
    hz = float(calibration["frequency_hz"])
    per = {"before": [], "after": []}
    agreement = []
    environment = "wsl2" if "microsoft" in platform.release().lower() else "bare_metal"
    for sample in range(args.samples):
        order = ("before", "after") if sample % 2 == 0 else ("after", "before")
        for name in order:
            call = calls[name]

            def region(call=call) -> None:
                for _ in range(iterations):
                    call()
            window = clock.measure(region, calibration)
            witness = clock.witness_sample(calibration, window, {"image": digests[name]},
                                           execution_environment=environment)
            tsc = witness["clocks"]["tsc_cycles"]
            if tsc.get("eligible_for_promotion") is not True:
                raise SystemExit(f"TSC witness refused: {tsc.get('provenance', {}).get('promotion_refused')}")
            reason = clock.verify_witness_sample(witness)
            if reason is not None:
                raise SystemExit(f"witness does not re-verify: {reason}")
            raw_ns = float(window["raw_end_ns"] - window["raw_start_ns"])
            tsc_ns = float(window["tsc_end"] - window["tsc_start"]) * 1e9 / hz
            err = abs(tsc_ns - raw_ns) / raw_ns
            if err > X86_AVX512_AGREEMENT_BAND:
                raise SystemExit(f"TSC disagrees with CLOCK_MONOTONIC_RAW by {err:.2%}")
            agreement.append(err)
            per[name].append(tsc_ns / iterations / 1e3)
    med = {name: statistics.median(v) for name, v in per.items()}
    print(json.dumps({
        "tag": args.tag, "host": os.uname().nodename, "pid": os.getpid(),
        "M": M, "N": N, "K": K, "b_mod64": b.ctypes.data % 64,
        "a_mod64": a.ctypes.data % 64, "o_mod64": outs["after"].ctypes.data % 64,
        "iterations_per_window": iterations, "samples": args.samples,
        "timing_source": "tsc_witness", "tsc_hz": hz,
        "max_tsc_raw_disagreement": max(agreement),
        "before_us": med["before"], "after_us": med["after"],
        "after_over_before": med["after"] / med["before"],
        "before_windows_us": per["before"], "after_windows_us": per["after"],
        "bitwise_before_eq_after": bitwise, "launch_eq_direct_after": launch_bitwise,
        "max_abs_err_vs_f64": max_err,
        "lib_sha256": digests, "production_package": bool(packaged),
        "after_path": after_path,
    }))


if __name__ == "__main__":
    main()
