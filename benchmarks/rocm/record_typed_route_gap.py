#!/usr/bin/env python3
"""The typed Tile route's f16 GEMM against the lanes it competes with, on the
host chip: a measured gap, never an admission.

Variants, per shape, each in a fresh process (three runs, medians):

* ``typed:<MxN>:<staging>`` -- the production scheduled package's Tile IR
  (Graph -> Schedule -> Tile through ``lower_scheduled_matmul``) with its
  ``tessera.macro_tile_*`` rewritten to the named panel, compiled by
  ``rocm_native._compile_native_tile_ir`` with the named staging, launched
  as the runtime's executor launches it (one wave per macro tile). A variant
  the pipeline refuses is recorded with its refusal, not skipped in silence.
* ``directive:<MxN>`` -- gfx11 only: the directive lane's production kernel
  (``select_rocm_gemm_schedule`` + ``_build_compiled_gemm_hsaco``).
* ``shipped:<register|lds|pipe>`` -- the hand-written HIP GEMM in
  ``libtessera_rocm_gemm.so`` (Tier 3), timed by its own bench entry, which
  reports whether it trusted the HIP event or fell back to the wall clock.

Timing for the module variants is the synchronized host wall clock over
``--iters`` launches, because HIP events on the fleet's WSL2 ``/dev/dxg``
hosts return ``hipSuccess`` with garbage intervals (runtime.py,
``_hip_resident_launch_latency``). Neither WSL2 box exposes ``/dev/kfd``, so
no counters exist: ``promotion.performance_eligible`` is False by
construction, and the packet is evidence for a *selection* decision only
when the gap is far outside run-to-run noise.

Usage (on a ROCm box, toolkit env sourced, TESSERA_ROCM_CHIP set)::

    python benchmarks/rocm/record_typed_route_gap.py --output \\
        benchmarks/baselines/typed_route_gap_20260918/gfx1201.json
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

SHAPES = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), (511, 513, 509)]
PANELS = [(16, 16), (32, 64)]
STAGINGS = ["register", "lds"]
ERROR_BUDGET = 2e-2  # relative to max |reference|, f16 storage / f32 accumulate


def _hip():
    from tessera import runtime as rt
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("no usable HIP runtime on this host")
    return hip


def _memref(pointer, size):
    return [ct.c_void_p(pointer.value), ct.c_void_p(pointer.value), ct.c_int64(0),
            ct.c_int64(size), ct.c_int64(1)]


def _launch_module(hip, hsaco, symbol, a, b, m, n, k, macro, iters):
    """Load ``hsaco``, run ``symbol`` as the executor would, return
    (per-launch ms, relative error)."""
    mod = ct.c_void_p()
    if hip.hipModuleLoadData(ct.byref(mod), hsaco) != 0:
        raise RuntimeError("hipModuleLoadData refused the image")
    fn = ct.c_void_p()
    if hip.hipModuleGetFunction(ct.byref(fn), mod, symbol.encode()) != 0:
        raise RuntimeError(f"kernel symbol {symbol!r} not found")
    da, db, dd = ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
    for dev, nbytes in ((da, a.nbytes), (db, b.nbytes), (dd, 4 * m * n)):
        if hip.hipMalloc(ct.byref(dev), nbytes) != 0:
            raise RuntimeError("hipMalloc failed")
    hip.hipMemcpy(da, a.ctypes.data_as(ct.c_void_p), a.nbytes, 1)
    hip.hipMemcpy(db, b.ctypes.data_as(ct.c_void_p), b.nbytes, 1)
    args = _memref(da, m * k) + _memref(db, k * n) + _memref(dd, m * n) + [
        ct.c_int64(m), ct.c_int64(n), ct.c_int64(k)]
    arr = (ct.c_void_p * len(args))()
    for i, v in enumerate(args):
        arr[i] = ct.cast(ct.byref(v), ct.c_void_p)
    macro_m, macro_n = macro
    gx, gy = (n + macro_n - 1) // macro_n, (m + macro_m - 1) // macro_m

    def once():
        return hip.hipModuleLaunchKernel(fn, gx, gy, 1, 32, 1, 1, 0, None, arr, None)

    for _ in range(3):
        if once() != 0:
            raise RuntimeError("warm-up launch failed")
    hip.hipDeviceSynchronize()
    out = np.zeros((m, n), np.float32)
    hip.hipMemcpy(out.ctypes.data_as(ct.c_void_p), dd, 4 * m * n, 2)
    ref = a.astype(np.float32) @ b.astype(np.float32)
    rel = float(np.max(np.abs(out - ref)) / (np.max(np.abs(ref)) + 1e-6))
    if not math.isfinite(rel) or rel > ERROR_BUDGET:
        raise RuntimeError(f"numerical budget failed: relative error {rel:.3e}")
    hip.hipDeviceSynchronize()
    started = time.perf_counter()
    for _ in range(iters):
        if once() != 0:
            raise RuntimeError("timed launch failed")
    hip.hipDeviceSynchronize()
    ms = (time.perf_counter() - started) * 1e3 / iters
    for dev in (da, db, dd):
        hip.hipFree(dev)
    hip.hipModuleUnload(mod)
    return ms, rel


def _typed_variants(chip, shape):
    """The production Tile IR for this shape, re-panelled and re-staged."""
    from tessera.compiler import rocm_native, scheduled_matmul
    from tests.unit.test_scheduled_matmul_consumers import _module

    m, n, k = shape
    artifact = scheduled_matmul.lower_scheduled_matmul(
        _module(target="rocm", shape=(m, k, n)), target=f"rocm_{chip}")
    for macro_m, macro_n in PANELS:
        tile_ir = re.sub(r"tessera\.macro_tile_m = \d+ : i64", f"tessera.macro_tile_m = {macro_m} : i64",
                         artifact.tile_ir)
        tile_ir = re.sub(r"tessera\.macro_tile_n = \d+ : i64", f"tessera.macro_tile_n = {macro_n} : i64", tile_ir)
        for staging in STAGINGS:
            name = f"typed:{macro_m}x{macro_n}:{staging}"
            try:
                target_ir, backend_ir, payload, *_ = rocm_native._compile_native_tile_ir(
                    tile_ir, directive="tessera_rocm.wmma", family="matmul",
                    architecture=chip, staging=staging)
            except Exception as exc:  # the refusal is the result
                yield name, None, dict(refused=str(exc)[:300])
                continue
            lds = re.search(r"tessera\.rocm\.lds_bytes = (\d+)", backend_ir)
            yield name, (payload, artifact.function_name, (macro_m, macro_n)), dict(
                backend_ir_sha256=hashlib.sha256(backend_ir.encode()).hexdigest(),
                lds_bytes=int(lds.group(1)) if lds else 0,
                production_panel=[artifact.macro_tile_m, artifact.macro_tile_n])


def _directive_variant(chip, shape):
    from tessera import runtime as rt
    from tessera.compiler.rocm_schedule import select_rocm_gemm_schedule

    m, n, k = shape
    schedule = select_rocm_gemm_schedule(m, n, k, dtype="f16", arch=chip)
    mt, nt = schedule.macro_tile
    hsaco = rt._build_compiled_gemm_hsaco(mt, nt, "f16", schedule=schedule)
    return f"directive:{16 * mt}x{16 * nt}", (hsaco, "gemm", (16 * mt, 16 * nt))


def _shipped_rows(shape, iters):
    from tests._support.rocm_build import rocm_gemm_lib_path

    path = rocm_gemm_lib_path()
    if path is None:
        return []
    lib = ct.CDLL(str(path))
    timer = lib.tessera_rocm_bench_last_timer_source
    timer.argtypes, timer.restype = [], ct.c_int
    rows = []
    m, n, k = shape
    for variant in ("register", "lds", "pipe"):
        staged = variant != "register"
        suffix = "_" + variant if staged else ""
        bench = getattr(lib, "tessera_rocm_wmma_gemm_f16_bench" + suffix)
        bench.argtypes = [ct.c_int] * (8 if staged else 6) + [ct.POINTER(ct.c_double)]
        bench.restype = ct.c_int
        value = ct.c_double()
        rc = bench(m, n, k, iters, *([2, 2, 2, 4] if staged else [1, 1]), ct.byref(value))
        source = timer()
        row = dict(shape=list(shape), variant=f"shipped:{variant}")
        if rc or source not in (0, 1) or not math.isfinite(value.value) or value.value <= 0:
            row["refused"] = f"bench rc={rc} timer={source} value={value.value}"
        else:
            row.update(per_launch_ms=value.value,
                       timing_source="hip_event_checked_against_wall" if source == 0 else "host_wall_launch_and_sync")
        rows.append(row)
    return rows


def worker(iters):
    from tessera import runtime as rt

    chip = rt._rocm_chip()
    if rt._rocm_live_arch() != chip:
        raise RuntimeError(f"pinned chip {chip} is not the live device {rt._rocm_live_arch()}")
    hip = _hip()
    rows = []
    for shape in SHAPES:
        m, n, k = shape
        rng = np.random.default_rng(1201)
        a = (rng.standard_normal((m, k)) * 0.25).astype(np.float16)
        b = (rng.standard_normal((k, n)) * 0.25).astype(np.float16)
        variants = list(_typed_variants(chip, shape))
        if chip.startswith("gfx11"):
            name, spec = _directive_variant(chip, shape)
            variants.append((name, spec, {}))
        for name, spec, extra in variants:
            row = dict(shape=list(shape), variant=name, **extra)
            if spec is not None:
                hsaco, symbol, macro = spec
                try:
                    ms, rel = _launch_module(hip, hsaco, symbol, a, b, m, n, k, macro, iters)
                    row.update(per_launch_ms=ms, relative_error=rel,
                               timing_source="host_wall_launch_and_sync")
                except Exception as exc:
                    row["refused"] = str(exc)[:300]
            rows.append(row)
        rows.extend(_shipped_rows(shape, iters))
    return dict(chip=chip, pid=os.getpid(), rows=rows)


def record(output, runs, iters):
    results = []
    with tempfile.TemporaryDirectory() as directory:
        for index in range(runs):
            child = Path(directory) / f"{index}.json"
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
                            "--iters", str(iters), "--output", str(child)], check=True)
            results.append(json.loads(child.read_text()))
    chip = results[0]["chip"]
    summary = []
    for index, row in enumerate(results[0]["rows"]):
        matches = [run["rows"][index] for run in results]
        if any((r["shape"], r["variant"]) != (row["shape"], row["variant"]) for r in matches):
            raise RuntimeError("cross-process variant order differs")
        entry = dict(shape=row["shape"], variant=row["variant"])
        timed = [r["per_launch_ms"] for r in matches if "per_launch_ms" in r]
        if timed:
            m, n, k = row["shape"]
            median = statistics.median(timed)
            entry.update(median_ms=median, run_ms=timed,
                         tflops=2.0 * m * n * k / (median * 1e-3) / 1e12,
                         timing_source=row.get("timing_source"))
        else:
            entry["refused"] = row.get("refused")
        for key in ("relative_error", "lds_bytes", "production_panel", "backend_ir_sha256"):
            if key in row:
                entry[key] = row[key]
        summary.append(entry)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(
        schema="tessera.typed_route_gap.v1", chip=chip, host=platform.platform(),
        iters=iters, runs=runs, summary=summary, raw=results,
        evidence_scope="measured_gap",
        promotion=dict(correctness_eligible=False, performance_eligible=False,
                       reason="host wall clock on a WSL2 /dev/dxg host; no /dev/kfd, no counters; "
                              "a selection input only where the gap dwarfs run-to-run spread"),
    ), indent=2, sort_keys=True) + "\n")
    for entry in summary:
        status = f"{entry['median_ms']:.4f} ms {entry['tflops']:.2f} TFLOP/s" if "median_ms" in entry else f"refused: {entry['refused']}"
        print(f"{tuple(entry['shape'])!s:>20} {entry['variant']:<24} {status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.worker:
        args.output.write_text(json.dumps(worker(args.iters)) + "\n")
    else:
        record(args.output, args.runs, args.iters)


if __name__ == "__main__":
    main()
