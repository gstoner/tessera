#!/usr/bin/env python3
"""The typed Tile route's GEMM against the lanes it competes with, on the
host chip: a measured gap, never an admission.

`--dtype` selects the operand storage (fp16 by default; also bf16, fp8_e4m3,
fp8_e5m2, int8, int4). This exists because the panel and K-unroll rules were
derived on f16 and applied to f16 alone, while the fp8 and integer branches
of `lower_scheduled_matmul` still hardcode the 1x1 tile -- and on RDNA4 those
storages carry 2x and 4x the f16 ceiling, so they are the rows most worth
tiling. Each storage brings its own reference and error budget: an integer
product is exact in i32, so an integer row with any error at all is a wrong
kernel rather than a tolerance question. The directive and shipped HIP lanes
are f16 kernels with no counterpart at another storage, so they appear only
in the fp16 packet instead of being compared against a different program.

Variants, per shape, each in a fresh process (three runs, medians):

* ``typed-lds:<MxN>:<WMxWN>`` -- the same Tile IR through the LDS-staged
  multi-wave body: WM x WN waves per workgroup each owning the panel, both
  operands staged through shared memory per 16-wide K slab (B transposed so
  every fragment pack is a contiguous vector load).
* ``typed:<MxN>`` -- the production scheduled package's Tile IR
  (Graph -> Schedule -> Tile through ``lower_scheduled_matmul``) with its
  ``tessera.macro_tile_*`` rewritten to the named panel, compiled by
  ``rocm_native._compile_native_tile_ir`` and launched as the runtime's
  executor launches it (one wave per macro tile). A variant the pipeline
  refuses is recorded with its refusal, not skipped in silence.
* ``directive:<MxN>`` -- gfx11 only: the directive lane's production kernel
  (``select_rocm_gemm_schedule`` + ``_build_compiled_gemm_hsaco``).
* ``shipped:<register|lds|pipe>`` -- the hand-written HIP GEMM in
  ``libtessera_rocm_gemm.so`` (Tier 3), timed by its own bench entry, which
  reports whether it trusted the HIP event or fell back to the wall clock.

Timing for the module variants is the synchronized host wall clock over
``--iters`` launches, because HIP events on the fleet's WSL2 ``/dev/dxg``
hosts return ``hipSuccess`` with garbage intervals (runtime.py,
``_hip_resident_launch_latency``). Every variant is compiled and loaded
before any is timed, the clocks are ramped with a burst of launches, and the
variants are timed in ``ROUNDS`` interleaved rounds (median per variant):
the first packet timed each variant once in program order and read a 3.4x
"gap" between two byte-identical kernels. Neither WSL2 box exposes ``/dev/kfd``, so
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
PANELS = [(16, 16), (32, 64), (64, 64)]
#: `staging` is a no-op on the typed route (the generator's LDS body is
#: reachable only from its canonical scf.for matcher): the first packet of
#: 2026-09-18 recorded byte-identical backend IR for every register/lds pair
#: on both chips, so the recorder no longer pretends to vary it.
STAGING = "register"
#: (waves_m, waves_n) workgroup shapes for the LDS-staged typed body; each wave
#: owns one `PANELS` entry, so the block tile is (WM*panel_m, WN*panel_n).
LDS_WAVES = [(2, 2), (4, 2)]
#: Full 16-wide K slabs per loop iteration for the register body (1 = the
#: established loop). Latency hiding without registers for a bigger tile.
K_UNROLL = [1]
ROUNDS = 3


class _Storage:
    """What one operand storage needs from the harness.

    The typed route admits five storages on these chips and they do not share
    a reference: an integer product is *exact* in i32, so an integer row with
    any relative error at all is a wrong kernel rather than a tolerance
    question, while f16 accumulates in f32 and earns a budget. Keeping the
    budget beside the dtype is what stops an f16 tolerance from being applied
    to an int4 row, which would hide exactly the packing defects the nibble
    order can produce.
    """

    def __init__(self, name, operand, out, budget, low=None, high=None):
        self.name = name        # the `dtype=` the Graph module is built with
        self.operand = operand  # numpy dtype the host buffers carry
        self.out = out          # numpy dtype of the device result
        self.budget = budget    # max relative error against the reference
        self.low, self.high = low, high  # integer operand range, when integral

    @property
    def integral(self):
        return self.low is not None

    def operands(self, rng, m, n, k):
        if self.integral:
            a = rng.integers(self.low, self.high + 1, size=(m, k), dtype=np.int8)
            b = rng.integers(self.low, self.high + 1, size=(k, n), dtype=np.int8)
            return a, b
        a = (rng.standard_normal((m, k)) * 0.25).astype(self.operand)
        b = (rng.standard_normal((k, n)) * 0.25).astype(self.operand)
        return a, b

    def reference(self, a, b):
        if self.integral:
            return a.astype(np.int32) @ b.astype(np.int32)
        return a.astype(np.float32) @ b.astype(np.float32)


def _storages():
    """Built lazily: ml_dtypes is not needed for the f16 lane."""
    import ml_dtypes
    return {
        "fp16": _Storage("fp16", np.float16, np.float32, 2e-2),
        "bf16": _Storage("bf16", ml_dtypes.bfloat16, np.float32, 6e-2),
        # An fp8 product is exact in f32; only the accumulation order moves it.
        "fp8_e4m3": _Storage("fp8_e4m3", ml_dtypes.float8_e4m3fn, np.float32, 2e-3),
        "fp8_e5m2": _Storage("fp8_e5m2", ml_dtypes.float8_e5m2, np.float32, 2e-3),
        # int4 rides an int8 container, one logical value per byte, [-8, 7].
        "int8": _Storage("int8", np.int8, np.int32, 0.0, -8, 7),
        "int4": _Storage("int4", np.int8, np.int32, 0.0, -8, 7),
    }


DTYPE = "fp16"


def _hip():
    from tessera import runtime as rt
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("no usable HIP runtime on this host")
    return hip


def _memref(pointer, size):
    return [ct.c_void_p(pointer.value), ct.c_void_p(pointer.value), ct.c_int64(0),
            ct.c_int64(size), ct.c_int64(1)]


class _Loaded:
    """A loaded module variant with its device buffers, timed in rounds."""

    def __init__(self, hip, hsaco, symbol, a, b, m, n, k, macro, threads=32,
                 storage=None):
        self.hip = hip
        self.storage = storage
        self.mod = ct.c_void_p()
        if hip.hipModuleLoadData(ct.byref(self.mod), hsaco) != 0:
            raise RuntimeError("hipModuleLoadData refused the image")
        self.fn = ct.c_void_p()
        if hip.hipModuleGetFunction(ct.byref(self.fn), self.mod, symbol.encode()) != 0:
            raise RuntimeError(f"kernel symbol {symbol!r} not found")
        self.dev = [ct.c_void_p(), ct.c_void_p(), ct.c_void_p()]
        # Every admitted accumulator is 4 bytes wide (f32 or i32), so the
        # output allocation does not vary with the operand storage.
        for dev, nbytes in zip(self.dev, (a.nbytes, b.nbytes, 4 * m * n)):
            if hip.hipMalloc(ct.byref(dev), nbytes) != 0:
                raise RuntimeError("hipMalloc failed")
        hip.hipMemcpy(self.dev[0], a.ctypes.data_as(ct.c_void_p), a.nbytes, 1)
        hip.hipMemcpy(self.dev[1], b.ctypes.data_as(ct.c_void_p), b.nbytes, 1)
        args = (_memref(self.dev[0], m * k) + _memref(self.dev[1], k * n)
                + _memref(self.dev[2], m * n) + [ct.c_int64(m), ct.c_int64(n), ct.c_int64(k)])
        self.keep = args
        self.arr = (ct.c_void_p * len(args))()
        for i, v in enumerate(args):
            self.arr[i] = ct.cast(ct.byref(v), ct.c_void_p)
        macro_m, macro_n = macro
        self.gx, self.gy = (n + macro_n - 1) // macro_n, (m + macro_m - 1) // macro_m
        self.m, self.n, self.threads = m, n, threads

    def launch(self):
        return self.hip.hipModuleLaunchKernel(self.fn, self.gx, self.gy, 1, self.threads,
                                              1, 1, 0, None, self.arr, None)

    def check(self, a, b):
        for _ in range(3):
            if self.launch() != 0:
                raise RuntimeError("warm-up launch failed")
        self.hip.hipDeviceSynchronize()
        out = np.zeros((self.m, self.n), self.storage.out)
        self.hip.hipMemcpy(out.ctypes.data_as(ct.c_void_p), self.dev[2], 4 * self.m * self.n, 2)
        ref = self.storage.reference(a, b)
        scale = float(np.max(np.abs(ref.astype(np.float64)))) + 1e-6
        rel = float(np.max(np.abs(out.astype(np.float64) - ref.astype(np.float64))) / scale)
        if not math.isfinite(rel) or rel > self.storage.budget:
            raise RuntimeError(
                f"numerical budget failed for {self.storage.name}: "
                f"relative error {rel:.3e} > {self.storage.budget:.3e}")
        return rel

    def time_batch(self, iters):
        self.hip.hipDeviceSynchronize()
        started = time.perf_counter()
        for _ in range(iters):
            if self.launch() != 0:
                raise RuntimeError("timed launch failed")
        self.hip.hipDeviceSynchronize()
        return (time.perf_counter() - started) * 1e3 / iters

    def close(self):
        for dev in self.dev:
            self.hip.hipFree(dev)
        self.hip.hipModuleUnload(self.mod)


def _typed_variants(chip, shape, storage):
    """The production Tile IR for this shape and storage, re-panelled."""
    from tessera.compiler import rocm_native, scheduled_matmul
    from tests.unit.test_scheduled_matmul_consumers import _module

    m, n, k = shape
    artifact = scheduled_matmul.lower_scheduled_matmul(
        _module(target="rocm", shape=(m, k, n), dtype=storage.name),
        target=f"rocm_{chip}")
    for macro_m, macro_n in PANELS:
        tile_ir = re.sub(r"tessera\.macro_tile_m = \d+ : i64", f"tessera.macro_tile_m = {macro_m} : i64",
                         artifact.tile_ir)
        tile_ir = re.sub(r"tessera\.macro_tile_n = \d+ : i64", f"tessera.macro_tile_n = {macro_n} : i64", tile_ir)
        builds = [(f"typed:{macro_m}x{macro_n}" + (f":k{u}" if u > 1 else ""),
                   STAGING, (1, 1), u) for u in K_UNROLL]
        builds += [(f"typed-lds:{macro_m}x{macro_n}:{wm}x{wn}", "lds", (wm, wn), 1)
                   for wm, wn in LDS_WAVES]
        for name, staging, (wm, wn), unroll in builds:
            try:
                target_ir, backend_ir, payload, *_ = rocm_native._compile_native_tile_ir(
                    tile_ir, directive="tessera_rocm.wmma", family="matmul",
                    architecture=chip, staging=staging, lds_waves=(wm, wn),
                    k_unroll=unroll)
            except Exception as exc:  # the refusal is the result
                yield name, None, dict(refused=str(exc)[:300])
                continue
            block = (macro_m * wm, macro_n * wn)
            # Derived, not scraped: the generator's attribute lives on the
            # gpu.func, which the binary stage has already serialized away, so
            # reading it back gave 0. One 16-wide K slab of each operand.
            lds_bytes = 0 if staging == "register" else (block[0] + block[1]) * 16 * 2
            yield name, (payload, artifact.function_name, block, 32 * wm * wn), dict(
                backend_ir_sha256=hashlib.sha256(backend_ir.encode()).hexdigest(),
                block_tile=list(block), threads=32 * wm * wn, lds_bytes=lds_bytes,
                k_unroll=unroll,
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
    storage = _storages()[DTYPE]
    hip = _hip()
    rows = []
    for shape in SHAPES:
        m, n, k = shape
        rng = np.random.default_rng(1201)
        a, b = storage.operands(rng, m, n, k)
        variants = list(_typed_variants(chip, shape, storage))
        # The directive lane and the shipped HIP library are f16 kernels. They
        # are competitors for the f16 row and have no counterpart at another
        # storage, so they are absent rather than silently compared against a
        # different program.
        if chip.startswith("gfx11") and storage.name == "fp16":
            name, (hsaco, symbol, macro) = _directive_variant(chip, shape)
            variants.append((name, (hsaco, symbol, macro, 32), {}))
        loaded = []
        shape_rows = []
        for name, spec, extra in variants:
            row = dict(shape=list(shape), variant=name, **extra)
            shape_rows.append(row)
            if spec is None:
                continue
            hsaco, symbol, macro, threads = spec
            try:
                module = _Loaded(hip, hsaco, symbol, a, b, m, n, k, macro, threads,
                                 storage=storage)
                row["relative_error"] = module.check(a, b)
                loaded.append((row, module))
            except Exception as exc:
                row["refused"] = str(exc)[:300]
        # Ramp the clocks before anything is timed, then interleave rounds so
        # no variant is systematically first.
        if loaded:
            ramp_until = time.perf_counter() + 0.3
            while time.perf_counter() < ramp_until:
                loaded[0][1].launch()
            hip.hipDeviceSynchronize()
        batches = {id(row): [] for row, _ in loaded}
        for _ in range(ROUNDS):
            for row, module in loaded:
                batches[id(row)].append(module.time_batch(iters))
        for row, module in loaded:
            row.update(per_launch_ms=statistics.median(batches[id(row)]),
                       batch_ms=batches[id(row)], timing_source="host_wall_launch_and_sync")
            module.close()
        rows.extend(shape_rows)
        if storage.name == "fp16":
            rows.extend(_shipped_rows(shape, iters))
    return dict(chip=chip, pid=os.getpid(), dtype=storage.name, rows=rows)


def record(output, runs, iters):
    results = []
    with tempfile.TemporaryDirectory() as directory:
        for index in range(runs):
            child = Path(directory) / f"{index}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                       "--iters", str(iters), "--output", str(child)]
            command += ["--panels", ",".join(f"{m}x{n}" for m, n in PANELS)]
            command += ["--shapes", ",".join("x".join(str(v) for v in s) for s in SHAPES)]
            command += ["--k-unroll", ",".join(str(u) for u in K_UNROLL)]
            command += ["--dtype", DTYPE]
            if not LDS_WAVES:
                command.append("--register-only")
            subprocess.run(command, check=True)
            results.append(json.loads(child.read_text()))
    chip = results[0]["chip"]
    dtype = results[0].get("dtype", "fp16")
    if any(run.get("dtype", "fp16") != dtype for run in results):
        raise RuntimeError("cross-process storage differs")
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
        for key in ("relative_error", "production_panel", "backend_ir_sha256", "batch_ms",
                    "block_tile", "threads", "lds_bytes", "k_unroll"):
            if key in row:
                entry[key] = row[key]
        summary.append(entry)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(
        schema="tessera.typed_route_gap.v1", chip=chip, dtype=dtype,
        host=platform.platform(),
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
    parser.add_argument("--panels", type=str, default=None,
                        help="comma-separated MxN wave panels, e.g. 64x64,64x128,128x128")
    parser.add_argument("--k-unroll", type=str, default=None,
                        help="comma-separated K-slab unroll factors, e.g. 1,2,4")
    parser.add_argument("--register-only", action="store_true",
                        help="skip the LDS-staged variants (a register panel sweep)")
    parser.add_argument("--shapes", type=str, default=None,
                        help="comma-separated MxNxK shapes")
    parser.add_argument("--dtype", type=str, default=None,
                        help="operand storage: fp16 (default), bf16, fp8_e4m3, "
                             "fp8_e5m2, int8, int4. The directive and shipped "
                             "lanes are f16 kernels and appear only for fp16.")
    args = parser.parse_args()
    global PANELS, LDS_WAVES, SHAPES, K_UNROLL, DTYPE
    if args.panels:
        PANELS = [tuple(int(v) for v in p.split("x")) for p in args.panels.split(",")]
    if args.register_only:
        LDS_WAVES = []
    if args.shapes:
        SHAPES = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    if args.dtype:
        if args.dtype not in _storages():
            raise SystemExit(f"unknown storage {args.dtype!r}; "
                             f"expected one of {', '.join(sorted(_storages()))}")
        DTYPE = args.dtype
    if args.k_unroll:
        K_UNROLL = [int(v) for v in args.k_unroll.split(",")]
    if args.worker:
        args.output.write_text(json.dumps(worker(args.iters)) + "\n")
    else:
        record(args.output, args.runs, args.iters)


if __name__ == "__main__":
    main()
