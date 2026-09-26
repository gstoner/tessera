#!/usr/bin/env python3
"""ROCM-SPLIT-K-1: split-K against the unsplit kernel for the same program,
paired and interleaved, on the host chip. A measurement, never an admission.

For each ``--shapes`` entry (``MxNxK``) and storage the production route is
lowered once (Graph -> Schedule -> Tile through ``lower_scheduled_matmul``) and
three kinds of variant are built from its Tile IR:

* ``split:S`` (production) -- the scheduled package exactly as
  ``rocm_native.package_scheduled_matmul`` emits it: the partial kernel over
  grid.z = S into an fp32 [S, M, N] workspace, then the ordered reduction.
  One timed iteration is BOTH launches. **It is not everything a caller of
  ``runtime.launch`` pays:** that path also ``hipMalloc``s and ``hipFree``s the
  S*M*N*4-byte workspace on every call, and this harness allocates it once per
  variant, outside the timed loop (as it does every other buffer). The
  per-call allocation cost is therefore excluded from every row here.
* ``unsplit`` (measurement-only control) -- the same Tile IR with the
  ``tessera.split_k`` pair removed, compiled by the same
  ``_compile_native_tile_ir`` call. It is the program the route ran before
  ROCM-SPLIT-K-1, and it is NOT a production route for a split schedule.
* ``split:S'`` for each ``--extra-slices`` (measurement-only) -- the same Tile
  IR with the slice count rewritten, to show where the selected S sits on the
  axis. The selection rule is not tuned from these rows.

Correctness first: every variant is checked against an f64 reference, and the
split variants against the unsplit control, before anything is timed; a
variant that fails is recorded with its failure, not timed.

Timing is the synchronized host wall clock over ``--iters`` iterations per
batch, because HIP events on the fleet's WSL2 ``/dev/dxg`` hosts are not a
trustworthy device clock (runtime.py, ``_hip_resident_launch_latency``) and
neither WSL2 ROCm box exposes counters. Variants are ramped, then timed in
``--rounds`` interleaved rounds so none is systematically first, each run in a
fresh process (``--runs``). Rows carry ``route`` and ``timing_source``
(Decision #12); ``performance_eligible`` is False by construction.

Usage (on Tajasarus, toolkit env sourced, TESSERA_ROCM_CHIP=gfx1201)::

    python benchmarks/rocm/record_split_k_router_gate.py --output \\
        benchmarks/baselines/rocm_split_k_20260926/gfx1201.json
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

REDUCE_THREADS = 256


def _memref(pointer, size):
    return [ct.c_void_p(pointer.value), ct.c_void_p(pointer.value), ct.c_int64(0),
            ct.c_int64(size), ct.c_int64(1)]


def _pack(args):
    arr = (ct.c_void_p * len(args))()
    for i, v in enumerate(args):
        arr[i] = ct.cast(ct.byref(v), ct.c_void_p)
    return arr


def _graph_module(m, n, k, dtype):
    """A one-matmul Graph module, built here rather than borrowed from the
    test suite so the recorder does not depend on a test helper's signature."""
    from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType

    element = {"fp16": "f16", "bf16": "bf16"}[dtype]
    a = IRType(f"tensor<{m}x{k}x{element}>", (str(m), str(k)), dtype)
    b = IRType(f"tensor<{k}x{n}x{element}>", (str(k), str(n)), dtype)
    out = IRType(f"tensor<{m}x{n}xf32>", (str(m), str(n)), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="rocm_split_k_router_gate",
        args=[IRArg("a", a), IRArg("b", b)],
        result_types=[out],
        body=[IROp(result="o", op_name="tessera.matmul", operands=["%a", "%b"],
                   operand_types=[str(a), str(b)], result_type=str(out),
                   kwargs={"activation": "none"})],
        return_values=["%o"],
    )])


def _check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed rc={rc}")


class _Variant:
    """One loaded kernel family: unsplit (one launch) or split (two).

    Owns its module and device buffers from the first line of ``__init__``,
    so ``close()`` is safe on a variant whose construction or correctness
    check failed part-way."""

    def __init__(self, hip, hsaco, symbol, a, b, m, n, k, macro, split_k):
        self.hip, self.m, self.n, self.split_k = hip, m, n, split_k
        self.mod = ct.c_void_p()
        self.dev = [ct.c_void_p() for _ in range(4)]
        try:
            self._init(hsaco, symbol, a, b, m, n, k, macro, split_k)
        except Exception:
            self.close()
            raise

    def _init(self, hsaco, symbol, a, b, m, n, k, macro, split_k):
        hip = self.hip
        if hip.hipModuleLoadData(ct.byref(self.mod), hsaco) != 0:
            raise RuntimeError("hipModuleLoadData refused the image")
        self.fn = ct.c_void_p()
        if hip.hipModuleGetFunction(ct.byref(self.fn), self.mod, symbol.encode()) != 0:
            raise RuntimeError(f"kernel symbol {symbol!r} not found")
        self.reduce_fn = ct.c_void_p()
        if split_k > 1 and hip.hipModuleGetFunction(
                ct.byref(self.reduce_fn), self.mod, f"{symbol}_splitk_reduce".encode()) != 0:
            raise RuntimeError("split-K reduce symbol not found")
        workspace = split_k * m * n if split_k > 1 else 0
        for dev, nbytes in zip(self.dev, (a.nbytes, b.nbytes, 4 * m * n, 4 * max(workspace, 1))):
            if hip.hipMalloc(ct.byref(dev), nbytes) != 0:
                raise RuntimeError("hipMalloc failed")
        _check(hip.hipMemcpy(self.dev[0], a.ctypes.data_as(ct.c_void_p), a.nbytes, 1), "hipMemcpy A")
        _check(hip.hipMemcpy(self.dev[1], b.ctypes.data_as(ct.c_void_p), b.nbytes, 1), "hipMemcpy B")
        target = self.dev[3] if split_k > 1 else self.dev[2]
        target_size = workspace if split_k > 1 else m * n
        self.keep = (_memref(self.dev[0], m * k) + _memref(self.dev[1], k * n)
                     + _memref(target, target_size) + [ct.c_int64(m), ct.c_int64(n), ct.c_int64(k)])
        self.arr = _pack(self.keep)
        if split_k > 1:
            self.rkeep = (_memref(self.dev[3], workspace) + _memref(self.dev[2], m * n)
                          + [ct.c_int64(m), ct.c_int64(n)])
            self.rarr = _pack(self.rkeep)
            self.rgrid = (m * n + REDUCE_THREADS - 1) // REDUCE_THREADS
        macro_m, macro_n = macro
        self.gx, self.gy = (n + macro_n - 1) // macro_n, (m + macro_m - 1) // macro_m

    def launch(self):
        rc = self.hip.hipModuleLaunchKernel(self.fn, self.gx, self.gy, self.split_k, 32, 1, 1,
                                            0, None, self.arr, None)
        if rc == 0 and self.split_k > 1:
            rc = self.hip.hipModuleLaunchKernel(self.reduce_fn, self.rgrid, 1, 1, REDUCE_THREADS,
                                                1, 1, 0, None, self.rarr, None)
        return rc

    def result(self):
        for _ in range(3):
            if self.launch() != 0:
                raise RuntimeError("warm-up launch failed")
        _check(self.hip.hipDeviceSynchronize(), "hipDeviceSynchronize")
        out = np.zeros((self.m, self.n), np.float32)
        _check(self.hip.hipMemcpy(out.ctypes.data_as(ct.c_void_p), self.dev[2],
                                  4 * self.m * self.n, 2), "hipMemcpy D")
        return out

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
            if dev.value:
                self.hip.hipFree(dev)
                dev.value = None
        if self.mod.value:
            self.hip.hipModuleUnload(self.mod)
            self.mod.value = None


def _variants(chip, shape, dtype, extra_slices):
    from tessera.compiler import rocm_native, scheduled_matmul

    m, n, k = shape
    artifact = scheduled_matmul.lower_scheduled_matmul(
        _graph_module(m, n, k, dtype), target=f"rocm_{chip}")
    package = rocm_native.package_scheduled_matmul(artifact, pipeline_name="tessera-lower-to-rocm")
    provenance = package.descriptor.provenance
    macro = tuple(provenance["macro_tile"])
    k_unroll = int(provenance["k_unroll"])
    selected = int(artifact.split_k)
    yield (f"split:{selected}" if selected > 1 else "unsplit", package.image.payload,
           artifact.function_name, macro, selected,
           dict(route=provenance["route"], physical_route=provenance["physical_route"],
                production=True, schedule_digest=artifact.schedule_digest))

    def compile_variant(tile_ir):
        _t, backend_ir, payload, *_ = rocm_native._compile_native_tile_ir(
            tile_ir, directive="tessera_rocm.wmma", family="matmul", architecture=chip,
            staging="register", lds_waves=(2, 2), k_unroll=k_unroll)
        return payload, hashlib.sha256(backend_ir.encode()).hexdigest()

    if selected > 1:
        unsplit_ir = re.sub(r', tessera\.split_k = \d+ : i64, tessera\.split_k_reduction = "ordered"',
                            "", artifact.tile_ir)
        if unsplit_ir == artifact.tile_ir or "split_k" in unsplit_ir:
            raise RuntimeError("could not derive the unsplit control from the Tile IR")
        payload, digest = compile_variant(unsplit_ir)
        yield ("unsplit", payload, artifact.function_name, macro, 1,
               dict(route="measurement_only_unsplit_control",
                    physical_route=provenance["physical_route"].rsplit("_splitk", 1)[0],
                    production=False, backend_ir_sha256=digest))
        for slices in extra_slices:
            if slices == selected:
                continue
            rewritten = artifact.tile_ir.replace(
                f"tessera.split_k = {selected} : i64", f"tessera.split_k = {slices} : i64")
            try:
                payload, digest = compile_variant(rewritten)
            except Exception as exc:  # the refusal is the result
                yield (f"split:{slices}", None, None, macro, slices,
                       dict(route="measurement_only_slice_sweep", production=False,
                            refused=str(exc)[:300]))
                continue
            yield (f"split:{slices}", payload, artifact.function_name, macro, slices,
                   dict(route="measurement_only_slice_sweep",
                        physical_route=provenance["physical_route"].rsplit("_splitk", 1)[0]
                        + f"_splitk{slices}_ordered",
                        production=False, backend_ir_sha256=digest))


def worker(shapes, dtypes, iters, rounds, extra_slices):
    import ml_dtypes
    from tessera import runtime as rt

    chip = rt._rocm_chip()
    if rt._rocm_live_arch() != chip:
        raise RuntimeError(f"pinned chip {chip} is not the live device {rt._rocm_live_arch()}")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) != 0:
        raise RuntimeError("no usable HIP runtime on this host")
    rows = []
    for dtype in dtypes:
        storage = np.float16 if dtype == "fp16" else ml_dtypes.bfloat16
        for shape in shapes:
            m, n, k = shape
            rng = np.random.default_rng(1201)
            a = (rng.standard_normal((m, k)) * 0.25).astype(storage)
            b = (rng.standard_normal((k, n)) * 0.25).astype(storage)
            ref = a.astype(np.float64) @ b.astype(np.float64)
            scale = float(np.max(np.abs(ref))) + 1e-6
            loaded, shape_rows, outputs = [], [], {}
            for name, payload, symbol, macro, slices, extra in _variants(chip, shape, dtype, extra_slices):
                row = dict(backend="rocm", op="matmul", shape=[m, n, k], dtype=dtype,
                           device=chip, variant=name, split_k=slices, **extra)
                shape_rows.append(row)
                if payload is None:
                    continue
                variant = None
                try:
                    variant = _Variant(hip, payload, symbol, a, b, m, n, k, macro, slices)
                    out = variant.result()
                    rel = float(np.max(np.abs(out.astype(np.float64) - ref))) / scale
                    row["relative_error_vs_f64"] = rel
                    if not math.isfinite(rel) or rel > 2e-3:
                        raise RuntimeError(f"relative error {rel:.3e} exceeds 2e-3")
                    outputs[name] = out
                    loaded.append((row, variant))
                except Exception as exc:
                    row["refused"] = str(exc)[:300]
                    if variant is not None:
                        variant.close()
            if "unsplit" in outputs:
                for row, _ in loaded:
                    if row["variant"] != "unsplit":
                        diff = np.abs(outputs[row["variant"]].astype(np.float64)
                                      - outputs["unsplit"].astype(np.float64))
                        row["max_abs_diff_vs_unsplit"] = float(np.max(diff))
                        row["relative_diff_vs_unsplit"] = float(np.max(diff)) / scale
            if loaded:
                ramp_until = time.perf_counter() + 0.3
                while time.perf_counter() < ramp_until:
                    for _, variant in loaded:
                        variant.launch()
                hip.hipDeviceSynchronize()
            batches = {id(row): [] for row, _ in loaded}
            for r in range(rounds):
                order = loaded if r % 2 == 0 else list(reversed(loaded))
                for row, variant in order:
                    batches[id(row)].append(variant.time_batch(iters))
            flops = 2.0 * m * n * k
            for row, variant in loaded:
                latency = statistics.median(batches[id(row)])
                row.update(latency_ms=latency, batch_ms=batches[id(row)],
                           tflops=flops / (latency * 1e-3) / 1e12,
                           timing_source="host_wall_launch_and_sync",
                           launches_per_iteration=2 if variant.split_k > 1 else 1)
                variant.close()
            base = next((row for row, _ in loaded if row["variant"] == "unsplit"), None)
            if base is not None:
                for row, _ in loaded:
                    if row is base:
                        continue
                    paired = [u / s for u, s in zip(batches[id(base)], batches[id(row)])]
                    row["speedup_vs_unsplit_median"] = statistics.median(paired)
                    row["speedup_vs_unsplit_per_round"] = paired
                    row["rounds_split_faster"] = sum(p > 1.0 for p in paired)
            rows.extend(shape_rows)
    return dict(chip=chip, pid=os.getpid(), rows=rows)


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def record(output, runs, argv_tail):
    results = []
    with tempfile.TemporaryDirectory() as directory:
        for run in range(runs):
            child = Path(directory) / f"run{run}.json"
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
                            "--output", str(child), *argv_tail], check=True)
            results.append(json.loads(child.read_text()))
    from tessera.compiler.scheduled_matmul import find_tessera_opt
    tool = find_tessera_opt()
    packet = dict(
        item="ROCM-SPLIT-K-1",
        host=platform.node(),
        chip=results[0]["chip"],
        git_head=_git("rev-parse", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain")),
        tessera_opt=str(tool) if tool else None,
        tessera_opt_sha256=hashlib.sha256(tool.read_bytes()).hexdigest() if tool else None,
        tessera_version=_git("describe", "--always", "--dirty"),
        timing_source="host_wall_launch_and_sync",
        promotion=dict(performance_eligible=False,
                       reason="WSL2 /dev/dxg host: no device clock witness, no counters"),
        runs=results,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet, indent=2) + "\n")
    # A one-screen summary: per (dtype, shape, variant), the median over runs.
    summary = {}
    for result in results:
        for row in result["rows"]:
            key = (row["dtype"], "x".join(map(str, row["shape"])), row["variant"])
            summary.setdefault(key, []).append(row)
    for (dtype, shape, variant), rows in sorted(summary.items()):
        if any("refused" in r for r in rows):
            print(f"{dtype} {shape} {variant}: refused: {rows[0].get('refused')}")
            continue
        lat = statistics.median(r["latency_ms"] for r in rows)
        speed = [r["speedup_vs_unsplit_median"] for r in rows if "speedup_vs_unsplit_median" in r]
        wins = sum(r.get("rounds_split_faster", 0) for r in rows)
        total = sum(len(r.get("speedup_vs_unsplit_per_round", [])) for r in rows)
        extra = (f" speedup_vs_unsplit(median of run medians)={statistics.median(speed):.3f}"
                 f" rounds_split_faster={wins}/{total}") if speed else ""
        print(f"{dtype} {shape} {variant}: {lat * 1e3:.2f} us/iter"
              f" rel_err={statistics.median(r['relative_error_vs_f64'] for r in rows):.2e}{extra}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shapes", default="16x256x2048")
    parser.add_argument("--dtypes", default="fp16,bf16")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--extra-slices", default="4,8")
    args = parser.parse_args()
    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",") if s]
    dtypes = [d for d in args.dtypes.split(",") if d]
    extra = [int(v) for v in args.extra_slices.split(",") if v]
    if args.worker:
        result = worker(shapes, dtypes, args.iters, args.rounds, extra)
        args.output.write_text(json.dumps(result))
        return
    tail = ["--shapes", args.shapes, "--dtypes", args.dtypes, "--iters", str(args.iters),
            "--rounds", str(args.rounds), "--extra-slices", args.extra_slices]
    record(args.output, args.runs, tail)


if __name__ == "__main__":
    main()
