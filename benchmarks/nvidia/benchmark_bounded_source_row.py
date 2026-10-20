"""Bounded/static native pullback member A/B; not a public composed-VJP benchmark."""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
import subprocess
import time

import ml_dtypes
import numpy as np

from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.nvidia_native import _compile_tile_ir
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests.device.nvidia.test_ranked_reverse_row_member import reference
from tests.device.nvidia.test_scalar_adjoint_contraction import driver
from tests.unit.test_sm120_bounded_source_rows import bounded_source
from tests.unit.test_sm120_ranked_reverse_rows import ranked_source
from tests.unit.test_sm120_reverse_row_schedule import invoke


def compile_graph(tool, graph):
    started = time.perf_counter()
    parts = [graph]
    for flag in ("--tessera-graph-to-schedule", "--tessera-schedule-to-tile"):
        result = invoke(tool, parts[-1], flag)
        if result.returncode:
            raise RuntimeError(result.stderr)
        parts.append(result.stdout)
    entry = re.search(r'name = "([^"]+)"', parts[-1])[1]
    native, ptx, resources, compiler, toolchain, *_ = _compile_tile_ir(parts[-1], entry)
    return entry, ptx, {
        "compile_host_ms": (time.perf_counter() - started) * 1e3,
        "stage_sha256": {name: hashlib.sha256(text.encode()).hexdigest()
                         for name, text in zip(("graph", "schedule", "tile", "native", "ptx"),
                                               (*parts, native, ptx), strict=True)},
        "compiler": compiler, "toolchain": toolchain, "packager_resources": dict(resources)}


def run(tool, storage, capacity, samples, reps, *, reverse=False):
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    lib = driver()
    bounded = compile_graph(tool, bounded_source(tool, storage, capacity))
    records = []
    modules = []
    with NvidiaDeviceSession() as session:
        try:
            bound_module, bound_function = ct.c_void_p(), ct.c_void_p()
            encoded = ct.create_string_buffer(bounded[1].encode())
            if lib.cuModuleLoadData(ct.byref(bound_module), encoded):
                raise RuntimeError("bounded member module load failed")
            modules.append(bound_module)
            if lib.cuModuleGetFunction(ct.byref(bound_function), bound_module, bounded[0].encode()):
                raise RuntimeError("bounded member entry lookup failed")
            queries = (capacity[2], 3, 1) if reverse else (1, 3, capacity[2])
            for sq in queries:
                shape = (*capacity[:2], sq, capacity[-1])
                rows, columns = math.prod(shape[:-1]), shape[-1]
                count = rows * columns
                rng = np.random.default_rng(20261019 + sq)
                x = rng.normal(0, .75, shape).astype(dtype)
                g = rng.normal(0, .25, shape).astype(dtype)
                expected = reference(x, g).reshape(-1)
                rounded = expected.astype(dtype)
                ulp = np.abs(np.nextafter(rounded, dtype(np.inf)).astype(np.float64)
                             - rounded.astype(np.float64))
                budget = columns * 32 * np.finfo(np.float32).eps * (np.abs(expected) + .01) + ulp
                a, b = session.upload(x), session.upload(g)
                static = compile_graph(tool, ranked_source(tool, storage, shape))
                static_module, static_function = ct.c_void_p(), ct.c_void_p()
                encoded = ct.create_string_buffer(static[1].encode())
                if lib.cuModuleLoadData(ct.byref(static_module), encoded):
                    raise RuntimeError("static member module load failed")
                modules.append(static_module)
                if lib.cuModuleGetFunction(ct.byref(static_function), static_module, static[0].encode()):
                    raise RuntimeError("static member entry lookup failed")
                arms = {}
                for label, function, compiled in (("bounded", bound_function, bounded),
                                                   ("static", static_function, static)):
                    out = session.upload(np.full(count + 16, 123, dtype=dtype))
                    values = [ct.c_uint64(a.ptr), ct.c_uint64(b.ptr), ct.c_uint64(out.ptr),
                              ct.c_int64(rows), ct.c_int64(columns)]
                    params = (ct.c_void_p * 5)(*(ct.cast(ct.pointer(v), ct.c_void_p) for v in values))

                    def launch(function=function, params=params):
                        if lib.cuLaunchKernel(function, (rows + 127) // 128, 1, 1,
                                              128, 1, 1, 0, ct.c_void_p(session.stream), params, None):
                            raise RuntimeError("native member launch failed")

                    launch()
                    if session.synchronize():
                        raise RuntimeError("native member synchronization failed")
                    record = {**compiled[2], "entry": compiled[0],
                              "event_dispatch_samples_ms": [], "completed_raw_host_samples_ms": []}
                    arms[label] = (record, launch, out, values, params)

                def check():
                    for record, _, out, *_ in arms.values():
                        actual = session.download(out)
                        np.testing.assert_array_equal(actual[count:], dtype(123))
                        values = actual[:count].astype(np.float64)
                        if not np.isfinite(values).all() or not np.all(np.abs(values - expected) <= budget):
                            raise AssertionError("independent native pullback oracle mismatch")
                        record["max_abs_error"] = float(np.max(np.abs(values - expected)))
                    np.testing.assert_array_equal(session.download(arms["bounded"][2]).view(np.uint8),
                                                  session.download(arms["static"][2]).view(np.uint8))
                    np.testing.assert_array_equal(session.download(a).view(np.uint8), x.view(np.uint8))
                    np.testing.assert_array_equal(session.download(b).view(np.uint8), g.view(np.uint8))

                check()
                for sample in range(samples):
                    order = ("bounded", "static") if (sample % 2 == 0) != reverse else ("static", "bounded")
                    for label in order:
                        record, launch, *_ = arms[label]
                        record["event_dispatch_samples_ms"].append(session.measure(launch, reps=reps, warmup=3))
                        started = time.perf_counter()
                        for _ in range(reps):
                            launch()
                        if session.synchronize():
                            raise RuntimeError("completed raw launch window failed")
                        record["completed_raw_host_samples_ms"].append((time.perf_counter() - started) * 1e3 / reps)
                check()
                for record, *_ in arms.values():
                    record["event_dispatch_median_ms"] = median(record["event_dispatch_samples_ms"])
                    record["completed_raw_host_median_ms"] = median(record["completed_raw_host_samples_ms"])
                    record["correctness"] = "passed_before_and_after_timing"
                records.append({"storage": storage, "runtime_shape": list(shape),
                                "capacity_shape": list(capacity), "block_threads": 128,
                                "arms": {name: arm[0] for name, arm in arms.items()}})
        finally:
            session.synchronize()
            for module in reversed(modules):
                if module.value and lib.cuModuleUnload(module):
                    raise RuntimeError("member module unload failed")
    assert len({r["arms"]["bounded"]["stage_sha256"]["ptx"] for r in records}) == 1
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--reps", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true",
                        help="reverse profile/query order and initial A/B arm")
    args = parser.parse_args()
    if args.samples < 3 or args.reps < 1:
        parser.error("at least three samples and positive reps required")
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,uuid",
         "--format=csv,noheader"], text=True).strip().splitlines()
    if len(devices) != 1 or devices[0].split(",")[1].strip() != "12.0":
        raise RuntimeError("requires one exact SM120 device")
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("requires a matching compiler")
    provider = rt._load_nvidia_gemm_runtime()
    if provider is None:
        raise RuntimeError("requires the native session runtime")
    paths = ("src/compiler/programming_model/lib/NativeRowScalarizer.h",
             "src/compiler/programming_model/lib/NativeRowSchedule.h",
             "src/compiler/ir/include/Tessera/IR/RowRegionContract.h",
             "src/compiler/ir/include/Tessera/IR/TileBufferViewContract.h",
             "src/compiler/ir/include/Tessera/Dialect/Tile/TileOps.td",
             "src/compiler/codegen/tessera_gpu_backend_NVIDIA/include/tessera/gpu/IR/TesseraNVIDIADialect.td",
             "tests/unit/test_sm120_ranked_reverse_rows.py",
             "tests/unit/test_sm120_reverse_row_schedule.py",
             "tests/fixtures/native_sm120/reverse_row_ssa.mlir",
             "tests/device/nvidia/test_ranked_reverse_row_member.py",
             "src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
             "tests/unit/test_sm120_bounded_source_rows.py",
             "tests/device/nvidia/test_bounded_source_row_member.py",
             "benchmarks/nvidia/benchmark_bounded_source_row.py")
    packet = {"schema": "tessera.bounded_source_row_member.v1", "device": devices[0],
              "scope": "native raw member event dispatch and completed raw host windows; not isolated profiler kernel time, public JIT or composed VJP",
              "order": "alternating bounded/static with identical inputs",
              "reverse_profile_and_initial_arm_order": args.reverse,
              "compiler_sha256": hashlib.sha256(Path(tool).read_bytes()).hexdigest(),
              "session_runtime_sha256": hashlib.sha256(Path(provider._name).read_bytes()).hexdigest(),
              "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
              "samples": args.samples, "reps": args.reps, "rows": []}
    profiles = [(storage, capacity) for storage in ("f16", "bf16")
                for capacity in ((2, 3, 7, 35), (1, 2, 17, 257))]
    if args.reverse:
        profiles.reverse()
    for storage, capacity in profiles:
        rows = run(tool, storage, capacity, args.samples, args.reps, reverse=args.reverse)
        packet["rows"].extend(rows)
        for row in rows:
            print(storage, row["runtime_shape"],
                  {k: v["event_dispatch_median_ms"] for k, v in row["arms"].items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
