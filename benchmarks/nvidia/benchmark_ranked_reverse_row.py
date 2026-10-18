"""Correctness-gated native row-member diagnostic; not public composed VJP."""
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
from tests.unit.test_sm120_ranked_reverse_rows import ranked_source
from tests.unit.test_sm120_reverse_row_schedule import invoke


def compile_member(tool, storage, shape):
    parts = [ranked_source(tool, storage, shape)]
    for flag in ("--tessera-graph-to-schedule", "--tessera-schedule-to-tile"):
        result = invoke(tool, parts[-1], flag)
        if result.returncode:
            raise RuntimeError(result.stderr)
        parts.append(result.stdout)
    entry = re.search(r'name = "([^"]+)"', parts[-1])[1]
    native, ptx, resources, compiler, toolchain, *_ = _compile_tile_ir(parts[-1], entry)
    return entry, ptx, {"stage_sha256": {
        name: hashlib.sha256(text.encode()).hexdigest()
        for name, text in zip(("graph", "schedule", "tile", "native", "ptx"),
                              (*parts, native, ptx), strict=True)},
        "compiler": compiler, "toolchain": toolchain, "packager_resources": dict(resources)}


def run(tool, storage, shape, samples, reps):
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    rows, columns = math.prod(shape[:-1]), shape[-1]
    count = rows * columns
    rng = np.random.default_rng(20261018)
    x = rng.normal(0, .75, shape).astype(dtype)
    g = rng.normal(0, .25, shape).astype(dtype)
    expected = reference(x, g).reshape(-1)
    rounded = expected.astype(dtype)
    ulp = np.abs(np.nextafter(rounded, dtype(np.inf)).astype(np.float64)
                 - rounded.astype(np.float64))
    budget = columns * 32 * np.finfo(np.float32).eps * (np.abs(expected) + .01) + ulp
    lib = driver()
    owners = []
    records = {}
    with NvidiaDeviceSession() as session:
        try:
            a, b = session.upload(x), session.upload(g)
            for label, logical in (("ranked", shape), ("rank2_control", (rows, columns))):
                started = time.perf_counter()
                entry, ptx, record = compile_member(tool, storage, logical)
                record["compile_host_ms"] = (time.perf_counter() - started) * 1e3
                module, function = ct.c_void_p(), ct.c_void_p()
                encoded = ct.create_string_buffer(ptx.encode())
                if lib.cuModuleLoadData(ct.byref(module), encoded):
                    raise RuntimeError("row-member module load failed")
                owners.append(module)
                if lib.cuModuleGetFunction(ct.byref(function), module, entry.encode()):
                    raise RuntimeError("row-member entry lookup failed")
                out = session.upload(np.full(count + 16, 123, dtype=dtype))
                args = [ct.c_uint64(a.ptr), ct.c_uint64(b.ptr), ct.c_uint64(out.ptr),
                        ct.c_int64(rows), ct.c_int64(columns)]
                params = (ct.c_void_p * 5)(*(ct.cast(ct.pointer(value), ct.c_void_p) for value in args))

                def launch(function=function, params=params):
                    if lib.cuLaunchKernel(function, (rows + 127) // 128, 1, 1,
                                          128, 1, 1, 0, ct.c_void_p(session.stream), params, None):
                        raise RuntimeError("row-member launch failed")

                launch()
                if session.synchronize():
                    raise RuntimeError("row-member synchronization failed")
                record.update(logical_shape=list(logical), entry=entry, samples_ms=[])
                records[label] = (record, launch, out, args, params)

            def check():
                for record, _, out, *_ in records.values():
                    actual = session.download(out)
                    np.testing.assert_array_equal(actual[count:], dtype(123))
                    values = actual[:count].astype(np.float64)
                    if not np.isfinite(values).all() or not np.all(np.abs(values - expected) <= budget):
                        raise AssertionError("independent pullback oracle mismatch")
                    record["max_abs_error"] = float(np.max(np.abs(values - expected)))

            check()
            for sample in range(samples):
                order = ("ranked", "rank2_control") if sample % 2 == 0 else ("rank2_control", "ranked")
                for label in order:
                    record, launch, *_ = records[label]
                    record["samples_ms"].append(session.measure(launch, reps=reps, warmup=3))
            check()
            for record, *_ in records.values():
                record["median_ms"] = median(record["samples_ms"])
                record["correctness"] = "passed_before_and_after_timing"
            return {"storage": storage, "logical_shape": list(shape), "rows": rows,
                    "columns": columns, "block_threads": 128,
                    "arms": {label: value[0] for label, value in records.items()}}
        finally:
            session.synchronize()
            for module in reversed(owners):
                if module.value and lib.cuModuleUnload(module):
                    raise RuntimeError("row-member module unload failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--reps", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 3 or args.reps < 1:
        parser.error("requires at least three samples and positive repetitions")
    device = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,uuid",
                                      "--format=csv,noheader"], text=True).strip().splitlines()
    if len(device) != 1 or device[0].split(",")[1].strip() != "12.0":
        raise RuntimeError("requires one owning SM120 GPU")
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("requires the matching candidate compiler")
    sources = ("src/compiler/programming_model/lib/NativeRowScalarizer.h",
               "src/compiler/programming_model/lib/NativeRowSchedule.h",
               "tests/unit/test_sm120_ranked_reverse_rows.py",
               "tests/device/nvidia/test_ranked_reverse_row_member.py",
               "tests/fixtures/native_sm120/reverse_row_ssa.mlir",
               "benchmarks/nvidia/benchmark_ranked_reverse_row.py",
               "python/tessera/compiler/nvidia_native.py",
               "python/tessera/compiler/emit/nvidia_cuda.py")
    provider = rt._load_nvidia_gemm_runtime()
    if provider is None:
        raise RuntimeError("requires the native session runtime")
    record = {"schema": "tessera.native_ranked_row_diagnostic.v1", "device": device[0],
              "route": "native paired-AD row fixture -> Schedule/Tile -> NVIDIA/NVVM/PTX -> raw CUDA member",
              "scope": "resident member event dispatch windows; enqueue gaps possible; not public/composed reverse",
              "order": "alternating ranked/control, identical inputs, 128-thread blocks",
              "compiler_sha256": hashlib.sha256(Path(tool).read_bytes()).hexdigest(),
              "session_runtime_sha256": hashlib.sha256(Path(provider._name).read_bytes()).hexdigest(),
              "source_hashes": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources},
              "samples": args.samples, "reps": args.reps,
              "rows": [run(tool, storage, shape, args.samples, args.reps)
                       for storage in ("f16", "bf16") for shape in
                       ((2, 3, 35), (2, 3, 5, 35), (2, 1, 3, 257), (2, 4, 17, 257))]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    for row in record["rows"]:
        print(row["storage"], row["logical_shape"],
              {key: arm["median_ms"] for key, arm in row["arms"].items()}, flush=True)


if __name__ == "__main__":
    main()
