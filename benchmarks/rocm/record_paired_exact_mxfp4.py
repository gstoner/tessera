"""Paired exact MXFP4 native-program timing, separate from captured member timing."""
from __future__ import annotations

import argparse
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from benchmarks.rocm.record_gfx1201_interleaved_compiler_formats import collect_interleaved_windows
from tessera import runtime
from tessera.compiler.native_scaled_program import NativeScaledProgram, PreparedScaledProgram
from tests.device.rocm.test_mxfp4_residual_native_execution import coefficient, frame
from tests.unit.test_native_mxfp4_residual_gate import typed_case


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ProgramWindows:
    """Use the native program owner; HIP spans include host dispatch gaps."""

    def window(self, engine, launches, *, bracketed):
        assert bracketed
        start = time.perf_counter_ns()
        engine.generation, elapsed = engine.prepared.invoke(repeats=launches, timed=True)
        return {"device_window_ms": elapsed * launches,
                "host_window_ms": (time.perf_counter_ns() - start) / 1e6,
                "native_event_ms_per_invocation": elapsed, "launches": launches}


class Engine:
    def __init__(self, prepared):
        self.prepared = prepared
        self.generation = None


def record(shape, compilers, *, windows, minimum):
    values, _ = frame(shape)
    expected = coefficient(values) * values[4]
    engines, owners, metadata = {}, {}, {}
    try:
        for label, compiler in compilers.items():
            os.environ["TESSERA_OPT"] = str(compiler)
            os.environ["TESSERA_NVIDIA_OPT"] = str(compiler)
            owner, _ = typed_case(shape)
            owners[label] = owner
            np.testing.assert_array_equal(owner(*values), expected)
            receipt = owner._native_descriptor_last_receipt
            if receipt["execution_kind"] != "native_gpu":
                raise RuntimeError("ordinary exact MXFP4 primal lacked owning native execution")
            package = next(iter(owner._native_composed_scaled_cache.values()))[0]
            if not isinstance(package, NativeScaledProgram) or package.target != "rocm_gfx1201":
                raise RuntimeError("compiler-owned native program required")
            prepared = PreparedScaledProgram(package, values,
                runtime_library=os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"])
            engines[label] = Engine(prepared)
            metadata[label] = {
                "program_digest": hashlib.sha256(package.program_json.encode()).hexdigest(),
                "image_sha256": [hashlib.sha256(image).hexdigest() for image in package.images],
                "member_operations": [step["operation"] for step in json.loads(package.program_json)["steps"]],
            }

        def verify(engine):
            outputs = engine.prepared.read(engine.generation)
            assert len(outputs) == 1
            np.testing.assert_array_equal(outputs[0], expected)

        samples, orders, admission = collect_interleaved_windows(
            ProgramWindows(), engines, windows=windows, min_window_ms=minimum,
            verify_output=verify)
        ratios = [a["native_event_ms_per_invocation"] / b["native_event_ms_per_invocation"]
                  for a, b in zip(samples["candidate"], samples["reference"], strict=True)]
        return {"shape_mkn": list(shape), "metadata": metadata,
                "input_sha256": [hashlib.sha256(value.tobytes()).hexdigest() for value in values],
                "correctness": "exact independent FP64 coefficient, required BF16 store, then FP32 gate; before and after every window",
                "samples": samples, "orders": orders, "admission": admission,
                "median_paired_ratio": float(np.median(ratios))}
    finally:
        for engine in engines.values():
            engine.prepared.close()
        for owner in owners.values():
            owner.close_native_storage()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--shapes", nargs="+", default=("256,4096,1024", "256,4096,2048"))
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--min-window-ms", type=float, default=10.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        shapes = [tuple(map(int, value.split(","))) for value in args.shapes]
    except ValueError:
        parser.error("shapes require integer M,K,N")
    if any(len(s) != 3 or min(s) <= 0 or s[1] % 32 for s in shapes):
        parser.error("shapes require positive M,K,N and K divisible by 32")
    if args.windows < 2 or not np.isfinite(args.min_window_ms) or args.min_window_ms < 10:
        parser.error("at least two windows and a finite duration floor of at least 10 ms are required")
    compilers = {"candidate": args.compiler.resolve(), "reference": args.reference.resolve()}
    if any(not path.is_file() for path in compilers.values()):
        parser.error("both compiler files must exist")
    if runtime._rocm_live_arch() != "gfx1201":
        raise RuntimeError("owning gfx1201 required")
    hip = c.CDLL("/opt/rocm/lib/libamdhip64.so")
    ordinal, name = c.c_int(), c.create_string_buffer(256)
    if (hip.hipGetDevice(c.byref(ordinal)) or hip.hipDeviceGetName(name, 256, ordinal.value)):
        raise RuntimeError("active HIP device identity query failed")
    sources = ("benchmarks/rocm/record_paired_exact_mxfp4.py",
               "benchmarks/rocm/record_gfx1201_interleaved_compiler_formats.py",
               "tests/device/rocm/test_mxfp4_residual_native_execution.py",
               "tests/unit/test_native_mxfp4_residual_gate.py",
               "python/tessera/compiler/native_scaled_program.py",
               "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp",
               "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateWMMAGemmKernel.cpp")
    packet = {"schema": "tessera.paired_exact_mxfp4_program.v1", "architecture": "gfx1201",
              "device": name.value.decode(), "device_ordinal": ordinal.value,
              "compiler_sha256": {label: sha(path) for label, path in compilers.items()},
              "source_sha256": {path: sha(path) for path in sources},
              "runtime_sha256": sha(os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]),
              "timing_scope": "HIP event span around repeated native primal programs including host dispatch gaps; preparation, uploads, readback and compilation excluded; not kernel-only or public-call latency",
              "memory_scope": "separate checked native owners with byte-identical inputs; allocations stay live across alternating windows",
              "claim": "exact-device attribution only; no selector promotion, model-quality claim or Radiance comparison", "rows": []}
    for shape in shapes:
        packet["rows"].append(record(shape, compilers, windows=args.windows, minimum=args.min_window_ms))
        print("measured", shape, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
