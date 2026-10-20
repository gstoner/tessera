"""Paired exact MXFP4 native-program spans or captured member windows."""
from __future__ import annotations

import argparse
import ctypes as c
import hashlib
import json
import os
import subprocess
from pathlib import Path
import time

import numpy as np

from benchmarks.rocm.record_gfx1201_interleaved_compiler_formats import collect_interleaved_windows
from benchmarks.rocm.benchmark_gfx1201_fp8_blockscale import paired_completed_host_windows
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
                "native_event_ms_per_invocation": elapsed,
                "comparison_ms_per_invocation": elapsed, "launches": launches}


class MemberWindows:
    """Capture pure SSA members in groups; compare the first matmul only."""

    def window(self, engine, launches, *, bracketed):
        assert bracketed
        start = time.perf_counter_ns()
        engine.generation, elapsed = engine.prepared.profile_members(repeats=launches)
        if not elapsed:
            raise RuntimeError("captured native program returned no member times")
        return {"device_window_ms": elapsed[0] * launches,
                "host_window_ms": (time.perf_counter_ns() - start) / 1e6,
                "captured_member_ms_per_invocation": list(elapsed),
                "comparison_ms_per_invocation": elapsed[0], "launches": launches}


class Engine:
    def __init__(self, prepared):
        self.prepared = prepared
        self.generation = None


def record(shape, compilers, *, windows, minimum, captured_members=False,
           public_windows=0, public_minimum=10.):
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
            operations = [step["operation"] for step in json.loads(package.program_json)["steps"]]
            if captured_members and operations[0] != "tessera.scaled_matmul":
                raise RuntimeError("captured comparison requires scaled matmul as first member")
            metadata[label] = {
                "program_digest": hashlib.sha256(package.program_json.encode()).hexdigest(),
                "image_sha256": [hashlib.sha256(image).hexdigest() for image in package.images],
                "member_operations": operations,
            }

        def verify(engine):
            outputs = engine.prepared.read(engine.generation)
            assert len(outputs) == 1
            np.testing.assert_array_equal(outputs[0], expected)

        samples, orders, admission = collect_interleaved_windows(
            MemberWindows() if captured_members else ProgramWindows(), engines,
            windows=windows, min_window_ms=minimum,
            verify_output=verify)
        ratios = [a["comparison_ms_per_invocation"] / b["comparison_ms_per_invocation"]
                  for a, b in zip(samples["candidate"], samples["reference"], strict=True)]
        completed = None
        if public_windows:
            arms = {}
            def public_arm(label):
                owner, compiler = owners[label], compilers[label]
                last = [None]
                def launch():
                    # Owner caches are compiler-keyed. Charge identical selector
                    # work to both arms rather than accidentally timing one image.
                    os.environ["TESSERA_OPT"] = str(compiler)
                    os.environ["TESSERA_NVIDIA_OPT"] = str(compiler)
                    last[0] = owner(*values)
                def validate():
                    if last[0] is not None:
                        np.testing.assert_array_equal(last[0], expected)
                    selected = owner._native_composed_scaled_last_program
                    images = [hashlib.sha256(image).hexdigest() for image in selected.images]
                    if (images != metadata[label]["image_sha256"] or
                            owner._native_descriptor_last_receipt["execution_kind"] != "native_gpu"):
                        raise RuntimeError("warm public call changed compiler image or execution route")
                return launch, validate
            for label in owners:
                arms[label] = public_arm(label)
            original_run = subprocess.run
            def forbidden(*args, **kwargs):
                raise RuntimeError("warm paired public call invoked compiler subprocess")
            try:
                subprocess.run = forbidden
                completed = paired_completed_host_windows(
                    arms, windows=public_windows, min_window_ms=public_minimum)
            finally:
                subprocess.run = original_run
            completed["domain"] = ("completed ordinary @jit calls including frontend binding, "
                "checked native preparation, transfers, execution and readback; "
                "diagnostic compiler-environment selection charged equally to both arms")
            completed["compiler_subprocesses"] = "forbidden"
        return {"shape_mkn": list(shape), "metadata": metadata,
                "input_sha256": [hashlib.sha256(value.tobytes()).hexdigest() for value in values],
                "correctness": "exact independent FP64 coefficient, required BF16 store, then FP32 gate; before and after every window",
                "samples": samples, "orders": orders, "admission": admission,
                "median_paired_ratio": float(np.median(ratios)),
                "completed_public": completed}
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
    parser.add_argument("--captured-members", action="store_true",
                        help="compare captured first-matmul kernel windows; other member times are diagnostic")
    parser.add_argument("--min-window-ms", type=float, default=10.)
    parser.add_argument("--public-windows", type=int, default=0,
                        help="also counterbalance warm completed ordinary @jit calls")
    parser.add_argument("--public-min-window-ms", type=float, default=10.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        shapes = [tuple(map(int, value.split(","))) for value in args.shapes]
    except ValueError:
        parser.error("shapes require integer M,K,N")
    if any(len(s) != 3 or min(s) <= 0 or s[1] % 32 for s in shapes):
        parser.error("shapes require positive M,K,N and K divisible by 32")
    if (args.windows < 2 or args.windows % 2
            or not np.isfinite(args.min_window_ms) or args.min_window_ms < 10):
        parser.error("positive even windows and a finite duration floor of at least 10 ms are required")
    if (args.public_windows < 0 or args.public_windows % 2
            or not np.isfinite(args.public_min_window_ms)
            or args.public_min_window_ms < 10):
        parser.error("public windows must be zero or positive even and their finite duration floor at least 10 ms")
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
    scope = ("HIP events around grouped pure-SSA member captures; compare first scaled-matmul kernel only; capture, instantiation and copies excluded; captures remove per-kernel host enqueue gaps, while device graph dispatch is included; other short member windows are diagnostic"
             if args.captured_members else
             "HIP event span around repeated native primal programs including host dispatch gaps; preparation, uploads, readback and compilation excluded; not kernel-only or public-call latency")
    schema = ("tessera.paired_exact_mxfp4_members.v1" if args.captured_members else
              "tessera.paired_exact_mxfp4_program.v1")
    packet = {"schema": schema, "architecture": "gfx1201",
              "profile_mode": "captured_members" if args.captured_members else "native_program",
              "device": name.value.decode(), "device_ordinal": ordinal.value,
              "compiler_sha256": {label: sha(path) for label, path in compilers.items()},
              "source_sha256": {path: sha(path) for path in sources},
              "runtime_sha256": sha(os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]),
              "timing_scope": scope,
              "memory_scope": "separate checked native owners with byte-identical inputs; allocations stay live across alternating windows",
              "claim": "exact-device attribution only; no selector promotion, model-quality claim or Radiance comparison", "rows": []}
    for shape in shapes:
        packet["rows"].append(record(shape, compilers, windows=args.windows, minimum=args.min_window_ms,
                                     captured_members=args.captured_members, public_windows=args.public_windows,
                                     public_minimum=args.public_min_window_ms))
        print("measured", shape, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
