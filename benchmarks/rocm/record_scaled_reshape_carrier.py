"""Record native reshape/product packages on their owning gfx1201 device."""
import argparse
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time

import numpy as np
from tessera import runtime
from tessera.compiler.native_scaled_program import NativeScaledProgram, PreparedScaledProgram
from tests.unit.test_scaled_reshape_carrier import case
from tests.device.rocm.test_scaled_reshape_carrier import expected


def find_program(value):
    if isinstance(value, dict):
        if value.get("schema") == "tessera.native_scaled_program.v1":
            return NativeScaledProgram.from_manifest(value)
        for child in value.values():
            result = find_program(child)
            if result is not None:
                return result
    elif isinstance(value, (list, tuple)):
        for child in value:
            result = find_program(child)
            if result is not None:
                return result
    return None


def record(output, mode):
    owner, _, values = case(mode, output)
    directions = tuple(np.full_like(value, .03125) for value in values)
    primal, tangent, _ = expected(values, directions, output)
    seed = np.random.default_rng(19073).uniform(-.5, .5, primal.shape).astype(np.float32)

    def call(frame, cotangent):
        if mode == "reverse":
            return owner.native_backward(*frame, out_cotangents=cotangent)
        if mode == "forward":
            return owner.native_jvp(*frame, tangents=directions)
        return (owner(*frame),)

    def wanted(frame, cotangent):
        value, dvalue, gradients = expected(frame, directions, output, cotangent)
        return gradients if mode == "reverse" else (value, dvalue) if mode == "forward" else (value,)

    def check(actual, oracle):
        errors = []
        for got, want in zip(actual, oracle, strict=True):
            np.testing.assert_allclose(got, want, rtol=4e-5, atol=3e-6)
            errors.append(float(np.max(np.abs(got-want))))
        return max(errors)

    start = time.perf_counter()
    result = call(values, seed)
    cold = (time.perf_counter()-start)*1000
    error = check(result, wanted(values, seed))
    if mode == "reverse":
        package = owner._native_backward_artifact
    elif mode == "forward":
        artifacts = tuple(owner._native_jvp_packages.values())
        if len(artifacts) != 1:
            raise RuntimeError("expected one actual executed JVP artifact")
        package = find_program(artifacts[0].runtime_metadata())
        if package is None:
            raise RuntimeError("executed JVP artifact lacks a native program")
    else:
        package = owner._native_composed_scaled_last_program
    package.validate()

    def arguments(frame, cotangent):
        return (*frame, cotangent) if mode == "reverse" else (*frame, *directions) if mode == "forward" else frame

    native, members, public = [], [], []
    with PreparedScaledProgram(package, arguments(values, seed),
                               runtime_library=os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]) as prepared:
        for index in range(7):
            factor = np.float32(1 if index % 2 == 0 else -.875)
            frame = tuple(np.ascontiguousarray(value*factor) for value in values)
            cotangent = np.ascontiguousarray(seed*(1 if index % 2 == 0 else -.5))
            oracle = wanted(frame, cotangent)
            prepared.update(arguments(frame, cotangent))
            generation, elapsed = prepared.invoke(repeats=128, timed=True)
            error = max(error, check(prepared.read(generation), oracle))
            native.append(elapsed)
            generation, parts = prepared.profile_members(repeats=128)
            error = max(error, check(prepared.read(generation), oracle))
            members.append(parts)
            start = time.perf_counter()
            result = call(frame, cotangent)
            public.append((time.perf_counter()-start)*1000)
            error = max(error, check(result, oracle))
    return {"output_shape": list(output), "mode": mode or "primal",
            "input_shapes": [list(value.shape) for value in values], "max_abs_error": error,
            "cold_compile_and_call_ms": cold, "native_program_samples_ms": native,
            "native_program_median_ms": median(native), "member_samples_ms": members,
            "public_compile_warm_samples_ms": public, "public_compile_warm_median_ms": median(public),
            "program": json.loads(package.program_json),
            "image_sha256": [hashlib.sha256(image).hexdigest() for image in package.images]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if runtime._rocm_live_arch() != "gfx1201":
        raise RuntimeError("exact gfx1201 required")
    hip = c.CDLL("/opt/rocm/lib/libamdhip64.so")
    name = c.create_string_buffer(256)
    uuid = (c.c_ubyte*16)()
    if hip.hipInit(0) or hip.hipDeviceGetName(name, 256, 0) or hip.hipDeviceGetUuid(c.byref(uuid), 0):
        raise RuntimeError("device identity query failed")
    paths = ["python/tessera/compiler/trace.py", "python/tessera/compiler/rocm_typed_scaled_native.py",
             "python/tessera/compiler/native_scaled_program.py",
             "src/transforms/lib/NativeScaledCarrier.h", "src/transforms/lib/NativeScaledMatmulProgram.h",
             "src/compiler/programming_model/lib/NativeScaleTranspose.h",
             "python/tessera/compiler/paged_host_span.py", "python/tessera/compiler/jit.py",
             "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_program_runtime.cpp",
             "tests/unit/test_scaled_reshape_carrier.py", "tests/device/rocm/test_scaled_reshape_carrier.py",
             "benchmarks/rocm/record_scaled_reshape_carrier.py"]
    packet = {"architecture": "gfx1201", "device": name.value.decode(), "uuid_hex": bytes(uuid).hex(),
              "route": "Frontend semantic projection->Graph->native AD->Schedule->Tile->ROCm Target->LLVM->HSACO->checked HIP ABI",
              "correctness": "Independent float64 block-product/JVP/transpose oracle before and after every timing window",
              "member_timing_scope": "Grouped repeated members including HIP graph dispatch; not additive interleaved program time",
              "public_scope": "Compilation warm; includes preparation and allocation; no speedup claim",
              "sources": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths},
              "tools": {key: hashlib.sha256(Path(os.environ[key]).read_bytes()).hexdigest()
                        for key in ("TESSERA_OPT", "TESSERA_ROCM_NATIVE_MOVEMENT_LIB")},
              "rows": [record(output, mode) for output in ((15,), (5,3), (1,3,5)) for mode in (None, "forward", "reverse")]}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(packet, indent=2)+"\n")
    print(json.dumps([{key: row[key] for key in ("output_shape", "mode", "max_abs_error",
                                               "native_program_median_ms", "public_compile_warm_median_ms")}
                      for row in packet["rows"]]))


if __name__ == "__main__":
    main()
