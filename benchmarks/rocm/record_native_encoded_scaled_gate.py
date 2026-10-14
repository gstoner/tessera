"""Exact gfx1201 encoded tensor-gate primal/JVP/VJP characterization."""
import argparse
import ctypes as c
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from statistics import median

import numpy as np

from tessera import runtime
from tessera.autodiff import jvp
from tessera.compiler.native_scaled_program import NativeScaledProgram, PreparedScaledProgram
from tests.unit.test_native_encoded_scaled_gate import typed_case
from tests.device.rocm.test_native_encoded_scaled_gate_execution import (
    typed_oracle)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(actual, expected):
    errors = []
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=6e-5, atol=7e-5)
        errors.append(float(np.max(np.abs(got-want))))
    return max(errors)


def record(fmt, shape=None):
    ta, tb, prefix, out_axes = False, False, (), 0
    if fmt == "mxfp4":
        from tests.unit.test_native_mxfp4_residual_gate import typed_case as mx_case
        from tests.device.rocm.test_mxfp4_residual_native_execution import frame, coefficient
        owner, _ = mx_case(shape)
        values, direction = frame(shape)
        directions = (None, None, None, None, direction)
        evaluate = lambda frame: coefficient(frame)*frame[4]
    elif fmt == "continuous":
        from tests.unit.test_native_scaled_multiply_carrier import case
        from tests.device.rocm.test_native_scaled_tensor_gate_execution import oracle
        owner, _, values, directions = case("reverse", shape=(7, 37, 19))
        evaluate = lambda frame: oracle(frame, False, False)
    else:
        owner, values, directions = typed_case(fmt)
        evaluate = lambda frame: typed_oracle(frame, fmt)
    tangent = np.zeros_like(evaluate(values))
    for slot, direction in enumerate(directions):
        if direction is None:
            continue
        frame = list(values)
        frame[slot] = direction
        tangent += evaluate(frame)
    expected = (evaluate(values), tangent)
    active = tuple(index for index, direction in enumerate(directions) if direction is not None)
    seeds = directions
    start = time.perf_counter()
    actual = jvp(owner, values, seeds)
    cold_ms = (time.perf_counter()-start)*1000
    error = check(actual, expected)
    primal = owner(*values)
    error = max(error, check((primal,), (expected[0],)))
    if owner._native_descriptor_last_receipt["execution_kind"] != "native_gpu":
        raise RuntimeError("ordinary primal did not execute natively")
    receipt_keys = ("compiler_path", "execution_kind", "execution_mode", "artifact_hash", "evidence_target")
    primal_receipt = {k: owner._native_descriptor_last_receipt.get(k) for k in receipt_keys}
    cotangent = np.random.default_rng(61194).uniform(-.5,.5,expected[0].shape).astype(np.float32)
    gradients = []
    for role in active:
        if fmt == "mxfp4":
            # Independent exact coefficient, after the required BF16 store.
            # Gate-only derivatives are linear; no derivative crosses rounding.
            gradients.append(coefficient(values)*cotangent)
            continue
        gradient = np.empty(values[role].shape, np.float64)
        for index in np.ndindex(gradient.shape):
            plus, minus = list(values), list(values)
            plus[role] = values[role].astype(np.float64)
            minus[role] = values[role].astype(np.float64)
            plus[role][index] += .0001
            minus[role][index] -= .0001
            gradient[index] = np.sum(cotangent * (evaluate(plus)-evaluate(minus))) / .0002
        gradients.append(gradient)
    backward = owner.native_backward(*values, out_cotangents=cotangent)
    error = max(error, check(backward, gradients))
    if owner.last_backward_execution["execution_kind"] != "native_gpu":
        raise RuntimeError("reverse did not execute natively")
    reverse_receipt = {k: owner.last_backward_execution.get(k) for k in receipt_keys}
    receipt = dict(owner.last_jvp_execution)
    if receipt["execution_kind"] != "native_gpu" or receipt["evidence_target"] != "rocm_gfx1201":
        raise RuntimeError("public JVP lacked exact gfx1201 native receipt")
    child = owner._native_public_jvp_owners[active][1]
    artifact = next(iter(child._native_jvp_packages.values()))
    steps = artifact.runtime_metadata()["native_jvp"]["steps"]
    if len(steps) != 1 or steps[0]["outputs"] != ["primal", "tangent"]:
        raise RuntimeError("recorder requires one paired native scaled program")
    package = NativeScaledProgram.from_manifest(steps[0]["child_metadata"]["native_scaled_program"])
    frames = {
        1.: [*values, *(directions[index] for index in active)],
        -.5: [*values, *(directions[index]*np.float32(-.5) for index in active)],
    }
    expected_frames = {1.: expected, -.5: (expected[0], expected[1]*-.5)}
    public, primal_times, reverse_times, events, members = [], [], [], [], []
    primal_events, reverse_events, primal_members, reverse_members = [], [], [], []
    primal_package = next(iter(owner._native_composed_scaled_cache.values()))[0]
    reverse_package = owner._native_backward_artifact
    if not isinstance(primal_package, NativeScaledProgram) or not isinstance(reverse_package, NativeScaledProgram):
        raise RuntimeError("native primal/reverse program ownership missing")
    original_run = subprocess.run
    def forbidden(*args, **kwargs):
        raise AssertionError("warm public JVP invoked a compiler subprocess")
    try:
        library = os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]
        with PreparedScaledProgram(package, frames[1.], runtime_library=library) as prepared, \
                PreparedScaledProgram(primal_package, values, runtime_library=library) as prepared_primal, \
                PreparedScaledProgram(reverse_package, [*values, cotangent], runtime_library=library) as prepared_reverse:
            generation, _ = prepared.invoke()
            error = max(error, check(prepared.read(generation), expected))
            subprocess.run = forbidden
            for index in range(7):
                factor = 1. if index % 2 == 0 else -.5
                selected = tuple(None if seed is None else seed*np.float32(factor) for seed in seeds)
                start = time.perf_counter()
                actual = jvp(owner, values, selected)
                public.append((time.perf_counter()-start)*1000)
                error = max(error, check(actual, expected_frames[factor]))
                start = time.perf_counter()
                primal = owner(*values)
                primal_times.append((time.perf_counter()-start)*1000)
                error = max(error, check((primal,), (expected[0],)))
                start = time.perf_counter()
                backward = owner.native_backward(*values, out_cotangents=cotangent*np.float32(factor))
                reverse_times.append((time.perf_counter()-start)*1000)
                error = max(error, check(backward, [g*factor for g in gradients]))
                prepared.update(frames[factor])
                generation, elapsed = prepared.invoke(repeats=128, timed=True)
                events.append(elapsed)
                error = max(error, check(prepared.read(generation), expected_frames[factor]))
                generation, per_member = prepared.profile_members(repeats=2048)
                members.append(list(per_member))
                error = max(error, check(prepared.read(generation), expected_frames[factor]))
                prepared_reverse.update([*values, cotangent*np.float32(factor)])
                for native, wanted, event_rows, member_rows in (
                        (prepared_primal, (expected[0],), primal_events, primal_members),
                        (prepared_reverse, [g*factor for g in gradients], reverse_events, reverse_members)):
                    generation, elapsed = native.invoke(repeats=128, timed=True)
                    event_rows.append(elapsed)
                    error = max(error, check(native.read(generation), wanted))
                    generation, per_member = native.profile_members(repeats=2048)
                    member_rows.append(list(per_member))
                    error = max(error, check(native.read(generation), wanted))
    finally:
        subprocess.run = original_run
        owner.close_native_storage()
    program = json.loads(package.program_json)
    return {
        "data_format": "mxfp4_w4a8_exact" if fmt == "mxfp4" else "fp32" if fmt == "continuous" else "fp8_e4m3",
        "physical_contract": "rocm_mxfp4_w4a8_exact_v1" if fmt == "mxfp4" else None,
        "scale_format": "fp32" if fmt == "continuous" else "e8m0" if fmt == "mxfp4" else fmt, "transposeA": ta, "transposeB": tb, "map_prefix": list(prefix),
        "out_axes": out_axes, "active_indices": list(active),
        "raw_input_shapes": [list(value.shape) for value in values],
        "correctness": "independent_fp64_before_and_after_every_timing_domain",
        "max_abs_error": error, "first_public_call_ms": cold_ms,
        "completed_primal_samples_ms": primal_times, "completed_primal_median_ms": median(primal_times),
        "completed_reverse_samples_ms": reverse_times, "completed_reverse_median_ms": median(reverse_times),
        "completed_public_call_samples_ms": public, "completed_public_call_median_ms": median(public),
        "interleaved_native_program_event_samples_ms": events,
        "interleaved_native_program_event_median_ms": median(events),
        "captured_member_event_samples_ms": members,
        "captured_member_event_median_ms": [median(row[index] for row in members)
                                          for index in range(len(members[0]))],
        "primal_program_event_samples_ms": primal_events, "primal_program_event_median_ms": median(primal_events),
        "reverse_program_event_samples_ms": reverse_events, "reverse_program_event_median_ms": median(reverse_events),
        "primal_captured_member_samples_ms": primal_members,
        "reverse_captured_member_samples_ms": reverse_members,
        "primal_member_operations": [r["operation"] for r in json.loads(primal_package.program_json)["steps"]],
        "reverse_member_operations": [r["operation"] for r in json.loads(reverse_package.program_json)["steps"]],
        "primal_program_digest": digest_bytes(primal_package.program_json.encode()),
        "reverse_program_digest": digest_bytes(reverse_package.program_json.encode()),
        "primal_image_digests": [digest_bytes(i) for i in primal_package.images],
        "reverse_image_digests": [digest_bytes(i) for i in reverse_package.images],
        "captured_member_repeats": 2048,
        "captured_min_window_ms": min(min(row)*2048 for row in members),
        "member_operations": [row["operation"] for row in program["steps"]],
        "native_program_digest": digest_bytes(package.program_json.encode()),
        "native_member_image_digests": [digest_bytes(image) for image in package.images],
        "compiler_receipt": receipt, "primal_receipt": primal_receipt, "reverse_receipt": reverse_receipt,
    }


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale-vjp-schedule",
                        choices=("serial_per_scale_element", "wave_per_scale_element", "auto"),
                        default="serial_per_scale_element")
    parser.add_argument("--formats", nargs="+", choices=("fp32", "e8m0", "continuous", "mxfp4"),
                        default=("fp32", "e8m0"))
    parser.add_argument("--mxfp4-shapes", nargs="+", default=("17,64,19", "256,128,80", "256,128,160"),
                        help="positive M,K,N exact packed-K32 diagnostic rows")
    options = parser.parse_args()
    mx_shapes = [tuple(map(int, row.split(","))) for row in options.mxfp4_shapes]
    if any(len(row) != 3 or min(row) <= 0 or row[1] % 32 for row in mx_shapes):
        parser.error("MXFP4 shapes require positive M,K,N with K divisible by 32")
    os.environ["TESSERA_ROCM_SCALE_VJP_SCHEDULE"] = options.scale_vjp_schedule
    if runtime._rocm_live_arch() != "gfx1201":
        raise RuntimeError("owning gfx1201 required")
    hip = c.CDLL("/opt/rocm/lib/libamdhip64.so")
    device = c.c_int()
    name = c.create_string_buffer(256)
    uuid = (c.c_ubyte*16)()
    if (hip.hipInit(0) or hip.hipGetDevice(c.byref(device)) or
            hip.hipDeviceGetName(name, 256, device.value) or
            hip.hipDeviceGetUuid(c.byref(uuid), device.value)):
        raise RuntimeError("active HIP device identity query failed")
    runtime_version = c.c_int()
    if hip.hipRuntimeGetVersion(c.byref(runtime_version)):
        raise RuntimeError("HIP runtime version query failed")
    paths = [
        "python/tessera/autodiff/jvp.py", "python/tessera/compiler/native_public_jvp.py",
        "python/tessera/compiler/jit.py", "python/tessera/compiler/native_jvp.py",
        "python/tessera/compiler/native_jvp_plugins.py", "python/tessera/compiler/native_scaled_program.py",
        "python/tessera/compiler/native_vmap.py", "python/tessera/compiler/rocm_typed_scaled_native.py",
        "python/tessera/runtime.py", "src/compiler/ir/TangentInterface.cpp",
        "src/transforms/lib/NativeFloatingScaledProduct.h", "src/transforms/lib/NativeScaledMatmulProgram.h",
        "src/compiler/programming_model/lib/NativeScaleTranspose.h",
        "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_image_cache.cpp",
        "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_program_runtime.cpp",
        "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_program_runtime.h",
        "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_movement_runtime.cpp",
        "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/MovementPhysicalSpan.h",
        "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_nvfp4_runtime.cpp",
        "tests/unit/test_public_floating_scaled_primal_jvp.py",
        "tests/device/rocm/test_public_native_scaled_jvp_transform.py",
        "tests/device/rocm/test_floating_scaled_product.py",
    ]
    paths.extend([
        "src/transforms/lib/AutodiffPairedPass.cpp",
        "tests/unit/test_native_encoded_scaled_gate.py",
        "tests/device/rocm/test_native_encoded_scaled_gate_execution.py",
        "src/compiler/programming_model/lib/NativeROCMMath.h",
        "python/tessera/compiler/rocm_math_native.py",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateROCMBinaryKernel.cpp",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/ROCMKernelIdentity.cpp",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp",
    ])
    paths.extend(["python/tessera/compiler/capabilities.py", "python/tessera/compiler/backend_manifest.py"])
    paths.extend([
        "src/compiler/ir/include/Tessera/IR/StructuredReductionContract.h",
        "tests/unit/test_native_scaled_multiply_carrier.py",
        "tests/device/rocm/test_native_scaled_tensor_gate_execution.py",
    ])
    paths.extend([
        "python/tessera/compiler/graph_ir.py", "python/tessera/compiler/trace.py",
        "python/tessera/compiler/rocm_nvfp4_program.py", "src/transforms/lib/NativeScaledCarrier.h",
        "src/compiler/ir/TileOps.cpp",
        "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateWMMAGemmKernel.cpp",
        "tests/unit/test_native_mxfp4_residual_gate.py",
        "tests/device/rocm/test_mxfp4_residual_native_execution.py",
    ])
    paths.append(str(Path(__file__).relative_to(Path.cwd())))
    packet = {
        "schema": "tessera.native_encoded_scaled_gate.packet.v1",
        "scale_vjp_schedule": options.scale_vjp_schedule, "architecture": runtime._rocm_live_arch(), "device": name.value.decode(),
        "hip_runtime_version_raw": runtime_version.value, "device_index": device.value, "device_uuid_hex": bytes(uuid).hex(),
        "visibility": {name: os.environ.get(name) for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")},
        "source_sha256": {path: digest(path) for path in paths},
        "binary_sha256": {name: digest(os.environ[name]) for name in (
            "TESSERA_OPT", "TESSERA_ROCM_NATIVE_MOVEMENT_LIB", "TESSERA_ROCM_NATIVE_IMAGE_LIB")},
        "compiler_version": subprocess.check_output([os.environ["TESSERA_OPT"], "--version"], text=True),
        "rows": [],
        "timing_scope": {
            "first_public_call": "first invocation includes specialization, certificate, compilation and preparation; process caches may be warm",
            "completed_primal_reverse": "warm native ordinary primal and reverse include frontend binding, checked preparation/upload/launch/readback; excluded from device-event domains",
            "completed_public_call": "warm public JVP includes intent witness, binding, native preparation/upload/launch/readback and independent outputs; compiler subprocesses forbidden",
            "interleaved_native_program": "HIP events around 128 native program repetitions include host dispatch gaps; preparation, update and readback excluded; value is already per invocation",
            "captured_members": "2048-repeat grouped pure-SSA member device windows; capture, graph instantiation and host copies excluded; distinct diagnostic schedule from ordinary interleaved execution",
        },
        "claim": "named exact-device correctness and latency characterization; no generic AD closure or performance promotion",
    }
    for fmt in options.formats:
        for shape in mx_shapes if fmt == "mxfp4" else (None,):
            packet["rows"].append(record(fmt, shape))
            print("measured", len(packet["rows"]), flush=True)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(packet, indent=2)+"\n")


if __name__ == "__main__":
    main()
