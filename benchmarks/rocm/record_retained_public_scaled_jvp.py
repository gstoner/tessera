"""Exact gfx1201 public JVP and native event characterization."""
import argparse
import ctypes as c
import hashlib
import itertools
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
from tests.unit.test_public_floating_scaled_primal_jvp import case
from tests.device.rocm.test_floating_scaled_product import reference


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(actual, expected):
    errors = []
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=4e-5, atol=3e-6)
        errors.append(float(np.max(np.abs(got-want))))
    return max(errors)


def record(ta, tb, prefix, out_axes, active):
    _, owner, values, directions = case(ta, tb, prefix=prefix, out_axes=out_axes)
    seeds = tuple(value if index in active else None for index, value in enumerate(directions))
    oracle_seeds = tuple(value if index in active else np.zeros_like(value)
                         for index, value in enumerate(directions))
    expected = reference([*values, *oracle_seeds], ta, tb, True)
    if owner._frontend_output_permutation is not None:
        expected = tuple(np.transpose(value, owner._frontend_output_permutation) for value in expected)
    start = time.perf_counter()
    actual = jvp(owner, values, seeds)
    cold_ms = (time.perf_counter()-start)*1000
    error = check(actual, expected)
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
    public, events, members = [], [], []
    original_run = subprocess.run
    def forbidden(*args, **kwargs):
        raise AssertionError("warm public JVP invoked a compiler subprocess")
    try:
        with PreparedScaledProgram(package, frames[1.],
                runtime_library=os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]) as prepared:
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
                prepared.update(frames[factor])
                generation, elapsed = prepared.invoke(repeats=128, timed=True)
                events.append(elapsed)
                error = max(error, check(prepared.read(generation), expected_frames[factor]))
                generation, per_member = prepared.profile_members(repeats=2048)
                members.append(list(per_member))
                error = max(error, check(prepared.read(generation), expected_frames[factor]))
    finally:
        subprocess.run = original_run
        owner.close_native_storage()
    program = json.loads(package.program_json)
    return {
        "transposeA": ta, "transposeB": tb, "map_prefix": list(prefix),
        "out_axes": out_axes, "active_indices": list(active),
        "raw_input_shapes": [list(value.shape) for value in values],
        "correctness": "independent_fp64_before_and_after_every_timing_domain",
        "max_abs_error": error, "first_public_call_ms": cold_ms,
        "completed_public_call_samples_ms": public, "completed_public_call_median_ms": median(public),
        "interleaved_native_program_event_samples_ms": events,
        "interleaved_native_program_event_median_ms": median(events),
        "captured_member_event_samples_ms": members,
        "captured_member_event_median_ms": [median(row[index] for row in members)
                                          for index in range(len(members[0]))],
        "captured_member_repeats": 2048,
        "captured_min_window_ms": min(min(row)*2048 for row in members),
        "member_operations": [row["operation"] for row in program["steps"]],
        "native_program_digest": digest_bytes(package.program_json.encode()),
        "native_member_image_digests": [digest_bytes(image) for image in package.images],
        "compiler_receipt": receipt,
    }


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
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
    if Path("python/tessera/compiler/native_scaled_jvp_call.py").exists():
        paths.append("python/tessera/compiler/native_scaled_jvp_call.py")
    paths.append(str(Path(__file__).relative_to(Path.cwd())))
    packet = {
        "schema": "tessera.retained_public_scaled_jvp.packet.v1",
        "architecture": runtime._rocm_live_arch(), "device": name.value.decode(),
        "device_index": device.value, "device_uuid_hex": bytes(uuid).hex(),
        "visibility": {name: os.environ.get(name) for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")},
        "source_sha256": {path: digest(path) for path in paths},
        "binary_sha256": {name: digest(os.environ[name]) for name in (
            "TESSERA_OPT", "TESSERA_ROCM_NATIVE_MOVEMENT_LIB", "TESSERA_ROCM_NATIVE_IMAGE_LIB")},
        "compiler_version": subprocess.check_output([os.environ["TESSERA_OPT"], "--version"], text=True),
        "rows": [],
        "timing_scope": {
            "first_public_call": "first invocation includes specialization, certificate, compilation and preparation; process caches may be warm",
            "completed_public_call": "warm public JVP includes intent witness, binding, native preparation/upload/launch/readback and independent outputs; compiler subprocesses forbidden",
            "interleaved_native_program": "HIP events around 128 native program repetitions include host dispatch gaps; preparation, update and readback excluded; value is already per invocation",
            "captured_members": "2048-repeat grouped pure-SSA member device windows; capture, graph instantiation and host copies excluded; distinct diagnostic schedule from ordinary interleaved execution",
        },
        "claim": "named exact-device correctness and latency characterization; no generic AD closure or performance promotion",
    }
    for ta, tb, (prefix, out_axes), active in itertools.product(
            (False, True), (False, True), (((), 0), ((2,), -1), ((2, 3), 0)),
            ((0, 1, 2, 3), (0,), (2, 3))):
        packet["rows"].append(record(ta, tb, prefix, out_axes, active))
        print("measured", len(packet["rows"]), flush=True)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(packet, indent=2)+"\n")


if __name__ == "__main__":
    main()

