"""Correctness-gated gfx1201 parent-view public/package timing boundaries."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
from statistics import median
import time

import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.native_scaled_program import PreparedScaledProgram
from tessera.compiler.rocm_parent_view_native import validate_parent_view_runtime_artifact
from tests.device.rocm.test_native_parent_view_jit import function, oracle


def record():
    assert rt._rocm_live_arch() == "gfx1201", "owning gfx1201 required"
    hip = rt._load_hip_from_toolkit()
    assert hip is not None
    name = ct.create_string_buffer(256)
    hip.hipDeviceGetName.argtypes = [ct.c_void_p, ct.c_int, ct.c_int]
    assert hip.hipDeviceGetName(name, 256, 0) == 0
    uuid = (ct.c_ubyte * 16)()
    hip.hipDeviceGetUuid.argtypes = [ct.c_void_p, ct.c_int]
    assert hip.hipDeviceGetUuid(ct.byref(uuid), 0) == 0
    library = rt._load_rocm_native_movement_runtime()
    assert library is not None
    rows = []
    for m, width in ((3, 17), (256, 12288)):
        parent = np.random.default_rng(1712).uniform(-5, 5, (m, 2 * width)).astype(np.float32)
        expected = oracle(parent, False)
        fn = ts.jit(target="rocm_gfx1201")(function(m, width, False))
        start = time.perf_counter_ns()
        actual = fn(parent)
        cold_ms = (time.perf_counter_ns() - start) / 1e6
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-7)
        artifact = fn.runtime_artifact()
        package = validate_parent_view_runtime_artifact(artifact)
        device = []
        with PreparedScaledProgram(package, [parent], runtime_library=library._name) as owner:
            owner.invoke()
            for _ in range(5):
                generation, elapsed = owner.invoke(repeats=8192, timed=True)
                assert elapsed > 0 and elapsed * 8192 >= 10
                device.append(elapsed)
            np.testing.assert_allclose(owner.read(generation)[0], expected, rtol=3e-6, atol=2e-7)
        warm = []
        for _ in range(5):
            start = time.perf_counter_ns()
            output = fn(parent)
            warm.append((time.perf_counter_ns() - start) / 1e6)
            np.testing.assert_allclose(output, expected, rtol=3e-6, atol=2e-7)
        rows.append({"shape_m_i": [m, width], "correctness": "passed_before_and_after_timing",
            "max_abs_error": float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))),
            "image_digest": artifact.native_image.image_digest,
            "abi": artifact.launch_descriptor.abi_id,
            "compiler_fingerprint": artifact.native_image.compiler_fingerprint,
            "cold_public_call_ms": cold_ms, "device_repeats": 8192,
            "device_launch_window_samples_ms": device, "device_launch_window_median_ms": median(device),
            "warm_completed_public_samples_ms": warm, "warm_completed_public_median_ms": median(warm)})
    return {"architecture": "gfx1201", "device": name.value.decode(), "device_uuid": bytes(uuid).hex(),
        "runtime_library_sha256": hashlib.sha256(Path(library._name).read_bytes()).hexdigest(),
        "timing_boundary": "HIP-event windows include device dispatch; warm public wall time includes "
            "validation, owner preparation, upload, invocation, completion and readback. Cold wall time "
            "also includes compiler work; process/compiler caches are not cleared.",
        "scope": "static f32 parent-view activation only; no resident matmul composition, low-storage "
            "model-quality, speedup or default-route promotion claim", "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = record()
    args.output.write_text(json.dumps(packet, indent=2) + "\n")
    print(json.dumps(packet, indent=2))
