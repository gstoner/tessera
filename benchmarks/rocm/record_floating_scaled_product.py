"""Correctness-gated continuous native scaled product and JVP timing."""
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
from tessera.compiler.native_scaled_program import (
    PreparedScaledProgram, package_native_scaled_primal, package_native_scaled_jvp)
from tests.unit.test_native_floating_scaled_product import product_source
from tests.device.rocm.test_floating_scaled_product import reference
from tests.device.rocm.test_floating_scaled_adjoint import inputs, batch_inputs


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(actual, expected):
    errors = []
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=4e-5, atol=3e-6)
        errors.append(float(np.max(np.abs(got-want))))
    return max(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if runtime._rocm_live_arch() != "gfx1201":
        raise RuntimeError("owning gfx1201 required")
    library = os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]
    hip = c.CDLL("/opt/rocm/lib/libamdhip64.so")
    name = c.create_string_buffer(256)
    uuid = (c.c_ubyte*16)()
    if hip.hipInit(0) or hip.hipDeviceGetName(name, 256, 0) or hip.hipDeviceGetUuid(c.byref(uuid), 0):
        raise RuntimeError("HIP device identity query failed")
    rows = []
    for ta, tb, policy, jvp in itertools.product((False, True), (False, True),
                                                (None, "broadcast"), (False, True)):
        values = list((inputs(ta, tb) if policy is None else batch_inputs(policy, ta, tb))[:4])
        if jvp:
            rng = np.random.default_rng(938)
            values.extend(rng.uniform(-.5, .5, value.shape).astype(np.float32)
                          for value in tuple(values))
        source = product_source(ta, tb, policy, jvp)
        start = time.perf_counter()
        package = (package_native_scaled_jvp(source) if jvp else package_native_scaled_primal(source))
        compile_ms = (time.perf_counter()-start)*1000
        expected = reference(values, ta, tb, jvp)
        changed = [np.ascontiguousarray(value*np.float32(-.625)) for value in values]
        expected_changed = reference(changed, ta, tb, jvp)
        with PreparedScaledProgram(package, values, runtime_library=library) as owner:
            generation, _ = owner.invoke()
            error = check(owner.read(generation), expected)
            device, end_to_end = [], []
            for round_index in range(7):
                selected, wanted = ((values, expected) if round_index % 2 == 0
                                    else (changed, expected_changed))
                owner.update(selected)
                generation, elapsed = owner.invoke(repeats=128, timed=True)
                error = max(error, check(owner.read(generation), wanted))
                device.append(elapsed)
                start = time.perf_counter()
                owner.update(selected)
                generation, _ = owner.invoke()
                actual = owner.read(generation)
                end_to_end.append((time.perf_counter()-start)*1000)
                error = max(error, check(actual, wanted))
            program = json.loads(package.program_json)
            members = [json.loads(raw) for raw in package.members_json]
            rows.append({"transposeA": ta, "transposeB": tb, "batching": policy,
                         "kind": program["kind"], "input_shapes": [list(value.shape) for value in values],
                         "max_abs_error": error, "correctness": "checked_before_and_after_every_round",
                         "compile_ms": compile_ms, "native_event_samples_ms": device,
                         "native_event_median_ms": median(device), "end_to_end_samples_ms": end_to_end,
                         "end_to_end_median_ms": median(end_to_end),
                         "members": [{"entry": member["entry"], "geometry": member["geometry"],
                                      "image_sha256": hashlib.sha256(image).hexdigest()}
                                     for member, image in zip(members, package.images, strict=True)],
                         "graph_sha256": hashlib.sha256(source.encode()).hexdigest()})
            print("measured", len(rows), flush=True)
    sources = [
        "src/transforms/lib/NativeFloatingScaledProduct.h",
        "src/transforms/lib/NativeScaledMatmulProgram.h",
        "src/compiler/programming_model/lib/NativeScaleTranspose.h",
        "python/tessera/compiler/native_scaled_program.py",
        "tests/unit/test_native_floating_scaled_product.py",
        "tests/device/rocm/test_floating_scaled_product.py",
    ]
    packet = {"architecture": "gfx1201", "device": name.value.decode(),
              "device_uuid_hex": bytes(uuid).hex(),
              "route": "original Graph witness -> native structured product -> Schedule -> Tile -> ROCm/LLVM -> HSACO",
              "native_event_domain": "ordinary HIP program events include host dispatch gaps; excludes compilation, upload and readback",
              "end_to_end_domain": "warm prepared owner update + invoke + read; excludes compilation/preparation",
              "claim": "numerical and timing characterization; no default or performance promotion",
              "compiler_sha256": digest(os.environ["TESSERA_OPT"]),
              "compiler_version": subprocess.check_output([os.environ["TESSERA_OPT"], "--version"], text=True),
              "provider_sha256": digest(library), "recorder_sha256": digest(__file__),
              "source_sha256": {path: digest(path) for path in sources}, "rows": rows}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(packet, indent=2)+"\n")


if __name__ == "__main__":
    main()
