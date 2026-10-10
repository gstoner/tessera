"""Correctness-gated public continuous scaled primal/JVP latency on gfx1201."""
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
from tests.unit.test_public_floating_scaled_primal_jvp import case, mixed_case
from tests.device.rocm.test_floating_scaled_product import reference


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(actual, expected):
    error = 0.
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=4e-5, atol=3e-6)
        error = max(error, float(np.max(np.abs(got-want))))
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if runtime._rocm_live_arch() != "gfx1201":
        raise RuntimeError("owning gfx1201 required")
    hip = c.CDLL("/opt/rocm/lib/libamdhip64.so")
    name = c.create_string_buffer(256)
    uuid = (c.c_ubyte*16)()
    if hip.hipInit(0) or hip.hipDeviceGetName(name, 256, 0) or hip.hipDeviceGetUuid(c.byref(uuid), 0):
        raise RuntimeError("HIP device identity query failed")
    rows = []
    for ta, tb, envelope, jvp in itertools.product((False, True), (False, True),
                                                   ("direct", "nested_mixed"), (False, True)):
        mode = "forward" if jvp else None
        if envelope == "direct":
            _, owner, values, directions = case(ta, tb, mode, prefix=())
            canonical, seeds, permutation = values, directions, None
        else:
            _, owner, values, directions, canonical, seeds, permutation = mixed_case(ta, tb, mode, True)
            # Independent logical map coordinates: B/sb vary only on outer axis.
            canonical = (canonical[0], canonical[1][:, None, ...], canonical[2], canonical[3][:, None, ...])
            seeds = (seeds[0], seeds[1][:, None, ...], seeds[2], seeds[3][:, None, ...])
        def expected(factor):
            operands = [value*factor for value in canonical]
            if jvp:
                operands += [value*factor for value in seeds]
            result = reference(operands, ta, tb, jvp)
            return tuple(np.transpose(value, permutation) if permutation is not None else value
                         for value in result)
        def invoke(args, tangents):
            return owner.native_jvp(*args, tangents=tangents) if jvp else (owner(*args),)
        start = time.perf_counter()
        actual = invoke(values, directions)
        cold_ms = (time.perf_counter()-start)*1000
        error = check(actual, expected(1.))
        receipt = owner.last_jvp_execution if jvp else owner._native_descriptor_last_receipt
        if receipt["execution_kind"] != "native_gpu":
            raise RuntimeError("public route did not execute natively")
        samples = []
        original_run = subprocess.run
        def forbidden(*args, **kwargs):
            raise AssertionError("warm public timing invoked a compiler subprocess")
        subprocess.run = forbidden
        try:
            for index in range(7):
                factor = np.float32(1. if index % 2 == 0 else -.625)
                args = tuple(value*factor for value in values)
                tangents = tuple(value*factor for value in directions)
                start = time.perf_counter()
                actual = invoke(args, tangents)
                samples.append((time.perf_counter()-start)*1000)
                error = max(error, check(actual, expected(factor)))
        finally:
            subprocess.run = original_run
        rows.append({"transposeA": ta, "transposeB": tb, "envelope": envelope,
                     "kind": "paired_jvp" if jvp else "primal",
                     "raw_input_shapes": [list(value.shape) for value in values],
                     "first_public_call_ms": cold_ms,
                     "warm_public_call_samples_ms": samples,
                     "warm_public_call_median_ms": median(samples),
                     "max_abs_error": error,
                     "correctness": "checked_before_and_after_every_round",
                     "warm_compiler_subprocesses": "forbidden",
                     "compiler_path": receipt["compiler_path"],
                     "execution_kind": receipt["execution_kind"]})
        print("measured", len(rows), flush=True)
    sources = [
        "src/transforms/lib/NativeFloatingScaledProduct.h",
        "src/transforms/lib/NativeScaledMatmulProgram.h",
        "src/compiler/programming_model/lib/NativeScaleTranspose.h",
        "python/tessera/compiler/native_scaled_program.py",
        "python/tessera/compiler/jit.py",
        "python/tessera/compiler/native_jvp_plugins.py",
        "python/tessera/compiler/native_vmap.py",
        "python/tessera/compiler/rocm_typed_scaled_native.py",
        "tests/unit/test_public_floating_scaled_primal_jvp.py",
        "tests/device/rocm/test_public_floating_scaled_primal_jvp.py",
        "tests/device/rocm/test_floating_scaled_product.py",
    ]
    packet = {"architecture": "gfx1201", "device": name.value.decode(),
              "device_uuid_hex": bytes(uuid).hex(),
              "route": "public textual frontend -> typed Graph -> native MLIR differentiation/product expansion -> Schedule/Tile -> ROCm/LLVM -> HSACO -> checked ABI",
              "first_public_call_domain": "first owner invocation includes trace/certificate, compilation and package preparation; process caches may already be warm",
              "warm_public_call_domain": "public call includes host checks/map packing, upload, dispatch, readback and output placement; compiler subprocesses forbidden",
              "native_event_packet": "benchmarks/baselines/continuous_scaled_product_20261009/gfx1201.json",
              "claim": "named numerical/latency characterization; no default or performance promotion",
              "compiler_sha256": digest(os.environ["TESSERA_OPT"]),
              "compiler_version": subprocess.check_output([os.environ["TESSERA_OPT"], "--version"], text=True),
              "provider_sha256": digest(os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]),
              "recorder_sha256": digest(__file__),
              "source_sha256": {path: digest(path) for path in sources}, "rows": rows}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(packet, indent=2)+"\n")


if __name__ == "__main__":
    main()
