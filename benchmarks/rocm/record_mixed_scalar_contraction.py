"""Correctness-gated native scalar packages; event and completed-call regions.

No speedup or general-shape claim: this records the named 17x19x35 envelope.
The public JIT is exercised separately by the owning-device package suite.
"""
from __future__ import annotations

import argparse
import ctypes as c
import hashlib
import itertools
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time

import numpy as np

from tessera import runtime as rt
from tessera.compiler.rocm_scalar_matmul import prepare
from tests.device.rocm.test_scalar_matmul_package import packaged, values


def record(architecture: str, *, repeats: int = 256) -> dict:
    if architecture not in {"gfx1151", "gfx1201"} or rt._rocm_live_arch() != architecture:
        raise ValueError("mixed scalar evidence requires its exact owning ROCm architecture")
    if type(repeats) is not int or not 0 < repeats <= 65536:
        raise ValueError("native event repetitions must be an integer in [1,65536]")
    hip = c.CDLL("/opt/rocm/lib/libamdhip64.so")
    name = c.create_string_buffer(256)
    hip.hipDeviceGetName.argtypes = [c.c_void_p, c.c_int, c.c_int]
    if hip.hipDeviceGetName(name, 256, 0):
        raise RuntimeError("HIP device identity query failed")
    rows = []
    for half, side, output, ta, tb in itertools.product(
            ("fp16", "bf16"), ("a", "b"), ("fp32", "fp16", "bf16"), (False, True), (False, True)):
        artifact = packaged(architecture, half, side, output, (ta, tb))
        roots, reference = values(half, side, output, (ta, tb), 1729)
        descriptor = artifact.launch_descriptor
        names = [b.name for b in descriptor.buffers]
        out = np.empty_like(reference)
        args = dict(zip(names, (*roots, out), strict=True), M=17, N=19, K=35)
        result = rt.launch(artifact, args)
        if result.get("ok") is not True or result.get("execution_kind") != "native_gpu":
            raise RuntimeError(f"checked native launch failed: {result}")
        np.testing.assert_array_equal(out, reference)
        with prepare(artifact.native_image, descriptor, roots) as owner:
            generation, _ = owner.invoke()
            np.testing.assert_array_equal(owner.read(generation)[0], reference)
            changed, expected = values(half, side, output, (ta, tb), 1733)
            original = subprocess.run

            def forbidden(*args, **kwargs):
                raise AssertionError("prepared/serialized timing attempted compilation")

            subprocess.run = forbidden
            try:
                owner.update(changed)
                generation, _ = owner.invoke()
                np.testing.assert_array_equal(owner.read(generation)[0], expected)
                completed, event_windows, prepared = [], [], []
                for _ in range(5):
                    generation, elapsed = owner.invoke(repeats=repeats, timed=True)
                    event_windows.append(elapsed)  # Native ABI already divides by repeats.
                    np.testing.assert_array_equal(owner.read(generation)[0], expected)
                    start = time.perf_counter_ns()
                    owner.update(changed)
                    generation, _ = owner.invoke()
                    actual = owner.read(generation)[0]
                    prepared.append((time.perf_counter_ns() - start)/1e6)
                    np.testing.assert_array_equal(actual, expected)
                    args.update(zip(names[:2], changed, strict=True))
                    start = time.perf_counter_ns()
                    receipt = rt.launch(artifact, args)
                    completed.append((time.perf_counter_ns() - start)/1e6)
                    if receipt.get("ok") is not True:
                        raise RuntimeError(f"completed replay failed: {receipt}")
                    np.testing.assert_array_equal(out, expected)
            finally:
                subprocess.run = original
        rows.append({
            "shape_mnk": [17, 19, 35], "half": half, "wide_side": side, "output": output,
            "transposeA": ta, "transposeB": tb, "correctness": "exact_dyadic_oracle_before_and_during_timing",
            "cancellation_witness": float(expected[0, 0]),
            "image_sha256": artifact.native_image.image_digest,
            "payload_sha256": hashlib.sha256(artifact.native_image.payload).hexdigest(),
            "abi": descriptor.abi_id, "entry": descriptor.entry_symbol,
            "schedule_digest": descriptor.provenance["schedule_digest"],
            "tile_digest": descriptor.provenance["tile_ir_digest"],
            "native_event_window_ms": event_windows, "native_event_window_median_ms": median(event_windows),
            "prepared_update_invoke_read_ms": prepared, "prepared_update_invoke_read_median_ms": median(prepared),
            "completed_descriptor_ms": completed, "completed_descriptor_median_ms": median(completed),
        })
    root = Path(__file__).resolve().parents[2]
    sources = (
        "python/tessera/compiler/rocm_scalar_matmul.py",
        "python/tessera/compiler/native_scaled_program.py",
        "python/tessera/compiler/scheduled_matmul.py", "python/tessera/runtime.py",
        "python/tessera/compiler/jit.py", "python/tessera/compiler/rocm_pipeline.py",
        "python/tessera/compiler/capabilities.py", "python/tessera/compiler/pipeline_registry.py",
        "python/tessera/compiler/rocm_native.py",
        "tests/unit/test_scheduled_matmul_consumers.py",
        "benchmarks/rocm/record_mixed_scalar_contraction.py",
        "tests/device/rocm/test_scalar_matmul_package.py", "tests/unit/test_rocm_scalar_package.py",
    )
    source_hashes = {path: hashlib.sha256((root/path).read_bytes()).hexdigest() for path in sources}
    library = rt._load_rocm_native_movement_runtime()
    return {
        "schema": 1, "architecture": architecture, "device": name.value.decode(),
        "scope": "static compact mixed scalar contraction 17x19x35; no speedup or full AD closure",
        "route": "frontend Graph->native Schedule->Tile->ROCm Target->ROCDL/LLVM->HSACO->checked C++ owner",
        "timing_regions": {
            "native_event_window_ms": "native owner HIP-event region divided by repeats; may include submission gaps",
            "prepared_update_invoke_read_ms": "host completed root upload, invoke and independent result read",
            "completed_descriptor_ms": "serialized descriptor validation, owner acquire/upload, invoke/read and release",
        },
        "repeats": repeats, "source_sha256": source_hashes,
        "compiler_sha256": hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        "provider_sha256": hashlib.sha256(Path(library._name).read_bytes()).hexdigest(),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True, choices=("gfx1151", "gfx1201"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=256)
    args = parser.parse_args()
    packet = record(args.architecture, repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")
    print(f"{packet['device']}: {len(packet['rows'])} correctness-gated rows -> {args.output}")


if __name__ == "__main__":
    main()
