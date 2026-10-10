"""Record correctness-gated gfx1201 non-leading map calls and native event windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import time

import numpy as np

from tessera import runtime as rt
from tessera.compiler.native_scaled_program import NativeScaledProgram, PreparedScaledProgram
from tests.device.rocm.test_scaled_map_axes import axis_oracle
from tests.unit.test_native_scaled_map_axes import axis_case

SOURCES = (
    "python/tessera/compiler/jit.py",
    "python/tessera/compiler/native_vmap.py",
    "python/tessera/compiler/paged_host_span.py",
    "python/tessera/compiler/native_scaled_program.py",
    "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_movement_runtime.cpp",
    "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_nvfp4_runtime.cpp",
    "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_program_runtime.cpp",
    "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_program_runtime.h",
    "tests/unit/test_native_scaled_map_axes.py",
    "tests/unit/test_native_host_view_pack.py",
    "tests/device/rocm/test_scaled_map_axes.py",
    "benchmarks/rocm/benchmark_scaled_map_axes.py",
)


def run_profile(fmt, nk, profile):
    _, owner, values, expected = axis_case(fmt, nk, nested=profile != "single",
                                           cartesian=profile == "cartesian")
    np.testing.assert_allclose(axis_oracle(values, owner._frontend_batch_policies, fmt, nk),
                               expected, rtol=4e-5, atol=2e-5)
    np.testing.assert_allclose(owner(*values), expected, rtol=4e-5, atol=2e-5)
    descriptor = owner.compile_result.launch_descriptor
    package = NativeScaledProgram.from_manifest(
        descriptor.provenance["native_scaled_primal_program"])
    library = rt._load_rocm_native_movement_runtime()
    normalized = owner._ordered_inputs(values, {})
    with PreparedScaledProgram(package, normalized, runtime_library=library._name) as resident:
        generation, _ = resident.invoke()
        np.testing.assert_allclose(resident.read(generation)[0], expected, rtol=4e-5, atol=2e-5)
        # Every recorder arm is compiled before its warm scope starts.
        original = subprocess.run
        def forbidden(*args, **kwargs):
            raise AssertionError("warm benchmark invoked the compiler")
        subprocess.run = forbidden
        try:
            public, events = [], []
            repetitions = 4096
            while True:
                _, value = resident.invoke(repeats=repetitions, timed=True)
                if value * repetitions >= 20:
                    break
                repetitions *= 2
                if repetitions > 262144:
                    raise RuntimeError("native event window remains too short")
            for _ in range(7):
                start = time.perf_counter()
                for _ in range(32):
                    result = owner(*values)
                    if owner._native_descriptor_last_receipt["execution_kind"] != "native_gpu":
                        raise RuntimeError("public call lost native execution")
                public.append((time.perf_counter()-start)*1000/32)
                generation, value = resident.invoke(repeats=repetitions, timed=True)
                events.append(value)
            np.testing.assert_allclose(result, expected, rtol=4e-5, atol=2e-5)
            np.testing.assert_allclose(resident.read(generation)[0], expected, rtol=4e-5, atol=2e-5)
        finally:
            subprocess.run = original
    return {
        "format": fmt, "rhs_transposed": nk, "profile": profile,
        "axes": owner._frontend_batch_policies,
        "input_shapes": [list(value.shape) for value in values],
        "input_byte_strides": [list(value.strides) for value in values],
        "output_shape": list(expected.shape),
        "max_abs_error": float(np.max(np.abs(result.astype(np.float64)-expected))),
        "correctness": "passed_before_and_after_timing",
        "route": "Graph->Schedule->Tile->ROCm Target->LLVM->HSACO",
        "abi_id": descriptor.abi_id,
        "graph_sha256": hashlib.sha256(package.graph_ir.encode()).hexdigest(),
        "image_sha256": [hashlib.sha256(image).hexdigest() for image in package.images],
        "public_completed_call_samples_ms": public,
        "public_completed_call_median_ms": median(public),
        "public_calls_per_window": 32,
        "native_resident_program_event_samples_ms": events,
        "native_resident_program_event_median_ms": median(events),
        "native_event_repetitions": repetitions,
        "native_event_windows_ms": [sample*repetitions for sample in events],
        "scope": "public includes view packing/transfers/readback; native event is resident program enqueue loop",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    architecture = rt._rocm_live_arch()
    if architecture != "gfx1201":
        raise RuntimeError(f"owning gfx1201 required, got {architecture}")
    info = subprocess.run(["/opt/rocm/bin/rocminfo"], capture_output=True, text=True, check=True).stdout
    library = rt._load_rocm_native_movement_runtime()
    root = Path(__file__).resolve().parents[2]
    packet = {
        "schema": "tessera.rocm.scaled_map_axes.v1",
        "architecture": architecture, "python": sys.version,
        "device_lines": [line.strip() for line in info.splitlines()
                         if "Marketing Name:" in line or "Name:                    gfx" in line],
        "source_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCES},
        "runtime_path": library._name,
        "runtime_sha256": hashlib.sha256(Path(library._name).read_bytes()).hexdigest(),
        "compiler_sha256": {name: hashlib.sha256(Path(os.environ[name]).read_bytes()).hexdigest()
                            for name in ("TESSERA_OPT", "TESSERA_ROCM_OPT")},
        "rows": [run_profile(fmt, nk, profile) for fmt in ("fp32", "e8m0")
                 for nk in (False, True) for profile in ("single", "nested", "cartesian")],
        "closure": "named static positive-stride map extension; generic closure remains unproved",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2)+"\n")
    for row in packet["rows"]:
        print(row["format"], row["rhs_transposed"], row["profile"],
              row["public_completed_call_median_ms"],
              row["native_resident_program_event_median_ms"])


if __name__ == "__main__":
    main()
