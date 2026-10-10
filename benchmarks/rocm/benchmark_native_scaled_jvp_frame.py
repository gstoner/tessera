"""Paired JVP frame attribution with frozen Python preparation as diagnostic control."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import textwrap
import time
from types import MethodType

import numpy as np

from tessera import runtime as rt
from tessera.compiler.native_scaled_program import NativeScaledProgram, PreparedScaledProgram
from tessera.compiler.native_vmap import mixed_batch_policies, normalize_mixed_batch_inputs
from tests.device.rocm.test_public_scaled_jvp import oracle
from tests.device.rocm.test_native_scaled_jvp_frame import padded
from tests.unit.test_composed_scaled_jvp import case
from tests.unit.test_native_scaled_map_axes import axis_case

SOURCES = (
    "python/tessera/compiler/jit.py", "python/tessera/compiler/native_vmap.py",
    "python/tessera/compiler/paged_host_span.py",
    "python/tessera/compiler/native_scaled_program.py",
    "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_program_runtime.cpp",
    "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_program_runtime.h",
    "tests/unit/test_native_scaled_jvp_frame.py",
    "tests/device/rocm/test_native_scaled_jvp_frame.py",
    "benchmarks/rocm/benchmark_native_scaled_jvp_frame.py",
)


def reference_method(path):
    source = path.read_text()
    parsed = ast.parse(source)
    cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "JitFn")
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "native_jvp")
    method_source = textwrap.dedent(ast.get_source_segment(source, node))
    scope = dict(importlib.import_module("tessera.compiler.jit").__dict__)
    exec(compile(method_source, str(path), "exec"), scope)
    return scope["native_jvp"], method_source


def scenario(name, nk=False):
    if name in ("single", "nested", "cartesian"):
        _, owner, values, expected = axis_case(
            "fp32", nk, mode="forward", nested=name != "single", cartesian=name == "cartesian"
        )
        seeds = tuple(padded(value*factor) for value, factor in
                      zip(values[2:], (.125, -.0625), strict=True))
        return owner, values, seeds, (expected, expected*.0625)
    shape = {"composed": (17, 19, 256), "partial": (3, 5, 37),
             "large": (200, 129, 1536)}[name]
    owner, values, seeds = case(shape, ("sb1", "sa0", "sb0", "sa1"))
    values, seeds = tuple(map(padded, values)), tuple(map(padded, seeds))
    a, b, sa0, sb0, sa1, sb1 = values
    mapping = dict(zip(owner.differentiation_request.wrt, seeds, strict=True))
    first = oracle(a, b, sa0, sb0, mapping["sa0"], mapping["sb0"])
    second = oracle(a, b, sa1, sb1, mapping["sa1"], mapping["sb1"])
    expected = tuple(left+right for left, right in zip(first, second, strict=True))
    return owner, values, seeds, expected


def run_profile(name, nk, old_method):
    owner, values, seeds, expected = scenario(name, nk)
    native_method = owner.native_jvp
    reference = MethodType(old_method, owner)
    hashes = []
    for method in (native_method, reference):
        result = method(*values, tangents=seeds)
        for got, want in zip(result, expected, strict=True):
            np.testing.assert_allclose(got, want, rtol=4e-5, atol=1e-4)
        hashes.append(owner.last_jvp_execution["artifact_hash"])
    if hashes[0] != hashes[1] or len(owner._native_jvp_packages) != 1:
        raise RuntimeError("paired arms did not reuse one identical compiled package")
    outer = next(iter(owner._native_jvp_packages.values()))
    package = NativeScaledProgram.from_manifest(
        outer.contract["steps"][0]["child_metadata"]["native_scaled_program"])
    primals = owner._ordered_inputs(values, {})
    if mixed_batch_policies(owner):
        frame = list(values)
        for role, seed in zip(owner.differentiation_request.wrt_indices, seeds, strict=True):
            frame[role] = seed
        frame = normalize_mixed_batch_inputs(frame, owner._frontend_batch_policies)
        tangent_values = tuple(frame[role] for role in owner.differentiation_request.wrt_indices)
    else:
        tangent_values = seeds
    library = rt._load_rocm_native_movement_runtime()
    public = {"reference_python_frame": [], "native_frame": []}
    events = []
    with PreparedScaledProgram(package, (*primals, *tangent_values),
                               runtime_library=library._name) as resident:
        repetitions = 128
        while True:
            _, sample = resident.invoke(repeats=repetitions, timed=True)
            if sample*repetitions >= 20:
                break
            repetitions *= 2
            if repetitions > 262144:
                raise RuntimeError("resident event window remains too short")
        original_run = subprocess.run
        original_compact = np.ascontiguousarray
        def forbidden(*args, **kwargs):
            raise AssertionError("warm candidate called a compiler or Python compactor")
        subprocess.run = forbidden
        try:
            for round_index in range(7):
                arms = (("native_frame", native_method), ("reference_python_frame", reference))
                if round_index % 2:
                    arms = tuple(reversed(arms))
                for label, method in arms:
                    np.ascontiguousarray = forbidden if label == "native_frame" else original_compact
                    start = time.perf_counter()
                    for _ in range(32):
                        result = method(*values, tangents=seeds)
                    public[label].append((time.perf_counter()-start)*1000/32)
                    if owner.last_jvp_execution["artifact_hash"] != hashes[0]:
                        raise RuntimeError("timed package identity changed")
                    for got, want in zip(result, expected, strict=True):
                        np.testing.assert_allclose(got, want, rtol=4e-5, atol=1e-4)
                generation, sample = resident.invoke(repeats=repetitions, timed=True)
                events.append(sample)
            np.ascontiguousarray = original_compact
            for got, want in zip(resident.read(generation), expected, strict=True):
                np.testing.assert_allclose(got, want, rtol=4e-5, atol=1e-4)
        finally:
            subprocess.run = original_run
            np.ascontiguousarray = original_compact
    ratios = [left/right for left, right in zip(public["native_frame"],
                                               public["reference_python_frame"], strict=True)]
    return {
        "profile": name, "rhs_transposed": nk,
        "input_shapes": [list(value.shape) for value in values],
        "input_byte_strides": [list(value.strides) for value in values],
        "seed_byte_strides": [list(seed.strides) for seed in seeds],
        "wrt": owner.differentiation_request.wrt,
        "correctness": "passed_before_and_after_timing",
        "artifact_hash": hashes[0],
        "image_sha256": [hashlib.sha256(image).hexdigest() for image in package.images],
        "program_sha256": hashlib.sha256(package.program_json.encode()).hexdigest(),
        "public_completed_samples_ms": public,
        "public_completed_medians_ms": {name: median(samples) for name, samples in public.items()},
        "paired_native_over_reference_samples": ratios,
        "paired_native_over_reference_median": median(ratios),
        "calls_per_window": 32,
        "native_resident_event_samples_ms": events,
        "native_event_repetitions": repetitions,
        "native_event_windows_ms": [sample*repetitions for sample in events],
        "scope": "completed public JVP includes all frame preparation/transfers; resident event excludes them",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-jit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    architecture = rt._rocm_live_arch()
    if architecture != "gfx1201":
        raise RuntimeError(f"owning gfx1201 required, got {architecture}")
    old_method, method_source = reference_method(args.reference_jit)
    root = Path(__file__).resolve().parents[2]
    library = rt._load_rocm_native_movement_runtime()
    info = subprocess.run(["/opt/rocm/bin/rocminfo"], check=True,
                          capture_output=True, text=True).stdout
    packet = {
        "schema": "tessera.rocm.native_scaled_jvp_frame.v1",
        "architecture": architecture,
        "device_lines": [line.strip() for line in info.splitlines()
                         if "Marketing Name:" in line or "Name:                    gfx" in line],
        "source_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCES},
        "reference_commit": "747c3364e4c20a55c57b112c3032a08838ee60c1",
        "reference_file_sha256": hashlib.sha256(args.reference_jit.read_bytes()).hexdigest(),
        "reference_method_sha256": hashlib.sha256(method_source.encode()).hexdigest(),
        "runtime_path": library._name,
        "runtime_sha256": hashlib.sha256(Path(library._name).read_bytes()).hexdigest(),
        "compiler_sha256": {name: hashlib.sha256(Path(os.environ[name]).read_bytes()).hexdigest()
                            for name in ("TESSERA_OPT", "TESSERA_ROCM_OPT")},
        "rows": [run_profile(name, nk, old_method)
                 for name, nk in [(name, nk) for name in ("single", "nested", "cartesian")
                                  for nk in (False, True)] +
                 [(name, False) for name in ("composed", "partial", "large")]],
        "closure": "native frame preparation extension; generic batching/transpose closure remains open",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2)+"\n")
    args.output.with_suffix(".reference-method.txt").write_text(method_source+"\n")
    for row in packet["rows"]:
        print(row["profile"], row["rhs_transposed"], row["public_completed_medians_ms"],
              row["paired_native_over_reference_median"])


if __name__ == "__main__":
    main()
