"""Exact-device admission comparison for compiled and retained ROCm movement."""
from __future__ import annotations

import argparse
import cProfile
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

from tessera import runtime as rt
from tessera.compiler import rocm_native
from tessera.compiler.canonical_compile import canonical_compile
from tessera.compiler.emit.rocm_hip import run_paged_kv_cache_read_f32
from benchmarks.rocm.benchmark_native_movement import device_identity, native_stats
from benchmarks.rocm.benchmark_rocm_e2e_movement import (
    _paged_module, _moe_module, _ResidentDescriptor, _event_ms,
)


def run_case(hip, movement, arch, family, shape, repeats, directory):
    rng = np.random.default_rng(120_509)
    if family == "paged_kv":
        p, page, h, d, start, tokens = shape
        module = _paged_module(*shape)
        x = rng.normal(size=(p, page, h, d)).astype(np.float32)
        indices = rng.permutation(p).astype(np.int32)
        output = np.zeros((tokens, h, d), np.float32)
        expected = x[indices].reshape(p*page, h, d)[start:start+tokens]
        dimensions = (p, p, page, h, d, start, tokens)
        args = dict(zip(("P", "LP", "PageSize", "H", "D", "Start", "Tokens"),
                        dimensions, strict=True), pages=x, page_table=indices, slice=output)
        token_indices = np.arange(start, start+tokens, dtype=np.int64)

        def retained():
            return run_paged_kv_cache_read_f32(x, indices, token_indices)
    else:
        t, s, h = shape
        module = _moe_module(*shape)
        x = rng.normal(size=(t, h)).astype(np.float32)
        indices = rng.integers(0, t, size=s, dtype=np.int32)
        output = np.zeros((s, h), np.float32)
        expected = x[indices]
        dimensions = shape
        args = dict(T=t, S=s, H=h, x=x, token=indices, o=output)
        row_indices = indices.astype(np.int64)

        def retained():
            return rt._rocm_gather_rows(x, row_indices, np)
    result = canonical_compile(module, target="rocm_"+arch, enable_tool_validation=True)
    if not result.executable:
        raise RuntimeError(f"canonical movement compile refused: {result.reason}")
    bundle = result.bundle
    stages = (bundle.graph, bundle.schedule, bundle.tile, bundle.target_ir, bundle.backend)
    for previous, current in zip(stages[:-1], stages[1:], strict=True):
        if current.input_digest != previous.output_digest:
            raise RuntimeError("canonical movement artifact chain is not adjacent")
    if bundle.schedule.producer != "tessera-opt.tessera-graph-to-schedule":
        raise RuntimeError("canonical movement is missing its native Schedule producer")
    package = rocm_native.ROCMNativePackage(
        tile_ir=bundle.tile.text, target_ir=bundle.target_ir.text,
        backend_ir=bundle.backend.text, image=result.native_image,
        descriptor=result.launch_descriptor)
    artifact = result.to_runtime_artifact()

    def compiled():
        result = rt.launch(artifact, args)
        if not result.get("ok") or result.get("execution_kind") != "native_gpu":
            raise RuntimeError(f"compiled launch failed: {result}")
        return output

    os.environ["TESSERA_ROCM_NATIVE_MOVEMENT"] = "1"
    os.environ["TESSERA_ROCM_MOVEMENT_STAGING_REUSE"] = "1"
    rt._clear_rocm_native_image_cache()
    samples = {"retained": [], "compiled": []}
    functions = dict(retained=retained, compiled=compiled)
    counters = []
    try:
        for name in functions:
            for _ in range(3):
                np.testing.assert_array_equal(functions[name](), expected)
        for trial in range(10):
            order = ("retained", "compiled") if trial % 2 == 0 else ("compiled", "retained")
            for name in order:
                np.testing.assert_array_equal(functions[name](), expected)
                before = native_stats(movement, "tessera_rocm_movement_stats",
                                      ("allocations", "frees", "reuses", "launches"))
                start_ns = time.perf_counter_ns()
                for _ in range(repeats):
                    value = functions[name]()
                elapsed = (time.perf_counter_ns()-start_ns)/1e6/repeats
                after = native_stats(movement, "tessera_rocm_movement_stats",
                                     ("allocations", "frees", "reuses", "launches"))
                np.testing.assert_array_equal(value, expected)
                if name == "compiled":
                    delta = {key: after[key]-before[key] for key in before}
                    if (delta["allocations"] or delta["frees"] or
                            delta["reuses"] != 3*repeats or delta["launches"] != repeats):
                        raise RuntimeError(f"native warm staging not proved: {delta}")
                    counters.append(delta)
                samples[name].append(elapsed)
        # Keep event dispatch diagnostics separate from end-to-end admission:
        # the retained page wrapper batches native HIP submissions, whereas the
        # descriptor event helper submits each kernel from Python.
        session = _ResidentDescriptor(hip, package, (x, indices),
                                      np.zeros_like(output), dimensions, output.size)
        try:
            session.launch()
            np.testing.assert_array_equal(session.read(), expected)
            events = [_event_ms(hip, session, 64) for _ in range(5)]
        finally:
            session.close()
        retained_events = None
        if family == "paged_kv":
            retained_events = []
            for _ in range(5):
                value, event = run_paged_kv_cache_read_f32(
                    x, indices, token_indices, return_device_ms=True, reps=64)
                np.testing.assert_array_equal(value, expected)
                retained_events.append(event)
        label = family+"_"+"_".join(map(str, shape))
        profile = cProfile.Profile()
        profile.enable()
        for _ in range(100):
            compiled()
        profile.disable()
        profile.dump_stats(str(directory/(label+".pstats")))
        for suffix, value in (("graph", bundle.graph.text), ("schedule", bundle.schedule.text),
                              ("tile", package.tile_ir), ("target", package.target_ir),
                              ("backend", package.backend_ir)):
            (directory/(label+"."+suffix+".mlir")).write_text(value)
        (directory/(label+".hsaco")).write_bytes(package.image.payload)
        medians = {name: statistics.median(values) for name, values in samples.items()}
        return dict(
            family=family, shape=list(shape), correctness="bit_exact_before_after_each_trial",
            canonical_executable=result.executable,
            artifact_chain=[dict(producer=stage.producer, input_digest=stage.input_digest,
                                 output_digest=stage.output_digest) for stage in stages],
            host_wall_samples_ms=samples, host_wall_medians_ms=medians,
            retained_over_compiled=medians["retained"]/medians["compiled"],
            non_regression_10pct=medians["compiled"] <= 1.1*medians["retained"],
            compiler_resident_dispatch_ms=events, retained_native_dispatch_ms=retained_events,
            native_staging_counter_deltas=counters,
            payload_sha256=hashlib.sha256(package.image.payload).hexdigest(),
            image_digest=package.image.image_digest, abi_id=package.descriptor.abi_id,
            schedule_digest=package.descriptor.provenance["schedule_digest"])
    finally:
        rt._clear_rocm_native_image_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("gfx1151", "gfx1201"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    hip = rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) or hip.hipDeviceSynchronize():
        raise RuntimeError("owning HIP device unavailable")
    identity = device_identity(hip, args.architecture)
    movement = rt._load_rocm_native_movement_runtime()
    if movement is None:
        raise RuntimeError("native movement service is required")
    directory = args.output.parent/"artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    cases = [("paged_kv", shape) for shape in
             ((4, 4, 3, 8, 1, 5), (32, 16, 4, 64, 3, 31), (64, 32, 8, 128, 7, 249))]
    if args.architecture == "gfx1151":
        cases += [("moe_dispatch", shape) for shape in
                  ((7, 9, 13), (64, 128, 256), (512, 768, 1024))]
    rows = []
    for family, shape in cases:
        row = run_case(hip, movement, args.architecture, family, shape, args.repeats, directory)
        rows.append(row)
        print(f'{family} {shape}: retained/compiled={row["retained_over_compiled"]:.3f}', flush=True)
    sources = ("python/tessera/runtime.py", "python/tessera/compiler/driver.py",
               "python/tessera/compiler/canonical_compile.py",
               "python/tessera/compiler/capabilities.py",
               "python/tessera/compiler/backend_manifest.py",
               "python/tessera/compiler/execution_matrix.py",
               "python/tessera/compiler/scheduled_paged_kv.py",
               "python/tessera/compiler/scheduled_moe_dispatch.py",
               "python/tessera/compiler/emit/rocm_hip.py",
               "python/tessera/compiler/rocm_native.py",
               "benchmarks/rocm/benchmark_movement_route_admission.py",
               "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_movement_runtime.cpp")
    result = dict(
        device=identity, architecture=args.architecture, rows=rows, repeats=args.repeats,
        admission_scope="ten alternating warm end-to-end trials; compiled pooled native service vs retained production helper",
        event_scope="diagnostic only: Python descriptor dispatch windows and native HIP-batched retained event windows have different host submission overhead",
        all_non_regression=all(row["non_regression_10pct"] for row in rows),
        selector_changed=False,
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        native_movement_library_sha256=hashlib.sha256(Path(movement._name).read_bytes()).hexdigest(),
        source_sha256={path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sources})
    args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
