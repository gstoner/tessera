"""Matched full-update versus import-once native NVFP4 program measurements."""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.nvfp4_expansion import nvfp4_expansion_policy
from tessera.compiler.rocm_nvfp4_expansion_program import package_native_nvfp4_expansion_program
from tessera.compiler.trace import trace, to_graph_ir_module


def measure(m, n, k, provider, output):
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("requires owning gfx1201")
    if min(m, n, k) <= 0 or n < 4 or k % 16:
        raise ValueError("positive M/N and whole K16 groups required")
    rng = np.random.default_rng(20261015)
    codes = rng.integers(0, 256, (n, k // 2), np.uint8)
    scales = rng.choice([.5, 1., 2., 4.], (n, k // 16)).astype(ml_dtypes.float8_e4m3fn)
    globals_ = np.array([.25, .125], np.float64)
    a = rng.integers(-3, 4, (m, k)).astype(ml_dtypes.float8_e4m3fn)
    sa = rng.choice([.5, 1., 2.], (m, k // 16)).astype(np.float32)
    roots = [codes, scales, globals_, a, sa]

    def product(codes, scales, globals_, activation, activation_scales):
        b, sb = ts.ops.nvfp4_expand(codes, scales, globals_,
            row_offsets=[0, n // 2, n], numeric_policy=nvfp4_expansion_policy())
        return ts.ops.scaled_matmul(activation, b, activation_scales, sb,
            transposeB=True, numeric_policy={"accum": "fp32", "execution_mode": "exact_per_block"},
            scale_layout={"granularity": "block", "block": [1, 16], "format": "fp32"})

    # Independent nibble decoding, without the expansion/reference-matmul helpers.
    levels = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float64)
    weights = np.empty((n, k), np.float64)
    weights[:, 0::2], weights[:, 1::2] = levels[codes & 15], levels[codes >> 4]
    plane = (scales.astype(np.float64) * np.where(np.arange(n) < n // 2, globals_[0], globals_[1])[:, None]).astype(np.float32)
    weights *= np.repeat(plane.astype(np.float64), 16, axis=1)
    expected = ((a.astype(np.float64) * np.repeat(sa.astype(np.float64), 16, axis=1)) @ weights.T).astype(np.float32)
    graph = to_graph_ir_module(trace(product, *roots), target="rocm_gfx1201")
    started = time.perf_counter_ns()
    package = package_native_nvfp4_expansion_program(graph)
    compile_ms = (time.perf_counter_ns() - started) / 1e6
    samples = {"full_update_all_stages": [], "activation_update_consumer_only": []}
    events = {key: [] for key in samples}
    prepare_started = time.perf_counter_ns()
    with package.prepare(roots, runtime_library=str(provider)) as owner:
        prepare_ms = (time.perf_counter_ns() - prepare_started) / 1e6
        import_started = time.perf_counter_ns()
        imported, import_event_ms = owner.import_weights(timed=True)
        import_host_ms = (time.perf_counter_ns() - import_started) / 1e6
        generation, _ = owner.invoke_consumer()
        np.testing.assert_array_equal(owner.read(generation)[0], expected)
        _, pilot = owner.profile_members(repeats=1)
        repeats = min(65536, max(1, math.ceil(30. / min(pilot))))
        stage_samples = []
        for _ in range(3):
            generation, times = owner.profile_members(repeats=repeats)
            stage_samples.append(list(times))
            np.testing.assert_array_equal(owner.read(generation)[0], expected)
        for window in range(3):
            order = list(samples) if window % 2 == 0 else list(reversed(samples))
            for mode in order:
                started = time.perf_counter_ns()
                if mode == "full_update_all_stages":
                    owner.update(roots)
                    generation, elapsed = owner.invoke(timed=True)
                else:
                    owner.update_activations(a, sa)
                    generation, elapsed = owner.invoke_consumer(timed=True)
                actual, = owner.read(generation)
                samples[mode].append((time.perf_counter_ns() - started) / 1e6)
                events[mode].append(elapsed)
                np.testing.assert_array_equal(actual, expected)
    hip = rt._load_hip_for_launch()
    device, name = C.c_int(), C.create_string_buffer(256)
    if hip.hipGetDevice(C.byref(device)) or hip.hipDeviceGetName(name, len(name), device.value):
        raise RuntimeError("missing exact-device identity")
    sha = lambda data: hashlib.sha256(data).hexdigest()
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root / "python/tessera/compiler/rocm_nvfp4_expansion_program.py",
             root / "python/tessera/compiler/native_scaled_program.py",
             root / "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_program_runtime.cpp",
             root / "src/transforms/lib/NativeNVFP4Program.h"]
    packet = {
        "schema": "tessera.gfx1201.nvfp4_prepared_import_once.v1",
        "architecture": rt._rocm_live_arch(), "device": name.value.decode(), "shape_mnk": [m, n, k],
        "input_kind": "synthetic_exact_dyadic_scales", "model_quality_acceptance": False,
        "public_jit_composition": False, "frontend_capture": True,
        "compile_package_ms": compile_ms, "prepare_upload_ms": prepare_ms,
        "initial_import_completed_host_ms": import_host_ms,
        "initial_import_native_event_ms": import_event_ms,
        "member_graph_dispatch_samples_ms": stage_samples, "member_repeats": repeats,
        "completed_update_invoke_read_samples_ms": samples,
        "completed_update_invoke_read_median_ms": {key: statistics.median(value) for key, value in samples.items()},
        "native_event_with_enqueue_samples_ms": events,
        "per_call_upload_bytes": {"full_update_all_stages": sum(x.nbytes for x in roots),
                                  "activation_update_consumer_only": a.nbytes + sa.nbytes},
        "provider_sha256": sha(provider.read_bytes()),
        "program_sha256": sha(package.program_json.encode()),
        "compiler_sha256": sha(Path(os.environ["TESSERA_OPT"]).read_bytes()),
        "images": [p.image.image_digest for p in (package.producer.native, package.consumer)],
        "binding_digest": package.binding_digest,
        "source_sha256": {str(p.relative_to(root)): sha(p.read_bytes()) for p in paths},
        "correctness": "bitwise_fp32_independent_nibble_oracle_before_and_after_each_window",
        "timing_scope": "host includes boundary/upload/invoke/read; native event includes enqueue gaps; graph windows include graph dispatch",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet, indent=2) + "\n")
    print(json.dumps(packet["completed_update_invoke_read_median_ms"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=200)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--k", type=int, default=1536)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measure(args.m, args.n, args.k, args.provider, args.output)


if __name__ == "__main__":
    main()
