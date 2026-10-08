"""Correctness-gated physical page layouts; device and public clocks differ."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from statistics import median
import time
from unittest.mock import patch

import numpy as np
import tessera as ts
from tessera import runtime as rt
from benchmarks.rocm.benchmark_captured_movement import paged_full
from benchmarks.rocm.benchmark_native_movement import device_identity
from tessera.compiler.paged_host_span import checked_page_span

SOURCES = (
    "python/tessera/compiler/scheduled_paged_kv.py",
    "python/tessera/compiler/graph_ir.py",
    "python/tessera/compiler/paged_host_span.py",
    "python/tessera/compiler/prepared_rocm_movement.py",
    "python/tessera/compiler/resident_rocm_movement.py",
    "python/tessera/compiler/rocm_native.py",
    "python/tessera/compiler/jit.py",
    "python/tessera/runtime.py",
    "src/compiler/programming_model/lib/NativePagedKV.h",
    "src/compiler/ir/TileOps.cpp",
    "src/compiler/ir/include/Tessera/Dialect/Tile/TileOps.td",
    "src/compiler/codegen/Tessera_ROCM_Backend/include/TesseraROCM/IR/TesseraROCMOps.td",
    "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp",
    "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateROCMPagedKVReadKernel.cpp",
    "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/MovementPhysicalSpan.h",
    "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_movement_runtime.cpp",
    "benchmarks/rocm/benchmark_strided_paged_kv.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def storage(dense, layout):
    if layout == "compact":
        return dense.copy()
    p, page, h, d = dense.shape
    if layout == "padded":
        result = np.empty((p+1, page+1, h, 2*d+2), np.float32)[1:, 1:, :, 1:2*d+1:2]
    elif layout == "permuted":
        result = np.empty((d, h, page, p), np.float32).transpose(3, 2, 1, 0)
    else:
        result = np.empty(dense.shape, np.float32, order="F")
    result[...] = dense
    return result


def record(architecture, directory, rounds, repeats):
    root = Path(__file__).resolve().parents[2]
    identity = device_identity(rt._load_hip_for_launch(), architecture)
    cases = []
    try:
        for label, shape, logical in (("small", (4, 256, 3, 8), 4),
                                      ("large", (32, 16, 8, 128), 64)):
            rng = np.random.default_rng(120_518)
            dense = rng.normal(size=shape).astype(np.float32)
            table = rng.integers(0, shape[0], size=logical, dtype=np.int32)
            expected = dense[table].reshape(1024, shape[2], shape[3])
            for layout in ("compact", "padded", "permuted", "fortran"):
                x = storage(dense, layout)
                fn = ts.jit(target="rocm_"+architecture, native_required=True)(paged_full)
                np.testing.assert_array_equal(fn(x, table), expected)
                owner = fn.prepare_native_movement(x, table)
                bundle = fn.compile_bundle
                stages = (bundle.graph, bundle.schedule, bundle.tile, bundle.target_ir, bundle.backend)
                assert bundle.schedule.producer == "tessera-opt.tessera-graph-to-schedule"
                for a, b in zip(stages[:-1], stages[1:], strict=True):
                    assert b.input_digest == a.output_digest
                stem = label+"_"+layout
                for stage in stages:
                    (directory/(stem+"."+stage.level+".mlir")).write_text(stage.text)
                span, strides = checked_page_span(x)
                data = dict(profile=label, layout=layout, page_shape=list(shape),
                            logical_pages=logical, output_shape=list(expected.shape),
                            element_strides=list(strides), addressed_bytes=span,
                            logical_input_bytes=x.nbytes, entry=bundle.launch_descriptor.entry_symbol,
                            abi_id=bundle.launch_descriptor.abi_id,
                            image_digest=bundle.native_image.image_digest,
                            image_cache_key=bundle.native_image.cache_key,
                            payload_digest=bundle.native_image.payload_digest,
                            schedule_digest=bundle.launch_descriptor.provenance["schedule_digest"],
                            lineage=[dict(level=s.level, producer=s.producer, input_digest=s.input_digest,
                                          output_digest=s.output_digest) for s in stages],
                            device_event_ms=[], public_complete_ms=[], public_window_ms=[],
                            correctness="passed_before_and_after_timing", warm_compiler_calls=0)
                cases.append((fn, owner, x, table.copy(), expected, data))
                # Keep the public/prepared and resident input tables identical.
                owner.upload((x, cases[-1][3]))
        def forbidden(*args, **kwargs):
            raise AssertionError("compiler invocation during warm benchmark")
        with ExitStack() as guards:
            for method in ("subprocess.run", "subprocess.Popen", "subprocess.check_output"):
                guards.enter_context(patch(method, forbidden))
            for fn, owner, x, table, expected, _ in cases:
                for _ in range(4):
                    np.testing.assert_array_equal(fn(x, table), expected)
                    owner.execute(download=False)
            # Rotate order each round. Retain raw paired samples; no route promotion.
            for round_index in range(rounds):
                ordered = cases[round_index % len(cases):] + cases[:round_index % len(cases)]
                for fn, owner, x, table, expected, data in ordered:
                    for _ in range(repeats):
                        _, receipt = owner.execute(download=False)
                        assert receipt["submission"] == "native_direct"
                        data["device_event_ms"].append(receipt["kernel_elapsed_ms"])
                    start = time.perf_counter_ns()
                    for _ in range(repeats):
                        output = fn(x, table)
                    window = (time.perf_counter_ns()-start)/1e6
                    data["public_window_ms"].append(window)
                    data["public_complete_ms"].append(window/repeats)
                    np.testing.assert_array_equal(output, expected)
            for fn, owner, x, table, expected, _ in cases:
                np.testing.assert_array_equal(owner.execute()[0], expected)
                retained = fn(x, table)
                original = retained.copy()
                x[...] *= np.float32(-0.75)
                table[:] = table[::-1]
                changed = x[table].reshape(expected.shape)
                np.testing.assert_array_equal(fn(x, table), changed)
                np.testing.assert_array_equal(retained, original)
                owner.upload((x, table))
                np.testing.assert_array_equal(owner.execute()[0], changed)
        rows = []
        for _, _, _, _, _, data in cases:
            data["device_event_median_ms"] = median(data["device_event_ms"])
            data["public_complete_median_ms"] = median(data["public_complete_ms"])
            rows.append(data)
        import os
        return dict(schema="tessera.rocm.strided_paged_kv.v1", architecture=architecture,
                    device=identity, source_hashes={name: sha(root/name) for name in SOURCES},
                    compiler_sha256=sha(os.environ["TESSERA_OPT"]),
                    target_compiler_sha256=sha(os.environ["TESSERA_ROCM_OPT"]),
                    runtime_sha256=sha(os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]),
                    device_event_scope="single resident kernel; upload/download excluded",
                    public_scope="completed public JIT with upload/download; compilation excluded",
                    default_route_promotion=False, rounds=rounds, repeats=repeats, cases=rows)
    finally:
        for _, owner, *_ in cases:
            owner.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", required=True, choices=("gfx1151", "gfx1201"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=32)
    args = parser.parse_args()
    if args.rounds <= 0 or args.repeats <= 0:
        raise ValueError("rounds and repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    packet = record(args.architecture, args.output, args.rounds, args.repeats)
    (args.output/"device.json").write_text(json.dumps(packet, indent=2)+"\n")


if __name__ == "__main__":
    main()
