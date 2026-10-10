"""Ordinary frontend movement JIT and checked-launch overhead on exact ROCm GPUs."""
import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler import rocm_native
from tessera.compiler.emit.rocm_hip import run_paged_kv_cache_read_f32
from benchmarks.rocm.benchmark_native_movement import device_identity
from tests.unit.test_public_movement_frontend import inputs, paged, paged_default, dispatched


def record(arch, family, large, directory, repeats):
    fn = ts.jit(target="rocm_" + arch, native_required=True)(
        dict(paged=paged, paged_default=paged_default, dispatched=dispatched)[family])
    args, expected = inputs(family, large)
    begin = time.perf_counter_ns()
    actual = fn(*args)
    first_ms = (time.perf_counter_ns() - begin) / 1e6
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert fn.execution_kind == "native_gpu" and fn.last_fallback_reason is None
    module, _ = fn._trace_frontend_capture(args, {})
    contract = (rocm_native._moe_dispatch_contract(module) if family == "dispatched"
                else rocm_native._paged_kv_contract(module))
    names = ("T", "S", "H") if family == "dispatched" else (
        "P", "LP", "PageSize", "H", "D", "Start", "Tokens")
    artifact = fn.runtime_artifact()
    buffers = dict(zip((arg.name for arg in module.functions[0].args), args, strict=True))
    output = np.empty_like(expected)
    buffers[contract[2]] = output
    scalar_values = dict(zip(names, contract[3], strict=True))

    def descriptor():
        receipt = rt.launch(artifact, {"buffers": buffers, "scalars": scalar_values})
        assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
        return output

    if family == "dispatched":
        token, x = args
        token64 = token.astype(np.int64)
        def retained():
            return rt._rocm_gather_rows(x, token64, np)
    else:
        x, table = args if family == "paged" else (args[1], args[0])
        start, count = contract[3][-2:]
        logical = np.arange(start, start + count, dtype=np.int64)
        def retained():
            return run_paged_kv_cache_read_f32(x, table, logical)

    def prepared():
        os.environ["TESSERA_ROCM_PREPARED_MOVEMENT"] = "1"
        value = fn(*args)
        assert fn._native_descriptor_last_receipt.get("native_call_binding") == "prepared_cpp_movement"
        return value

    def unprepared():
        os.environ["TESSERA_ROCM_PREPARED_MOVEMENT"] = "0"
        try:
            value = fn(*args)
            assert "native_call_binding" not in fn._native_descriptor_last_receipt
            return value
        finally:
            os.environ["TESSERA_ROCM_PREPARED_MOVEMENT"] = "1"

    functions = dict(jit=prepared, jit_unprepared=unprepared, descriptor=descriptor, retained=retained)
    samples = {name: [] for name in functions}
    counters = []
    for function in functions.values():
        for _ in range(3):
            np.testing.assert_array_equal(function().view(np.uint32), expected.view(np.uint32))
    labels = list(functions)
    for trial in range(10):
        order = labels[trial % len(labels):] + labels[:trial % len(labels)]
        if trial % 2:
            order.reverse()
        for name in order:
            function = functions[name]
            np.testing.assert_array_equal(function().view(np.uint32), expected.view(np.uint32))
            before = rt._rocm_native_movement_stats()
            begin = time.perf_counter_ns()
            for _ in range(repeats):
                value = function()
            samples[name].append((time.perf_counter_ns() - begin) / 1e6 / repeats)
            after = rt._rocm_native_movement_stats()
            if name != "retained":
                delta = {key: after[key] - before[key] for key in before}
                if delta["allocations"] or delta["frees"] or delta["launches"] != repeats or delta["reuses"] != 3*repeats:
                    raise RuntimeError(f"warm native staging differs: {delta}")
                counters.append(dict(arm=name, trial=trial, **delta))
            np.testing.assert_array_equal(value.view(np.uint32), expected.view(np.uint32))
    label = family + ("_large" if large else "_small")
    bundle = fn.compile_bundle
    stages = [bundle.graph, bundle.schedule, bundle.tile, bundle.target_ir, bundle.backend]
    for previous, current in zip(stages[:-1], stages[1:], strict=True):
        assert current.input_digest == previous.output_digest
    assert bundle.schedule.producer == "tessera-opt.tessera-graph-to-schedule"
    for name, stage in zip(("graph", "schedule", "tile", "target", "backend"), stages, strict=True):
        (directory / (label + "." + name + ".mlir")).write_text(stage.text)
    (directory / (label + ".hsaco")).write_bytes(artifact.native_image.payload)
    for name, function in functions.items():
        profile = cProfile.Profile()
        profile.enable()
        for _ in range(100):
            function()
        profile.disable()
        profile.dump_stats(str(directory / (label + "." + name + ".pstats")))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return dict(family=family, large=large, shape=list(contract[3]),
        correctness="bit_exact_before_after_every_trial", first_call_wall_ms=first_ms,
        warm_call_samples_ms=samples, warm_call_medians_ms=medians,
        retained_over_jit=medians["retained"]/medians["jit"],
        jit_over_descriptor=medians["jit"]/medians["descriptor"],
        native_staging_counter_deltas=counters,
        prepared_over_unprepared=medians["jit"]/medians["jit_unprepared"],
        retained_non_regression_10pct=medians["jit"] <= 1.1*medians["retained"],
        execution_kind=fn.execution_kind, fallback_reason=fn.last_fallback_reason,
        image_digest=artifact.native_image.image_digest,
        schedule_digest=artifact.launch_descriptor.provenance["schedule_digest"],
        artifact_chain=[dict(producer=s.producer, input_digest=s.input_digest,
                             output_digest=s.output_digest) for s in stages])


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
        raise RuntimeError("owning HIP GPU is unavailable")
    identity = device_identity(hip, args.architecture)
    directory = args.output.parent / "artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    families = ["paged", "paged_default"] + (["dispatched"] if args.architecture == "gfx1151" else [])
    try:
        rows = [record(args.architecture, family, large, directory, args.repeats)
                for family in families for large in (False, True)]
        files = ["python/tessera/autodiff/tape.py", "python/tessera/autodiff/jvp.py", "python/tessera/autodiff/vjp.py",
                 "python/tessera/autodiff/law_inputs.py", "python/tessera/__init__.py", "python/tessera/compiler/jit.py",
                 "python/tessera/compiler/graph_ir.py", "python/tessera/compiler/trace.py",
                 "python/tessera/compiler/prepared_rocm_movement.py",
                 "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_movement_runtime.cpp",
                 "python/tessera/compiler/op_catalog.py", "python/tessera/compiler/rocm_native.py",
                 "src/compiler/programming_model/lib/NativePagedKV.h",
                 "src/compiler/programming_model/lib/NativeMoeDispatch.h",
                 "benchmarks/rocm/benchmark_public_movement_jit.py",
                 "tests/unit/test_public_movement_frontend.py"]
        from tessera.compiler.scheduled_matmul import find_tessera_opt
        tool = find_tessera_opt()
        files.append(str(tool))
        files.append(rt._load_rocm_native_movement_runtime()._name)
        hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files}
        args.output.write_text(json.dumps(dict(device=identity, architecture=args.architecture,
            scope="host_wall_full_calls_not_isolated_kernel_time", repeats=args.repeats,
            source_and_compiler_sha256=hashes, rows=rows), indent=2) + "\n")
        print(json.dumps([dict(family=row["family"], large=row["large"],
            medians=row["warm_call_medians_ms"], retained_over_jit=row["retained_over_jit"])
            for row in rows], indent=2))
    finally:
        rt._clear_rocm_native_image_cache()


if __name__ == "__main__":
    main()
