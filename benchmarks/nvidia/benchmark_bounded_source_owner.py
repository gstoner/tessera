"""Correctness-gated paired native bounded source reverse owner measurements.

This diagnostic recorder uses compiler-exported members and the retained C++
owner. It is not the ordinary public JIT or portable runtime adapter.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess
import time

import ml_dtypes
import numpy as np

from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests.device.nvidia.test_bounded_source_reverse_owner import RawBoundedSourceOwner, guarded_outputs
from tests.device.nvidia.test_source_attention_reverse import oracle


def paired(storage, capacity, samples, reps, reverse):
    dtype = np.dtype(np.float16 if storage == "f16" else ml_dtypes.bfloat16)
    sq_cap, sk_cap, d, dv = capacity
    trace = (np.zeros((1, 2, 1, d), dtype), np.zeros((1, 1, 1, d), dtype),
             np.zeros((1, 1, 1, dv), dtype))
    owners, programs, records = {}, {}, []
    order = ("rows", "inline") if reverse else ("inline", "rows")
    try:
        for label in order:
            started = time.perf_counter()
            owner = RawBoundedSourceOwner(trace, storage, label == "rows", "normal", (0, 1, 2), capacity[:2])
            owners[label] = owner
            programs[label] = {
                "source_compile_host_ms": owner.compile_host_ms,
                "native_prepare_host_ms": owner.native_prepare_host_ms,
                "constructor_total_host_ms": (time.perf_counter() - started) * 1000,
                "program_graph_sha256": owner.plan["program_graph_sha256"],
                "image_sha256": [hashlib.sha256(bytes(x)[:-1]).hexdigest() for x in owner.images],
                "native_prepare_scope": "one registration per capacity and policy; reused across all frames",
            }
        shapes = [(1, 1), (3, 5), (sq_cap, sk_cap)]
        if reverse:
            shapes.reverse()
        for sq, sk in shapes:
            rng = np.random.default_rng(20261019 + sq * 100 + sk)
            dims = (1, 2, 1, sq, sk, d, dv)
            shapes_ = ((1, 2, sq, d), (1, 1, sk, d), (1, 1, sk, dv), (1, 2, sq, dv), (1, 2, sq))
            values = tuple(rng.normal(0, .2, shape).astype(np.float32 if i == 4 else dtype)
                           for i, shape in enumerate(shapes_))
            frozen = tuple(x.tobytes() for x in values)
            for x in values:
                x.flags.writeable = False
            expected = oracle(*values)
            outputs = {label: guarded_outputs(values, (0, 1, 2)) for label in owners}
            records_ = {label: {"device_event_samples_ms": [], "completed_ctypes_host_samples_ms": []}
                        for label in owners}

            def check():
                for label, owner in owners.items():
                    arenas, results = outputs[label]
                    owner.check(owner.call(dims, values, results))
                    errors = []
                    for actual, ref in zip(results, expected, strict=True):
                        np.testing.assert_allclose(actual.astype(np.float32), ref.astype(np.float32),
                                                   rtol=.03 if storage == "bf16" else .006, atol=.002)
                        errors.append(float(np.max(np.abs(actual.astype(np.float32) - ref.astype(np.float32)))))
                    records_[label]["max_abs_error"] = errors
                    for arena in arenas:
                        np.testing.assert_array_equal(arena[:8], arena.dtype.type(23))
                        np.testing.assert_array_equal(arena[-8:], arena.dtype.type(23))
                for value, original in zip(values, frozen, strict=True):
                    assert value.tobytes() == original

            check()
            for trial in range(samples):
                trial_order = ("inline", "rows") if (trial % 2 == 0) != reverse else ("rows", "inline")
                for label in trial_order:
                    owner = owners[label]
                    for _ in range(reps):
                        started = time.perf_counter()
                        owner.check(owner.call(dims, values, outputs[label][1]))
                        records_[label]["completed_ctypes_host_samples_ms"].append(
                            (time.perf_counter() - started) * 1000)
                        times = (ct.c_float * (6 if owner.rows else 5))()
                        owner.check(owner.times(owner.handle, times, len(times)))
                        assert all(np.isfinite(x) and x >= 0 for x in times)
                        records_[label]["device_event_samples_ms"].append(list(times))
            check()
            for record in records_.values():
                events = record["device_event_samples_ms"]
                record["stage_event_median_ms"] = [median(x[i] for x in events)
                                                   for i in range(len(events[0]) - 1)]
                record["program_event_median_ms"] = median(x[-1] for x in events)
                record["completed_ctypes_host_median_ms"] = median(record["completed_ctypes_host_samples_ms"])
                record["correctness"] = "independent_source_vjp_oracle_and_host_guards_before_after"
            records.append({"storage": storage, "capacity": list(capacity), "runtime_dims": list(dims),
                            "input_sha256": [hashlib.sha256(x).hexdigest() for x in frozen], "arms": records_})
            print(storage, capacity, sq, sk,
                  {label: r["program_event_median_ms"] for label, r in records_.items()}, flush=True)
    finally:
        for owner in reversed(tuple(owners.values())):
            owner.close()
    return programs, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 3 or args.reps < 1:
        parser.error("at least three samples and positive repetitions required")
    device = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,compute_cap,driver_version,uuid",
        "--format=csv,noheader"], text=True).strip()
    if len(device.splitlines()) != 1 or device.split(",")[1].strip() != "12.0":
        raise RuntimeError("requires one owning SM120 GPU")
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("requires the matching native compiler")
    profiles = [(storage, cap) for storage in ("f16", "bf16")
                for cap in ((7, 11, 4, 3), (17, 19, 35, 7), (17, 19, 257, 13))]
    if args.reverse:
        profiles.reverse()
    packet = {"schema": "tessera.bounded_source_owner_measurement.v1", "device": device,
              "scope": "native retained source C ABI; not ordinary public JIT/portable or isolated profiler kernel time",
              "route": "frontend Graph -> paired native AD -> bounded program -> Schedule -> Tile -> NVIDIA/NVVM -> PTX -> native owner",
              "samples": args.samples, "reps": args.reps, "reverse": args.reverse,
              "compiler_sha256": hashlib.sha256(Path(tool).read_bytes()).hexdigest(),
              "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
                  "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp",
                  "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.h",
                  "tests/device/nvidia/test_bounded_source_reverse_owner.py",
                  "benchmarks/nvidia/benchmark_bounded_source_owner.py")},
              "programs": [], "rows": []}
    for storage, capacity in profiles:
        programs, rows = paired(storage, capacity, args.samples, args.reps, args.reverse)
        packet["programs"].append({"storage": storage, "capacity": list(capacity), "arms": programs})
        packet["rows"].extend(rows)
    from tessera import runtime as rt
    packet["provider_sha256"] = hashlib.sha256(Path(rt._load_nvidia_ptx_launch()._name).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
