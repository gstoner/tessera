"""Correctness-gated semantic DAG public-call and device-event characterization."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import time
from unittest.mock import patch

import numpy as np
import tessera as ts
from benchmarks.nvidia.record_native_tensor_binary import event_windows
from benchmarks.nvidia.record_native_tensor_jit import identity
from tests.device.nvidia import test_native_semantic_dag_ownership as semantic
from tests.device.nvidia.test_native_semantic_dag_ownership import CASES, inputs, check

FUNCTIONS = {name: getattr(semantic, name) for name in CASES}


def options(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", nargs="+", choices=CASES, default=list(CASES))
    p.add_argument("--storage", nargs="+", choices=("f16", "bf16", "f32"), default=["f16", "bf16", "f32"])
    p.add_argument("--rows", nargs="+", type=int, default=[1, 17, 65])
    p.add_argument("--bound", type=int, default=65)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--calls", type=int, default=16)
    args = p.parse_args(argv)
    if not 17 <= args.bound < 2**31 or any(not 0 < m <= args.bound for m in args.rows):
        raise ValueError("rows must fit a positive bound containing the M1 trace")
    if not 2 <= args.rounds <= 1000 or args.rounds % 2 or not 1 <= args.calls <= 10000:
        raise ValueError("invalid round or call count")
    if any(len(set(values)) != len(values) for values in (args.rows, args.case, args.storage)):
        raise ValueError("case, storage and row selections must be unique")
    return args


def record_program(kind, storage, args, packet, save):
    fn = ts.jit(target="nvidia_sm120", shape_bounds={"M": args.bound})(FUNCTIONS[kind])
    try:
        initial = inputs(storage, 1, 19301)
        started = time.perf_counter_ns()
        first = fn(*initial)
        cold_ms = (time.perf_counter_ns() - started) / 1e6
        check(kind, first, initial)
        program, owner = fn._native_tensor_last_program, fn._native_tensor_last_owner
        if fn.execution_kind != "native_gpu" or program.frame["schema"] != "tessera.native.tensor_program_frame.v2":
            raise RuntimeError("recorder requires completed bounded native execution")
        saved_first = first.copy()
        before = owner.stats()
        for m in args.rows:
            values = inputs(storage, m, 19301 + m)
            snapshots = tuple(value.copy() for value in values)
            check(kind, fn(*values), values)
            windows = event_windows(owner, args.rounds)
            check(kind, fn(*values), values)

            def forbidden(*unused, **keywords):
                raise RuntimeError("timed public call traced, compiled or used eager execution")

            samples, elapsed_windows = [], []
            with patch.object(fn, "_fn", forbidden), patch.object(fn, "_trace_frontend_capture", forbidden), \
                    patch("subprocess.run", forbidden), patch("subprocess.Popen", forbidden):
                for _ in range(args.rounds):
                    started = time.perf_counter_ns()
                    for _ in range(args.calls):
                        actual = fn(*values)
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    elapsed_windows.append(elapsed)
                    samples.append(elapsed / args.calls)
            check(kind, actual, values)
            np.testing.assert_array_equal(first, saved_first)
            for value, snapshot in zip(values, snapshots, strict=True):
                np.testing.assert_array_equal(value, snapshot)
            after = owner.stats()
            if (fn._native_tensor_last_program is not program or fn._native_tensor_last_owner is not owner
                    or before["allocations"] != 1 or after["allocations"] != 1
                    or before["capacity"] != after["capacity"]):
                raise RuntimeError("changed-M recording lost its retained native program or arena")
            packet["rows"].append({"case": kind, "storage": storage, "active_m": m,
                "root_shapes": [list(value.shape) for value in values],
                "cold_public_trace_compile_prepare_execute_ms": cold_ms,
                "cold_sample_group": f"{kind}:{storage}", "cold_sample_repeated_across_rows": True,
                "correctness": "independent_oracle_before_after_timing",
                "device_windows": windows, "completed_public_samples_ms": samples,
                "completed_public_median_ms": median(samples), "completed_public_window_ms": elapsed_windows,
                "calls_per_sample": args.calls, "before": before, "after": after,
                "frame": program.frame,
                "source_ir_sha256": hashlib.sha256(program.source_ir.encode()).hexdigest(),
                "image_sha256": [hashlib.sha256(image).hexdigest() for image in program.images]})
            save()
            print(kind, storage, m, "passed and recorded", flush=True)
    finally:
        fn.close_native_storage()


def main(argv=None):
    args = options(argv)
    source = identity()
    for name in ("benchmarks/nvidia/record_native_semantic_dag.py",
                 "python/tessera/compiler/bounded_nvidia_lhs.py",
                 "tests/device/nvidia/test_native_semantic_dag_ownership.py",
                 "tests/unit/test_native_semantic_dag_ownership.py"):
        source["source_sha256"][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    packet = {"schema": "tessera.benchmark.native_semantic_dag.v1", "source": source,
              "status": "recording", "row_bound": args.bound, "rounds": args.rounds, "rows": [],
              "scope": "One-root compact semantic DAGs; M-only bounded shapes. One cold public sample per case/storage group is repeated for context, not independent observations. Independent member/program CUDA regions are not additive; under-10ms windows are diagnostic. Warm public calls include binding, transfers and readback; exclude compilation/prepare. Same images and native arena across active rows. No speedup or general closure claim."}

    def save():
        args.output.write_text(json.dumps(packet, indent=2) + "\n")

    save()
    try:
        for kind in args.case:
            for storage in args.storage:
                record_program(kind, storage, args, packet, save)
    except Exception as error:
        packet.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    packet["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
