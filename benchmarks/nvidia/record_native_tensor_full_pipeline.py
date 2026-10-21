"""Correctness-gated complete Graph-to-NVVM native owner characterization."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import time
from unittest.mock import patch

import numpy as np
from benchmarks.nvidia.record_native_tensor_binary import event_windows
from benchmarks.nvidia.record_native_tensor_jit import identity
from tessera.compiler.prepared_native_tensor_frame import PreparedNativeTensorFrame
from tests.device.nvidia.test_native_tensor_pipeline_layout import (
    check_semantic, full_native_program,
)
from tests.device.nvidia.test_native_tensor_binary_producers import roots as binary_roots
from tests.device.nvidia.test_native_tensor_program_package_execution import values
from tests.unit.test_native_tensor_mlp_program import mlp_source, mlp_roots, assert_mlp_numeric
from tests.unit.test_native_tensor_pipeline_layout import annotated_source


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--storage", nargs="+", choices=("f16", "bf16"), default=["f16", "bf16"])
    parser.add_argument("--case", nargs="+", choices=("add", "relu", "rmsnorm", "softmax", "mlp_f16", "mlp_f32"),
                        default=["add", "relu", "rmsnorm", "softmax", "mlp_f16", "mlp_f32"])
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--calls", type=int, default=16)
    args = parser.parse_args(argv)
    if not 2 <= args.rounds <= 1000 or args.rounds % 2:
        raise ValueError("rounds must be even and between 2 and 1000")
    if not 1 <= args.calls <= 10000:
        raise ValueError("calls must be between 1 and 10000")
    if len(set(args.storage)) != len(args.storage) or len(set(args.case)) != len(args.case):
        raise ValueError("storage and case selections must be unique")
    return args


def record_case(kind, storage, rounds, calls):
    intermediate = "f32" if kind == "mlp_f32" else "f16"
    mlp = kind.startswith("mlp_")
    source = (mlp_source(storage, intermediate=intermediate) if mlp else
              annotated_source(kind, storage, "tessera.layout", '"row_major"'))
    args = (mlp_roots(storage, intermediate=intermediate, seed=15051) if mlp else
            binary_roots("lhs", storage, (17, 32, 11), 15051) if kind == "add" else
            values(storage, 15051))
    snapshots = tuple(value.copy() for value in args)

    def check(output):
        if mlp:
            assert_mlp_numeric(output, args, False, intermediate)
        else:
            check_semantic(kind, output, args)

    started = time.perf_counter_ns()
    frame, images = full_native_program(source)
    compile_ms = (time.perf_counter_ns() - started) / 1e6
    started = time.perf_counter_ns()
    owner = PreparedNativeTensorFrame(frame, images)
    prepare_ms = (time.perf_counter_ns() - started) / 1e6
    with owner:
        check(owner(args))
        before = owner.stats()
        windows = event_windows(owner, rounds)
        check(owner(args))

        def forbidden(*args, **kwargs):
            raise RuntimeError("timed retained owner invoked compiler subprocess")

        samples, elapsed_windows = [], []
        with patch("subprocess.run", forbidden), patch("subprocess.Popen", forbidden):
            for _ in range(rounds):
                started = time.perf_counter_ns()
                for _ in range(calls):
                    output = owner(args)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                elapsed_windows.append(elapsed)
                samples.append(elapsed / calls)
        check(output)
        for value, saved in zip(args, snapshots, strict=True):
            np.testing.assert_array_equal(value, saved)
        after = owner.stats()
        if before["allocations"] != 1 or after["allocations"] != 1 or before["capacity"] != after["capacity"]:
            raise RuntimeError("timing changed the retained arena")
    return {"case": kind, "storage": storage, "seed": 15051,
            "root_shapes": [list(value.shape) for value in args],
            "route": "registered Graph pipeline -> Schedule/Tile -> NVIDIA Target -> NVVM -> PTX",
            "strict_cpp_source_replay": "passed", "correctness": "independent_oracle_before_after_timing",
            "compile_and_replay_and_member_image_ms": compile_ms, "native_owner_prepare_ms": prepare_ms,
            "device_windows": windows, "completed_owner_samples_ms": samples,
            "completed_owner_median_ms": median(samples), "completed_owner_window_ms": elapsed_windows,
            "calls_per_sample": calls, "before": before, "after": after, "frame": frame,
            "source_ir_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "image_sha256": [hashlib.sha256(image).hexdigest() for image in images]}


def main(argv=None):
    args = options(argv)  # Invalid arguments must not touch the GPU.
    source = identity()
    for name in ("benchmarks/nvidia/record_native_tensor_full_pipeline.py",
                 "benchmarks/nvidia/record_native_tensor_binary.py",
                 "tests/device/nvidia/test_native_tensor_pipeline_layout.py",
                 "tests/unit/test_native_tensor_pipeline_layout.py",
                 "tests/unit/test_native_tensor_mlp_program.py"):
        source["source_sha256"][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    packet = {"schema": "tessera.benchmark.native_tensor_full_pipeline.v1",
              "source": source, "status": "recording", "rounds": args.rounds, "rows": [],
              "scope": "Static compact validation shapes. Compile includes strict replay and member image construction; prepare is separate. CUDA member/program event windows are independent, not additive. Under-10ms regions are diagnostic. Completed owner time includes bindings, transfers, output allocation and readback; excludes compilation/prepare and is not ordinary public JIT timing. No speedup, general shape, bounded DAG or route promotion claim."}

    def save():
        args.output.write_text(json.dumps(packet, indent=2) + "\n")

    save()
    try:
        for storage in args.storage:
            for kind in args.case:
                packet["rows"].append(record_case(kind, storage, args.rounds, args.calls))
                save()
                print(storage, kind, "passed and recorded", flush=True)
    except Exception as error:
        packet.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    packet["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
