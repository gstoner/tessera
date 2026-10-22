"""Correctness-gated independent-axis native kernel/program/public timings."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import time
from unittest.mock import patch

import numpy as np
import tessera as ts
from tessera import runtime
from benchmarks.nvidia.record_native_tensor_binary import event_windows
from benchmarks.nvidia.record_native_tensor_jit import identity
from tests.unit import test_native_tensor_shape_frontend as frontend
from tests.device.nvidia.test_native_tensor_independent_shapes import roots, check

CASES = ("norm_lhs", "rhs_add", "chain", "two_matmuls")


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--storage", nargs="+", choices=("f16", "bf16"), default=["f16", "bf16"])
    parser.add_argument("--shapes", nargs="+", default=["17x11x23", "65x47x64", "5x31x7"])
    parser.add_argument("--bounds", nargs=3, type=int, default=[65, 47, 64], metavar=("M", "N", "K"))
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--calls", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        shapes = [tuple(int(axis) for axis in shape.split("x")) for shape in args.shapes]
    except ValueError as error:
        raise ValueError("shapes must be MxNxK integer triples") from error
    if (any(not 0 < bound < 2**31 for bound in args.bounds)
            or any(len(shape) != 3 or any(not 0 < axis <= bound for axis, bound in zip(shape, args.bounds)) for shape in shapes)):
        raise ValueError("active M/N/K must fit positive named capacities")
    if not 2 <= args.rounds <= 1000 or args.rounds % 2 or not 1 <= args.calls <= 10000:
        raise ValueError("invalid round or completed-call count")
    if any(len(set(values)) != len(values) for values in (args.case, args.storage, shapes)):
        raise ValueError("case, storage and active shape selections must be unique")
    args.shapes, args.bounds = shapes, dict(zip("MNK", args.bounds, strict=True))
    if args.output.exists():
        raise FileExistsError("use a fresh output path to preserve previous evidence")
    return args


def record_program(case, storage, args, packet, artifacts, save):
    fn = ts.jit(target="nvidia_sm120", shape_bounds=args.bounds)(getattr(frontend, case))
    try:
        initial = roots(case, storage, 1, 1, 1, seed=23401)
        start = time.perf_counter_ns()
        first = fn(*initial)
        cold_ms = (time.perf_counter_ns() - start) / 1e6
        check(case, first, initial)
        program, owner = fn._native_tensor_last_program, fn._native_tensor_last_owner
        if fn.execution_kind != "native_gpu" or program.frame["schema"] != "tessera.native.tensor_program_frame.v3":
            raise RuntimeError("recording requires completed compiler-owned independent-shape execution")
        images = tuple(hashlib.sha256(image).hexdigest() for image in program.images)
        prefix = f"{case}-{storage}"
        for suffix, data in (("graph.mlir", program.source_ir.encode()), ("schedule.mlir", program.schedule_ir.encode()),
                             ("tile.mlir", program.tile_ir.encode()), ("target.mlir", program.target_ir.encode()),
                             ("shape.mlir", program.frame["shape_program"]["ir"].encode()),
                             ("shape.so", program.shape_companion.image)):
            (artifacts / f"{prefix}.{suffix}").write_bytes(data)
        for index, (member, image) in enumerate(zip(program.members, program.images, strict=True)):
            (artifacts / f"{prefix}.member{index}.mlir").write_text(member)
            (artifacts / f"{prefix}.member{index}.ptx").write_bytes(image)
        kept = first.copy()
        before = owner.stats()
        for ordinal, (m, n, k) in enumerate(args.shapes):
            values = roots(case, storage, m, n, k, seed=23402 + ordinal)
            snapshots = tuple(value.copy() for value in values)
            check(case, fn(*values), values)
            windows = event_windows(owner, args.rounds)
            check(case, fn(*values), values)
            def forbidden(*unused, **kwargs):
                raise RuntimeError("measured public call traced, compiled or ran eager")
            samples, elapsed_windows = [], []
            with patch.object(fn, "_fn", forbidden), patch.object(fn, "_trace_frontend_capture", forbidden), \
                    patch("subprocess.run", forbidden), patch("subprocess.Popen", forbidden):
                for _ in range(args.rounds):
                    start = time.perf_counter_ns()
                    for _ in range(args.calls):
                        actual = fn(*values)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    samples.append(elapsed / args.calls)
                    elapsed_windows.append(elapsed)
            error = check(case, actual, values)
            np.testing.assert_array_equal(first, kept)
            for value, snapshot in zip(values, snapshots, strict=True):
                np.testing.assert_array_equal(value, snapshot)
            after = owner.stats()
            if (fn._native_tensor_last_program is not program or fn._native_tensor_last_owner is not owner
                    or before["allocations"] != 1 or after["allocations"] != 1
                    or before["capacity"] != after["capacity"]):
                raise RuntimeError("independent shapes lost their retained program/arena")
            packet["rows"].append({"case": case, "storage": storage, "shape_mnk": [m, n, k],
                "root_shapes": [list(value.shape) for value in values],
                "cold_public_trace_compile_prepare_execute_ms": cold_ms,
                "cold_sample_group": prefix, "cold_sample_repeated_across_shapes": True,
                "correctness": "independent_oracle_before_after_timing", "max_abs_error": error,
                "device_windows": windows, "completed_public_samples_ms": samples,
                "completed_public_median_ms": median(samples), "completed_public_window_ms": elapsed_windows,
                "calls_per_sample": args.calls, "before": before, "after": after,
                "frame": program.frame, "image_sha256": list(images),
                "shape_library_sha256": program.shape_companion.image_sha256,
                "shape_compiler_tools": list(program.shape_companion.tools), "artifact_prefix": prefix})
            save()
            print(case, storage, [m, n, k], "passed and recorded", flush=True)
    finally:
        fn.close_native_storage()


def main(argv=None):
    args = options(argv)
    source = identity()
    actual = Path(runtime._load_nvidia_ptx_launch()._name)
    if hashlib.sha256(actual.read_bytes()).hexdigest() != source["provider_sha256"]:
        raise RuntimeError("recorded provider differs from the loaded native owner")
    for name in ("benchmarks/nvidia/record_native_independent_shapes.py",
                 "python/tessera/compiler/native_tensor_shape.py", "python/tessera/compiler/bounded_nvidia_lhs.py",
                 "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/NativeTensorFrameRuntime.h",
                 "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_native_frame.h",
                 "tests/unit/test_native_tensor_shape_frontend.py",
                 "tests/device/nvidia/test_native_tensor_independent_shapes.py"):
        source["source_sha256"][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    artifacts = args.output.parent / (args.output.stem + "-artifacts")
    artifacts.mkdir(parents=True, exist_ok=False)
    packet = {"schema": "tessera.benchmark.native_independent_shapes.v1", "source": source,
              "status": "recording", "bounds": args.bounds, "rounds": args.rounds, "rows": [],
              "oracle_contract": "FP64 semantics with FP32 arithmetic/storage-rounding intervals and propagated final-product error; finite recorded inputs only",
              "scope": "Opt-in compact rank-two independent M/N/K frames on the recorded SM120 device. Cold samples repeat by case/storage group and are not independent observations. Member/program CUDA timing regions are non-additive; under-10ms windows are diagnostic. Public samples include shape query, binding, transfers and completed readback; exclude cold compilation/preparation. Same images/arena across active shapes. No baseline speedup, physical-schedule promotion or general program closure claim."}
    def save():
        pending = args.output.with_suffix(args.output.suffix + ".partial")
        pending.write_text(json.dumps(packet, indent=2) + "\n")
        pending.replace(args.output)
    save()
    try:
        for case in args.case:
            for storage in args.storage:
                record_program(case, storage, args, packet, artifacts, save)
        for name, digest in source["source_sha256"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise RuntimeError("recorded source changed during measurement")
    except Exception as error:
        packet.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    packet["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
