"""Correctness-gated native MLP timings with explicit intermediate precision."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import time

import numpy as np
import tessera as ts
from tessera import runtime as rt
from benchmarks.nvidia.record_native_tensor_jit import identity
from tests.unit.test_native_tensor_mlp_program import default_mlp, mlp, mlp_roots, assert_mlp_numeric
from tests.device.nvidia.test_native_tensor_mlp_program import reference


def record_case(shape, storage, intermediate, rounds, repeats, profile_repeats):
    function = mlp if intermediate == "f16" else default_mlp
    fn = ts.jit(target="nvidia_sm120")(function)
    args = mlp_roots(storage, shape=shape, intermediate=intermediate)
    snapshots = tuple(value.copy() for value in args)
    expected = reference(args, False, intermediate)
    try:
        started = time.perf_counter_ns()
        actual = fn(*args)
        cold = (time.perf_counter_ns() - started) / 1e6
        assert_mlp_numeric(actual, args, False, intermediate)
        if fn.execution_kind != "native_gpu":
            raise RuntimeError("MLP recording requires completed native GPU execution")
        program = fn._native_tensor_last_program
        owner = fn._native_tensor_last_owner
        if program is None or owner is None or len(program.members) != 3:
            raise RuntimeError("MLP recording requires three actual native members")
        before = owner.stats()
        windows = {}
        for captured in (False, True):
            raw = owner.profile(profile_repeats, captured=captured)
            if "output" in raw:
                assert_mlp_numeric(raw["output"], args, False, intermediate)
            windows["captured" if captured else "enqueued"] = {
                key: value for key, value in raw.items() if key != "output"}
        samples = []
        for _ in range(rounds):
            started = time.perf_counter_ns()
            for _ in range(repeats):
                actual = fn(*args)
            samples.append((time.perf_counter_ns() - started) / repeats / 1e6)
        assert_mlp_numeric(actual, args, False, intermediate)
        for original, saved in zip(args, snapshots, strict=True):
            np.testing.assert_array_equal(original, saved)
        after = owner.stats()
        if before["allocations"] != 1 or after["allocations"] != 1:
            raise RuntimeError("retained MLP owner changed allocation count")
        return {
            "shape_mkhn": list(shape), "root_profile": storage,
            "intermediate": intermediate,
            "correctness": "passed_before_after_and_profile",
            "cold_call_ms": cold,
            "completed_public_samples_ms": samples,
            "completed_public_median_ms": median(samples),
            "rounds": rounds, "calls_per_sample": repeats,
            "profile_repeats": profile_repeats, "device_windows": windows,
            "frame": program.frame, "before": before, "after": after,
            "ir_sha256": {name: hashlib.sha256(getattr(program, name).encode()).hexdigest()
                          for name in ("source_ir", "schedule_ir", "tile_ir", "target_ir")},
            "image_sha256": [hashlib.sha256(image).hexdigest() for image in program.images],
            "max_abs_error": float(np.max(np.abs(actual.astype(np.float64) - expected))),
        }
    finally:
        fn.close_native_storage()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--profile-repeats", type=int, default=64)
    parser.add_argument("--shape", nargs=4, type=int, action="append", metavar=("M", "K", "H", "N"))
    args = parser.parse_args()
    if (not 1 <= args.rounds <= 1000 or not 1 <= args.repeats <= 10000
            or not 1 <= args.profile_repeats <= 10000):
        raise ValueError("MLP benchmark counts exceed the bounded recording envelope")
    shapes = args.shape or [(17, 32, 48, 11), (17, 35, 29, 19), (64, 256, 128, 64)]
    if any(any(extent <= 0 for extent in shape) for shape in shapes):
        raise ValueError("MLP benchmark shapes must be positive")
    if rt._nvidia_device_name() != "sm_120":
        raise RuntimeError("MLP benchmark requires owning SM120 execution")
    source = identity()
    source["recorder_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name in ("tests/unit/test_native_tensor_mlp_program.py",
                 "tests/device/nvidia/test_native_tensor_mlp_program.py"):
        source["source_sha256"][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    packet = {
        "schema": "tessera.benchmark.native_tensor_mlp.v1",
        "source": source, "status": "recording", "rows": [],
        "scope": "Explicit FP16/FP32 precision profiles; no cross-precision speedup claim. "
                 "Captured/enqueued stage and program windows are independent event regions; "
                 "completed public calls and cold compilation are separate host timings.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for shape in shapes:
        for storage in ("f16", "bf16"):
            for intermediate in ("f16", "f32"):
                row = record_case(tuple(shape), storage, intermediate, args.rounds,
                                  args.repeats, args.profile_repeats)
                packet["rows"].append(row)
                args.output.write_text(json.dumps(packet, indent=2) + "\n")
                print(storage, intermediate, shape, row["completed_public_median_ms"], flush=True)
    packet["status"] = "complete"
    args.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
