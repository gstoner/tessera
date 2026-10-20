"""Correctness-gated SM120 binary producer timings in separate measurement domains."""
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
from benchmarks.nvidia.record_native_tensor_jit import identity
from tests.unit.test_native_tensor_binary_producers import CASES
from tests.device.nvidia.test_native_tensor_binary_producers import roots, check


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shape", nargs=3, type=int, action="append", metavar=("M", "K", "N"))
    parser.add_argument("--storage", nargs="+", choices=("f16", "bf16", "f32"), default=["f16", "bf16"])
    parser.add_argument("--case", nargs="+", choices=tuple(CASES), default=["lhs", "rhs", "both"])
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--calls", type=int, default=16)
    args = parser.parse_args(argv)
    if not 2 <= args.rounds <= 1000 or args.rounds % 2:
        raise ValueError("rounds must be positive even, between 2 and 1000")
    if not 1 <= args.calls <= 10000:
        raise ValueError("calls must be between 1 and 10000")
    args.shape = args.shape or [(256, 4096, 256)]
    for m, k, n in args.shape:
        if k >= 2**24:
            raise ValueError("the FP32 gamma-K oracle requires K < 2**24")
        if (not all(0 < extent < 2**31 for extent in (m, k, n))
                or max(m*k, k*n, m*n) >= 2**31):
            raise ValueError("shape exceeds the static compact native frame envelope")
    if len(set(args.storage)) != len(args.storage) or len(set(args.case)) != len(args.case):
        raise ValueError("storage and case selections must be unique")
    return args


def event_windows(owner, rounds):
    windows = []
    for window in range(rounds):
        for captured in ((False, True) if window % 2 == 0 else (True, False)):
            repeats = 256
            maximum = 4096 if captured else 10000
            while True:
                raw = owner.profile(repeats, captured=captured)
                regions = raw["stage_ms"] + [raw["program_ms"]]
                if any(not np.isfinite(value) or value <= 0 for value in regions):
                    raise RuntimeError("native event timing must be finite and positive")
                duration = min(regions) * repeats
                if duration >= 10 or repeats == maximum:
                    break
                repeats = min(maximum, repeats * 2)
            windows.append({"window": window, "captured": captured, "repeats": repeats,
                            "minimum_region_window_ms": duration,
                            "duration_admitted": duration >= 10, **raw})
    return windows


def record_case(case, storage, shape, rounds, calls):
    values = roots(case, storage, shape, 13051)
    snapshots = tuple(value.copy() for value in values)
    fn = ts.jit(target="nvidia_sm120")(CASES[case][0])
    try:
        actual = fn(*values)
        check(actual, case, values)
        if fn.execution_kind != "native_gpu":
            raise RuntimeError("binary recording requires completed native GPU execution")
        owner, program = fn._native_tensor_last_owner, fn._native_tensor_last_program
        before = owner.stats()
        windows = event_windows(owner, rounds)
        # Profiling submits the complete program after measuring isolated members.
        # Rebind through the ordinary route and independently check that result.
        check(fn(*values), case, values)
        samples, elapsed_windows = [], []
        def forbidden(*args, **kwargs):
            raise RuntimeError("measured public call used eager execution, tracing or tools")
        with ExitStack() as guard:
            guard.enter_context(patch.object(fn, "_fn", forbidden))
            guard.enter_context(patch.object(fn, "_trace_frontend_capture", forbidden))
            guard.enter_context(patch("subprocess.run", forbidden))
            guard.enter_context(patch("subprocess.Popen", forbidden))
            for _ in range(rounds):
                started = time.perf_counter_ns()
                for _ in range(calls):
                    actual = fn(*values)
                elapsed = (time.perf_counter_ns()-started)/1e6
                elapsed_windows.append(elapsed)
                samples.append(elapsed/calls)
        check(actual, case, values)
        for value, snapshot in zip(values, snapshots, strict=True):
            np.testing.assert_array_equal(value, snapshot)
        after = owner.stats()
        if before["allocations"] != 1 or after["allocations"] != 1 or before["capacity"] != after["capacity"]:
            raise RuntimeError("binary recorder changed retained allocation")
        return {"case": case, "storage": storage, "shape_mkn": list(shape), "seed": 13051,
                "correctness": "independent_fp64_gamma_k_before_after",
                "device_windows": windows, "completed_public_samples_ms": samples,
                "completed_public_median_ms": median(samples),
                "completed_public_window_ms": elapsed_windows, "calls_per_sample": calls,
                "frame": program.frame, "before": before, "after": after,
                "ir_sha256": {name: hashlib.sha256(getattr(program, name).encode()).hexdigest()
                              for name in ("source_ir", "schedule_ir", "tile_ir", "target_ir")},
                "image_sha256": [hashlib.sha256(image).hexdigest() for image in program.images]}
    finally:
        fn.close_native_storage()


def main(argv=None):
    args = options(argv)  # Argument errors must not probe hardware.
    source = identity()
    source["recorder_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name in ("tests/unit/test_native_tensor_binary_producers.py",
                 "tests/device/nvidia/test_native_tensor_binary_producers.py"):
        source["source_sha256"][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    packet = {"schema": "tessera.benchmark.native_tensor_binary.v1", "source": source,
              "status": "recording", "rows": [], "rounds": args.rounds,
              "scope": "Independent producer, consumer and program CUDA event regions; do not add isolated regions. Completed public wall time includes binding, transfers and readback. Under-duration event regions are diagnostic only. No baseline speedup or general promotion."}
    def save():
        args.output.write_text(json.dumps(packet, indent=2)+"\n")
    save()
    try:
        for shape in args.shape:
            for storage in args.storage:
                for case in args.case:
                    packet["rows"].append(record_case(case, storage, shape, args.rounds, args.calls))
                    save()
                    print(storage, case, tuple(shape), "passed and recorded", flush=True)
    except Exception as error:
        packet.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    packet["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
