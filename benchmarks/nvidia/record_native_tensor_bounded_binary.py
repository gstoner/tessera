"""Correctness-gated variable-M timing of one retained compiler-owned program."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time
from unittest.mock import patch

import tessera as ts
import numpy as np

from benchmarks.nvidia.record_native_tensor_binary import event_windows
from benchmarks.nvidia.record_native_tensor_jit import identity
from tests.device.nvidia.test_native_tensor_binary_producers import check, roots
from tests.unit.test_native_tensor_binary_producers import lhs_add


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bound", type=int, default=256)
    parser.add_argument("--rows", type=int, nargs="+")
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--storage", nargs="+", choices=("f16", "bf16"), default=["f16", "bf16"])
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--calls", type=int, default=16)
    args = parser.parse_args(argv)
    if not 0 < args.k < 2**24:
        raise ValueError("the FP32 gamma-K oracle requires 0 < K < 2**24")
    if (not 0 < args.bound <= 16 * 65535 or not 0 < args.n < 2**31
            or max(args.bound * args.k, args.k * args.n, args.bound * args.n) >= 2**31):
        raise ValueError("capacity exceeds the compact bounded benchmark envelope")
    if not 2 <= args.rounds <= 1000 or args.rounds % 2:
        raise ValueError("rounds must be even, between 2 and 1000")
    if not 1 <= args.calls <= 10000:
        raise ValueError("calls must be between 1 and 10000")
    if args.rows is None:
        args.rows = list(dict.fromkeys([min(17, args.bound), 1, args.bound,
                                       min(65, args.bound), min(5, args.bound), min(200, args.bound)]))
    if len(set(args.rows)) != len(args.rows) or any(not 0 < m <= args.bound for m in args.rows):
        raise ValueError("active rows must be unique and within capacity")
    if len(set(args.storage)) != len(args.storage):
        raise ValueError("storage selections must be unique")
    return args


def forbidden(*args, **kwargs):
    raise RuntimeError("measured bounded call used eager execution, tracing or tools")


def record_profile(storage, args, profile, save):
    fn = ts.jit(target="nvidia_sm120", shape_bounds={"M": args.bound})(lhs_add)
    try:
        trace_rows = min(17, args.bound)
        initial = roots("lhs", storage, (trace_rows, args.k, args.n), 14200)
        started = time.perf_counter_ns()
        explicit = fn.compile_native_tensor_program(*initial)
        profile["explicit_frontend_and_compile_ms"] = (time.perf_counter_ns() - started) / 1e6
        started = time.perf_counter_ns()
        probe = explicit.prepare()
        profile["explicit_prepare_ms"] = (time.perf_counter_ns() - started) / 1e6
        probe.close()
        started = time.perf_counter_ns()
        first = fn(*initial)
        profile["first_public_after_compile_prepare_probe_ms"] = (time.perf_counter_ns() - started) / 1e6
        check(first, "lhs", initial)
        program, owner = fn._native_tensor_last_program, fn._native_tensor_last_owner
        if (fn.execution_kind != "native_gpu" or program is None or owner is None
                or program.images != explicit.images or program.frame != explicit.frame):
            raise RuntimeError("public owner differs from the explicit compiler product")
        before = owner.stats()
        profile.update(frame=program.frame, allocation_before=before,
                       ir_sha256={name: hashlib.sha256(getattr(program, name).encode()).hexdigest()
                                  for name in ("source_ir", "schedule_ir", "tile_ir", "target_ir")},
                       image_sha256=[hashlib.sha256(image).hexdigest() for image in program.images])
        save()
        for m in args.rows:
            values = roots("lhs", storage, (m, args.k, args.n), 14201 + m)
            snapshots = tuple(value.copy() for value in values)
            check(fn(*values), "lhs", values)
            windows = event_windows(owner, args.rounds)
            check(fn(*values), "lhs", values)
            samples, elapsed_windows = [], []
            with ExitStack() as guard:
                guard.enter_context(patch.object(fn, "_fn", forbidden))
                guard.enter_context(patch.object(fn, "_trace_frontend_capture", forbidden))
                guard.enter_context(patch("subprocess.run", forbidden))
                guard.enter_context(patch("subprocess.Popen", forbidden))
                for _ in range(args.rounds):
                    started = time.perf_counter_ns()
                    for _ in range(args.calls):
                        actual = fn(*values)
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    samples.append(elapsed / args.calls)
                    elapsed_windows.append(elapsed)
            check(actual, "lhs", values)
            for value, snapshot in zip(values, snapshots, strict=True):
                np.testing.assert_array_equal(value, snapshot)
            after = owner.stats()
            if (fn._native_tensor_last_owner is not owner or fn._native_tensor_last_program is not program
                    or before["allocations"] != 1 or after["allocations"] != 1
                    or after["capacity"] != before["capacity"]):
                raise RuntimeError("active rows changed the retained program or capacity allocation")
            profile["rows"].append({"shape_mkn": [m, args.k, args.n], "seed": 14201 + m,
                "correctness": "independent_fp64_stored_add_then_gamma_k_before_after",
                "device_windows": windows, "completed_public_samples_ms": samples,
                "completed_public_median_ms": median(samples),
                "completed_public_window_ms": elapsed_windows, "calls_per_sample": args.calls,
                "allocation_after": after})
            save()
            print(storage, (m, args.k, args.n), "checked on retained bounded owner", flush=True)
    finally:
        fn.close_native_storage()


def main(argv=None):
    args = options(argv)  # Reject invalid measurement contracts before hardware.
    source = identity()
    source["recorder_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name in ("python/tessera/compiler/bounded_nvidia_lhs.py",
                 "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/NativeTensorFrameRuntime.h",
                 "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_native_frame.h",
                 "tests/device/nvidia/test_native_tensor_bounded_producers.py"):
        source["source_sha256"][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    build = Path(os.environ["TESSERA_NATIVE_FRAME_TEST_LIB"]).parent / "production-identity.json"
    source["provider_build"] = json.loads(build.read_text())
    if source["provider_build"]["provider_sha256"] != source["provider_sha256"]:
        raise RuntimeError("provider build receipt differs from live native library")
    packet = {"schema": "tessera.benchmark.native_tensor_bounded_binary.v1", "status": "recording",
              "source": source, "bound_mkn": [args.bound, args.k, args.n], "active_rows": args.rows,
              "rounds": args.rounds, "profiles": [],
              "scope": "One retained compiler-owned program per storage across active M. Separate producer, consumer and whole-program CUDA event windows; isolated regions are not additive. Under-10ms event windows are diagnostic only. Completed public wall time includes marshaling, transfer, allocation and readback. Explicit compile/prepare probes are separate; first-public timing follows those probes and is not process-cold. No baseline speedup or universal promotion."}

    def save():
        args.output.write_text(json.dumps(packet, indent=2) + "\n")

    save()
    try:
        for storage in args.storage:
            profile = {"storage": storage, "rows": []}
            packet["profiles"].append(profile)
            save()
            record_profile(storage, args, profile, save)
    except Exception as error:
        packet.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    packet["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
