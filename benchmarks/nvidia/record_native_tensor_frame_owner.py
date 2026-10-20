"""Record retained native frame execution; no route/performance promotion."""
from contextlib import ExitStack
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import median
import subprocess
import time

import numpy as np

from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.prepared_native_tensor_frame import PreparedNativeTensorFrame, _descriptor
from tests.device.nvidia.test_native_tensor_frame_owner import CASES, oracle, package, values


def run(packet, provider, output, *, rounds=9, repetitions=32):
    os.environ["TESSERA_NATIVE_FRAME_TEST_PACKET"] = str(packet)
    device = subprocess.check_output([
        "/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,compute_cap,uuid,driver_version", "--format=csv,noheader",
    ], text=True).strip()
    if len(device.splitlines()) != 1 or "RTX 5070, 12.0" not in device:
        raise RuntimeError("native frame benchmark requires the owning RTX5070")
    library = ct.CDLL(str(provider))
    records = {}
    rng = random.Random(20261020)
    with ExitStack() as stack:
        stack.enter_context(NvidiaDeviceSession())
        owners, inputs, raw_bindings = {}, {}, {}
        for name in CASES:
            frame, images = package(name)
            a, b = values(name, 137)
            expected = oracle(name, a, b)
            start = time.perf_counter_ns()
            owner = stack.enter_context(PreparedNativeTensorFrame(frame, images, provider=library))
            prepare_ms = (time.perf_counter_ns() - start) / 1e6
            first_start = time.perf_counter_ns()
            actual = owner((a, b))
            first_ms = (time.perf_counter_ns() - first_start) / 1e6
            np.testing.assert_allclose(actual, expected, rtol=.004, atol=2e-5)
            before = owner.stats()
            raw = json.loads((packet / (name + ".frame.json")).read_text())
            records[name] = {
                "source_sha256": raw["source_sha256"], "program_sha256": raw["program_sha256"],
                "member_ptx_sha256": [s["ptx_sha256"] for s in raw["stages"]],
                "shape_mnk": [17, 11, 32],
                "cold_owner_prepare_ms": prepare_ms,
                "first_completed_host_call_ms": first_ms,
                "completed_host_samples_ms": [], "raw_native_host_samples_ms": [],
                "captured_program_samples_ms": [],
                "captured_stage_samples_ms": [], "native_stats_before": before,
                "max_abs_error": float(np.max(np.abs(actual.astype(np.float64) - expected))),
            }
            owners[name], inputs[name] = owner, (a, b, expected)
            root_views, _ = owner._root_views((a, b), False)
            raw_output = np.empty(frame["buffers"][frame["output"]]["shape"], np.float32)
            raw_view = _descriptor(frame["buffers"][frame["output"]],
                                   raw_output.ctypes.data, raw_output.strides)
            raw_bindings[name] = root_views, raw_output, raw_view
        orders = []
        for _ in range(rounds):
            order = list(CASES)
            rng.shuffle(order)
            orders.append(order)
            for name in order:
                owner = owners[name]
                a, b, expected = inputs[name]
                modes = ["adapter", "raw"]
                rng.shuffle(modes)
                records[name].setdefault("host_mode_orders", []).append(modes)
                for mode in modes:
                    root_views, raw_output, raw_view = raw_bindings[name]
                    start = time.perf_counter_ns()
                    for _ in range(repetitions):
                        if mode == "adapter":
                            actual = owner((a, b))
                        else:
                            owner._check(library.tessera_nvidia_native_frame_invoke_v1(
                                owner.handle, root_views, len(root_views), ct.byref(raw_view), 1))
                            actual = raw_output
                    elapsed = (time.perf_counter_ns() - start) / 1e6 / repetitions
                    np.testing.assert_allclose(actual, expected, rtol=.004, atol=2e-5)
                    records[name]["completed_host_samples_ms" if mode == "adapter"
                                  else "raw_native_host_samples_ms"].append(elapsed)
                timing = owner.profile(repetitions, captured=True)
                records[name]["captured_program_samples_ms"].append(timing["program_ms"])
                records[name]["captured_stage_samples_ms"].append(timing["stage_ms"])
        for name in CASES:
            record = records[name]
            after = owners[name].stats()
            if after["allocations"] != record["native_stats_before"]["allocations"] or (
                    after["capacity"] != record["native_stats_before"]["capacity"]):
                raise RuntimeError("retained frame allocation changed during steady state")
            record["native_stats_after"] = after
            record["completed_host_median_ms"] = median(record["completed_host_samples_ms"])
            record["raw_native_host_median_ms"] = median(record["raw_native_host_samples_ms"])
            record["captured_program_median_ms"] = median(record["captured_program_samples_ms"])
            record["captured_stage_median_ms"] = [
                median(sample[i] for sample in record["captured_stage_samples_ms"])
                for i in range(len(record["member_ptx_sha256"]))]
    result = {
        "scope": "named static FP16/BF16 native frame owner; no public JIT/portable or performance promotion",
        "device": device, "provider_sha256": hashlib.sha256(provider.read_bytes()).hexdigest(),
        "rounds": rounds, "repetitions": repetitions, "case_orders": orders,
        "timing_contract": {
            "completed_host": "ABI marshaling, host result allocation, snapshot, H2D, all kernels, D2H and completion",
            "raw_native_host": "same checked C++ invocation with prebound host views/output; includes snapshot, transfers, kernels and completion",
            "captured_program": "CUDA graph device-event window, excludes capture/prime/retirement, uploads and readback",
            "captured_stages": "independent grouped stage windows; do not sum to program time",
            "cold_prepare": "Python ABI binding and C++ module JIT/stream/device+pinned arena on an existing context; excludes frontend/compiler time",
        },
        "cases": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    for name, record in records.items():
        print(name, "host_ms", round(record["completed_host_median_ms"], 6),
              "raw_host_ms", round(record["raw_native_host_median_ms"], 6),
              "device_ms", round(record["captured_program_median_ms"], 6), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.packet.resolve(), args.provider.resolve(), args.output.resolve())
