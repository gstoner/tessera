#!/usr/bin/env python3
"""Measure host-array versus resident HIP lifecycle for one identical HSACO.

Wall timings include their respective host I/O; event timing measures only
the resident kernel. This is an opt-in manual route, not selector evidence.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import socket
import statistics
import subprocess
import time

import ml_dtypes
import numpy as np

from benchmarks.rocm.benchmark_gfx1201_mxfp4_production import (
    Case,
    _logical_inputs,
    _sampled_exact_reference,
)
from tessera import runtime as rt
from tessera.compiler.rocm_mxfp4_packed_folded import (
    package_mxfp4_packed_folded_prefill,
    prepare_packed_folded_payload,
)
from tessera.compiler.rocm_mxfp4_resident import PackedFoldedResidentSession


ROOT = Path(__file__).resolve().parents[2]


def _device_identity() -> tuple[str, int]:
    hip = rt._load_hip_for_launch()
    if hip is None:
        raise RuntimeError("HIP runtime is unavailable")
    hip.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    hip.hipDeviceGetName.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    device = ctypes.c_int()
    name = ctypes.create_string_buffer(256)
    if hip.hipGetDevice(ctypes.byref(device)) != 0:
        raise RuntimeError("hipGetDevice failed")
    if hip.hipDeviceGetName(name, len(name), device.value) != 0:
        raise RuntimeError("hipDeviceGetName failed")
    return name.value.decode(), device.value


def _median_us(fn: object, count: int = 7) -> float:
    samples = []
    for _ in range(count):
        start = time.perf_counter_ns()
        fn()  # type: ignore[operator]
        samples.append((time.perf_counter_ns() - start) / 1000)
    return statistics.median(samples)


def _kernel_event_us(session: PackedFoldedResidentSession, repetitions: int = 10) -> float:
    hip = session._hip
    hip.hipEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    hip.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    hip.hipEventSynchronize.argtypes = [ctypes.c_void_p]
    hip.hipEventElapsedTime.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p,
    ]
    hip.hipEventDestroy.argtypes = [ctypes.c_void_p]
    start, end = ctypes.c_void_p(), ctypes.c_void_p()
    if hip.hipEventCreate(ctypes.byref(start)) != 0:
        raise RuntimeError("hipEventCreate(start) failed")
    try:
        if hip.hipEventCreate(ctypes.byref(end)) != 0:
            raise RuntimeError("hipEventCreate(end) failed")
        try:
            values = []
            for _ in range(7):
                if hip.hipEventRecord(start, session._stream) != 0:
                    raise RuntimeError("hipEventRecord(start) failed")
                for _ in range(repetitions):
                    session.launch_resident()
                if hip.hipEventRecord(end, session._stream) != 0:
                    raise RuntimeError("hipEventRecord(end) failed")
                if hip.hipEventSynchronize(end) != 0:
                    raise RuntimeError("hipEventSynchronize failed")
                elapsed = ctypes.c_float()
                if hip.hipEventElapsedTime(ctypes.byref(elapsed), start, end) != 0:
                    raise RuntimeError("hipEventElapsedTime failed")
                values.append(elapsed.value * 1000 / repetitions)
            session.synchronize()
            return statistics.median(values)
        finally:
            hip.hipEventDestroy(end)
    finally:
        hip.hipEventDestroy(start)


def measure(m: int, n: int, k: int) -> dict[str, object]:
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("benchmark requires the selected gfx1201 device")
    case = Case("prefill", m, n, k)
    inputs = _logical_inputs(case)
    payload = prepare_packed_folded_payload(
        inputs["packed_row_major"], inputs["b_scale"], allow_approximate=True,
    )
    if not payload.lossless:
        raise RuntimeError("matched benchmark inputs must fold losslessly")
    package = package_mxfp4_packed_folded_prefill(
        m, payload, permute_decode=True, batched_loads=True,
    )
    a, a_scale = inputs["a"], inputs["a_scale"]
    rows, cols, expected = _sampled_exact_reference(case, inputs)

    def host_array() -> np.ndarray:
        output = np.empty((m, n), dtype=ml_dtypes.bfloat16)
        rt._submit_rocm_mxfp4_w4a8(
            package.image, package.descriptor,
            {"a": a, "b_packed": payload.weight_bytes, "a_scale": a_scale,
             "scale_plane": payload.scale_plane, "output": output},
            {"M": m, "N": n, "K": k},
        )
        return output

    legacy = host_array()
    np.testing.assert_array_equal(legacy[np.ix_(rows, cols)], expected)
    with PackedFoldedResidentSession(package, payload, m) as session:
        resident = session.run_host(a, a_scale)
        np.testing.assert_array_equal(resident, legacy)
        legacy_wall = _median_us(host_array)
        resident_wall = _median_us(lambda: session.run_host(a, a_scale))
        session.upload_activations(a, a_scale)
        session.synchronize()
        kernel_event = _kernel_event_us(session)
        receipt = session.receipt()
    device_name, device_ordinal = _device_identity()
    return {
        "sync_key": "GFX1201-MXFP4-RESIDENT-HIP-2026-09-23",
        "revision": subprocess.check_output(
            ("git", "-C", str(ROOT), "rev-parse", "HEAD"), text=True,
        ).strip(),
        "host": socket.gethostname(),
        "device_name": device_name,
        "device_ordinal": device_ordinal,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "python/tessera/compiler/rocm_mxfp4_resident.py",
                "benchmarks/rocm/benchmark_gfx1201_mxfp4_resident.py",
            )
        },
        "shape": [m, n, k],
        "target": "rocm_gfx1201",
        "hsaco_sha256": hashlib.sha256(package.image.payload).hexdigest(),
        "weight_sha256": payload.receipt()["weight_sha256"],
        "legacy_host_wall_us_median": legacy_wall,
        "resident_host_wall_us_median": resident_wall,
        "resident_kernel_event_us_median": kernel_event,
        "resident_receipt": receipt,
        "sampled_bf16_exact": True,
        "full_output_equals_legacy": True,
        "timing_note": (
            "Host-wall includes pageable NumPy transfers; HIP async copies may block. "
            "Event timing excludes H2D/D2H and is not directly comparable to host-wall."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=3, type=int, default=[256, 5120, 8704])
    args = parser.parse_args()
    print(json.dumps(measure(*args.shape), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
