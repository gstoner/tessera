#!/usr/bin/env python3
"""Same-HSACO HIP-event and host-enqueue comparison: direct versus graph."""
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
from typing import Callable

import numpy as np

from benchmarks.rocm.benchmark_gfx1201_mxfp4_production import (
    Case, _logical_inputs, _sampled_exact_reference,
)
from benchmarks.rocm.benchmark_gfx1201_mxfp4_resident import _device_identity
from tessera import runtime as rt
from tessera.compiler.rocm_mxfp4_graph import PackedFoldedGraphSession
from tessera.compiler.rocm_mxfp4_packed_folded import (
    package_mxfp4_packed_folded_prefill, prepare_packed_folded_payload,
)
from tessera.compiler.rocm_mxfp4_resident import PackedFoldedResidentSession


ROOT = Path(__file__).resolve().parents[2]


def _time_launch(
    hip: object, stream: ctypes.c_void_p, launch: Callable[[], None],
    *, repetitions: int = 20, samples: int = 7,
) -> dict[str, float]:
    hip.hipEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]  # type: ignore[attr-defined]
    hip.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]  # type: ignore[attr-defined]
    hip.hipEventSynchronize.argtypes = [ctypes.c_void_p]  # type: ignore[attr-defined]
    hip.hipEventElapsedTime.argtypes = [  # type: ignore[attr-defined]
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p,
    ]
    hip.hipEventDestroy.argtypes = [ctypes.c_void_p]  # type: ignore[attr-defined]
    host_samples: list[float] = []
    event_samples: list[float] = []
    start, end = ctypes.c_void_p(), ctypes.c_void_p()
    if hip.hipEventCreate(ctypes.byref(start)) != 0:  # type: ignore[attr-defined]
        raise RuntimeError("hipEventCreate(start) failed")
    try:
        if hip.hipEventCreate(ctypes.byref(end)) != 0:  # type: ignore[attr-defined]
            raise RuntimeError("hipEventCreate(end) failed")
        try:
            for _ in range(samples):
                if hip.hipEventRecord(start, stream) != 0:  # type: ignore[attr-defined]
                    raise RuntimeError("hipEventRecord(start) failed")
                host_start = time.perf_counter_ns()
                for _ in range(repetitions):
                    launch()
                host_samples.append(
                    (time.perf_counter_ns() - host_start) / 1000 / repetitions
                )
                if hip.hipEventRecord(end, stream) != 0:  # type: ignore[attr-defined]
                    raise RuntimeError("hipEventRecord(end) failed")
                if hip.hipEventSynchronize(end) != 0:  # type: ignore[attr-defined]
                    raise RuntimeError("hipEventSynchronize failed")
                elapsed = ctypes.c_float()
                if hip.hipEventElapsedTime(  # type: ignore[attr-defined]
                    ctypes.byref(elapsed), start, end,
                ) != 0:
                    raise RuntimeError("hipEventElapsedTime failed")
                event_samples.append(elapsed.value * 1000 / repetitions)
        finally:
            hip.hipEventDestroy(end)  # type: ignore[attr-defined]
    finally:
        hip.hipEventDestroy(start)  # type: ignore[attr-defined]
    return {
        "host_enqueue_us_median": statistics.median(host_samples),
        "kernel_event_us_median": statistics.median(event_samples),
    }


def measure(m: int, n: int, k: int) -> dict[str, object]:
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("graph benchmark requires the selected gfx1201 device")
    case = Case("prefill", m, n, k)
    inputs = _logical_inputs(case)
    payload = prepare_packed_folded_payload(
        inputs["packed_row_major"], inputs["b_scale"], allow_approximate=True,
    )
    if not payload.lossless:
        raise RuntimeError("matched graph inputs must fold losslessly")
    package = package_mxfp4_packed_folded_prefill(
        m, payload, permute_decode=True, batched_loads=True,
    )
    a, a_scale = inputs["a"], inputs["a_scale"]
    rows, cols, expected = _sampled_exact_reference(case, inputs)
    with PackedFoldedResidentSession(package, payload, m) as direct:
        direct.upload_activations(a, a_scale)
        direct.synchronize()
        direct.launch_resident()
        direct_output = direct.read_output()
        np.testing.assert_array_equal(direct_output[np.ix_(rows, cols)], expected)
        direct_timing = _time_launch(direct._hip, direct._stream, direct.launch_resident)
        direct.synchronize()
        with PackedFoldedGraphSession(package, payload, m) as graph:
            graph.upload_inputs(a, a_scale)
            graph.capture()
            graph.replay()
            graph_output = graph.read_output()
            np.testing.assert_array_equal(graph_output, direct_output)
            graph_timing = _time_launch(
                graph._hip, ctypes.c_void_p(graph.stream_pointer), graph.replay,
            )
            graph.synchronize()
            direct_second = _time_launch(
                direct._hip, direct._stream, direct.launch_resident,
            )
            direct.synchronize()
            graph_second = _time_launch(
                graph._hip, ctypes.c_void_p(graph.stream_pointer), graph.replay,
            )
            graph.synchronize()
            receipt = graph.receipt()
    direct_medians = {
        metric: statistics.median((direct_timing[metric], direct_second[metric]))
        for metric in direct_timing
    }
    graph_medians = {
        metric: statistics.median((graph_timing[metric], graph_second[metric]))
        for metric in graph_timing
    }
    device_name, device_ordinal = _device_identity()
    return {
        "sync_key": "GFX1201-MXFP4-DEVICE-GRAPH-2026-09-23",
        "revision": subprocess.check_output(
            ("git", "-C", str(ROOT), "rev-parse", "HEAD"), text=True,
        ).strip(),
        "host": socket.gethostname(),
        "target": "rocm_gfx1201",
        "device_name": device_name,
        "device_ordinal": device_ordinal,
        "shape": [m, n, k],
        "hsaco_sha256": hashlib.sha256(package.image.payload).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "python/tessera/compiler/rocm_mxfp4_graph.py",
                "benchmarks/rocm/benchmark_gfx1201_mxfp4_graph.py",
            )
        },
        "direct": direct_medians,
        "graph": graph_medians,
        "rounds": {
            "direct_first": direct_timing, "graph_first": graph_timing,
            "direct_second": direct_second, "graph_second": graph_second,
        },
        "graph_receipt": receipt,
        "capture_nodes": list(receipt["capture_nodes"]),
        "sampled_bf16_exact": True,
        "full_output_equals_direct": True,
        "timing_note": (
            "Two alternating direct/graph rounds, each seven samples of 20 "
            "launches; same HSACO and device-resident inputs, separate "
            "streams/sessions. Host enqueue excludes synchronization; HIP "
            "event timing includes device work and dispatch, not transfers."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=3, type=int, default=[256, 5120, 8704])
    args = parser.parse_args()
    print(json.dumps(measure(*args.shape), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
