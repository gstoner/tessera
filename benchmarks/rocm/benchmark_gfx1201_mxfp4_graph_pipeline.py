#!/usr/bin/env python3
"""Same-HSACO direct triple-launch versus three-node HIP graph timing."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import socket
import statistics
import subprocess

import ml_dtypes
import numpy as np

from benchmarks.rocm.benchmark_gfx1201_mxfp4_graph import _time_launch
from benchmarks.rocm.benchmark_gfx1201_mxfp4_production import (
    Case, _logical_inputs, _sampled_exact_reference,
)
from benchmarks.rocm.benchmark_gfx1201_mxfp4_resident import _device_identity
from tessera import runtime as rt
from tessera.compiler.rocm_mxfp4_graph_pipeline import PackedFoldedGraphPipeline
from tessera.compiler.rocm_mxfp4_packed_folded import (
    package_mxfp4_packed_folded_prefill, prepare_packed_folded_payload,
)


ROOT = Path(__file__).resolve().parents[2]


def measure(m: int, n: int, k: int) -> dict[str, object]:
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("graph pipeline benchmark requires selected gfx1201")
    case = Case("prefill", m, n, k)
    inputs = _logical_inputs(case)
    payload = prepare_packed_folded_payload(
        inputs["packed_row_major"], inputs["b_scale"], allow_approximate=True,
    )
    if not payload.lossless:
        raise RuntimeError("matched pipeline timing requires a lossless fold")
    package = package_mxfp4_packed_folded_prefill(
        m, payload, permute_decode=True, batched_loads=True,
    )
    x = np.ascontiguousarray(
        inputs["a"].view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    )
    rows, cols, expected = _sampled_exact_reference(case, inputs)
    expected_relu = np.maximum(
        expected.astype(np.float32), 0,
    ).astype(ml_dtypes.bfloat16)
    with PackedFoldedGraphPipeline(package, payload, m) as pipeline:
        pipeline.upload_fp32(x)
        pipeline.capture()
        pipeline.replay()
        graph_output = pipeline.read_final()
        np.testing.assert_array_equal(graph_output[np.ix_(rows, cols)], expected_relu)

        def direct_triple() -> None:
            pipeline._enqueue_producer()
            pipeline._enqueue_kernel()
            pipeline._enqueue_consumer()

        direct_triple()
        pipeline.synchronize()
        direct_output = pipeline.read_final()
        np.testing.assert_array_equal(direct_output, graph_output)
        stream = ctypes.c_void_p(pipeline.stream_pointer)
        direct_first = _time_launch(pipeline._hip, stream, direct_triple)
        graph_first = _time_launch(pipeline._hip, stream, pipeline.replay)
        direct_second = _time_launch(pipeline._hip, stream, direct_triple)
        graph_second = _time_launch(pipeline._hip, stream, pipeline.replay)
        pipeline.synchronize()
        receipt = pipeline.receipt()
    device_name, device_ordinal = _device_identity()
    return {
        "sync_key": "GFX1201-MXFP4-GRAPH-PIPELINE-2026-09-23",
        "revision": subprocess.check_output(
            ("git", "-C", str(ROOT), "rev-parse", "HEAD"), text=True,
        ).strip(),
        "host": socket.gethostname(),
        "device_name": device_name,
        "device_ordinal": device_ordinal,
        "target": "rocm_gfx1201",
        "shape": [m, n, k],
        "gemm_hsaco_sha256": hashlib.sha256(package.image.payload).hexdigest(),
        "aux_hsaco_sha256": receipt["aux_hsaco_sha256"],
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "python/tessera/compiler/rocm_mxfp4_graph_pipeline.py",
                "benchmarks/rocm/benchmark_gfx1201_mxfp4_graph_pipeline.py",
            )
        },
        "direct": {
            key: statistics.median((direct_first[key], direct_second[key]))
            for key in direct_first
        },
        "graph": {
            key: statistics.median((graph_first[key], graph_second[key]))
            for key in graph_first
        },
        "rounds": {
            "direct_first": direct_first, "graph_first": graph_first,
            "direct_second": direct_second, "graph_second": graph_second,
        },
        "capture_nodes": list(receipt["capture_nodes"]),
        "sampled_bf16_exact": True,
        "full_output_equals_direct": True,
        "automatic_selection": False,
        "timing_note": (
            "Two alternating direct/graph rounds, each seven samples of "
            "20 three-kernel sequences on the same stream, buffers, and code "
            "objects. Host enqueue excludes sync; HIP event excludes transfers."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=3, type=int, default=[256, 5120, 8704])
    args = parser.parse_args()
    print(json.dumps(measure(*args.shape), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
