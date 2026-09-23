#!/usr/bin/env python3
"""Matched gfx1201 block versus wave FP8 producer timing and correctness."""
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
        raise RuntimeError("producer comparison requires selected gfx1201")
    case = Case("prefill", m, n, k)
    inputs = _logical_inputs(case)
    payload = prepare_packed_folded_payload(
        inputs["packed_row_major"], inputs["b_scale"], allow_approximate=True,
    )
    if not payload.lossless:
        raise RuntimeError("producer comparison requires lossless weight folding")
    package = package_mxfp4_packed_folded_prefill(
        m, payload, permute_decode=True, batched_loads=True,
    )
    x = np.ascontiguousarray(
        inputs["a"].view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    )
    rows, cols, expected = _sampled_exact_reference(case, inputs)
    expected_relu = np.maximum(expected.astype(np.float32), 0).astype(
        ml_dtypes.bfloat16
    )
    rounds: list[dict[str, object]] = []
    for variant in ("block", "wave", "wave", "block"):
        with PackedFoldedGraphPipeline(
            package, payload, m, producer_variant=variant,
        ) as pipeline:
            pipeline.upload_fp32(x)
            pipeline.capture()
            pipeline.replay()
            output = pipeline.read_final()
            np.testing.assert_array_equal(output[np.ix_(rows, cols)], expected_relu)
            stream = ctypes.c_void_p(pipeline.stream_pointer)

            def direct() -> None:
                pipeline._enqueue_producer()
                pipeline._enqueue_kernel()
                pipeline._enqueue_consumer()

            producer = _time_launch(
                pipeline._hip, stream, pipeline._enqueue_producer,
            )
            direct_timing = _time_launch(pipeline._hip, stream, direct)
            graph_timing = _time_launch(pipeline._hip, stream, pipeline.replay)
            pipeline.synchronize()
            rounds.append({
                "variant": variant,
                "aux_hsaco_sha256": pipeline.receipt()["aux_hsaco_sha256"],
                "producer_stage": producer,
                "direct_triple": direct_timing,
                "graph_triple": graph_timing,
                "sampled_bf16_exact": True,
            })
    summary = {
        variant: {
            stage: {
                metric: statistics.median(
                    row[stage][metric] for row in rounds if row["variant"] == variant
                )
                for metric in ("host_enqueue_us_median", "kernel_event_us_median")
            }
            for stage in ("producer_stage", "direct_triple", "graph_triple")
        }
        for variant in ("block", "wave")
    }
    device_name, device_ordinal = _device_identity()
    return {
        "sync_key": "GFX1201-MXFP4-GRAPH-MODEL-PRODUCER-2026-09-23",
        "revision": subprocess.check_output(
            ("git", "-C", str(ROOT), "rev-parse", "HEAD"), text=True,
        ).strip(),
        "host": socket.gethostname(),
        "device_name": device_name,
        "device_ordinal": device_ordinal,
        "target": "rocm_gfx1201",
        "shape": [m, n, k],
        "gemm_hsaco_sha256": hashlib.sha256(package.image.payload).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "python/tessera/compiler/rocm_mxfp4_graph_pipeline.py",
                "benchmarks/rocm/benchmark_gfx1201_mxfp4_graph_producer.py",
            )
        },
        "rounds": rounds,
        "summary": summary,
        "automatic_selection": False,
        "timing_note": (
            "Block/wave/wave/block order; each round uses seven samples of twenty "
            "launches and the same packed GEMM HSACO, shape, logical inputs and "
            "device. HIP events exclude host transfers."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", nargs=3, type=int, default=[256, 5120, 8704])
    args = parser.parse_args()
    print(json.dumps(measure(*args.shape), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
