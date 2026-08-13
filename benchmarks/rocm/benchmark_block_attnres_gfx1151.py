#!/usr/bin/env python3
"""Record exact-device Block AttnRes Phase-5 correctness/timing evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import statistics
import time

import numpy as np

from tessera import runtime as rt
from tessera._block_attnres_ops import depth_attn
from tessera.compiler.driver import compile_graph_module
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType


def _module(sources: int, rows: int, width: int) -> GraphIRModule:
    query = IRType(f"tensor<{width}xf32>", (str(width),), "fp32")
    stack = IRType(
        f"tensor<{sources}x{rows}x{width}xf32>",
        (str(sources), str(rows), str(width)),
        "fp32",
    )
    output = IRType(f"tensor<{rows}x{width}xf32>", (str(rows), str(width)), "fp32")
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name=f"block_attnres_s{sources}_r{rows}_d{width}",
                args=[IRArg("query", query), IRArg("sources", stack)],
                result_types=[output],
                body=[
                    IROp(
                        result="result",
                        op_name="tessera.depth_attn",
                        operands=["%query", "%sources"],
                        operand_types=[str(query), str(stack)],
                        result_type=str(output),
                        kwargs={
                            "eps": 1.0e-6,
                            "numeric_policy": {
                                "storage": "fp32",
                                "softmax": "fp32",
                                "accum": "fp32",
                            },
                        },
                    )
                ],
                return_values=["%result"],
            )
        ]
    )


def record(*, warmup: int = 3, repetitions: int = 10) -> dict[str, object]:
    rng = np.random.default_rng(20260813)
    rows_out: list[dict[str, object]] = []
    for sources, rows, width in ((7, 3, 8), (17, 31, 64), (33, 127, 128)):
        bundle = compile_graph_module(
            _module(sources, rows, width),
            source_origin="block-attnres-phase5-evidence",
            target="rocm_gfx1151",
            options={"package_native": True},
            enable_tool_validation=False,
        )
        assert bundle.native_image is not None
        assert bundle.launch_descriptor is not None
        assert bundle.tile is not None and bundle.target_ir is not None
        runtime_artifact = rt.RuntimeArtifact(
            metadata={"target": "rocm_gfx1151"},
            native_image=bundle.native_image,
            launch_descriptor=bundle.launch_descriptor,
            tile_ir=bundle.tile.text,
            target_ir=bundle.target_ir.text,
        )
        query = np.ascontiguousarray(rng.normal(size=(width,)), dtype=np.float32)
        source_values = np.ascontiguousarray(rng.normal(size=(sources, rows, width)), dtype=np.float32)
        expected = depth_attn(query, source_values)
        samples: list[int] = []
        maximum_error = 0.0
        for iteration in range(warmup + repetitions):
            output = np.zeros((rows, width), dtype=np.float32)
            start = time.perf_counter_ns()
            result = rt.launch(
                runtime_artifact,
                {"query": query, "sources": source_values, "result": output},
            )
            elapsed = time.perf_counter_ns() - start
            if not result.get("ok"):
                raise RuntimeError(str(result))
            maximum_error = max(maximum_error, float(np.max(np.abs(output - expected))))
            if iteration >= warmup:
                samples.append(elapsed)
        if maximum_error > 3.0e-5:
            raise RuntimeError(f"depth-attention maximum error {maximum_error}")
        rows_out.append(
            {
                "shape": {"sources": sources, "rows": rows, "width": width},
                "schedule_digest": bundle.launch_descriptor.provenance["schedule_digest"],
                "tile_ir_digest": hashlib.sha256(bundle.tile.text.encode()).hexdigest(),
                "target_ir_digest": bundle.native_image.target_ir_digest,
                "image_digest": bundle.native_image.image_digest,
                "hsaco_sha256": hashlib.sha256(bundle.native_image.payload).hexdigest(),
                "compile_state": bundle.native_image.compile_state,
                "maximum_absolute_error": maximum_error,
                "correct": True,
                "host_wall_ns": {
                    "source": "synchronized_host_wall",
                    "samples": samples,
                    "median": int(statistics.median(samples)),
                    "minimum": min(samples),
                    "valid": True,
                    "selector_eligible": False,
                },
                "hip_event_ns": {
                    "source": "not_collected_by_descriptor_launcher",
                    "samples": [],
                    "median": None,
                    "valid": False,
                    "selector_eligible": False,
                },
            }
        )
    return {
        "schema": "tessera.block_attnres.gfx1151.phase5.v1",
        "work_item": "BLOCK-ATTNRES-ROCM-1",
        "sync_key": "BLOCK-ATTNRES-ROCM-2026-08-12",
        "architecture": "gfx1151",
        "device": "AMD Radeon 8060S / Strix Halo",
        "host": platform.platform(),
        "algorithm": "rms_key_online_softmax_stats_v1",
        "merge": "max_shifted_pairwise_merge_v1",
        "numeric_policy": {"storage": "fp32", "softmax": "fp32", "accum": "fp32"},
        "timing_provenance": {
            "environment": "WSL gfx1151",
            "operation_total": True,
            "includes_allocation_module_load_copy_and_synchronization": True,
            "selector_eligible": False,
            "promotion_blocker": "requires calibrated HIP events or ROCprofiler activity on bare metal",
        },
        "rows": rows_out,
        "verdict": {
            "correctness": "retain",
            "physical_package": "retain",
            "performance": "provisional_operation_total_only",
            "selector_eligible": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    packet = record(warmup=args.warmup, repetitions=args.repetitions)
    rendered = json.dumps(packet, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
