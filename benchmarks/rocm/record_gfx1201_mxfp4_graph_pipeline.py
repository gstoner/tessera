#!/usr/bin/env python3
"""Record an exact-device FP8→MXFP4→BF16 graph and exact-M lifetime proof."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import subprocess

import ml_dtypes
import numpy as np

from benchmarks.rocm.benchmark_gfx1201_mxfp4_resident import _device_identity
from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.rocm_mxfp4_graph_pipeline import PackedFoldedGraphPipelinePool
from tessera.compiler.rocm_mxfp4_packed_folded import (
    package_mxfp4_packed_folded_prefill,
    prepare_packed_folded_payload,
)


ROOT = Path(__file__).resolve().parents[2]


def record() -> dict[str, object]:
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("graph pipeline proof requires selected gfx1201")
    device_name, ordinal = _device_identity()
    if device_name != "AMD Radeon RX 9070 XT":
        raise RuntimeError(f"Tajasarus proof requires RX 9070 XT, got {device_name}")
    n, k = 48, 64
    payload = prepare_packed_folded_payload(
        mx.pack_e2m1_codes(np.ones((n, k), dtype=np.uint8)),
        np.full((k // 32, n), 127, dtype=np.uint8),
        allow_approximate=True,
    )

    def package_for_m(m: int, p: object) -> object:
        return package_mxfp4_packed_folded_prefill(
            m, p, permute_decode=True, batched_loads=True,
        )

    rows = []
    with PackedFoldedGraphPipelinePool(payload, package_for_m, max_shapes=2) as pool:
        sessions = [pool.get(m) for m in (65, 129)]
        pointers = [session.input_pointer for session in sessions]
        if len(set(pointers)) != 2:
            raise RuntimeError("different M shapes reused a live input pointer")
        for m, session in zip((65, 129), sessions, strict=True):
            session.upload_fp32(np.ones((m, k), dtype=np.float32))
            session.capture()
            session.replay()
            output = session.read_final()
            np.testing.assert_array_equal(
                output, np.full((m, n), k * 0.5, dtype=ml_dtypes.bfloat16),
            )
            receipt = session.receipt()
            if receipt["capture_nodes"] != (0, 0, 0):
                raise RuntimeError("pipeline capture is not exactly three kernel nodes")
            rows.append({
                "shape": [m, n, k],
                "gemm_hsaco_sha256": receipt["image_sha256"],
                "aux_hsaco_sha256": receipt["aux_hsaco_sha256"],
                "capture_nodes": list(receipt["capture_nodes"]),
                "producer": receipt["producer"],
                "consumer": receipt["consumer"],
                "bf16_oracle_exact": True,
            })
        pool_receipt = pool.receipt()
    for session in sessions:
        try:
            _ = session.input_pointer
        except RuntimeError as exc:
            if "closed" not in str(exc):
                raise
        else:
            raise RuntimeError("pool close left a device pointer lease usable")
    return {
        "sync_key": "GFX1201-MXFP4-GRAPH-PIPELINE-2026-09-23",
        "revision": subprocess.check_output(
            ("git", "-C", str(ROOT), "rev-parse", "HEAD"), text=True,
        ).strip(),
        "host": socket.gethostname(),
        "target": "rocm_gfx1201",
        "device_name": device_name,
        "device_ordinal": ordinal,
        "hipcc": subprocess.check_output(
            ("hipcc", "--version"), text=True,
        ).splitlines()[0],
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "python/tessera/compiler/rocm_mxfp4_graph.py",
                "python/tessera/compiler/rocm_mxfp4_graph_pipeline.py",
                "benchmarks/rocm/record_gfx1201_mxfp4_graph_pipeline.py",
                "tests/device/rocm/test_mxfp4_graph_pipeline.py",
            )
        },
        "rows": rows,
        "distinct_live_input_pointers": True,
        "leases_invalid_after_pool_close": True,
        "pool_receipt": pool_receipt,
        "automatic_selection": False,
    }


def main() -> None:
    print(json.dumps(record(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
