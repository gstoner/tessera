"""Correctness-gated public frontend benchmark for the gfx1201 resident edge."""

import json
import subprocess

import numpy as np
from tessera import runtime as rt
from tessera.compiler.from_text import from_text
from tessera.compiler.resident_rocm_norm_matmul import package_graph_rmsnorm_matmul

assert rt._rocm_live_arch() == rt._rocm_chip() == "gfx1201"
m, k, n = 128, 256, 256
rng = np.random.default_rng(91201)
x = rng.standard_normal((m, k)).astype(np.float16)
rhs = rng.standard_normal((k, n)).astype(np.float16)
producer = from_text("""
    def rmsnorm_frontend(x):
        return ts.ops.rmsnorm(x, eps=1e-5)
""")
consumer = from_text("""
    def matmul_frontend(normalized, weights):
        return ts.ops.matmul(normalized, weights, output_dtype="fp32")
""")
producer(x)
consumer(np.zeros((m, k), dtype=np.float16), rhs)
assert producer.frontend_authority == consumer.frontend_authority == "tracer"
assert consumer.graph_ir.functions[0].result_types[0].dtype == "fp32"

with package_graph_rmsnorm_matmul(
    producer.graph_ir,
    consumer.graph_ir,
    x,
    rhs,
    pipeline_name="tessera-lower-to-rocm",
) as session:
    check = session.run(warmup=0, iterations=1)
    epsilon = float(session._norm_package.descriptor.provenance["epsilon"])
    x32 = x.astype(np.float32)
    normalized = (
        x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + epsilon)
    ).astype(np.float16)
    expected = normalized.astype(np.float32) @ rhs.astype(np.float32)
    np.testing.assert_allclose(check["outputs"][0], expected, rtol=2e-3, atol=2e-3)
    result = session.run(warmup=25, iterations=100)
    for output in result["outputs"]:
        np.testing.assert_allclose(output, expected, rtol=2e-3, atol=2e-3)
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    )
    packet = {
        "schema": "tessera.resident_norm_matmul.benchmark.v1",
        "architecture": "gfx1201",
        "device_architecture": rt._rocm_live_arch(),
        "compiler_architecture": rt._rocm_chip(),
        "device_id": result["device_id"],
        "route": "public_from_text_tracer_graph_schedule_tile_resident_package",
        "frontend_authority": [producer.frontend_authority, consumer.frontend_authority],
        "producer_image_digest": session._norm_package.image.image_digest,
        "consumer_image_digest": session._matmul_package.image.image_digest,
        "shape_mkn": [m, k, n],
        "storage": "fp16",
        "accumulation": "fp32",
        "output": "fp32",
        "correctness_checked": True,
        "oracle": "f32 RMSNorm, fp16 materialized edge, f32 matmul",
        "warmup": 25,
        "iterations": 100,
        "producer_device_event_ms": result["producer_device_event_ms"],
        "consumer_device_event_ms": result["consumer_device_event_ms"],
        "producer_median_ms": result["producer_median_ms"],
        "consumer_median_ms": result["consumer_median_ms"],
        "resident_buffer_addresses": result["buffer_addresses"],
        "package_reused": True,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "worktree_dirty": bool(git_status.strip()),
    }
    print(json.dumps(packet, sort_keys=True, indent=2))
