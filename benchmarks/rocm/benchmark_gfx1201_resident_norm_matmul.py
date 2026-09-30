"""Correctness-gated public frontend benchmark for the gfx1201 resident edge."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from tessera import runtime as rt
from tessera.compiler.from_text import from_text
from tessera.compiler.resident_rocm_norm_matmul import package_graph_rmsnorm_matmul

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
parser.add_argument("--warmup", type=int, default=25)
parser.add_argument("--iterations", type=int, default=100)
parser.add_argument("--dynamic-n", action="store_true")
parser.add_argument("--dynamic-m", action="store_true")
parser.add_argument("--dynamic-k", action="store_true")
parser.add_argument("--dynamic-mk", action="store_true")
parser.add_argument("--dynamic-mnk", action="store_true")
args = parser.parse_args()
if sum((args.dynamic_m, args.dynamic_n, args.dynamic_k, args.dynamic_mk, args.dynamic_mnk)) > 1:
    parser.error("choose one dynamic extent envelope; --dynamic-mnk combines M, N, and K")
if args.warmup < 0 or args.iterations <= 0:
    parser.error("warmup must be nonnegative and iterations must be positive")
assert rt._rocm_live_arch() == rt._rocm_chip() == "gfx1201"
storage_dtype = np.float16
if args.dtype == "bf16":
    import ml_dtypes
    storage_dtype = ml_dtypes.bfloat16
m, k, n = 128, 256, 256
rng = np.random.default_rng(91201)
x = rng.standard_normal((m, k)).astype(storage_dtype)
rhs = rng.standard_normal((k, n)).astype(storage_dtype)
producer = from_text("""
    def rmsnorm_frontend(x):
        return ts.ops.rmsnorm(x, eps=1e-5)
""")
consumer = from_text("""
    def matmul_frontend(normalized, weights):
        return ts.ops.matmul(normalized, weights, output_dtype="fp32")
""")
producer(x)
consumer(np.zeros((m, k), dtype=storage_dtype), rhs)
assert producer.frontend_authority == consumer.frontend_authority == "tracer"
assert consumer.graph_ir.functions[0].result_types[0].dtype == "fp32"

with package_graph_rmsnorm_matmul(
    producer.graph_ir,
    consumer.graph_ir,
    x,
    rhs,
    dynamic_m_bound=m if args.dynamic_m or args.dynamic_mk or args.dynamic_mnk else None,
    dynamic_n_bound=n if args.dynamic_n or args.dynamic_mnk else None,
    dynamic_k_bound=k if args.dynamic_k or args.dynamic_mk or args.dynamic_mnk else None,
    pipeline_name="tessera-lower-to-rocm",
) as session:
    check = session.run(warmup=0, iterations=1)
    epsilon = float(session._norm_package.descriptor.provenance["epsilon"])
    x32 = x.astype(np.float32)
    normalized = (
        x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + epsilon)
    ).astype(storage_dtype)
    expected = normalized.astype(np.float32) @ rhs.astype(np.float32)
    tolerance = 3e-2 if args.dtype == "bf16" else 2e-3
    np.testing.assert_allclose(check["outputs"][0], expected, rtol=tolerance, atol=tolerance)
    active_n_runs = []
    active_m_runs = []
    active_k_runs = []
    active_mk_runs = []
    active_mnk_runs = []
    if args.dynamic_mnk:
        active_shapes = ((m // 2, k // 2, n // 2), (3 * m // 4, 3 * k // 4, 3 * n // 4), (m, k, n))
        for active_m, active_k, active_n in active_shapes:
            active_x_backing = np.zeros((active_m, active_k + 3), dtype=storage_dtype)
            active_x_backing[:, :active_k] = x[:active_m, :active_k]
            active_x = active_x_backing[:, :active_k]
            active_rhs_backing = np.zeros((k + 5, n + 3), dtype=storage_dtype, order="F")
            active_rhs_backing[:active_k, :active_n] = rhs[:active_k, :active_n]
            active_rhs = active_rhs_backing[:active_k, :active_n]
            active_x32 = active_x.astype(np.float32)
            active_normalized = (
                active_x32 / np.sqrt(
                    np.mean(active_x32 * active_x32, axis=-1, keepdims=True) + epsilon
                )
            ).astype(storage_dtype)
            active_expected = active_normalized.astype(np.float32) @ active_rhs.astype(np.float32)
            result = session.run(
                warmup=args.warmup, iterations=args.iterations,
                x=active_x, rhs=active_rhs,
            )
            for output in result["outputs"]:
                np.testing.assert_allclose(
                    output, active_expected, rtol=tolerance, atol=tolerance
                )
            active_mnk_runs.append({
                "active_m": active_m, "active_n": active_n, "active_k": active_k,
                "correctness_checked": True,
                "producer_device_event_ms": result["producer_device_event_ms"],
                "consumer_device_event_ms": result["consumer_device_event_ms"],
                "producer_median_ms": result["producer_median_ms"],
                "consumer_median_ms": result["consumer_median_ms"],
                "resident_buffer_addresses": result["buffer_addresses"],
            })
    elif args.dynamic_mk:
        active_shapes = (
            (max(1, m // 2), max(1, k // 2)),
            (m, max(1, (3 * k) // 4)),
            (m, k),
        )
        for active_m, active_k in active_shapes:
            active_x_backing = np.zeros((active_m, active_k + 3), dtype=storage_dtype)
            active_x_backing[:, :active_k] = x[:active_m, :active_k]
            active_x = active_x_backing[:, :active_k]
            active_rhs_backing = np.zeros((k + 5, n + 3), dtype=storage_dtype)
            active_rhs_backing[:active_k, :n] = rhs[:active_k, :]
            active_rhs = active_rhs_backing[:active_k, :n]
            active_x32 = active_x.astype(np.float32)
            active_normalized = (
                active_x32 / np.sqrt(
                    np.mean(active_x32 * active_x32, axis=-1, keepdims=True) + epsilon
                )
            ).astype(storage_dtype)
            active_expected = active_normalized.astype(np.float32) @ active_rhs.astype(np.float32)
            result = session.run(
                warmup=args.warmup, iterations=args.iterations,
                x=active_x, rhs=active_rhs,
            )
            for output in result["outputs"]:
                np.testing.assert_allclose(
                    output, active_expected, rtol=tolerance, atol=tolerance
                )
            active_mk_runs.append({
                "active_m": active_m,
                "active_k": active_k,
                "correctness_checked": True,
                "producer_device_event_ms": result["producer_device_event_ms"],
                "consumer_device_event_ms": result["consumer_device_event_ms"],
                "producer_median_ms": result["producer_median_ms"],
                "consumer_median_ms": result["consumer_median_ms"],
                "resident_buffer_addresses": result["buffer_addresses"],
            })
    elif args.dynamic_m:
        for active_m in (max(1, m // 2), m):
            active_x = x[:active_m]
            active_x32 = active_x.astype(np.float32)
            active_normalized = (
                active_x32 / np.sqrt(
                    np.mean(active_x32 * active_x32, axis=-1, keepdims=True) + epsilon
                )
            ).astype(storage_dtype)
            active_expected = active_normalized.astype(np.float32) @ rhs.astype(np.float32)
            result = session.run(
                warmup=args.warmup, iterations=args.iterations, x=active_x
            )
            for output in result["outputs"]:
                np.testing.assert_allclose(
                    output, active_expected, rtol=tolerance, atol=tolerance
                )
            active_m_runs.append({
                "active_m": active_m,
                "correctness_checked": True,
                "producer_device_event_ms": result["producer_device_event_ms"],
                "consumer_device_event_ms": result["consumer_device_event_ms"],
                "producer_median_ms": result["producer_median_ms"],
                "consumer_median_ms": result["consumer_median_ms"],
                "resident_buffer_addresses": result["buffer_addresses"],
            })
    elif args.dynamic_k:
        active_ks = (max(1, k // 2), max(1, (3 * k) // 4), k)
        for active_k in active_ks:
            active_x_backing = np.zeros((m, active_k + 3), dtype=storage_dtype)
            active_x_backing[:, :active_k] = x[:, :active_k]
            active_x = active_x_backing[:, :active_k]
            active_rhs_backing = np.zeros((k + 5, n + 3), dtype=storage_dtype)
            active_rhs_backing[:active_k, :n] = rhs[:active_k, :]
            active_rhs = active_rhs_backing[:active_k, :n]
            active_x32 = active_x.astype(np.float32)
            active_normalized = (
                active_x32 / np.sqrt(
                    np.mean(active_x32 * active_x32, axis=-1, keepdims=True) + epsilon
                )
            ).astype(storage_dtype)
            active_expected = active_normalized.astype(np.float32) @ active_rhs.astype(np.float32)
            result = session.run(
                warmup=args.warmup, iterations=args.iterations,
                x=active_x, rhs=active_rhs,
            )
            for output in result["outputs"]:
                np.testing.assert_allclose(
                    output, active_expected, rtol=tolerance, atol=tolerance
                )
            active_k_runs.append({
                "active_k": active_k,
                "correctness_checked": True,
                "producer_device_event_ms": result["producer_device_event_ms"],
                "consumer_device_event_ms": result["consumer_device_event_ms"],
                "producer_median_ms": result["producer_median_ms"],
                "consumer_median_ms": result["consumer_median_ms"],
                "resident_buffer_addresses": result["buffer_addresses"],
            })
    else:
        active_ns = (max(1, n // 2), n) if args.dynamic_n else (n,)
        for active_n in active_ns:
            active_rhs = rhs[:, :active_n]
            active_expected = normalized.astype(np.float32) @ active_rhs.astype(np.float32)
            result = session.run(
                warmup=args.warmup, iterations=args.iterations, rhs=active_rhs
            )
            for output in result["outputs"]:
                np.testing.assert_allclose(
                    output, active_expected, rtol=tolerance, atol=tolerance
                )
            active_n_runs.append({
                "active_n": active_n,
                "correctness_checked": True,
                "producer_device_event_ms": result["producer_device_event_ms"],
                "consumer_device_event_ms": result["consumer_device_event_ms"],
                "producer_median_ms": result["producer_median_ms"],
                "consumer_median_ms": result["consumer_median_ms"],
                "resident_buffer_addresses": result["buffer_addresses"],
            })
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    )
    dirty_diff = bytearray(subprocess.check_output(["git", "diff", "--binary", "HEAD"]))
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"]
    )
    for raw_path in untracked.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode()
        dirty_diff.extend(b"\0" + raw_path + b"\0")
        dirty_diff.extend(Path(relative).read_bytes())
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
        "producer_compiler_fingerprint": session._norm_package.image.compiler_fingerprint,
        "consumer_compiler_fingerprint": session._matmul_package.image.compiler_fingerprint,
        "producer_toolchain_fingerprint": session._norm_package.image.toolchain_fingerprint,
        "consumer_toolchain_fingerprint": session._matmul_package.image.toolchain_fingerprint,
        "shape_mkn": [m, k, n],
        "dynamic_n_bound": n if args.dynamic_n or args.dynamic_mnk else None,
        "dynamic_m_bound": m if args.dynamic_m or args.dynamic_mk or args.dynamic_mnk else None,
        "dynamic_k_bound": k if args.dynamic_k or args.dynamic_mk or args.dynamic_mnk else None,
        "active_n_runs": active_n_runs,
        "active_m_runs": active_m_runs,
        "active_k_runs": active_k_runs,
        "active_mk_runs": active_mk_runs,
        "active_mnk_runs": active_mnk_runs,
        "storage": args.dtype,
        "accumulation": "fp32",
        "output": "fp32",
        "correctness_checked": True,
        "oracle": f"f32 RMSNorm, {args.dtype} materialized edge, f32 matmul",
        "warmup": args.warmup,
        "iterations": args.iterations,
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
        "worktree_diff_sha256": hashlib.sha256(dirty_diff).hexdigest(),
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    print(json.dumps(packet, sort_keys=True, indent=2))
