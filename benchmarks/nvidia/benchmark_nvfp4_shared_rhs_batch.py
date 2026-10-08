"""Correctness-gated native shared-RHS row-batch attribution on SM120."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
import numpy as np
from tessera.compiler.canonical_compile import compile_result_from_bundle
from tessera.compiler.driver import compile_graph_module
from tessera.runtime import launch, _nvidia_native_descriptor_device_latency, _nvidia_native_descriptor_resources
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.unit.test_nvfp4_shared_rhs_batch import frontend_batch_module
from tests.device.nvidia.test_e2e_spine_native import _pack_nvfp4, _decode_e2m1, _decode_ue4m3


def package(batch, rows, n, k, batching="shared_rhs_rows"):
    module = frontend_batch_module(batch, rows, n, k, batching)
    bundle = compile_graph_module(module, source_origin="W1.1", target="nvidia_sm120",
                                 options={"package_native": True}, enable_tool_validation=False)
    if bundle.launch_descriptor is None or bundle.native_image is None:
        raise RuntimeError("batch compilation omitted its native image or checked ABI")
    return compile_result_from_bundle(bundle, module=module).to_runtime_artifact()


def run_shape(shape, samples, reps, warmup, batching="shared_rhs_rows"):
    batch, rows, n, k = shape
    independent_rhs = batching == "independent_rhs"
    sk = (k + 15) // 16
    rng = np.random.default_rng(120608 + sum(shape))
    ac = rng.integers(0, 16, (batch, rows, k), dtype=np.uint8)
    bc = rng.integers(0, 16, (batch, k, n) if independent_rhs else (k, n), dtype=np.uint8)
    choices = np.asarray([0x30, 0x31, 0x33, 0x35, 0x38, 0x3A, 0x40], np.uint8)
    sa = np.ascontiguousarray(choices[(np.arange(batch)[:, None, None] * 3 + np.arange(rows)[None, :, None]
                                     + np.arange(sk)[None, None, :]) % choices.size])
    sb = np.ascontiguousarray(choices[(2 * np.arange(sk)[:, None] + np.arange(n)[None, :]) % choices.size])
    if independent_rhs:
        sb = np.ascontiguousarray(choices[(3 * np.arange(batch)[:, None, None]
            + 2 * np.arange(sk)[None, :, None] + np.arange(n)[None, None, :]) % choices.size])
    ap, bp = _pack_nvfp4(ac, 2), _pack_nvfp4(bc, 1 if independent_rhs else 0)
    expected = ((_decode_e2m1(ac) * np.repeat(_decode_ue4m3(sa), 16, axis=2)[:, :, :k]).astype(np.float64)
                @ (_decode_e2m1(bc) * (np.repeat(_decode_ue4m3(sb), 16, axis=1)[:, :k, :]
                    if independent_rhs else np.repeat(_decode_ue4m3(sb), 16, axis=0)[:k])).astype(np.float64))
    fused = package(batch, rows, n, k, batching)
    separate = package(1, rows, n, k, batching)
    out = np.full((batch, rows, n), np.nan, np.float32)
    args = {"a": ap, "b": bp, "sa": sa, "sb": sb, fused.launch_descriptor.buffers[4].name: out, "M": batch * rows, "N": n, "K": k}
    separate_args = [{"a": ap[i:i+1], "b": bp[i:i+1] if independent_rhs else bp,
                      "sa": sa[i:i+1], "sb": sb[i:i+1] if independent_rhs else sb,
                      separate.launch_descriptor.buffers[4].name: np.full((1, rows, n), np.nan, np.float32), "M": rows, "N": n, "K": k}
                     for i in range(batch)]
    if independent_rhs:
        args.update(BatchRows=rows, BatchCount=batch)
        for one in separate_args:
            one.update(BatchRows=rows, BatchCount=1)
    def checked(artifact, arguments):
        result = launch(artifact, arguments)
        if not result.get("ok") or result.get("execution_kind") != "native_gpu":
            raise RuntimeError(f"native batch launch failed: {result}")
    checked(fused, args)
    np.testing.assert_allclose(out, expected, rtol=0, atol=2e-3)
    for i, one in enumerate(separate_args):
        checked(separate, one)
        np.testing.assert_allclose(one[separate.launch_descriptor.buffers[4].name][0], expected[i], rtol=0, atol=2e-3)
    device, serial_device, wall, serial_wall = [], [], [], []
    for _ in range(samples):
        device.append(_nvidia_native_descriptor_device_latency(
            fused.native_image, fused.launch_descriptor, args, reps=reps, warmup=warmup))
        # Sum owning event intervals; this is total device work for a logical
        # batch, distinct from the wall interval around all serial submissions.
        serial_device.append(sum(_nvidia_native_descriptor_device_latency(
            separate.native_image, separate.launch_descriptor, one, reps=reps, warmup=warmup)
            for one in separate_args))
        start = time.perf_counter()
        for _ in range(reps): checked(fused, args)
        wall.append((time.perf_counter() - start) * 1e3 / reps)
        start = time.perf_counter()
        for _ in range(reps):
            for one in separate_args: checked(separate, one)
        serial_wall.append((time.perf_counter() - start) * 1e3 / reps)
    d = fused.launch_descriptor
    return {"shape_bmnk": list(shape), "max_abs_error": float(np.max(np.abs(out - expected))),
            "correctness": "both_arms_passed_before_timing", "batch_rows": d.provenance["batch_rows"],
            "schedule_digest": d.provenance["schedule_digest"], "tile_ir_digest": d.provenance["tile_ir_digest"],
            "image_digest": fused.native_image.image_digest, "entry": d.entry_symbol, "abi_id": d.abi_id,
            "native_batch_event_samples_ms": device, "serial_event_sum_samples_ms": serial_device,
            "native_batch_wall_samples_ms": wall, "serial_batch_wall_samples_ms": serial_wall,
            "native_batch_event_median_ms": statistics.median(device), "serial_event_sum_median_ms": statistics.median(serial_device),
            "native_batch_wall_median_ms": statistics.median(wall), "serial_batch_wall_median_ms": statistics.median(serial_wall),
            "resources": _nvidia_native_descriptor_resources(fused.native_image, d, block_size=32)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--batching", choices=("shared_rhs_rows", "independent_rhs"), default="shared_rhs_rows")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if min(args.samples, args.reps) <= 0 or args.warmup < 0:
        raise ValueError("positive sample/repetition counts and nonnegative warmup required")
    if not nvidia_cuda_host_ready(): raise SystemExit("owning SM120 device/compiler required")
    paths = ["python/tessera/__init__.py", "python/tessera/compiler/driver.py", "python/tessera/compiler/graph_ir.py", "python/tessera/compiler/capabilities.py",
             "python/tessera/compiler/scheduled_matmul.py", "python/tessera/compiler/nvidia_native.py",
             "python/tessera/runtime.py", "src/compiler/ir/TesseraOps.cpp", "src/compiler/ir/TesseraOps.td",
             "src/compiler/programming_model/lib/PMPasses.cpp", "tests/unit/test_nvfp4_shared_rhs_batch.py",
             "tests/device/nvidia/test_nvfp4_shared_rhs_batch.py", "tests/device/nvidia/test_nvfp4_independent_rhs_native.py",
             "src/compiler/ir/TileOps.cpp",
             "src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
             "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp",
             "benchmarks/nvidia/benchmark_nvfp4_shared_rhs_batch.py"]
    packet = {"schema": "tessera.nvidia.independent_rhs_nvfp4_batch.v1" if args.batching == "independent_rhs" else "tessera.nvidia.shared_rhs_nvfp4_batch.v1",
              "batching": args.batching, "owner": "W1.1",
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
              "gpu_reported": subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"], text=True).strip(),
              "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in paths},
              "compiler_binary_sha256": {name: hashlib.sha256(Path(os.environ[name]).read_bytes()).hexdigest()
                                         for name in ("TESSERA_OPT", "TESSERA_NVIDIA_OPT", "TESSERA_NVIDIA_PTX_LAUNCH_LIB")},
              "frontend": "typed_python_source_public_scaled_matmul",
              "method": {"samples": args.samples, "reps": args.reps, "warmup": args.warmup,
                         "baseline": "one checked native package call per member under the same RHS batching policy",
                         "timing_domains": ["cuda_event", "host_wall"], "selector_changed": False},
              "rows": [run_shape(shape, args.samples, args.reps, args.warmup, args.batching)
                       for shape in ((3, 7, 5, 31), (2, 17, 19, 129), (5, 9, 11, 64))]}
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
