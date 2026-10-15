"""Exact SM120 native producer-chain correctness and separated timing domains."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.nvidia_tensor_lhs import package_traced_lhs
from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall

SOURCE = """def norm_relu(a, b):
    normalized = tessera.ops.rmsnorm(a, eps=1e-5)
    edge = tessera.ops.relu(normalized)
    return tessera.ops.matmul(edge, b, output_dtype="fp32")
"""


def oracle(a, b):
    wide = a.astype(np.float32)
    normalized = (wide / np.sqrt(np.mean(wide * wide, axis=-1, keepdims=True)
                                 + np.float32(1e-5))).astype(a.dtype)
    return np.maximum(0, normalized).astype(np.float32) @ b.astype(np.float32)


def check(output, expected, storage):
    np.testing.assert_allclose(output, expected,
        atol=0.02 if storage == "bf16" else 0.003,
        rtol=0.02 if storage == "bf16" else 0.003)


def wall_samples(call, samples, repetitions):
    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(repetitions):
            call()
        timings.append((time.perf_counter() - start) * 1000 / repetitions)
    return timings


def run_case(shape, storage, samples, repetitions, warmup):
    m, k, n = shape
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    rng = np.random.default_rng(47017 + m + k + n)
    a = rng.uniform(-1, 1, (m, k)).astype(dtype)
    b = rng.uniform(-0.25, 0.25, (k, n)).astype(dtype)
    expected = oracle(a, b)
    public = ts.from_text(SOURCE, target="nvidia_sm120")
    graph, _ = public._trace_frontend_capture((a, b), {})
    start = time.perf_counter()
    program = package_traced_lhs(graph)
    package_ms = (time.perf_counter() - start) * 1000
    native = json.loads(program.native_plan_json)
    assert [step["operation"] for step in native["steps"]] == [
        "tessera.rmsnorm", "tessera.relu", "tessera.matmul"]
    owner = PreparedLhsCall(program)
    try:
        output, receipt = owner((a, b))
        assert receipt["execution_kind"] == "native_gpu"
        assert len(receipt["component_receipts"]) == 3
        check(output, expected, storage)
        check(public(a, b), expected, storage)
        assert public.execution_kind == "native_gpu"
        for _ in range(warmup):
            owner((a, b))
            public(a, b)
        # Public calls use another owner of the native shared arena. Reacquire
        # this owner's frame before profiling; keep the stale-lease guard intact.
        owner((a, b))
        profiles = [owner.profile(repeats=repetitions) for _ in range(samples)]
        for profile in profiles:
            check(profile["output"], expected, storage)
        stages = [profile["grouped_stage_ms"] for profile in profiles]
        assert all(len(row) == 3 for row in stages)
        device = [profile["program_ms"] for profile in profiles]
        prepared_wall = wall_samples(lambda: owner((a, b)), samples, repetitions)
        public_wall = wall_samples(lambda: public(a, b), samples, repetitions)
        return {"shape_mkn": list(shape), "storage": storage,
            "correctness": "passed_before_timing", "max_abs_error": float(np.max(np.abs(output - expected))),
            "package_ms": package_ms, "device_program_samples_ms": device,
            "device_program_median_ms": statistics.median(device),
            "device_grouped_stage_samples_ms": stages,
            "device_grouped_stage_medians_ms": dict(zip(("rmsnorm", "relu", "matmul"),
                (statistics.median(row[index] for row in stages) for index in range(3)), strict=True)),
            "prepared_host_samples_ms": prepared_wall,
            "prepared_host_median_ms": statistics.median(prepared_wall),
            "public_jit_samples_ms": public_wall,
            "public_jit_median_ms": statistics.median(public_wall),
            "scratch_stats": owner.scratch_stats(),
            "program_digest": hashlib.sha256(program.native_plan_json.encode()).hexdigest(),
            "images": [p.image.image_digest for p in (*program.producer_chain, program.edge.consumer)],
            "ptx_sha256": [hashlib.sha256(p.image.payload).hexdigest()
                for p in (*program.producer_chain, program.edge.consumer)],
            "tile_ir_sha256": [hashlib.sha256(p.tile_ir.encode()).hexdigest()
                for p in (*program.producer_chain, program.edge.consumer)],
            "entries": [p.descriptor.entry_symbol for p in (*program.producer_chain, program.edge.consumer)]}
    finally:
        owner.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples <= 0 or args.repetitions <= 0 or args.warmup < 0:
        parser.error("samples/repetitions must be positive and warmup nonnegative")
    if rt._nvidia_device_name() != "sm_120":
        raise RuntimeError("requires the owning SM120 device")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version",
                                   "--format=csv,noheader"], text=True).strip()
    files = ["src/compiler/programming_model/lib/PMPasses.cpp",
             "src/compiler/programming_model/ir/ScheduleDialect.cpp",
             "src/compiler/programming_model/ir/schedule/ScheduleMeshPipelineOps.td",
             "src/compiler/ir/TileOps.cpp", "src/compiler/ir/include/Tessera/Dialect/Tile/TileOps.td",
             "src/transforms/lib/NativeSM120TensorProgram.h",
             "src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
             "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/matmul_prepared.cpp",
             "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp",
             "python/tessera/compiler/native_sm120_tensor_program.py",
             "python/tessera/compiler/nvidia_tensor_lhs.py", "python/tessera/compiler/nvidia_tensor_dag.py",
             "python/tessera/compiler/nvidia_native.py", "python/tessera/compiler/prepared_nvidia_lhs.py",
             "python/tessera/runtime.py", "benchmarks/nvidia/benchmark_norm_relu_matmul.py"]
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    cache = (Path(os.environ["TESSERA_OPT"]).parents[2] / "CMakeCache.txt").read_text()
    configuration = {name: f"{name}:BOOL=ON" in cache for name in (
        "TESSERA_BUILD_NVIDIA_BACKEND", "TESSERA_BUILD_ROCM_BACKEND",
        "TESSERA_BUILD_X86_BACKEND", "TESSERA_BUILD_EBM_BACKEND",
        "TESSERA_BUILD_CLIFFORD_BACKEND")}
    packet = {"schema": "tessera.nvidia.norm_relu_matmul_benchmark.v1", "work_item": "W1.1",
        "sync_key": "W11-RELU-PRODUCER-20261014", "gpu_reported": gpu,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_state": "sha256_pinned_working_tree",
        "compiler_configuration": configuration,
        "source_sha256": {path: sha(path) for path in files},
        "binary_sha256": {name: sha(os.environ[name]) for name in
            ("TESSERA_OPT", "TESSERA_NVIDIA_OPT", "TESSERA_NVIDIA_PTX_LAUNCH_LIB")},
        "method": {"samples": args.samples, "repetitions": args.repetitions, "warmup": args.warmup,
            "device_grouped_stage_definition": "CUDA event mean per member; isolated repeated stage groups",
            "device_program_definition": "CUDA event mean for the complete three-member native program",
            "prepared_host_definition": "completed prepared host call including binding, copies and synchronization",
            "public_jit_definition": "completed warm public JIT call including admission, binding, copies and synchronization",
            "package_definition": "first traced Graph package wall time in this recorder process",
            "selector_changed": False, "performance_comparison_claim": False},
        "cases": [run_case(shape, storage, args.samples, args.repetitions, args.warmup)
            for shape in ((7, 32, 9), (17, 64, 19), (129, 256, 33)) for storage in ("f16", "bf16")]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
