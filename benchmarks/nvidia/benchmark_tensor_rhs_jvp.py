"""Correctness-gated public paired-JVP timings on the owning SM120 device."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time
import numpy as np
import ml_dtypes
import tessera as ts
from tessera import runtime as rt


def product(a, b):
    return ts.ops.matmul(ts.ops.relu(ts.ops.rmsnorm(a, eps=1e-5)), b, output_dtype="fp32")


def record(storage, shape):
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    m, k, n = shape
    rng = np.random.default_rng(151070 + m + k + n)
    a = rng.uniform(-1, 1, (m, k)).astype(dtype)
    b = rng.uniform(-0.25, 0.25, (k, n)).astype(dtype)
    db = rng.uniform(-0.125, 0.125, (k, n)).astype(dtype)
    originals = tuple(v.copy() for v in (a, b, db))
    wide = a.astype(np.float32)
    normalized = (wide / np.sqrt(np.mean(wide * wide, axis=-1, keepdims=True) + np.float32(1e-5))).astype(dtype)
    edge = np.maximum(normalized, 0).astype(np.float32)
    expected = (edge @ b.astype(np.float32), edge @ db.astype(np.float32))
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("b",))(product)

    def check(outputs):
        errors = []
        for actual, reference in zip(outputs, expected, strict=True):
            np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
            errors.append(float(np.max(np.abs(actual - reference))))
        return errors

    replay = None
    try:
        start = time.perf_counter_ns()
        first = fn.native_jvp(a, b, tangents=(db,))
        first_ms = (time.perf_counter_ns() - start) / 1e6
        errors = check(first)
        saved = tuple(v.copy() for v in first)
        owner = next(iter(fn._native_prepared_jvp_calls.values()))
        for _ in range(3):
            check(fn.native_jvp(a, b, tangents=(db,)))
        from tessera.compiler.prepared_nvidia_tensor_jvp import PreparedTensorJVP
        portable = json.dumps(owner.manifest())
        scratch = owner.scratch_stats()
        samples = []
        for _ in range(5):
            start = time.perf_counter_ns()
            profile = owner.profile(repeats=128, captured=True)
            profile_host_ms = (time.perf_counter_ns() - start) / 1e6
            check(profile["outputs"])
            samples.append(
                {
                    "program_ms": profile["program_ms"],
                    "stage_ms": profile["stage_ms"],
                    "profile_call_host_ms": profile_host_ms,
                }
            )
        completed = []
        for _ in range(5):
            start = time.perf_counter_ns()
            for _ in range(8):
                outputs = fn.native_jvp(a, b, tangents=(db,))
            completed.append((time.perf_counter_ns() - start) / 1e6 / 8)
            check(outputs)
        public_artifact = rt.RuntimeArtifact.from_json(fn.native_jvp_runtime_artifact().to_json())
        from tessera.compiler import nvidia_tensor_jvp_runtime as binding
        public_replay, full_validation = [], []
        validation_order = []
        cached_validate = binding._validated_plan
        def revalidate(data):
            return binding.restore(data)[0].manifest
        try:
            for window in range(6):
                order = ("cached", "full") if window % 2 == 0 else ("full", "cached")
                validation_order.append(list(order))
                for mode in order:
                    binding._validated_plan = cached_validate if mode == "cached" else revalidate
                    start = time.perf_counter_ns()
                    for _ in range(8):
                        receipt = rt.launch(public_artifact, (a, b, db))
                        if not receipt.get("ok"):
                            raise RuntimeError(receipt)
                    (public_replay if mode == "cached" else full_validation).append(
                        (time.perf_counter_ns() - start) / 1e6 / 8)
                    check(receipt["output"])
        finally:
            binding._validated_plan = cached_validate
        # Warm ordinary JVP remains valid after RuntimeArtifact replay.
        check(fn.native_jvp(a, b, tangents=(db,)))
        assert owner.scratch_stats() == scratch
        # A matching prepare acquires a new native lease. Finish all original
        # owner timing before preparing replay; the old handle is then stale.
        start = time.perf_counter_ns()
        replay = PreparedTensorJVP.from_manifest(json.loads(portable))
        replay_prepare_ms = (time.perf_counter_ns() - start) / 1e6
        check(replay((a, b, db)))
        replay_samples = []
        for _ in range(5):
            start = time.perf_counter_ns()
            for _ in range(8):
                replay_outputs = replay((a, b, db))
            replay_samples.append((time.perf_counter_ns() - start) / 1e6 / 8)
            check(replay_outputs)
        replay_profile = replay.profile(repeats=128, captured=True)
        check(replay_profile["outputs"])
        for actual, old in zip(first, saved, strict=True):
            np.testing.assert_array_equal(actual, old)
        for actual, old in zip((a, b, db), originals, strict=True):
            np.testing.assert_array_equal(actual, old)
        return {
            "storage": storage,
            "shape_mkn": list(shape),
            "correctness": "passed_before_and_after_timing",
            "max_abs_error_primal_tangent": errors,
            "first_compile_prepare_call_ms": first_ms,
            "device_timing_domain": "cuda_graph_batch",
            "device_samples": samples,
            "device_program_median_ms": median(s["program_ms"] for s in samples),
            "stage_medians_ms": [median(s["stage_ms"][i] for s in samples) for i in range(len(owner.members))],
            "completed_public_call_samples_ms": completed,
            "completed_public_call_median_ms": median(completed),
            "completed_runtime_artifact_samples_ms": public_replay,
            "completed_runtime_artifact_median_ms": median(public_replay),
            "revalidated_runtime_artifact_samples_ms": full_validation,
            "revalidated_runtime_artifact_median_ms": median(full_validation),
            "runtime_validation_ab_order": validation_order,
            "runtime_artifact_hash": public_artifact.artifact_hash,
            "runtime_component_receipts": receipt["component_receipts"],
            "replay_prepare_ms": replay_prepare_ms,
            "portable_manifest_bytes": len(portable.encode()),
            "completed_prepared_replay_samples_ms": replay_samples,
            "completed_prepared_replay_median_ms": median(replay_samples),
            "replay_device_profile": {k: v for k, v in replay_profile.items() if k != "outputs"},
            "scratch": scratch,
            "receipt": dict(fn.last_jvp_execution),
            "native_program": owner.plan,
            "member_ir_digests": [
                {
                    "tile": hashlib.sha256(p.tile_ir.encode()).hexdigest(),
                    "target": hashlib.sha256(p.target_ir.encode()).hexdigest(),
                    "backend": hashlib.sha256(p.backend_ir.encode()).hexdigest(),
                    "image": p.image.image_digest,
                    "descriptor": p.descriptor.descriptor_digest,
                }
                for p in owner.members
            ],
        }
    finally:
        if replay is not None:
            replay.close()
        for owner in fn._native_prepared_jvp_calls.values():
            owner.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if rt._nvidia_device_name() != "sm_120":
        raise RuntimeError("requires owning SM120 device")
    device = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True
    ).strip()
    files = [
        "src/transforms/lib/NativeSM120TensorProgram.h",
        "src/transforms/lib/NativeScaledMatmulProgram.h",
        "src/transforms/lib/AutodiffForwardPass.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/matmul_prepared.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.h",
        "python/tessera/compiler/native_sm120_tensor_program.py",
        "python/tessera/compiler/prepared_nvidia_tensor_jvp.py",
        "python/tessera/compiler/portable_nvidia_tensor_jvp.py",
        "python/tessera/compiler/nvidia_tensor_jvp_runtime.py",
        "python/tessera/runtime.py",
        "python/tessera/compiler/jit.py",
        str(Path(__file__).relative_to(Path.cwd())),
    ]
    packet = {
        "schema": "tessera.benchmark.tensor_rhs_jvp.v4",
        "device": device,
        "scope": "named static public RHS-only JVP; no active-source, reverse, dynamic or sibling proof",
        "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
        "binary_sha256": {
            key: hashlib.sha256(Path(os.environ[key]).read_bytes()).hexdigest()
            for key in ("TESSERA_OPT", "TESSERA_NVIDIA_OPT", "TESSERA_NVIDIA_PTX_LAUNCH_LIB")
        },
        "cases": [],
    }
    for storage in ("f16", "bf16"):
        for shape in ((7, 32, 9), (17, 48, 19), (129, 256, 33)):
            row = record(storage, shape)
            packet["cases"].append(row)
            print(
                storage,
                shape,
                "captured_ms",
                row["device_program_median_ms"],
                "public_ms",
                row["completed_public_call_median_ms"],
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
