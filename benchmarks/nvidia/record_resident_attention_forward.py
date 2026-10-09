"""Matched native resident/host ordinary attention forward and saved LSE."""
import argparse
from contextlib import ExitStack
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
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed
from tests.device.nvidia.test_resident_attention_forward import plain, biased, saved, oracle

ROOT = Path(__file__).resolve().parents[2]


def run(with_lse, sk, repetitions, dtype="float32", with_bias=False):
    rng = np.random.default_rng(202610094)
    shapes = {"q": (1, 2, 3, 4), "k": (1, 1, sk, 4), "v": (1, 1, sk, 3)}
    storage = ml_dtypes.bfloat16 if dtype == "bfloat16" else np.dtype(dtype)
    values = {name: rng.normal(0, .2, shape).astype(storage) for name, shape in shapes.items()}
    if with_lse or with_bias:
        shapes["bias"] = (1, 2, 1, 1) if with_lse else (1, 2, 3, sk)
        values["bias"] = rng.normal(0, .1, shapes["bias"]).astype(np.float32)
    expected = oracle(tuple(values[name] for name in ("q", "k", "v")), values.get("bias"))
    expected = (expected[0].astype(storage), expected[1])
    fn = ts.jit(target="nvidia_sm120")(saved if with_lse else biased if with_bias else plain)
    samples = {"resident": [], "host": []}
    events = {"resident": [], "host": []}
    errors = []
    with ExitStack() as stack:
        sessions = {name: stack.enter_context(NvidiaDeviceSession()) for name in values}
        roots = {name: Borrowed(sessions[name].upload(value)) for name, value in values.items()}
        for session in sessions.values():
            assert session.synchronize() == 0
        stack.callback(fn.close_native_storage)
        def call(mode):
            return fn(**(roots if mode == "resident" else values))
        def verify(result):
            outputs = result if with_lse else (result,)
            for output, reference in zip(outputs, expected[:len(outputs)], strict=True):
                np.testing.assert_allclose(output, reference, rtol=3e-5, atol=3e-5)
                errors.append(float(np.max(np.abs(output - reference))))
        for mode in samples:
            verify(call(mode))
        owner = next(iter(fn._native_prepared_attention_calls.values()))
        handle = owner.handle
        for iteration in range(repetitions):
            for mode in (("resident", "host") if iteration % 2 else ("host", "resident")):
                start = time.perf_counter()
                result = call(mode)
                samples[mode].append((time.perf_counter() - start) * 1000)
                events[mode].append(owner.last_device_ms)
                assert owner.handle == handle
                verify(result)
        medians = {mode: statistics.median(items) for mode, items in samples.items()}
        descriptor = owner.descriptor
        return {
            "saved_lse": with_lse, "sk": sk, "causal": True,
            "input_storage": dtype, "output_storage": str(np.dtype(storage)), "dense_bias": with_bias,
            "shape": descriptor.provenance["shape"],
            "frontend_order": list(fn.arg_names),
            "correctness": "independent_fp64_oracle_rounded_to_declared_output_storage_before_timing",
            "max_abs_error": max(errors),
            "public_completed_call_samples_ms": samples,
            "public_completed_call_medians_ms": medians,
            "native_forward_event_samples_ms": events,
            "native_forward_event_medians_ms": {mode: statistics.median(items) for mode, items in events.items()},
            "resident_over_host": medians["resident"] / medians["host"],
            "image_digest": owner.image.image_digest,
            "launch_descriptor_digest": descriptor.descriptor_digest,
            "abi_id": descriptor.abi_id, "entry": descriptor.entry_symbol,
            "provenance": descriptor.provenance,
            "native_compiler_stages": [
                {"stage": name, "producer": stage.producer,
                 "input_digest": stage.input_digest, "output_digest": stage.output_digest}
                for name, stage in (
                    ("graph", owner.compiled.bundle.graph), ("schedule", owner.compiled.bundle.schedule),
                    ("tile", owner.compiled.bundle.tile), ("target", owner.compiled.bundle.target_ir),
                    ("backend", owner.compiled.bundle.backend))],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=9)
    args = parser.parse_args()
    if args.repetitions < 3:
        raise ValueError("requires at least three alternating rounds")
    gpu = subprocess.check_output([
        "/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version",
        "--format=csv,noheader"], text=True).strip()
    if len(gpu.splitlines()) != 1 or "RTX 5070" not in gpu or gpu.split(",")[2].strip() != "12.0":
        raise RuntimeError("requires owning RTX5070/SM120")
    files = (
        "python/tessera/compiler/nvidia_native.py", "python/tessera/compiler/scheduled_attention.py",
        "python/tessera/runtime.py", "src/transforms/lib/TileIRLoweringPass.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp",
        "python/tessera/compiler/jit.py", "python/tessera/compiler/prepared_attention_forward.py",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.h",
        "tests/device/nvidia/test_resident_attention_forward.py",
        "benchmarks/nvidia/record_resident_attention_forward.py",
        "src/compiler/programming_model/lib/PMPasses.cpp",
        "src/compiler/ir/TileOps.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
        "tests/unit/test_attention_native_result_storage.py")
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    rows = [run(saved_lse, sk, args.repetitions) for saved_lse in (False, True) for sk in (5, 129)]
    rows += [run(False, sk, args.repetitions, dtype, bias)
             for dtype in ("float16", "bfloat16") for sk in (5, 129) for bias in (False, True)]
    packet = {
        "architecture": "sm120", "device": gpu, "rows": rows,
        "fingerprints": {name: digest(ROOT / name) for name in files},
        "compiler_sha256": digest(os.environ["TESSERA_OPT"]),
        "nvidia_compiler_sha256": digest(os.environ["TESSERA_NVIDIA_OPT"]),
        "native_provider_sha256": digest(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]),
        "timing_scope": "kernel events exclude snapshots and waits; completed public calls include private snapshots and host output/LSE downloads",
        "scope": "static compact fp32 ordinary/saved-LSE forward and fp16/bf16 ordinary forward with matching native result storage and optional dense fp32 bias",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows),
                      "wall_ratios": [row["resident_over_host"] for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
