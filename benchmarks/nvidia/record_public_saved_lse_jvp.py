"""Correctness-gated public native saved-LSE JVP on exact RTX5070."""
from contextlib import ExitStack
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import tessera as ts
from tessera.autodiff import jvp
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_attention_jvp_runtime import prepared, clear_prepared
from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed
from tests.device.nvidia.test_resident_attention_forward import saved, oracle

ROOT = Path(__file__).resolve().parents[2]
PINS = (
    "python/tessera/__init__.py", "python/tessera/apple_gpu_ops_interception.py",
    "python/tessera/autodiff/vjp.py",
    "python/tessera/compiler/frontend_authority.py",
    "python/tessera/compiler/native_gpu_storage.py",
    "python/tessera/compiler/nvidia_native.py",
    "tests/device/nvidia/test_resident_attention_forward.py",
    "python/tessera/autodiff/jvp.py", "python/tessera/compiler/native_public_jvp.py",
    "python/tessera/compiler/jit.py", "python/tessera/compiler/native_jvp_plugins.py",
    "python/tessera/compiler/native_attention_jvp.py",
    "python/tessera/compiler/native_attention_program.py",
    "python/tessera/compiler/native_attention_jvp_artifact.py",
    "python/tessera/compiler/native_attention_jvp_runtime.py",
    "python/tessera/compiler/resident_attention.py",
    "src/compiler/ir/TangentInterface.cpp",
    "src/compiler/programming_model/lib/NativeAttentionJvp.h",
    "src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td",
    "src/compiler/tile_opt_fa4/lib/Dialect/Attn/AttnOps.cpp",
    "src/transforms/lib/AutodiffForwardPass.cpp",
    "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp",
    "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.h",
    "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp",
    "benchmarks/nvidia/record_public_saved_lse_jvp.py",
)


def run(activity, sk, repetitions):
    rng = np.random.default_rng(202610095)
    shapes = {"q": (1, 2, 3, 4), "k": (1, 1, sk, 4),
              "v": (1, 1, sk, 3), "bias": (1, 2, 1, sk)}
    order = ("v", "q", "k", "bias")
    values = {name: rng.normal(0, .2, shape).astype(np.float32) for name, shape in shapes.items()}
    seeds = {name: rng.normal(0, .1, shape).astype(np.float32) if name in activity else None
             for name, shape in shapes.items()}
    h = 1e-4
    plus = {name: value.astype(np.float64) + (h*seeds[name] if seeds[name] is not None else 0)
            for name, value in values.items()}
    minus = {name: value.astype(np.float64) - (h*seeds[name] if seeds[name] is not None else 0)
             for name, value in values.items()}
    primal = oracle(tuple(values[name] for name in ("q", "k", "v")), values["bias"])
    positive = oracle(tuple(plus[name] for name in ("q", "k", "v")), plus["bias"])
    negative = oracle(tuple(minus[name] for name in ("q", "k", "v")), minus["bias"])
    tangent = tuple((upper-lower)/(2*h) for upper, lower in zip(positive, negative, strict=True))
    fn = ts.jit(target="nvidia_sm120")(saved)
    wall = {"host": [], "resident": []}
    device = {"host": [], "resident": []}
    errors = []
    with ExitStack() as stack:
        sessions = {name: stack.enter_context(NvidiaDeviceSession()) for name in shapes}
        roots = {name: Borrowed(sessions[name].upload(value)) for name, value in values.items()}
        resident_seeds = {name: Borrowed(sessions[name].upload(value)) if value is not None else None
                          for name, value in seeds.items()}
        for session in sessions.values():
            assert session.synchronize() == 0
        def call(mode):
            inputs = roots if mode == "resident" else values
            directions = resident_seeds if mode == "resident" else seeds
            return jvp(fn, tuple(inputs[name] for name in order), tuple(directions[name] for name in order))
        def compare(result):
            for actual, expected in zip((*result[0], *result[1]), (*primal, *tangent), strict=True):
                np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=3e-6)
                errors.append(float(np.max(np.abs(actual-expected))))
        for mode in wall:
            compare(call(mode))
        child = next(iter(fn._native_public_jvp_owners.values()))[1]
        package = next(iter(child._native_jvp_packages.values()))
        owner = prepared(package.contract["steps"][0]["child_metadata"])
        handle = owner.handle
        for index in range(repetitions):
            for mode in (("host", "resident") if index % 2 else ("resident", "host")):
                start = time.perf_counter()
                result = call(mode)
                wall[mode].append((time.perf_counter()-start)*1e3)
                assert owner.handle == handle and owner.last_device_ms is not None
                device[mode].append(owner.last_device_ms)
                compare(result)
        compare(call("resident"))
        receipt = dict(fn.last_jvp_execution)
        medians = {mode: statistics.median(samples) for mode, samples in wall.items()}
        row = dict(sk=sk, activity=list(activity), correctness="passed_before_timing",
                   max_abs_error=max(errors), compiler_receipt=receipt,
                   package_digest=package.artifact_hash,
                   public_completed_call_samples_ms=wall,
                   public_completed_call_medians_ms=medians,
                   native_forward_jvp_event_samples_ms=device,
                   native_forward_jvp_event_medians_ms={
                       mode: [statistics.median(pair[index] for pair in samples) for index in range(2)]
                       for mode, samples in device.items()},
                   resident_over_host=medians["resident"]/medians["host"])
        fn.close_native_storage()
        clear_prepared()
        return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=9)
    args = parser.parse_args()
    if args.repetitions < 3:
        raise ValueError("requires at least three alternating timing rounds")
    gpu = subprocess.check_output([
        "/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version",
        "--format=csv,noheader"], text=True).strip()
    if len(gpu.splitlines()) != 1 or "RTX 5070" not in gpu or gpu.split(",")[2].strip() != "12.0":
        raise RuntimeError("requires the owning RTX5070/SM120")
    rows = [run(activity, sk, args.repetitions) for sk in (5, 129)
            for activity in (("v", "q", "k"), ("v",), ("bias",))]
    pins = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in PINS}
    binaries = {name: hashlib.sha256(Path(os.environ[name]).read_bytes()).hexdigest()
                for name in ("TESSERA_OPT", "TESSERA_NVIDIA_OPT", "TESSERA_NVIDIA_PTX_LAUNCH_LIB")}
    packet = dict(schema="tessera.public_saved_lse_jvp.packet.v1", device=gpu,
                  architecture="sm_120", rows=rows, source_sha256=pins, binary_sha256=binaries,
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  scope="public native Graph/Schedule/Tile/LLVM-PTX paired O/LSE/dO/dLSE; no speedup promotion")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2)+"\n")


if __name__ == "__main__":
    main()
