"""Correctness-gated bounded saved-LSE Graph AD on the exact RTX5070."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

from benchmarks.nvidia.benchmark_jit_attention_vjp import download
from benchmarks.record_device_ring_protocol import Device
from benchmarks.record_jit_attention_program import timed_native_product
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_attention_program import NativeAttentionJVPProgram
from tests.device.nvidia.test_bounded_saved_lse_jvp import function, inputs, reference

ROOT = Path(__file__).resolve().parents[2]
PINS = (
    "python/tessera/runtime.py",
    "python/tessera/compiler/jit.py",
    "python/tessera/compiler/graph_ir.py",
    "python/tessera/compiler/native_attention_program.py",
    "python/tessera/compiler/native_attention_jvp.py",
    "python/tessera/compiler/native_attention_jvp_artifact.py",
    "python/tessera/compiler/native_gpu_storage.py",
    "python/tessera/compiler/native_storage_contract.py",
    "python/tessera/compiler/resident_attention.py",
    "python/tessera/compiler/nvidia_native.py",
    "src/compiler/ir/AttentionADContract.h",
    "src/transforms/lib/NativeAttentionProjection.h",
    "src/transforms/lib/AutodiffForwardPass.cpp",
    "src/transforms/lib/AutodiffPairedPass.cpp",
    "src/compiler/programming_model/lib/NativeAttentionJvp.h",
    "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp",
    "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_gemm.cpp",
    "benchmarks/record_jit_attention_program.py",
    "benchmarks/record_device_ring_protocol.py",
    "benchmarks/nvidia/record_bounded_saved_lse_jvp.py",
    "tests/device/nvidia/test_bounded_saved_lse_jvp.py",
    "tests/device/nvidia/test_resident_attention_forward.py",
)


def compare(session, actual, expected):
    errors = []
    for got, wanted in zip(actual, expected, strict=True):
        host = download(session, got)
        np.testing.assert_allclose(host, wanted, rtol=4e-5, atol=3e-6)
        errors.append(float(np.max(np.abs(host - wanted))))
    return max(errors)


def pointer(value):
    return int(value.__cuda_array_interface__["data"][0])


def record(biased, activity, samples, repetitions):
    fn = function(biased, activity)
    traced, _ = inputs(3, 4, biased, activity)
    begin = time.perf_counter()
    program = fn.compile_native_attention_jvp(
        *traced.values(), compiler=Path(os.environ["TESSERA_OPT"]),
        llvm_bin=Path(os.environ["LLVM_BIN"]), sequence_bounds=(17, 129))
    compile_ms = (time.perf_counter() - begin) * 1e3
    program = NativeAttentionJVPProgram.from_json(
        program.to_json(), expected_digest=program.program_digest)
    rows = []
    with NvidiaDeviceSession() as session:
        device = Device("nvidia", use_current_context=True)
        for sq, sk in ((1, 1), (3, 4), (9, 11), (17, 129), (17, 1), (1, 129)):
            print(json.dumps(dict(sq=sq, sk=sk, biased=biased, activity=activity, phase='capture')), file=sys.stderr, flush=True)
            values, seeds = inputs(sq, sk, biased, activity)
            expected = reference(values, seeds)
            roots = {name: session.upload(value) for name, value in values.items()}
            directions = {name: session.upload(value) for name, value in seeds.items()}
            with program.capture(*roots.values()) as frame:
                tangent = frame.jvp(*(directions[name] for name in activity))
                error = compare(session, (*frame.primal, *tangent), (*expected[0], *expected[1]))
                if activity == ("v",):
                    np.testing.assert_array_equal(download(session, tangent[1]),
                                                  np.zeros((1, 2, sq), np.float32))
                native = frame._frame
                slots = dict(frame._zeros)
                slots.update(zip(program.active,
                                 (directions[name] for name in activity), strict=True))
                raw = [buf.pointer.value for buf in native._saved]
                raw.extend(pointer(slots[i]) for i in range(3))
                if biased:
                    raw.extend((native._bias.pointer.value, pointer(slots[3])))
                raw.extend((pointer(tangent[0]), pointer(tangent[1]), 128, sq, sk))
                print('jvp events', file=sys.stderr, flush=True)
                jvp_time = timed_native_product(device, program.tangent, raw, 2 * sq)
                error = max(error, compare(session, tangent, expected[1]))
                dims = (1, 2, 1, sq, sk, 4, 3)
                scalar_values = dict(zip(("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv"),
                                         dims, strict=True))
                if biased:
                    scalar_values.update(zip(("BiasB", "BiasH", "BiasQ", "BiasK"),
                                              values["bias"].shape, strict=True))
                output = session.empty(expected[0][0].shape, np.float32)
                lse = session.empty(expected[0][1].shape, np.float32)
                forward = program.pair.forward
                buffers = [roots[name] for name in ("q", "k", "v")]
                if biased:
                    buffers.append(roots["bias"])
                fargs = dict(zip((binding.name for binding in forward.descriptor.buffers),
                                 (*buffers, output, lse), strict=True))
                fargs.update(scalar_values)
                print('forward events', file=sys.stderr, flush=True)
                forward_time = [rt._nvidia_native_descriptor_resident_device_latency(
                    forward.image, forward.descriptor, fargs, stream=session.stream,
                    reps=repetitions, warmup=5) for _ in range(samples)]
                error = max(error, compare(session, (output, lse), expected[0]))
                print('checked completion', file=sys.stderr, flush=True)
                checked_jvp = []
                checked_pair = []
                for _ in range(samples):
                    start = time.perf_counter()
                    repeated = frame.jvp(*(directions[name] for name in activity))
                    checked_jvp.append((time.perf_counter() - start) * 1e3)
                    error = max(error, compare(session, repeated, expected[1]))
                    start = time.perf_counter()
                    fresh = program.capture(*roots.values())
                    try:
                        paired = fresh.jvp(*(directions[name] for name in activity))
                        checked_pair.append((time.perf_counter() - start) * 1e3)
                        error = max(error, compare(session, (*fresh.primal, *paired),
                                                   (*expected[0], *expected[1])))
                    finally:
                        fresh.close()
            rows.append(dict(
                shape_b_hq_hkv_sq_sk_d_dv=list(dims), biased=biased, activity=list(activity),
                schema=5, program_digest=program.program_digest,
                forward_image_digest=forward.image.image_digest,
                jvp_binding_digest=program.tangent.binding_digest,
                compile_wall_ms=compile_ms, correctness="passed_before_and_after_timing",
                max_abs_error=error,
                forward_device_event_samples_ms=forward_time,
                forward_device_event_median_ms=statistics.median(forward_time),
                jvp_preloaded_device_dispatch=jvp_time,
                checked_jvp_completed_samples_ms=checked_jvp,
                checked_jvp_completed_median_ms=statistics.median(checked_jvp),
                resident_capture_jvp_completed_samples_ms=checked_pair,
                resident_capture_jvp_completed_median_ms=statistics.median(checked_pair)))
    fn.close_native_storage()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=64)
    args = parser.parse_args()
    if args.samples < 3 or args.repetitions < 3:
        raise ValueError("requires at least three samples and repetitions")
    gpu = subprocess.check_output([
        "/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version",
        "--format=csv,noheader"], text=True).strip()
    fields = gpu.split(",")
    if len(gpu.splitlines()) != 1 or len(fields) != 4 or "RTX 5070" not in fields[0] or fields[2].strip() != "12.0":
        raise RuntimeError("requires the owning RTX5070 / SM120")
    rows = []
    for biased, activity in ((False, ("k", "v", "q")), (False, ("v",)),
                             (True, ("k", "v", "q", "bias")), (True, ("v",)),
                             (True, ("bias",))):
        rows.extend(record(biased, activity, args.samples, args.repetitions))
    packet = dict(
        schema="tessera.bounded_saved_lse_jvp.packet.v1", architecture="sm_120", device=gpu,
        git_base=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        route="Python frontend -> typed Graph AD -> Schedule -> Tile -> LLVM/PTX -> checked CUDA ABI",
        sequence_bounds=[17, 129], rows=rows,
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in PINS},
        binary_sha256={name: hashlib.sha256(Path(os.environ[name]).read_bytes()).hexdigest()
                       for name in ("TESSERA_OPT", "TESSERA_NVIDIA_OPT",
                                    "TESSERA_NVIDIA_PTX_LAUNCH_LIB", "TESSERA_NVIDIA_GEMM_LIB")},
        timing_scope={
            "forward": "preloaded native descriptor CUDA-event dispatch window",
            "jvp": "preloaded raw native CUDA-event dispatch window; actual 128-thread block",
            "checked": "synchronous resident capture/JVP or JVP only; includes allocations and binding, excludes input upload, download and close"},
        claim="named bounded paired O/LSE/dO/dLSE correctness and overhead characterization; no speedup or generic closure")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
