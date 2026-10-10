"""Measure public JIT wall time separately from resident native CUDA events."""
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_permuted_matmul_jit import plain, fused, half_fused


def elapsed(call):
    start = time.perf_counter_ns()
    result = call()
    return result, (time.perf_counter_ns() - start) / 1e6


def record():
    device = subprocess.check_output(
        ["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap",
         "--format=csv,noheader"], text=True).strip()
    if len(device.splitlines()) != 1 or device.split(",")[-1].strip() != "12.0":
        raise RuntimeError("one SM120 GPU required")
    rows = []
    for name, template in (("plain", plain), ("fused", fused), ("half_fused", half_fused)):
        for dtype, storage in (("fp16", np.float16), ("bf16", ml_dtypes.bfloat16)):
            for layout in ("row_major", "col_major"):
                fn = ts.jit(target="nvidia_sm120")(template._fn)
                rng = np.random.default_rng(120708)
                a = (rng.normal(size=(17, 35)) * .2).astype(storage)
                b = np.array((rng.normal(size=(35, 19)) * .2).astype(storage),
                             order="C" if layout == "row_major" else "F")
                bias = (rng.normal(size=19) * .1).astype(np.float32)
                residual = (rng.normal(size=(17, 19)) * .05).astype(np.float32)
                args = (b, a) if name == "plain" else (residual, b, a, bias)
                expected = a.astype(np.float64) @ b.astype(np.float64)
                if name != "plain":
                    expected = np.maximum(expected + bias.astype(np.float64), 0) + residual
                if name == "half_fused":
                    expected = expected.astype(np.float16)
                actual, cold = elapsed(lambda: fn(*args))
                np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=4e-5)
                artifact = rt.RuntimeArtifact.from_json(fn.runtime_artifact().to_json())
                descriptor = artifact.launch_descriptor
                assert descriptor.provenance["route"] == "canonical_scheduled_tile_consumer"
                assert descriptor.provenance["b_layout"] == layout
                warm = []
                for _ in range(5):
                    _, ms = elapsed(lambda: fn(*args))
                    warm.append(ms)
                output = next(x for x in descriptor.buffers if x.direction == "output")
                values = dict(zip(artifact.metadata["frontend_input_bindings"], args, strict=True))
                values[output.name] = np.empty_like(actual)
                with NvidiaDeviceSession() as session:
                    bindings = {"M": 17, "N": 19, "K": 35}
                    for item in descriptor.buffers:
                        value = values[item.name]
                        bindings[item.name] = (
                            session.empty(value.shape, value.dtype, layout=item.layout)
                            if item.direction == "output" else session.upload(value, layout=item.layout))
                    device_ms = [
                        rt._nvidia_native_descriptor_resident_device_latency(
                            artifact.native_image, descriptor, bindings, stream=session.stream,
                            reps=200, warmup=20) for _ in range(5)]
                    session.synchronize()
                    resident = session.download(bindings[output.name])
                    np.testing.assert_array_equal(actual, resident)
                rows.append({"mode": name, "dtype": dtype, "rhs_storage_order": layout,
                             "shape_mkn": [17, 35, 19],
                             "warm_native_binding": fn._native_descriptor_last_receipt.get("native_call_binding", "portable_descriptor"),
                             "image_digest": artifact.native_image.image_digest,
                             "cold_public_call_wall_ms": cold,
                             "warm_public_call_wall_samples_ms": warm,
                             "warm_public_call_wall_median_ms": statistics.median(warm),
                             "resident_device_event_samples_ms": device_ms,
                             "resident_device_event_median_ms": statistics.median(device_ms),
                             "max_abs_error": float(np.max(np.abs(actual.astype(np.float64) - expected))),
                             "schedule_digest": descriptor.provenance["schedule_digest"],
                             "tile_ir_digest": descriptor.provenance["tile_ir_digest"],
                             "correctness": "fp64_oracle_and_bitwise_public_resident_parity"})
    sources = ["python/tessera/compiler/jit.py", "python/tessera/compiler/scheduled_matmul.py",
               "src/compiler/programming_model/lib/PMPasses.cpp",
               "src/compiler/programming_model/ir/ScheduleDialect.cpp",
               "tests/device/nvidia/test_permuted_matmul_jit.py",
               "benchmarks/nvidia/benchmark_public_matmul_package.py",
               "python/tessera/compiler/prepared_nvidia_matmul.py",
               "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/matmul_prepared.cpp",
               "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.h",
               "tests/device/nvidia/test_prepared_matmul_owner.py"]
    return {"schema": "tessera.public_matmul_package.v1", "architecture": "sm_120",
            "device": device, "rows": rows,
            "prepared_matmul": os.environ.get("TESSERA_NVIDIA_PREPARED_MATMUL", "1"),
            "native_runtime_sha256": hashlib.sha256(
                Path(rt._load_nvidia_ptx_launch()._name).read_bytes()).hexdigest(),
            "compiler_sha256": {key: hashlib.sha256(Path(os.environ[key]).read_bytes()).hexdigest()
                                for key in ("TESSERA_OPT", "TESSERA_NVIDIA_OPT")},
            "timing_scope": "cold includes tracing/compilation/launch; warm includes host allocation/transfers and launch, plus tracing/portable checks only in the portable control; resident CUDA-event windows include driver dispatch gaps; no speedup claim",
            "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared", choices=("0", "1"), default="1")
    args = parser.parse_args()
    os.environ["TESSERA_NVIDIA_PREPARED_MATMUL"] = args.prepared
    Path(args.output).write_text(json.dumps(record(), indent=2) + "\n")
