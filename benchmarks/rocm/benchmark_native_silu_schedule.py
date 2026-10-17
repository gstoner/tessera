"""Correctness-gated native SiLU characterization on the owning ROCm GPU."""
import argparse
import ctypes as C
import hashlib
import json
import os
import subprocess
from pathlib import Path
from statistics import median
import time

import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from benchmarks.rocm.benchmark_native_math_schedule import MathCase


def silu(a, b):
    return ts.ops.silu_mul(a, b)


def wide(a, b):
    return ts.ops.silu_mul(ts.ops.cast(a, "fp32"), ts.ops.cast(b, "fp32"))


def record(hip, storage, widen, shape, samples, iterations, legacy_first):
    dtype = {"f32": np.float32, "f16": np.float16, "bf16": ml_dtypes.bfloat16}[storage]
    rng = np.random.default_rng(1610)
    arrays = [rng.uniform(-6, 6, shape).astype(dtype) for _ in range(2)]
    a, b = [v.astype(np.float64) for v in arrays]
    oracle = a / (1 + np.exp(-a)) * b
    output_dtype = np.float32 if widen or storage == "f32" else dtype
    if output_dtype == np.float32:
        budget = 3e-6 * np.abs(oracle) + 2e-7
    else:
        limit = np.finfo(np.float16) if storage == "f16" else ml_dtypes.finfo(ml_dtypes.bfloat16)
        budget = float(limit.eps) * np.maximum(np.abs(oracle), float(limit.tiny))
        budget += 8 * np.finfo(np.float32).eps * np.abs(oracle)

    def check(actual):
        assert actual.dtype == np.dtype(output_dtype)
        assert np.all(np.abs(actual.astype(np.float64) - oracle) <= budget)

    fn = ts.jit(target="rocm_gfx1201")(wide if widen else silu)
    start = time.perf_counter_ns()
    actual = fn(*arrays)
    cold_ms = (time.perf_counter_ns() - start) / 1e6
    check(actual)
    assert fn.execution_kind == "native_gpu"
    artifact = fn.runtime_artifact()
    image, descriptor = artifact.native_image, artifact.launch_descriptor
    assert artifact.metadata["compiler_path"] == "rocm_math_native_descriptor"
    images = [("native", image.payload, descriptor.entry_symbol)]
    if not widen:
        images.append(("legacy", rt._build_compiled_silu_mul_hsaco(storage), "sm"))
    if legacy_first:
        images.reverse()
    measurements = {}
    for label, payload, entry in images:
        case = MathCase(hip, payload, entry, arrays, False, output_dtype)
        try:
            check(case.download())
            values = case.measure(trials=samples, iterations=iterations, warmup=5)
            check(case.download())
        finally:
            case.close()
        measurements[label] = {"image_sha256": hashlib.sha256(payload).hexdigest(),
                               "device_event_samples_ms": values,
                               "device_event_median_ms": median(values)}
    device = measurements["native"]["device_event_samples_ms"]
    legacy = measurements.get("legacy")
    wall = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        actual = fn(*arrays)
        wall.append((time.perf_counter_ns() - start) / 1e6)
        check(actual)
    return {"shape": list(shape), "storage": storage, "explicit_widen": widen,
            "output_dtype": descriptor.buffers[-1].dtype, "abi_id": descriptor.abi_id,
            "image_sha256": image.image_digest,
            "ir_sha256": {k: hashlib.sha256(getattr(artifact, k).encode()).hexdigest()
                          for k in ("graph_ir", "schedule_ir", "tile_ir", "target_ir")},
            "correctness": "FP64 oracle checked before/after events and after every public call",
            "max_abs_error": float(np.max(np.abs(actual.astype(np.float64)-oracle))),
            "cold_public_call_ms": cold_ms, "warm_public_call_samples_ms": wall,
            "warm_public_call_median_ms": median(wall),
            "device_event_samples_ms": device, "device_event_median_ms": median(device),
            "legacy_target_only_same_store": legacy}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--legacy-first", action="store_true")
    args = parser.parse_args()
    if args.samples < 3 or args.iterations < 1:
        raise ValueError("requires at least three samples and positive iterations")
    arch = rt._rocm_live_arch()
    if arch != "gfx1201":
        raise RuntimeError(f"requires owning gfx1201, got {arch}")
    hip = rt._load_hip_for_launch()
    MathCase.check(hip.hipInit(0), "init")
    ordinal = C.c_int()
    name = C.create_string_buffer(256)
    MathCase.check(hip.hipGetDevice(C.byref(ordinal)), "ordinal")
    MathCase.check(hip.hipDeviceGetName(name, len(name), ordinal.value), "device name")
    rows = [record(hip, storage, widen, shape, args.samples, args.iterations, args.legacy_first)
            for shape in ((3, 17), (256, 1024), (256, 12288))
            for storage, widen in (("f32", False), ("f16", False), ("bf16", False),
                                   ("f16", True), ("bf16", True))]
    uuid = C.create_string_buffer(16)
    hip.hipDeviceGetUuid.argtypes = [C.c_void_p, C.c_int]
    hip.hipDeviceGetUuid.restype = C.c_int
    MathCase.check(hip.hipDeviceGetUuid(uuid, ordinal.value), "device UUID")
    root = Path(__file__).resolve().parents[2]
    sources = [Path(p) for p in subprocess.check_output(
        ["git", "diff", "--name-only", "154f8549620839ae97876e1f2210500a36239524"], text=True).splitlines()]
    sources.append(Path(__file__).relative_to(root))
    sources = sorted(set(p for p in sources if (root/p).is_file()))
    packet = {"schema": "tessera.rocm.native-silu.v1", "architecture": arch,
              "device": name.value.decode(), "device_ordinal": ordinal.value,
              "opaque_hip_uuid": uuid.raw.hex(), "legacy_first": args.legacy_first,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "compiler_sha256": hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
              "source_sha256": {str(p): hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
              "runtime_sha256": hashlib.sha256(Path(os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]).read_bytes()).hexdigest(),
              "samples": args.samples, "iterations": args.iterations, "rows": rows,
              "timing_domains": {"device": "HIP events over resident launches; excludes transfers and allocation; may include host submission gaps",
                                 "warm_public": "completed @jit call including validation, staging, transfers, launch and readback; compiler cached",
                                 "cold_public": "first public call in this process including compilation/cache lookup and execution"},
              "scope": "isolated activation; no full MLP/model-quality or sibling-architecture claim"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, allow_nan=False)+"\n")
    print(json.dumps([{k: row[k] for k in ("shape", "storage", "explicit_widen", "device_event_median_ms", "warm_public_call_median_ms")} for row in rows], indent=2))


if __name__ == "__main__":
    main()
