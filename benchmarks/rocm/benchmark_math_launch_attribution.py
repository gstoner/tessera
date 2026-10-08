"""Attribute warm native math host calls; instrumentation is diagnostic only."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time

import numpy as np
import tessera as ts
from tessera import runtime as rt
from benchmarks.rocm.benchmark_native_math_schedule import expected
from tests.device.rocm.test_native_math_package_jit import FUNCTIONS
from tests.device.rocm.test_native_math_widening import FUNCTIONS as WIDENING


class Calls:
    def __init__(self, library):
        self.library = library
        self.calls = {}

    def __getattr__(self, name):
        fn = getattr(self.library, name)
        if not callable(fn):
            return fn

        def invoke(*args):
            start = time.perf_counter_ns()
            try:
                return fn(*args)
            finally:
                row = self.calls.setdefault(name, {"count": 0, "elapsed_ns": 0})
                row["count"] += 1
                row["elapsed_ns"] += time.perf_counter_ns() - start
        return invoke


def record(architecture, kind, shape, storage, repetitions):
    import ml_dtypes
    dtype = {"f32": np.float32, "f16": np.float16, "bf16": ml_dtypes.bfloat16}[storage]
    rng = np.random.default_rng(606)
    values = [rng.uniform(.25, 1.5, shape).astype(dtype)]
    if kind == "add":
        values.append(rng.uniform(.5, 1.5, shape).astype(dtype))
    fn = ts.jit(target="rocm_" + architecture)((FUNCTIONS if storage == "f32" else WIDENING)[kind])

    def oracle():
        return expected(kind, [x.astype(np.float32) for x in values])

    np.testing.assert_allclose(fn(*values), oracle(), rtol=2e-5, atol=2e-5)
    assert fn.execution_kind == "native_gpu"
    artifact = fn.runtime_artifact()
    restored = rt.RuntimeArtifact.from_json(artifact.to_json())
    descriptor = restored.launch_descriptor
    info = descriptor.provenance["native_math"]
    buffers = dict(zip(info["bindings"][:-1], values, strict=True))
    output = next(b.name for b in descriptor.buffers if b.direction == "output")
    buffers[output] = np.empty(shape, np.float32)
    scalars = {"Rows": info["rows"], "Columns": info["columns"]} if kind == "cumsum" else {"N": info["elements"]}

    def launch(iteration):
        # Every input changes at the same address, including the binary RHS.
        for index, value in enumerate(values):
            value[:] = rng.uniform(.25 + index * .1, 1.5, shape).astype(dtype)
        start = time.perf_counter_ns()
        receipt = rt.launch(restored, {"buffers": buffers, "scalars": scalars})
        elapsed = time.perf_counter_ns() - start
        assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
        np.testing.assert_allclose(buffers[output], oracle(), rtol=2e-5, atol=2e-5)
        return elapsed

    # Uninstrumented measurements are kept separate from wrapper diagnostics.
    walls = [launch(i) / 1e6 for i in range(repetitions)]
    load_hip, load_image = rt._load_hip_for_launch, rt._load_rocm_native_image_runtime
    hip = Calls(load_hip())
    image_library = load_image()
    if image_library is None:
        raise RuntimeError("native image cache is required for this attribution envelope")
    images = Calls(image_library)
    rt._load_hip_for_launch = lambda: hip
    rt._load_rocm_native_image_runtime = lambda: images
    try:
        instrumented = [launch(i) / 1e6 for i in range(repetitions)]
    finally:
        rt._load_hip_for_launch, rt._load_rocm_native_image_runtime = load_hip, load_image
    for name, per_call in (("hipMalloc", len(values) + 1), ("hipFree", len(values) + 1),
                           ("hipMemcpy", len(values) + 1), ("hipModuleLaunchKernel", 1)):
        assert hip.calls[name]["count"] == repetitions * per_call, (name, hip.calls)
    for name in ("tessera_rocm_image_acquire", "tessera_rocm_image_release"):
        assert images.calls[name]["count"] == repetitions, (name, images.calls)
    return {"kind": kind, "storage": storage, "output_storage": "f32", "shape": list(shape),
            "image_sha256": artifact.native_image.image_digest, "abi_id": descriptor.abi_id,
            "correctness": "independent f32 oracle before and after every changed-input launch",
            "repetitions": repetitions, "uninstrumented_portable_wall_samples_ms": walls,
            "uninstrumented_portable_wall_median_ms": median(walls),
            "instrumented_wall_samples_ms": instrumented,
            "hip_calls": hip.calls, "native_image_calls": images.calls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("gfx1151", "gfx1201"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=9)
    args = parser.parse_args()
    if args.repetitions < 3:
        raise ValueError("at least three repetitions required")
    live = rt._rocm_live_arch()
    if live != args.architecture:
        raise RuntimeError(f"architecture mismatch: expected {args.architecture}, got {live}")
    hip = rt._load_hip_for_launch()
    ordinal, name = C.c_int(), C.create_string_buffer(256)
    if hip.hipGetDevice(C.byref(ordinal)) or hip.hipDeviceGetName(name, len(name), ordinal.value):
        raise RuntimeError("device identity query failed")
    previous = os.environ.get("TESSERA_ROCM_NATIVE_MATH")
    os.environ["TESSERA_ROCM_NATIVE_MATH"] = "0"
    try:
        rows = [record(live, kind, shape, storage, args.repetitions)
                for storage in ("f32", "f16", "bf16") for shape in ((3, 17), (256, 1024))
                for kind in ("sqrt", "add", "cumsum")]
    finally:
        if previous is None:
            os.environ.pop("TESSERA_ROCM_NATIVE_MATH", None)
        else:
            os.environ["TESSERA_ROCM_NATIVE_MATH"] = previous
    from tessera.compiler.scheduled_matmul import find_tessera_opt
    root = Path(__file__).resolve().parents[2]
    sources = ("python/tessera/runtime.py", "python/tessera/compiler/rocm_math_native.py",
               "benchmarks/rocm/benchmark_math_launch_attribution.py")
    packet = {"schema": "tessera.rocm.math-launch-attribution.v1", "architecture": live,
              "device": name.value.decode(), "device_ordinal": ordinal.value,
              "compiler_binary_sha256": hashlib.sha256(find_tessera_opt().read_bytes()).hexdigest(),
              "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
              "scope": "unreused Python baseline: static compact unary/binary/scan packages; all inputs updated every call",
              "timing_policy": "wrapper durations include Python instrumentation overhead; diagnostic attribution only, no kernel or speedup claim",
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"architecture": live, "rows": len(rows), "correctness": "passed", "call_counts": "passed"}))


if __name__ == "__main__":
    main()
