"""Matched native ROCm math staging reuse/control with changed-input oracles."""
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
from tests.device.rocm.test_native_math_staging import stats


def record(lib, architecture, kind, shape, storage, samples, iterations):
    import ml_dtypes
    dtype = {"f32": np.float32, "f16": np.float16, "bf16": ml_dtypes.bfloat16}[storage]
    values = [np.full(shape, .5, dtype)]
    if kind == "add":
        values.append(np.full(shape, .75, dtype))
    fn = ts.jit(target="rocm_" + architecture)((FUNCTIONS if storage == "f32" else WIDENING)[kind])
    fn(*values)
    assert fn.execution_kind == "native_gpu"
    artifact = fn.runtime_artifact()
    restored = rt.RuntimeArtifact.from_json(artifact.to_json())
    descriptor = restored.launch_descriptor
    info = descriptor.provenance["native_math"]
    buffers = dict(zip(info["bindings"][:-1], values, strict=True))
    output = next(b.name for b in descriptor.buffers if b.direction == "output")
    buffers[output] = np.empty(shape, np.float32)
    scalars = {"Rows": info["rows"], "Columns": info["columns"]} if kind == "cumsum" else {"N": info["elements"]}
    arms = {name: {"jit_wall_samples_ms": [], "portable_wall_samples_ms": [], "counter_deltas": []}
            for name in ("reuse", "control", "python")}
    for sample in range(samples):
        # Counterbalance order; both arms get identical values at identical addresses.
        order = ("reuse", "control", "python") if sample % 2 else ("python", "control", "reuse")
        for arm in order:
            os.environ["TESSERA_ROCM_NATIVE_MATH"] = "0" if arm == "python" else "1"
            os.environ["TESSERA_ROCM_MATH_STAGING_REUSE"] = "1" if arm == "reuse" else "0"
            fn(*values)  # Warm capacity before measuring, including after control clears.
            before = stats(lib)
            jit_wall, portable_wall = [], []
            for iteration in range(iterations):
                for index, value in enumerate(values):
                    value[:] = np.asarray(.25 + index * .2 + (sample * iterations + iteration) % 13 * .0625, dtype=dtype)
                oracle = expected(kind, [v.astype(np.float32) for v in values])
                start = time.perf_counter_ns()
                actual = fn(*values)
                jit_wall.append((time.perf_counter_ns() - start) / 1e6)
                np.testing.assert_allclose(actual, oracle, rtol=2e-5, atol=2e-5)
                start = time.perf_counter_ns()
                receipt = rt.launch(restored, {"buffers": buffers, "scalars": scalars})
                portable_wall.append((time.perf_counter_ns() - start) / 1e6)
                assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
                np.testing.assert_allclose(buffers[output], oracle, rtol=2e-5, atol=2e-5)
            after = stats(lib)
            delta = [b - a for a, b in zip(before, after, strict=True)]
            allocations = 0 if arm in {"reuse", "python"} else 2 * iterations * (len(values) + 1)
            assert delta[0] == allocations and delta[1] == allocations, (arm, delta)
            assert delta[3] == (0 if arm == "python" else 2 * iterations)
            assert delta[2] == (2 * iterations * (len(values) + 1) if arm == "reuse" else 0)
            arms[arm]["jit_wall_samples_ms"].append(median(jit_wall))
            arms[arm]["portable_wall_samples_ms"].append(median(portable_wall))
            arms[arm]["counter_deltas"].append(delta)
    for arm in arms.values():
        arm["jit_wall_median_ms"] = median(arm["jit_wall_samples_ms"])
        arm["portable_wall_median_ms"] = median(arm["portable_wall_samples_ms"])
    assert fn.runtime_artifact().native_image.image_digest == artifact.native_image.image_digest
    return {"kind": kind, "shape": list(shape), "storage": storage, "output_storage": "f32",
            "image_sha256": artifact.native_image.image_digest, "abi_id": descriptor.abi_id,
            "correctness": "independent f32 oracle after every changed-input JIT and portable launch",
            "arms": arms, "reuse_over_control_jit_wall": arms["reuse"]["jit_wall_median_ms"] / arms["control"]["jit_wall_median_ms"],
            "reuse_over_control_portable_wall": arms["reuse"]["portable_wall_median_ms"] / arms["control"]["portable_wall_median_ms"],
            "reuse_over_python_jit_wall": arms["reuse"]["jit_wall_median_ms"] / arms["python"]["jit_wall_median_ms"],
            "reuse_over_python_portable_wall": arms["reuse"]["portable_wall_median_ms"] / arms["python"]["portable_wall_median_ms"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("gfx1151", "gfx1201"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if args.samples < 3 or args.iterations < 1:
        raise ValueError("invalid sample/iteration count")
    live = rt._rocm_live_arch()
    if live != args.architecture:
        raise RuntimeError(f"architecture mismatch: {live}")
    os.environ["TESSERA_ROCM_NATIVE_MATH"] = "1"
    lib = rt._load_rocm_native_movement_runtime()
    if lib is None or not hasattr(lib, "tessera_rocm_math_launch"):
        raise RuntimeError("matching native math staging library required")
    hip = rt._load_hip_for_launch()
    ordinal, name = C.c_int(), C.create_string_buffer(256)
    if hip.hipGetDevice(C.byref(ordinal)) or hip.hipDeviceGetName(name, len(name), ordinal.value):
        raise RuntimeError("device identity query failed")
    try:
        rows = [record(lib, live, kind, shape, storage, args.samples, args.iterations)
                for storage in ("f32", "f16", "bf16") for shape in ((3, 17), (256, 1024))
                for kind in ("sqrt", "add", "cumsum")]
    finally:
        if lib.tessera_rocm_movement_clear_current():
            raise RuntimeError("native staging cleanup failed")
    from tessera.compiler.scheduled_matmul import find_tessera_opt
    root = Path(__file__).resolve().parents[2]
    sources = ("python/tessera/runtime.py", "python/tessera/compiler/rocm_math_native.py",
               "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_movement_runtime.cpp",
               "benchmarks/rocm/benchmark_native_math_staging.py")
    packet = {"schema": "tessera.rocm.math-staging-ab.v1", "architecture": live,
              "device": name.value.decode(), "device_ordinal": ordinal.value,
              "compiler_binary_sha256": hashlib.sha256(find_tessera_opt().read_bytes()).hexdigest(),
              "native_runtime_sha256": hashlib.sha256(Path(lib._name).read_bytes()).hexdigest(),
              "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
              "samples": args.samples, "iterations": args.iterations,
              "counter_order": ["allocations", "frees", "capacity_reuses", "launches"],
              "timing_policy": "native reuse/control and Python baseline, same image and changed identical inputs; warm host wall includes validation/upload/launch/readback; compiler and oracle excluded; no kernel speedup claim",
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"architecture": live, "rows": len(rows), "correctness": "passed", "allocation_counters": "passed"}))


if __name__ == "__main__":
    main()
