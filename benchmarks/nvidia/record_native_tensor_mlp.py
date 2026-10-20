"""Correctness-gated native MLP timings with explicit intermediate precision."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import time

import numpy as np
import tessera as ts
from tessera import runtime as rt
from benchmarks.nvidia.record_native_tensor_jit import identity
from tests.unit.test_native_tensor_mlp_program import default_mlp, mlp, mlp_roots, assert_mlp_numeric
from tests.device.nvidia.test_native_tensor_mlp_program import reference


def record_case(shape, storage, intermediate, rounds, repeats, profile_repeats):
    function = mlp if intermediate == "f16" else default_mlp
    fn = ts.jit(target="nvidia_sm120")(function)
    args = mlp_roots(storage, shape=shape, intermediate=intermediate)
    snapshots = tuple(value.copy() for value in args)
    expected = reference(args, False, intermediate)
    try:
        started = time.perf_counter_ns()
        actual = fn(*args)
        cold = (time.perf_counter_ns() - started) / 1e6
        assert_mlp_numeric(actual, args, False, intermediate)
        if fn.execution_kind != "native_gpu":
            raise RuntimeError("MLP recording requires completed native GPU execution")
        program = fn._native_tensor_last_program
        owner = fn._native_tensor_last_owner
        if program is None or owner is None or len(program.members) != 3:
            raise RuntimeError("MLP recording requires three actual native members")
        before = owner.stats()
        windows = {}
        for captured in (False, True):
            raw = owner.profile(profile_repeats, captured=captured)
            if "output" in raw:
                assert_mlp_numeric(raw["output"], args, False, intermediate)
            windows["captured" if captured else "enqueued"] = {
                key: value for key, value in raw.items() if key != "output"}
        samples = []
        for _ in range(rounds):
            started = time.perf_counter_ns()
            for _ in range(repeats):
                actual = fn(*args)
            samples.append((time.perf_counter_ns() - started) / repeats / 1e6)
        assert_mlp_numeric(actual, args, False, intermediate)
        for original, saved in zip(args, snapshots, strict=True):
            np.testing.assert_array_equal(original, saved)
        after = owner.stats()
        if before["allocations"] != 1 or after["allocations"] != 1:
            raise RuntimeError("retained MLP owner changed allocation count")
        return {
            "shape_mkhn": list(shape), "root_profile": storage,
            "intermediate": intermediate,
            "correctness": "passed_before_after_and_profile",
            "cold_call_ms": cold,
            "completed_public_samples_ms": samples,
            "completed_public_median_ms": median(samples),
            "rounds": rounds, "calls_per_sample": repeats,
            "profile_repeats": profile_repeats, "device_windows": windows,
            "frame": program.frame, "before": before, "after": after,
            "ir_sha256": {name: hashlib.sha256(getattr(program, name).encode()).hexdigest()
                          for name in ("source_ir", "schedule_ir", "tile_ir", "target_ir")},
            "image_sha256": [hashlib.sha256(image).hexdigest() for image in program.images],
            "max_abs_error": float(np.max(np.abs(actual.astype(np.float64) - expected))),
        }
    finally:
        fn.close_native_storage()


def record_standalone_relu(shape, rounds, repeats, profile_repeats):
    """Public host latency and separate resident CUDA-event regions."""
    import ctypes as ct
    from tessera.compiler import nvidia_native, scheduled_kernel
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from tests.unit.test_sm120_public_relu import relu
    from tests.device.nvidia.test_public_standalone_relu import values

    source = values("f32", shape)
    expected = np.maximum(0, source).view(np.uint32)
    fn = ts.jit(target="nvidia_sm120")(relu)
    try:
        started = time.perf_counter_ns()
        actual = fn(source)
        cold = (time.perf_counter_ns() - started) / 1e6
        np.testing.assert_array_equal(actual.view(np.uint32), expected)
        if fn.execution_kind != "native_gpu":
            raise RuntimeError("standalone ReLU requires completed native GPU execution")
        public = []
        for _ in range(rounds):
            started = time.perf_counter_ns()
            for _ in range(repeats):
                actual = fn(source)
            public.append((time.perf_counter_ns() - started) / repeats / 1e6)
        np.testing.assert_array_equal(actual.view(np.uint32), expected)
        # Trace this shape, not the unit helper's fixed 7x32 shape.
        graph = fn._traced_autodiff_module((source,), {})
        lib = rt._load_nvidia_ptx_launch()
        benchmark = lib.tessera_nvidia_ptx_benchmark_resident
        benchmark.argtypes = [ct.c_char_p, ct.POINTER(ct.c_void_p), ct.c_size_t,
            ct.POINTER(ct.c_int64), ct.c_size_t, ct.c_void_p, ct.c_int, ct.c_int,
            ct.POINTER(ct.c_float)]
        benchmark.restype = ct.c_int
        arms = {}
        for schedule in ("serial", "flat_128"):
            started = time.perf_counter_ns()
            artifact = scheduled_kernel.lower_scheduled_kernel(
                graph, target="nvidia_sm120", schedule=schedule)
            package = nvidia_native.package_scheduled_kernel(
                artifact, pipeline_name="tessera-nvidia-pipeline-sm120")
            setup_ms = (time.perf_counter_ns() - started) / 1e6
            runtime = rt.RuntimeArtifact(graph_ir=artifact.graph_ir,
                schedule_ir=artifact.schedule_ir, tile_ir=package.tile_ir,
                target_ir=package.target_ir, native_image=package.image,
                launch_descriptor=package.descriptor)
            with NvidiaDeviceSession() as session:
                x = session.upload(source)
                y = session.empty(source.shape, source.dtype)
                result = rt.launch(runtime, {artifact.input_name: x,
                    artifact.output_name: y, "Rows": shape[0], "Columns": shape[1]},
                    stream=session.stream)
                if not result["ok"]:
                    raise RuntimeError(f"resident ReLU failed: {result}")
                np.testing.assert_array_equal(session.download(y).view(np.uint32), expected)
                pointers = (ct.c_void_p * 2)(x.ptr, y.ptr)
                dims = (ct.c_int64 * 2)(*shape)
                samples = []
                for _ in range(rounds):
                    elapsed = ct.c_float()
                    code = benchmark(package.descriptor.entry_symbol.encode(), pointers, 2,
                        dims, 2, ct.c_void_p(session.stream), 8, profile_repeats,
                        ct.byref(elapsed))
                    if code:
                        raise RuntimeError(f"resident ReLU benchmark failed rc={code}")
                    samples.append(float(elapsed.value))
                np.testing.assert_array_equal(session.download(y).view(np.uint32), expected)
            arms[schedule] = {
                "package_compile_ms": setup_ms, "device_event_samples_ms": samples,
                "device_event_median_ms": median(samples),
                "measurement": "resident CUDA events including possible enqueue gaps",
                "geometry": package.descriptor.geometry.to_dict(),
                "abi_id": package.descriptor.abi_id,
                "entry": package.descriptor.entry_symbol,
                "schedule_digest": artifact.schedule_digest,
                "image_sha256": hashlib.sha256(package.image.payload).hexdigest(),
            }
        return {"shape": list(shape), "storage": "f32",
            "correctness": "bitwise_passed_before_and_after_timing",
            "cold_public_call_ms": cold,
            "completed_public_samples_ms": public,
            "completed_public_median_ms": median(public),
            "public_schedule": "serial", "resident_arms": arms}
    finally:
        fn.close_native_storage()


def record_fp32_root_case(shape, storage, rounds, repeats, profile_repeats):
    """Original FP32 producer roots; no precision-changing baseline."""
    from tests.unit.test_native_tensor_fp32_roots import source
    from tests.device.nvidia.test_native_tensor_fp32_roots import roots, check
    fn = ts.jit(target="nvidia_sm120")(source)
    values = roots(storage, shape, 12044, False)
    saved = tuple(value.copy() for value in values)
    try:
        started = time.perf_counter_ns()
        actual = fn(*values)
        cold = (time.perf_counter_ns() - started) / 1e6
        check(actual, values, False)
        if fn.execution_kind != "native_gpu":
            raise RuntimeError("FP32-root recorder requires native GPU completion")
        program, owner = fn._native_tensor_last_program, fn._native_tensor_last_owner
        if program is None or len(program.members) != 2:
            raise RuntimeError("FP32-root recorder requires both actual source stages")
        before = owner.stats()
        windows = {}
        for captured in (False, True):
            windows["captured" if captured else "enqueued"] = [
                owner.profile(profile_repeats, captured=captured) for _ in range(rounds)]
        samples = []
        for _ in range(rounds):
            started = time.perf_counter_ns()
            for _ in range(repeats):
                actual = fn(*values)
            samples.append((time.perf_counter_ns() - started) / repeats / 1e6)
        check(actual, values, False)
        for value, snapshot in zip(values, saved, strict=True):
            np.testing.assert_array_equal(value, snapshot)
        after = owner.stats()
        if before["allocations"] != after["allocations"] or before["capacity"] != after["capacity"]:
            raise RuntimeError("FP32-root recorder changed retained allocation")
        return {"shape_mkn": list(shape), "root_storage": ["f32", storage],
            "correctness": "independent_fp64_gamma_k_before_after",
            "cold_public_ms": cold, "completed_public_samples_ms": samples,
            "completed_public_median_ms": median(samples),
            "stage_roles": ["fp32_relu_producer", "fp32_scalar_fma_consumer"],
            "device_windows": windows, "profile_repeats": profile_repeats,
            "rounds": rounds, "calls_per_sample": repeats,
            "frame": program.frame, "before": before, "after": after,
            "ir_sha256": {name: hashlib.sha256(getattr(program, name).encode()).hexdigest()
                for name in ("source_ir", "schedule_ir", "tile_ir", "target_ir")},
            "image_sha256": [hashlib.sha256(image).hexdigest() for image in program.images]}
    finally:
        fn.close_native_storage()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--standalone-relu", action="store_true")
    parser.add_argument("--fp32-roots", action="store_true")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--profile-repeats", type=int, default=64)
    parser.add_argument("--shape", nargs=4, type=int, action="append", metavar=("M", "K", "H", "N"))
    args = parser.parse_args()
    if (not 1 <= args.rounds <= 1000 or not 1 <= args.repeats <= 10000
            or not 1 <= args.profile_repeats <= 10000):
        raise ValueError("MLP benchmark counts exceed the bounded recording envelope")
    if args.standalone_relu and args.fp32_roots:
        raise ValueError("standalone ReLU and FP32-root modes are mutually exclusive")
    if args.fp32_roots and args.shape:
        raise ValueError("FP32-root mode uses its fixed MKN sweep")
    if args.standalone_relu and args.shape:
        raise ValueError("--shape describes MLP matrices; standalone ReLU uses its fixed three-shape sweep")
    shapes = args.shape or [(17, 32, 48, 11), (17, 35, 29, 19), (64, 256, 128, 64)]
    if any(any(extent <= 0 for extent in shape) for shape in shapes):
        raise ValueError("MLP benchmark shapes must be positive")
    if rt._nvidia_device_name() != "sm_120":
        raise RuntimeError("MLP benchmark requires owning SM120 execution")
    source = identity()
    source["recorder_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name in ("tests/unit/test_native_tensor_mlp_program.py",
                 "tests/device/nvidia/test_native_tensor_mlp_program.py"):
        source["source_sha256"][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    if args.fp32_roots:
        for name in ("tests/unit/test_native_tensor_fp32_roots.py",
                     "tests/device/nvidia/test_native_tensor_fp32_roots.py",
                     "src/compiler/programming_model/lib/NativeTensorProgram.h"):
            source["source_sha256"][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        packet = {"schema": "tessera.benchmark.native_tensor_fp32_roots.v1",
            "source": source, "status": "recording", "rows": [],
            "scope": "Original FP32 roots with existing scalar-FMA consumers. Captured/enqueued "
                     "producer, consumer and program event windows are independent; do not add "
                     "them or compare them to completed public wall time. No promotion claim."}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for shape in ((17, 35, 19), (64, 256, 64), (256, 256, 128)):
            for storage in ("f16", "bf16", "f32"):
                row = record_fp32_root_case(shape, storage, args.rounds,
                    args.repeats, args.profile_repeats)
                packet["rows"].append(row)
                args.output.write_text(json.dumps(packet, indent=2) + "\n")
                print(storage, shape, row["completed_public_median_ms"], flush=True)
        packet["status"] = "complete"
        args.output.write_text(json.dumps(packet, indent=2) + "\n")
        return
    if args.standalone_relu:
        for name in ("python/tessera/compiler/nvidia_native.py",
                     "python/tessera/compiler/scheduled_kernel.py",
                     "tests/unit/test_sm120_public_relu.py",
                     "tests/device/nvidia/test_public_standalone_relu.py",
                     "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp"):
            source["source_sha256"][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        packet = {"schema": "tessera.benchmark.native_standalone_relu.v1",
            "source": source, "status": "recording", "rows": [],
            "scope": "Public serial host completion and serial/flat resident CUDA-event windows "
                     "are separate domains; no default schedule promotion."}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for shape in ((7, 32), (129, 33), (128, 4096)):
            row = record_standalone_relu(shape, args.rounds, args.repeats, args.profile_repeats)
            packet["rows"].append(row)
            args.output.write_text(json.dumps(packet, indent=2) + "\n")
            print(shape, row["completed_public_median_ms"], flush=True)
        packet["status"] = "complete"
        args.output.write_text(json.dumps(packet, indent=2) + "\n")
        return
    packet = {
        "schema": "tessera.benchmark.native_tensor_mlp.v1",
        "source": source, "status": "recording", "rows": [],
        "scope": "Explicit FP16/FP32 precision profiles; no cross-precision speedup claim. "
                 "Captured/enqueued stage and program windows are independent event regions; "
                 "completed public calls and cold compilation are separate host timings.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for shape in shapes:
        for storage in ("f16", "bf16"):
            for intermediate in ("f16", "f32"):
                row = record_case(tuple(shape), storage, intermediate, args.rounds,
                                  args.repeats, args.profile_repeats)
                packet["rows"].append(row)
                args.output.write_text(json.dumps(packet, indent=2) + "\n")
                print(storage, intermediate, shape, row["completed_public_median_ms"], flush=True)
    packet["status"] = "complete"
    args.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
