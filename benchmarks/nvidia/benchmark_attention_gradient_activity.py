#!/usr/bin/env python3
"""Native requested-gradient pruning: matched SM120 numerical and dispatch proof."""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import numpy as np
from benchmarks.record_device_ring_protocol import Device
from benchmarks.nvidia.benchmark_jit_attention_vjp import function, reference, download
from benchmarks.nvidia.benchmark_attention_argument_order import function as ordered_function, ORDERS
from benchmarks.nvidia.benchmark_checkpoint_bias_gradient import reference as biased_reference
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession


def prepared(device, frame, cotangent, results):
    native = frame._frame
    pointers = [cotangent.ptr, *(buf.pointer.value for buf in native._saved[:4])]
    if native._has_bias:
        pointers.append(native._bias.pointer.value)
    pointers.append(native._saved[4].pointer.value)
    pointers.extend(value.__cuda_array_interface__["data"][0] for value in results)
    args = [ct.c_void_p(p) for p in pointers] + [ct.c_int64(x) for x in native.dims]
    params = (ct.c_void_p*len(args))(*(ct.cast(ct.pointer(x), ct.c_void_p) for x in args))
    total = sum(math.prod(shape) for shape in native.shapes[:3])
    def launch():
        device.check(device.launch(native._functions[1], (total+127)//128, 1, 1,
                                   128, 1, 1, 0, None, params, None))
    resources = {"block_threads": 128, "launch_blocks": (total+127)//128}
    for name, attribute in (("registers", 4), ("local_bytes", 3), ("shared_bytes", 1)):
        value = ct.c_int()
        device.check(device.attribute(ct.byref(value), attribute, native._functions[1]))
        resources[name] = value.value
    active = ct.c_int()
    device.check(device.occupancy(ct.byref(active), native._functions[1], 128, 0))
    resources["active_blocks_per_sm"] = active.value
    return launch, resources, args, params


def paired_timings(device, arms, repetitions):
    start, end = ct.c_void_p(), ct.c_void_p()
    samples = {name: [] for name in arms}
    try:
        device.check(device.event_create(ct.byref(start), 0))
        device.check(device.event_create(ct.byref(end), 0))
        for launch in arms.values():
            for _ in range(3):
                launch()
        device.check(device.sync())
        for trial in range(5):
            order = tuple(arms) if trial % 2 == 0 else tuple(reversed(arms))
            for name in order:
                device.check(device.event_record(start, None))
                for _ in range(repetitions):
                    arms[name]()
                device.check(device.event_record(end, None))
                device.check(device.event_sync(end))
                elapsed = ct.c_float()
                device.check(device.event_elapsed(ct.byref(elapsed), start, end))
                samples[name].append(elapsed.value/repetitions)
        return samples
    finally:
        for event in (start, end):
            if event:
                device.check(device.event_destroy(event))


def run_case(device, shape, causal, wrt, repetitions, directory):
    b,hq,hkv,sq,sk,d,dv = shape
    rng = np.random.default_rng(120_505)
    values = [(rng.normal(size=s)*.3).astype(np.float32) for s in
              ((b,hq,sq,d), (b,hkv,sk,d), (b,hkv,sk,dv))]
    seed = (rng.normal(size=(b,hq,sq,dv))*.3).astype(np.float32)
    output, _, expected = reference(*values, seed, causal)
    compiler = Path(os.environ["TESSERA_OPT"])
    programs = {
        "all_gradients": function(("q","k","v"), causal).compile_native_attention_vjp(
            *values, compiler=compiler),
        "requested_gradients": function(wrt, causal).compile_native_attention_vjp(
            *values, compiler=compiler),
    }
    active = tuple(("q","k","v").index(name) for name in wrt)
    errors = []
    with NvidiaDeviceSession() as session:
        resident = [session.upload(value) for value in values]
        cotangent = session.upload(seed)
        session.synchronize()
        frames, handles, resources, owners = {}, {}, {}, []
        try:
            for name, program in programs.items():
                frame = program.capture(*resident)
                frames[name] = frame
                np.testing.assert_allclose(download(session, frame.primal), output, atol=4e-5, rtol=4e-5)
                actual = frame.backward(cotangent)
                indices = (0,1,2) if name == "all_gradients" else active
                for result, index in zip(actual, indices, strict=True):
                    host = download(session, result)
                    np.testing.assert_allclose(host, expected[index], atol=4e-5, rtol=4e-5)
                    errors.append(float(np.max(np.abs(host-expected[index]))))
                full = frame._frame.backward(cotangent)
                for index, result in enumerate(full):
                    host = download(session, result)
                    if name == "requested_gradients" and index not in active:
                        np.testing.assert_array_equal(host, np.zeros_like(host))
                    else:
                        np.testing.assert_allclose(host, expected[index], atol=4e-5, rtol=4e-5)
                launch, res, args, params = prepared(device, frame, cotangent, full)
                handles[name], resources[name] = launch, res
                owners.append((args, params, full))
            timing = paired_timings(device, handles, repetitions)
            # Recheck outputs after raw timed launches.
            for (name, frame), (_, _, full) in zip(frames.items(), owners, strict=True):
                for index, result in enumerate(full):
                    host = download(session, result)
                    if name == "requested_gradients" and index not in active:
                        np.testing.assert_array_equal(host, np.zeros_like(host))
                    else:
                        np.testing.assert_allclose(host, expected[index], atol=4e-5, rtol=4e-5)
            wall = []
            frame = frames["requested_gradients"]
            for _ in range(5):
                begin = time.perf_counter_ns()
                repeated = frame.backward(cotangent)
                wall.append((time.perf_counter_ns()-begin)/1e6)
                for result, index in zip(repeated, active, strict=True):
                    np.testing.assert_allclose(download(session, result), expected[index], atol=4e-5, rtol=4e-5)
        finally:
            for frame in frames.values():
                frame.close()
    label = "_".join(map(str,shape))+"_"+str(causal)+"_"+"_".join(wrt)
    image_hashes = {}
    for name, program in programs.items():
        package = program.pair.backward
        (directory/(label+"_"+name+".mlir")).write_text(package.tile_ir)
        (directory/(label+"_"+name+".target.mlir")).write_text(package.target_ir)
        (directory/(label+"_"+name+".ptx")).write_bytes(package.image.payload)
        image_hashes[name] = hashlib.sha256(package.image.payload).hexdigest()
    medians = {name: statistics.median(samples) for name, samples in timing.items()}
    return dict(shape=list(shape), causal=causal, wrt=list(wrt),
                gradient_activity=programs["requested_gradients"].pair.backward.descriptor.provenance["gradient_activity"],
                max_abs_gradient_error=max(errors), resources=resources, image_sha256=image_hashes,
                device_dispatch_samples_ms=timing, device_dispatch_medians_ms=medians,
                dispatch_ratio_all_over_requested=medians["all_gradients"]/medians["requested_gradients"],
                checked_backward_wall_samples_ms=wall,
                checked_backward_wall_median_ms=statistics.median(wall))



def nonfinite_value_case(shape, causal, kind):
    b,hq,hkv,sq,sk,d,dv = shape
    rng = np.random.default_rng(120_506)
    values = [(rng.normal(size=s)*.3).astype(np.float32) for s in
              ((b,hq,sq,d), (b,hkv,sk,d), (b,hkv,sk,dv))]
    seed = (rng.normal(size=(b,hq,sq,dv))*.3).astype(np.float32)
    # dV depends on Q/K and dO, so changing primal V must not contaminate it.
    _, _, expected = reference(*values, seed, causal)
    values[2].fill(np.inf if kind == "inf" else np.nan)
    program = function(("v",), causal).compile_native_attention_vjp(
        *values, compiler=Path(os.environ["TESSERA_OPT"]))
    error = 0.0
    with NvidiaDeviceSession() as session:
        resident = [session.upload(value) for value in values]
        with program.capture(*resident) as frame:
            for multiplier in (1,2):
                cotangent = session.upload(seed*multiplier)
                result, = frame.backward(cotangent)
                host = download(session, result)
                assert np.isfinite(host).all()
                np.testing.assert_allclose(host, expected[2]*multiplier, atol=4e-5, rtol=4e-5)
                error = max(error, float(np.max(np.abs(host-expected[2]*multiplier))))
                full = frame._frame.backward(cotangent)
                for inactive in full[:2]:
                    zeros = download(session, inactive)
                    np.testing.assert_array_equal(zeros, np.zeros_like(zeros))
    return dict(shape=list(shape), causal=causal, primal_value_kind=kind,
                max_abs_error=error, activity=[0,0,1], finite_value_gradient=True,
                inactive_gradients="exact_zero", repeated_cotangent_multipliers=[1,2])



def biased_value_case(physical_bias, causal):
    rng = np.random.default_rng(120_507)
    values = [(rng.normal(size=s)*.3).astype(np.float32) for s in
              ((1,2,3,4), (1,1,5,4), (1,1,5,3))]
    bias = (rng.normal(size=physical_bias)*.3).astype(np.float32)
    seed = (rng.normal(size=(1,2,3,3))*.3).astype(np.float32)
    output, _, expected = biased_reference(*values, bias, seed, causal)
    role_values = dict(zip(("q","k","v","bias"), (*values,bias), strict=True))
    order = "k_bias_v_q"
    frontend = [role_values[name] for name in ORDERS[order]]
    program = ordered_function(order, ("v",), causal).compile_native_attention_vjp(
        *frontend, compiler=Path(os.environ["TESSERA_OPT"]))
    assert program.pair.backward.descriptor.provenance["gradient_activity"] == [0,0,1,0]
    with NvidiaDeviceSession() as session:
        resident = [session.upload(value) for value in frontend]
        cotangent = session.upload(seed)
        with program.capture(*resident) as frame:
            np.testing.assert_allclose(download(session,frame.primal), output, atol=4e-5, rtol=4e-5)
            result, = frame.backward(cotangent)
            host = download(session, result)
            np.testing.assert_allclose(host, expected[2], atol=4e-5, rtol=4e-5)
            full = frame._frame.backward(cotangent)
            assert len(full) == 4
            for index in (0,1,3):
                zeros = download(session, full[index])
                np.testing.assert_array_equal(zeros, np.zeros_like(zeros))
    return dict(physical_bias=list(physical_bias), causal=causal,
                frontend_argument_order=list(ORDERS[order]), activity=[0,0,1,0],
                max_abs_error=float(np.max(np.abs(host-expected[2]))),
                inactive_gradients="Q,K,physical dBias exactly zero")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=32)
    args = parser.parse_args()
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    gpu = subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,driver_version", "--format=csv,noheader"], text=True).strip()
    if len(gpu.splitlines()) != 1 or "RTX 5070" not in gpu or gpu.split(",")[2].strip() != "12.0":
        raise RuntimeError("recorder requires the owning RTX 5070 / sm_120")
    device = Device("nvidia")
    directory = args.output.parent/"artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for shape in ((1,2,1,3,5,4,3), (1,2,1,8,129,8,6)):
        for causal in (False, True):
            for wrt in (("q",), ("k",), ("v",), ("v","q"), ("k","q"), ("q","k","v")):
                rows.append(run_case(device, shape, causal, wrt, args.repetitions, directory))
                print("verified", shape, causal, wrt, flush=True)
    nonfinite = [nonfinite_value_case(shape, causal, kind)
        for shape in ((1,2,1,3,5,4,3), (1,2,1,8,129,8,6))
        for causal in (False, True) for kind in ("inf", "nan")]
    biased = [biased_value_case(physical_bias, causal)
        for physical_bias in ((1,2,3,5),(1,2,1,5))
        for causal in (False,True)]
    sources = ("src/transforms/lib/AutodiffPairedPass.cpp",
        "src/compiler/programming_model/lib/NativeCheckpoint.h", "src/compiler/ir/TileOps.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
        "python/tessera/compiler/scheduled_checkpoint.py", "python/tessera/compiler/nvidia_native.py",
        "python/tessera/compiler/native_attention_program.py", "python/tessera/compiler/resident_attention.py")
    packet = dict(device=gpu, architecture="sm_120", rows=rows, nonfinite_value_rows=nonfinite, biased_value_rows=biased, repetitions=args.repetitions,
        timing_scope="alternating balanced preloaded CUDA-event dispatch windows include host dispatch gaps; checked allocating/synchronizing wall measured separately; inactive ABI outputs remain allocated and zero-filled",
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        nvidia_compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_OPT"]).read_bytes()).hexdigest(),
        source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(packet, indent=2)+"\n")
    print(len(rows), "native requested-gradient cases passed")


if __name__ == "__main__":
    main()
