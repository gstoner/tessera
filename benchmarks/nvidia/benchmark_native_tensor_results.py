"""Completed public JVP output-placement cost; no kernel-speedup comparison."""
import argparse
from contextlib import ExitStack
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from benchmarks.nvidia.active_source_jvp_reference import inputs, oracle, product
from benchmarks.rocm.benchmark_gfx1201_fp8_blockscale import paired_completed_host_windows


class Borrowed:
    def __init__(self, value):
        self.value, self.dtype = value, value.dtype

    @property
    def __cuda_array_interface__(self):
        result = dict(self.value.__cuda_array_interface__)
        result["data"] = (result["data"][0], True)
        return result

    def __array__(self, *args, **kwargs):
        raise AssertionError("resident input was coerced to host")


def _device():
    driver = ct.CDLL("libcuda.so.1")
    class UUID(ct.Structure):
        _fields_ = [("bytes", ct.c_ubyte * 16)]
    signatures = {
        "cuCtxGetDevice": [ct.POINTER(ct.c_int)],
        "cuDeviceGetName": [ct.c_char_p, ct.c_int, ct.c_int],
        "cuDeviceGetUuid_v2": [ct.POINTER(UUID), ct.c_int],
        "cuDeviceGetAttribute": [ct.POINTER(ct.c_int), ct.c_int, ct.c_int],
    }
    for name, args in signatures.items():
        fn = getattr(driver, name)
        fn.argtypes, fn.restype = args, ct.c_int
    device, name, uuid, major, minor = ct.c_int(), ct.create_string_buffer(256), UUID(), ct.c_int(), ct.c_int()
    assert driver.cuCtxGetDevice(ct.byref(device)) == 0
    assert driver.cuDeviceGetName(name, 256, device.value) == 0
    assert driver.cuDeviceGetUuid_v2(ct.byref(uuid), device.value) == 0
    assert driver.cuDeviceGetAttribute(ct.byref(major), 75, device.value) == 0
    assert driver.cuDeviceGetAttribute(ct.byref(minor), 76, device.value) == 0
    if (major.value, minor.value) != (12, 0) or "RTX 5070" not in name.value.decode():
        raise RuntimeError("this recorder requires the owning RTX 5070")
    return {"name": name.value.decode(), "uuid_bytes": bytes(uuid.bytes).hex(),
            "ordinal": device.value, "compute_capability": [major.value, minor.value]}


def _check(values, expected, budgets):
    for value, reference, budget in zip(values, expected, budgets, strict=True):
        assert value.dtype == np.float32 and np.isfinite(value).all()
        assert np.all(np.abs(value.astype(np.float64) - reference) <= budget)


def record(output):
    if rt._nvidia_device_name() != "sm_120":
        raise RuntimeError("owning sm_120 required")
    root = Path(__file__).resolve().parents[2]
    paths = [
        "python/tessera/compiler/native_tensor_result.py",
        "python/tessera/compiler/jit.py", "python/tessera/runtime.py",
        "src/compiler/programming_model/lib/PMPasses.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/matmul_prepared.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/resident_tensor_result.inc",
    ]
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    packet = {"schema": "sm120.public_native_tensor_results.v1",
              "scope": "static source-active FP16/BF16 roots; completed public calls; unpooled output generations",
              "provider_sha256": digest(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]),
              "compiler_sha256": digest(os.environ["TESSERA_OPT"]),
              "source_sha256": {path: digest(root / path) for path in paths},
              "rows": [], "rejected_windows": []}

    def checkpoint():
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(packet, indent=2) + "\n")
        temporary.replace(output)

    for storage in ("f16", "bf16"):
        for shape in ((17, 19, 35), (128, 128, 256), (128, 1024, 1024)):
            a, b, da = inputs({"shape_mnk": shape, "storage": storage})
            expected, budgets = oracle(a, b, da)
            fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("a",))(product)
            _check(fn.native_jvp(a, b, tangents=(da,)), expected, budgets)
            owner = next(iter(fn._native_prepared_jvp_calls.values()))
            state = {"host": None, "cuda": None}
            try:
                with ExitStack() as stack:
                    sessions = [stack.enter_context(NvidiaDeviceSession()) for _ in range(3)]
                    values = tuple(Borrowed(session.upload(value)) for session, value in
                                   zip(sessions, (a, b, da), strict=True))
                    live = _device()
                    if "device" in packet:
                        assert live == packet["device"]
                    packet["device"] = live
                    def call(mode):
                        if mode == "cuda" and state[mode] is not None:
                            for value in state[mode]:
                                value.close()
                        state[mode] = fn.native_jvp(
                            *values[:2], tangents=(values[2],), result_placement=mode)
                    def validate(mode):
                        pair = state[mode]
                        actual = pair if mode == "host" else tuple(value.to_host() for value in pair)
                        _check(actual, expected, budgets)
                    for mode in state:
                        call(mode)
                        validate(mode)
                    profiled, events = owner.profile(values, repeats=8, captured=False)
                    _check(profiled, expected, budgets)
                    members = [member["image_digest"] for member in owner._members]
                    for first in ("host", "cuda"):
                        names = [first, "cuda" if first == "host" else "host"]
                        arms = {name: (lambda mode=name: call(mode),
                                       lambda mode=name: validate(mode)) for name in names}
                        for floor in (40., 80., 160.):
                            with patch.object(subprocess, "run", side_effect=AssertionError("warm public call compiled")):
                                timing = paired_completed_host_windows(arms, windows=4, min_window_ms=floor)
                            if all(arm["admissible"] for arm in timing["arms"].values()):
                                break
                            packet["rejected_windows"].append({
                                "storage": storage, "shape_mnk": list(shape),
                                "first": first, "requested_min_ms": floor, "timing": timing})
                            checkpoint()
                        else:
                            raise RuntimeError("timing windows remained too short")
                        packet["rows"].append({
                            "storage": storage, "shape_mnk": list(shape), "first": first,
                            "inputs": "three resident roots; zero root uploads in both arms",
                            "outputs": {"host": "two completed host arrays",
                                        "cuda": "two completed device generations; prior generation release included"},
                            "member_images": members,
                            "independent_numerics": "passed outside timing windows",
                            "device_events_control_ms": events,
                            "device_events_comparison": "one unchanged program; no placement kernel-speedup claim",
                            "public_timing": timing,
                            "cuda_over_host": timing["arms"]["cuda"]["median_ms_per_call"] /
                                              timing["arms"]["host"]["median_ms_per_call"]})
                        checkpoint()
                        print(storage, shape, first, packet["rows"][-1]["cuda_over_host"], flush=True)
            finally:
                if state["cuda"] is not None:
                    for value in state["cuda"]:
                        value.close()
                fn.close_native_storage()
    checkpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    record(parser.parse_args().output)
