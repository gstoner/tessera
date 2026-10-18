"""Real CUDA event failure injection: native result pins drain enqueued consumers."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.hardware_nvidia

_SHIM = r"""
#include <cuda.h>
#include <dlfcn.h>
#include <cstdlib>
extern "C" CUresult cuEventCreate(CUevent *event, unsigned flags) {
  if (std::getenv("TESSERA_TEST_RESULT_EVENT_CREATE")) return CUDA_ERROR_OUT_OF_MEMORY;
  static auto real = reinterpret_cast<decltype(&cuEventCreate)>(dlsym(RTLD_NEXT, "cuEventCreate"));
  return real(event, flags);
}
extern "C" CUresult cuEventRecord(CUevent event, CUstream stream) {
  if (std::getenv("TESSERA_TEST_RESULT_EVENT_RECORD")) return CUDA_ERROR_OUT_OF_MEMORY;
  static auto real = reinterpret_cast<decltype(&cuEventRecord)>(dlsym(RTLD_NEXT, "cuEventRecord"));
  return real(event, stream);
}
extern "C" CUresult cuStreamWaitEvent(CUstream stream, CUevent event, unsigned flags) {
  if (std::getenv("TESSERA_TEST_RESULT_SKIP_WAIT")) return CUDA_SUCCESS;
  static auto real = reinterpret_cast<decltype(&cuStreamWaitEvent)>(dlsym(RTLD_NEXT, "cuStreamWaitEvent"));
  return real(stream, event, flags);
}
"""


@pytest.fixture(scope="module")
def event_fault_shim(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-result-event-fault")
    source = directory / "event_fault.cpp"
    source.write_text(_SHIM)
    library = directory / "event_fault.so"
    cuda = Path(os.environ["CUDA_HOME"])
    subprocess.run(["g++", "-shared", "-fPIC", "-std=c++17",
                    "-I" + str(cuda / "include"), str(source), "-ldl", "-o", str(library)],
                   check=True, capture_output=True, text=True)
    return library


@pytest.mark.parametrize("fault", ["CREATE", "RECORD"])
def test_event_retention_failure_drains_pending_consumer(event_fault_shim, fault):
    env = dict(os.environ, LD_PRELOAD=str(event_fault_shim))
    code = ("from tests.device.nvidia.test_native_tensor_result_faults import _exercise; "
            f"_exercise({fault!r})")
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def _exercise(fault):
    import ctypes as ct
    import gc
    import time
    import numpy as np
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from tests.device.nvidia.test_native_tensor_result import _result_pair
    fn, pair = _result_pair()
    expected = pair[0].to_host()
    pointer = pair[0].__cuda_array_interface__["data"][0]
    fn.close_native_storage()
    driver = ct.CDLL("libcuda.so.1")
    driver.cuLaunchHostFunc.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p]
    driver.cuLaunchHostFunc.restype = ct.c_int
    driver.cuMemcpyDtoDAsync_v2.argtypes = [ct.c_uint64, ct.c_uint64, ct.c_size_t, ct.c_void_p]
    driver.cuMemcpyDtoDAsync_v2.restype = ct.c_int
    completed = []
    @ct.CFUNCTYPE(None, ct.c_void_p)
    def delay(_):
        time.sleep(.1)
        completed.append(True)
    with NvidiaDeviceSession() as consumer:
        output = consumer.empty(expected.shape, expected.dtype)
        assert driver.cuLaunchHostFunc(ct.c_void_p(consumer.stream), ct.cast(delay, ct.c_void_p), None) == 0
        assert driver.cuMemcpyDtoDAsync_v2(
            output.ptr, pointer, expected.nbytes, ct.c_void_p(consumer.stream)) == 0
        assert not completed
        key = "TESSERA_TEST_RESULT_EVENT_" + fault
        os.environ[key] = "1"
        try:
            with pytest.raises(RuntimeError, match="resident result consumer"):
                pair[0].record_consumer(consumer.stream)
        finally:
            del os.environ[key]
        # This observation precedes any result free, driver download or stream
        # sync, so allocation/free implicit synchronization cannot rescue it.
        assert completed, "failed event retention returned with an unpinned consumer"
        for value in pair:
            value.close()
        gc.collect()
        np.testing.assert_array_equal(output.numpy(), expected)


def test_removing_only_native_stream_waits_breaks_independent_numerics(event_fault_shim):
    env = dict(os.environ, LD_PRELOAD=str(event_fault_shim),
               TESSERA_TEST_RESULT_SKIP_WAIT="1")
    code = """
import numpy as np
from tests.device.nvidia.test_native_tensor_result_ordering import pending_frame
actual, expected, budgets = pending_frame("f16")
assert all(np.isfinite(value).all() for value in actual)
assert any(np.any(np.abs(value.astype(np.float64)-ref)>budget)
           for value,ref,budget in zip(actual,expected,budgets,strict=True)), "negative control did not detect omitted native producer waits"
"""
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_low_stream_tokens_refuse_before_driver_and_preserve_generation():
    code = """
import resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import numpy as np
import pytest
from tests.device.nvidia.test_native_tensor_result import _result_pair
fn,pair = _result_pair()
saved = pair[0].to_host()
lib,handle = pair[0]._batch.lib,pair[0]._batch.handle
for token in (3,1024,4095):
    with pytest.raises(RuntimeError,match="consumer stream token is invalid"):
        pair[0].record_consumer(token)
    assert lib.tessera_nvidia_tensor_result_record_consumer(handle,token) != 0
    assert "consumer stream token is invalid" in lib.tessera_nvidia_matmul_last_error().decode()
    np.testing.assert_array_equal(pair[0].to_host(),saved)
for token in (0,1,2):
    pair[0].record_consumer(token)
fn.close_native_storage()
for value in pair: value.close()
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("form", ["wrapped", "user_dict", "proxy_groups", "dict_subclass"])
def test_wrapped_descriptor_inputs_retain_consumer_completion(event_fault_shim, form):
    env = dict(os.environ, LD_PRELOAD=str(event_fault_shim))
    code = ("from tests.device.nvidia.test_native_tensor_result_faults import _exercise_wrapped_launch; "
            f"_exercise_wrapped_launch({form!r})")
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def _exercise_wrapped_launch(form):
    from collections import UserDict
    from types import MappingProxyType
    from dataclasses import replace
    import ctypes as ct
    import time
    import numpy as np
    from tessera import runtime as rt
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from tests.device.nvidia.test_cooperative_norm import package
    from tests.device.nvidia.test_native_tensor_result import _result_pair
    fn, pair = _result_pair()
    value = pair[0]
    saved = value.to_host()
    fn.close_native_storage()
    _, native = package("rmsnorm", saved, "serial")
    descriptor = native.descriptor
    artifact = rt.RuntimeArtifact(metadata={"target": "nvidia_sm120"},
        tile_ir=native.tile_ir, target_ir=native.target_ir,
        native_image=native.image, launch_descriptor=descriptor)
    stream = value.__cuda_array_interface__["stream"]
    driver = ct.CDLL("libcuda.so.1")
    driver.cuLaunchHostFunc.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p]
    driver.cuLaunchHostFunc.restype = ct.c_int
    driver.cuStreamSynchronize.argtypes = [ct.c_void_p]
    driver.cuStreamSynchronize.restype = ct.c_int
    completed = []
    @ct.CFUNCTYPE(None, ct.c_void_p)
    def delay(_):
        time.sleep(.1)
        completed.append(True)
    with NvidiaDeviceSession() as session:
        output = session.empty(saved.shape, np.float32)
        class Destination:
            @property
            def __cuda_array_interface__(self):
                interface = dict(output.__cuda_array_interface__)
                interface["stream"] = stream
                interface["data"] = (interface["data"][0], False)
                return interface
        destination = rt.NativeBufferValue(Destination(), rt._native_buffer_value(output)[1])
        source = rt.NativeBufferValue(value, rt._native_buffer_value(value)[1])
        scalars = {"Rows": saved.shape[0], "Columns": saved.shape[1]}
        buffers = {descriptor.buffers[0].name: source, descriptor.buffers[1].name: destination}
        flat = {**buffers, **scalars}
        # Complete module/event/allocation warmup before the delayed consumer.
        assert rt.launch(artifact, flat, stream=stream)["ok"]
        assert driver.cuStreamSynchronize(ct.c_void_p(stream)) == 0
        if form == "wrapped":
            args = flat
        elif form == "user_dict":
            args = UserDict({**flat, descriptor.buffers[0].name: value})
        elif form == "proxy_groups":
            args = MappingProxyType({"buffers": MappingProxyType(buffers), "scalars": UserDict(scalars)})
        else:
            class Arguments(dict):
                pass
            args = Arguments({**flat, descriptor.buffers[0].name: value})
        original = rt._native_launchers["nvidia_sm120"]
        key = "TESSERA_TEST_RESULT_EVENT_RECORD"
        def submit(*args, **kwargs):
            result = original.submit(*args, **kwargs)
            # Inject only after the real kernel has been submitted, so the
            # missing-generation bug cannot be hidden by allocation fences.
            os.environ[key] = "1"
            return result
        rt._native_launchers["nvidia_sm120"] = replace(original, submit=submit)
        try:
            assert driver.cuLaunchHostFunc(ct.c_void_p(stream), ct.cast(delay, ct.c_void_p), None) == 0
            assert not completed
            receipt = rt.launch(artifact, args, stream=stream)
            assert not receipt["ok"] and receipt["runtime_status"] == "device_error", receipt
            assert "record resident result consumer" in receipt["reason"]
            assert completed, "consumer returned without retained completion"
            x = saved.astype(np.float64)
            expected = x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + 1e-5)
            np.testing.assert_allclose(output.numpy(), expected, rtol=3e-6, atol=3e-6)
        finally:
            os.environ.pop(key, None)
            rt._native_launchers["nvidia_sm120"] = original
            assert driver.cuStreamSynchronize(ct.c_void_p(stream)) == 0
            for result in pair:
                result.close()
