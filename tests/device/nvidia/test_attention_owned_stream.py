"""Pending external-stream seeds preserve the saved attention generation."""
import ctypes as ct
import os
import threading
import numpy as np
import pytest
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_jit_multiresult_attention_vjp import function,values,reference
from benchmarks.nvidia.benchmark_jit_attention_vjp import download

@pytest.mark.parametrize("bias",[False,True])
def test_pending_producer_seeds_wait_on_private_attention_stream(bias):
    if not nvidia_cuda_host_ready():pytest.skip("owning SM120 host/toolchain required")
    inputs,seeds=values((1,2,1,3,5,4,3),bias,"mixed")
    _,expected=reference(inputs,seeds,True)
    program=function(bias,True).compile_native_attention_vjp(
        *inputs,compiler=os.environ["TESSERA_OPT"],compact_gradients=True)
    with NvidiaDeviceSession() as source, NvidiaDeviceSession() as producer:
        resident=[source.upload(v) for v in inputs]
        ready=[producer.upload(v) for v in seeds]
        pending=[producer.upload(np.zeros_like(v)) for v in seeds]
        producer.synchronize()
        with program.capture(*resident) as frame:
            tape=frame._frame
            assert tape._stream.value not in (source.stream,producer.stream,0,1,2)
            entered,release=threading.Event(),threading.Event()
            callback_type=ct.CFUNCTYPE(None,ct.c_void_p)
            @callback_type
            def hold(_):
                entered.set()
                release.wait(timeout=5)
            launch=tape._driver.cuLaunchHostFunc
            launch.argtypes=[ct.c_void_p,callback_type,ct.c_void_p];launch.restype=ct.c_int
            copy=tape._driver.cuMemcpyDtoDAsync_v2
            copy.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_void_p];copy.restype=ct.c_int
            records,waits=[],[]
            event_record,stream_wait=tape._event_record,tape._stream_wait
            def record(event,stream):
                records.append(stream.value);return event_record(event,stream)
            def wait(stream,event,flags):
                waits.append(stream.value);return stream_wait(stream,event,flags)
            tape._event_record=record;tape._stream_wait=wait
            timer=None
            try:
                tape.check(launch(ct.c_void_p(producer.stream),hold,None))
                assert entered.wait(timeout=2)
                for dst,src in zip(pending,ready,strict=True):
                    tape.check(copy(ct.c_void_p(dst.ptr),ct.c_void_p(src.ptr),dst.nbytes,
                                    ct.c_void_p(producer.stream)))
                assert not release.is_set()
                timer=threading.Timer(.05,release.set);timer.start()
                gradients=frame.backward(tuple(pending))
                for got,role in zip(gradients,program.active,strict=True):
                    np.testing.assert_allclose(download(source,got),expected[role],rtol=4e-5,atol=4e-5)
                assert records==[producer.stream]
                assert waits==[tape._stream.value]
            finally:
                release.set()
                if timer is not None:timer.join()
                producer.synchronize()
