"""Owning SM120 asynchronous saved-LSE snapshots and reverse generations."""
import ctypes as ct
import os
import subprocess
import threading
import numpy as np
import pytest
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_attention_program import NativeAttentionVJPProgram
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_jit_multiresult_attention_vjp import function,values,reference
from benchmarks.nvidia.benchmark_jit_attention_vjp import download

@pytest.mark.parametrize("bias",[False,True])
@pytest.mark.parametrize("compact",[False,True])
@pytest.mark.parametrize("shape",[(1,2,1,3,5,4,3),(2,4,2,7,9,8,6)])
def test_async_attention_pending_seeds_snapshot_and_consumer_lifetime(shape,bias,compact,monkeypatch):
    if not nvidia_cuda_host_ready():pytest.skip("owning SM120 toolchain required")
    inputs,seeds=values(shape,bias,"mixed")
    expected,gradients=reference(inputs,seeds,True)
    program=function(bias,True).compile_native_attention_vjp(
        *inputs,compiler=os.environ["TESSERA_OPT"],compact_gradients=compact)
    program=NativeAttentionVJPProgram.from_json(program.to_json(),expected_digest=program.program_digest)
    with NvidiaDeviceSession() as source,NvidiaDeviceSession() as producer:
        resident=[source.upload(v) for v in inputs]
        ready=[producer.upload(v) for v in seeds]
        pending=[producer.upload(np.zeros_like(v)) for v in seeds]
        source.synchronize();producer.synchronize()
        def no_compiler(*args,**kwargs):raise AssertionError("portable async replay invoked compiler")
        monkeypatch.setattr(subprocess,"run",no_compiler)
        frame=program.capture(*resident,asynchronous=True)
        tape=frame._frame
        held=tuple(frame.primal)
        for value in held:
            assert value.__cuda_array_interface__["stream"]==tape._stream.value
        # Writes queued after capture must wait for its immutable input copies.
        for value in resident:
            changed=np.full(value.shape,19,np.float32)
            assert source.lib.tessera_nvidia_device_upload(ct.c_void_p(value.ptr),ct.c_void_p(changed.ctypes.data),changed.nbytes,ct.c_void_p(source.stream))==0
        frame.wait_on(source.stream);source.synchronize()
        for got,want in zip(held,expected,strict=True):np.testing.assert_allclose(download(source,got),want,rtol=4e-5,atol=4e-5)
        entered,release=threading.Event(),threading.Event()
        callback_type=ct.CFUNCTYPE(None,ct.c_void_p)
        @callback_type
        def hold(_):
            entered.set();release.wait(timeout=5)
        launch=tape._driver.cuLaunchHostFunc
        launch.argtypes=[ct.c_void_p,callback_type,ct.c_void_p];launch.restype=ct.c_int
        copy=tape._driver.cuMemcpyDtoDAsync_v2
        copy.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_void_p];copy.restype=ct.c_int
        timer=None
        original_sync=tape.sync
        def forbidden():raise AssertionError("async backward waited for host completion")
        try:
            tape.check(launch(ct.c_void_p(producer.stream),hold,None))
            assert entered.wait(timeout=2)
            for dst,src in zip(pending,ready,strict=True):
                tape.check(copy(ct.c_void_p(dst.ptr),ct.c_void_p(src.ptr),dst.nbytes,ct.c_void_p(producer.stream)))
            timer=threading.Timer(1,release.set);timer.start()
            tape.sync=forbidden
            first=frame.backward(tuple(pending))
            assert not release.is_set(),"async backward blocked until producer completion"
            assert tape._pending_sources
            tape.sync=original_sync
            frame.wait_on(source.stream)
            release.set();source.synchronize()
            for got,role in zip(first,program.active,strict=True):
                assert got.__cuda_array_interface__["stream"]==tape._stream.value
                np.testing.assert_allclose(download(source,got),gradients[role],rtol=4e-5,atol=4e-5)
            frame.synchronize()
            assert not tape._pending_sources
            changed_seeds=[v*-.7 for v in seeds]
            _,second_expected=reference(inputs,changed_seeds,True)
            second=frame.backward(tuple(producer.upload(v) for v in changed_seeds))
            frame.wait_on(source.stream);source.synchronize()
            for got,role in zip(second,program.active,strict=True):
                np.testing.assert_allclose(download(source,got),second_expected[role],rtol=4e-5,atol=4e-5)
            for got,role in zip(first,program.active,strict=True):
                np.testing.assert_allclose(download(source,got),gradients[role],rtol=4e-5,atol=4e-5)
            with pytest.raises(ValueError,match="consumer stream"):frame.wait_on(0)
            timer.cancel();timer.join();timer=None
            # Close must wait for a registered consumer still reading outputs.
            destination=source.empty(second[0].shape,np.float32)
            consumer_entered,consumer_release=threading.Event(),threading.Event()
            @callback_type
            def hold_consumer(_):
                consumer_entered.set();consumer_release.wait(timeout=5)
            frame.wait_on(source.stream)
            tape.check(launch(ct.c_void_p(source.stream),hold_consumer,None))
            assert consumer_entered.wait(timeout=2)
            tape.check(copy(ct.c_void_p(destination.ptr),
                            ct.c_void_p(second[0].__cuda_array_interface__["data"][0]),
                            destination.nbytes,ct.c_void_p(source.stream)))
            consumer_timer=threading.Timer(.05,consumer_release.set);consumer_timer.start()
            try:
                frame.close()
                assert consumer_release.is_set(),"close released outputs before consumer finished"
                np.testing.assert_allclose(download(source,destination),
                    second_expected[program.active[0]],rtol=4e-5,atol=4e-5)
            finally:
                consumer_release.set();consumer_timer.join()
        finally:
            tape.sync=original_sync;release.set()
            if timer is not None:timer.join()
            frame.close();producer.synchronize()
        for value in (*held,*first,*second):
            with pytest.raises(ValueError,match="closed"):_=value.__cuda_array_interface__
