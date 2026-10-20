"""Exact SM120 original FP32-root producer integration."""
import subprocess
import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tests.unit.test_native_tensor_fp32_roots import source, reversed_source

pytestmark = pytest.mark.hardware_nvidia

def roots(storage, shape, seed, reordered):
    m,k,n=shape
    rng=np.random.default_rng(seed)
    dtype={"f16":np.float16,"bf16":ml_dtypes.bfloat16,"f32":np.float32}[storage]
    a=rng.normal(0,.2,(m,k)).astype(np.float32)
    b=rng.normal(0,.2,(k,n)).astype(dtype)
    return (b,a) if reordered else (a,b)

def check(output, values, reordered):
    b,a=values if reordered else (values[1],values[0])
    x=np.maximum(a.astype(np.float64),0)
    expected=x@b.astype(np.float64)
    u=2**-24
    gamma=a.shape[1]*u/(1-a.shape[1]*u)
    bound=gamma*(np.abs(x)@np.abs(b.astype(np.float64)))+np.finfo(np.float32).tiny
    assert output.dtype==np.float32
    assert np.all(np.abs(output.astype(np.float64)-expected)<=bound), (
        np.max(np.abs(output.astype(np.float64)-expected)),np.max(bound))

@pytest.mark.parametrize("storage",["f16","bf16","f32"])
@pytest.mark.parametrize("shape",[(16,32,16),(17,35,19)])
@pytest.mark.parametrize("reordered",[False,True])
def test_public_fp32_root_program_preserves_precision_and_owner(storage,shape,reordered,monkeypatch):
    assert rt._nvidia_device_name()=="sm_120"
    fn=ts.jit(target="nvidia_sm120")(reversed_source if reordered else source)
    values=roots(storage,shape,12041,reordered)
    try:
        result=fn(*values)
        check(result,values,reordered)
        assert fn.execution_kind=="native_gpu"
        program=fn._native_tensor_last_program
        owner=fn._native_tensor_last_owner
        assert program is not None and len(program.members)==2
        assert program.frame["buffers"][1 if reordered else 0]["storage"]=="f32"
        before=owner.stats()
        saved=result.copy()
        def forbidden(*args,**kwargs):
            pytest.fail("warm FP32-root call invoked eager code, tracing, or compiler tools")
        with monkeypatch.context() as guard:
            guard.setattr(fn,"_fn",forbidden)
            guard.setattr(fn,"_trace_frontend_capture",forbidden)
            guard.setattr(subprocess,"run",forbidden)
            guard.setattr(subprocess,"Popen",forbidden)
            changed=roots(storage,shape,12042,reordered)
            copies=tuple(x.copy() for x in changed)
            actual=fn(*changed)
            check(actual,changed,reordered)
            assert fn._native_tensor_last_owner is owner
            assert owner.stats()["allocations"]==before["allocations"]
            np.testing.assert_array_equal(result,saved)
            for value,copy in zip(changed,copies,strict=True):
                np.testing.assert_array_equal(value,copy)
        artifact=rt.RuntimeArtifact.from_json(fn.runtime_artifact().to_json())
        args=dict(zip(artifact.metadata["arg_names"],changed,strict=True))
        receipt=rt.launch(artifact,args)
        assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
        check(receipt["output"],changed,reordered)
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("reordered", [False,True])
def test_fp32_resident_roots_order_independent_pending_producers(storage,reordered,monkeypatch):
    from contextlib import ExitStack
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed, queue_delayed_upload
    assert rt._nvidia_device_name()=="sm_120"
    values=roots(storage,(17,35,19),12043,reordered)
    fn=ts.jit(target="nvidia_sm120")(reversed_source if reordered else source)
    with ExitStack() as stack:
        sessions=[stack.enter_context(NvidiaDeviceSession()) for _ in values]
        buffers=[session.upload(np.zeros_like(value)) for session,value in zip(sessions,values,strict=True)]
        assert all(session.synchronize()==0 for session in sessions)
        resident=[Borrowed(buffer,session.stream) for buffer,session in zip(buffers,sessions,strict=True)]
        try:
            fn(*resident)
            owner=fn._native_tensor_last_owner
            before=owner.stats()
            done=[[],[]]
            callbacks=[queue_delayed_upload(session,buffer,value,session.stream,completed)
                for session,buffer,value,completed in zip(sessions,buffers,values,done,strict=True)]
            assert not any(done)
            def forbidden(*args,**kwargs):
                pytest.fail("resident FP32 roots invoked eager code, tracing, tools or host coercion")
            with monkeypatch.context() as guard:
                guard.setattr(fn,"_fn",forbidden)
                guard.setattr(fn,"_trace_frontend_capture",forbidden)
                guard.setattr(subprocess,"run",forbidden)
                guard.setattr(subprocess,"Popen",forbidden)
                actual=fn(*resident)
            assert all(done) and len(callbacks)==2
            assert fn._native_tensor_last_owner is owner
            assert owner.stats()["allocations"]==before["allocations"]==1
            assert owner.stats()["capacity"]==before["capacity"]
            check(actual,values,reordered)
            for session,buffer,expected in zip(sessions,buffers,values,strict=True):
                np.testing.assert_array_equal(session.download(buffer),expected)
            saved=actual.copy()
        finally:
            fn.close_native_storage()
    np.testing.assert_array_equal(actual,saved)
