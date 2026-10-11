"""RTX5070 public JIT resident roots, ordering, epilogues and retained results."""
from contextlib import ExitStack
import ctypes as ct
import subprocess

import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_native_tensor_dag import (
    public_dag,public_dag_reordered,public_dag_deep,public_dag_fused,oracle)
from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed,queue_delayed_upload

pytestmark=pytest.mark.hardware_nvidia


@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("variant",["ordinary","reordered","deep","fused"])
@pytest.mark.parametrize("streams",["same","different"])
@pytest.mark.parametrize("keywords",[False,True])
def test_public_jit_resident_roots_order_pending_writes(dtype,variant,streams,keywords,monkeypatch):
    if rt._nvidia_device_name()!="sm_120":
        pytest.skip("owning RTX5070 required")
    functions={"ordinary":public_dag,"reordered":public_dag_reordered,
               "deep":public_dag_deep,"fused":public_dag_fused}
    owner=ts.jit(target="nvidia_sm120")(functions[variant]._fn)
    storage=np.float16 if dtype=="fp16" else ml_dtypes.bfloat16
    rng=np.random.default_rng(19327)
    shapes=[(17,32),(32,11)]+([(11,),(17,11)] if variant=="fused" else [])
    dtypes=[storage,storage]+([np.float32,np.float32] if variant=="fused" else [])
    depth=2 if variant=="deep" else 1
    held=[]
    def wanted(values):
        output=oracle(*values[:2],depth)
        if variant=="fused":
            output=np.maximum(output+values[2],0)+values[3]
            output=output.astype(np.float16)
        return output
    with ExitStack() as stack:
        first=stack.enter_context(NvidiaDeviceSession())
        second=first if streams=="same" else stack.enter_context(NvidiaDeviceSession())
        sessions=[first,second]+([first,second] if variant=="fused" else [])
        zeros=[np.zeros(shape,dtype) for shape,dtype in zip(shapes,dtypes,strict=True)]
        buffers=[session.upload(value) for session,value in zip(sessions,zeros,strict=True)]
        for session in set(sessions):
            assert session.synchronize()==0
        roots=[Borrowed(buffer) for buffer in buffers]
        ordered=[roots[1],roots[0]] if variant=="reordered" else roots
        def call():
            return owner(**dict(zip(owner.arg_names,ordered,strict=True))) if keywords else owner(*ordered)
        cold=call()
        np.testing.assert_allclose(cold,wanted(zeros),rtol=.015,atol=.015)
        assert owner.frontend_authority=="tracer"
        program=owner._nvidia_lhs_last_program
        owners=tuple(owner._nvidia_lhs_prepared_calls.values())
        assert len(owners)==1
        scratch=owners[0].scratch_stats()
        held.append((cold,cold.copy()))
        def forbidden(*args,**kwargs):
            pytest.fail("warm public resident call invoked compiler/frontend/eager")
        for _ in range(2):
            values=[rng.normal(0,.2,shape).astype(dtype) for shape,dtype in zip(shapes,dtypes,strict=True)]
            completion=[]
            callbacks=[queue_delayed_upload(first,buffers[0],values[0],first.stream,completion)]
            for session,buffer,value in zip(sessions[1:],buffers[1:],values[1:],strict=True):
                assert session.lib.tessera_nvidia_device_upload(ct.c_void_p(buffer.ptr),
                    ct.c_void_p(value.ctypes.data),value.nbytes,ct.c_void_p(session.stream))==0
            assert not completion
            with monkeypatch.context() as guard:
                guard.setattr(subprocess,"run",forbidden)
                guard.setattr(owner,"_fn",forbidden)
                output=call()
            assert completion and len(callbacks)==1
            np.testing.assert_allclose(output,wanted(values),rtol=.015,atol=.015)
            assert owner._nvidia_lhs_last_program is program
            assert owners[0].scratch_stats()==scratch
            assert all(receipt["native_call_binding"]=="prepared_cpp_ordered_resident_tensor_dag"
                       for receipt in owner._nvidia_lhs_last_receipts)
            held.append((output,output.copy()))
    for value,snapshot in held:
        np.testing.assert_array_equal(value,snapshot)
    for prepared in owner._nvidia_lhs_prepared_calls.values():
        prepared.close()

@pytest.mark.parametrize("field",["bytes","dtype","rank","shape","stride","pointer"])
def test_native_completed_result_rejects_bad_storage_without_writes(field):
    from tessera.compiler.prepared_nvidia_matmul import HostView
    from tessera.compiler.resident_nvidia_tensor import ordered_resident_views
    if rt._nvidia_device_name()!="sm_120":
        pytest.skip("owning RTX5070 required")
    owner=ts.jit(target="nvidia_sm120")(public_dag._fn)
    with NvidiaDeviceSession() as session:
        roots=[Borrowed(session.upload(np.zeros(shape,np.float16))) for shape in ((17,32),(32,11))]
        owner(*roots)
        prepared=next(iter(owner._nvidia_lhs_prepared_calls.values()))
        try:
            output=np.full((17,11),91,np.float32)
            destination=HostView()
            destination.data,destination.bytes,destination.rank=output.ctypes.data,output.nbytes,2
            destination.dtype=1
            destination.shape[:]=output.shape;destination.strides[:]=output.strides
            if field=="bytes":destination.bytes-=4
            elif field=="dtype":destination.dtype=2
            elif field=="rank":destination.rank=1
            elif field=="shape":destination.shape[0]-=1
            elif field=="stride":destination.strides[0]+=4
            else:destination.data=1
            views,streams=ordered_resident_views(roots,None,writable_from=len(roots))
            declared=(ct.c_uint64*len(streams))(*streams)
            fn=prepared.lib.tessera_nvidia_matmul_invoke_dag_resident_to_host_ordered
            fn.argtypes=[ct.c_uint64,ct.POINTER(HostView),ct.c_size_t,
                         ct.POINTER(ct.c_uint64),ct.c_size_t,ct.POINTER(HostView)]
            fn.restype=ct.c_int
            assert fn(prepared.handle,views,len(roots),declared,len(streams),ct.byref(destination))!=0
            np.testing.assert_array_equal(output,np.full_like(output,91))
            np.testing.assert_array_equal(owner(*roots),np.zeros_like(output))
        finally:
            prepared.close()
