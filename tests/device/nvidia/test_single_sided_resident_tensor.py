"""Ordered single-sided resident producer -> raw RHS matmul proof."""
from contextlib import ExitStack
import ctypes as ct
import subprocess

import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_lhs_tensor_jit import (
    rms_lhs,layer_lhs,softmax_lhs,rms_lhs_fused,layer_lhs_fused,
    softmax_lhs_fused,layer_lhs_reordered,_oracle,_storage)
from tests.device.nvidia.test_bounded_producer_chain import two,three,three_fused,oracle as chain_oracle
from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed,queue_delayed_upload

pytestmark=pytest.mark.hardware_nvidia
VARIANTS=("rmsnorm","layernorm","softmax","rmsnorm_fused","layernorm_fused","softmax_fused",
          "two","three","three_fused","reordered")
BOUNDS={"M":65,"N":33,"K":513}


def function(variant,bounded):
    bodies={"rmsnorm":rms_lhs._fn,"layernorm":layer_lhs._fn,"softmax":softmax_lhs._fn,
            "rmsnorm_fused":rms_lhs_fused._fn,"layernorm_fused":layer_lhs_fused._fn,
            "softmax_fused":softmax_lhs_fused._fn,"two":two,"three":three,
            "three_fused":three_fused,"reordered":layer_lhs_reordered._fn}
    return ts.jit(target="nvidia_sm120",**({"shape_bounds":BOUNDS} if bounded else {}))(bodies[variant])


def oracle(values,variant):
    if variant in {"two","three","three_fused"}:
        return chain_oracle(values,2 if variant=="two" else 3)
    kind="layernorm" if variant=="reordered" else variant.removesuffix("_fused")
    return _oracle(*values[:2],kind,*(values[2:] if len(values)==4 else (None,None)))


def ordered(values,variant):
    return [values[3],values[1],values[0],values[2]] if variant=="reordered" else values


@pytest.mark.parametrize("variant",VARIANTS)
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("bounded",[False,True])
def test_single_sided_public_resident_chain_and_raw_rhs(variant,dtype,bounded,monkeypatch):
    if rt._nvidia_device_name()!="sm_120":pytest.skip("owning RTX5070 required")
    owner=function(variant,bounded)
    fused=variant.endswith("_fused") or variant=="reordered"
    storage=_storage(dtype)
    capacity=(65,33,513) if bounded else (17,19,35)
    frames=((17,19,35),capacity,(1,1,1),(17,19,35)) if bounded else ((17,19,35),)*2
    rng=np.random.default_rng(19332)
    held=[]
    with ExitStack() as stack:
        left=stack.enter_context(NvidiaDeviceSession())
        right=stack.enter_context(NvidiaDeviceSession())
        launch=stack.enter_context(NvidiaDeviceSession())
        sessions=[left,right]+([left,right] if fused else [])
        m,n,k=capacity
        dtypes=[storage,storage]+([np.float32,np.float32] if fused else [])
        shapes=[(m,k),(k,n)]+([(n,),(m,n)] if fused else [])
        buffers=[session.upload(np.zeros(shape,dtype)) for session,shape,dtype in zip(sessions,shapes,dtypes,strict=True)]
        output_capacity=launch.empty((m,n),np.float16 if fused else np.float32)
        for session in set(sessions):assert session.synchronize()==0
        program=None
        def forbidden(*args,**kwargs):pytest.fail("warm single-sided resident input traced/compiled/evaluated")
        for index,(m,n,k) in enumerate(frames):
            shapes=[(m,k),(k,n)]+([(n,),(m,n)] if fused else [])
            values=[rng.normal(0,.2,shape).astype(dtype) for shape,dtype in zip(shapes,dtypes,strict=True)]
            views=[buffer.view(0,value.shape,value.dtype) for buffer,value in zip(buffers,values,strict=True)]
            roots=[Borrowed(view) for view in views]
            args=ordered(roots,variant)
            if index==0:
                result=owner(*args)
                np.testing.assert_array_equal(result,np.zeros((m,n),result.dtype))
                held.append((result,result.copy()))
                program=owner._nvidia_lhs_last_program
                assert not program.rhs_chain
                assert program.edge.consumer.descriptor.provenance["b_layout"]=="row_major"
                prepared=next(iter(owner._nvidia_lhs_prepared_calls.values()))
                stack.callback(prepared.close)
                cap_m,cap_n,cap_k=capacity
                cap_shapes=[(cap_m,cap_k),(cap_k,cap_n)]+(
                    [(cap_n,),(cap_m,cap_n)] if fused else [])
                zeros=[np.zeros(shape,dtype) for shape,dtype in zip(cap_shapes,dtypes,strict=True)]
                owner(*ordered(zeros,variant))
                assert owner._nvidia_lhs_last_program is program
                scratch=prepared.scratch_stats()
            done=[]
            # Pending writes on the raw RHS must also be ordered, not merely
            # producer output on the LHS. Private allocation is already primed.
            callback=queue_delayed_upload(right,views[1],values[1],right.stream,done)
            for ordinal in (0,2,3) if fused else (0,):
                session=sessions[ordinal];view=views[ordinal];value=values[ordinal]
                assert session.lib.tessera_nvidia_device_upload(ct.c_void_p(view.ptr),
                    ct.c_void_p(value.ctypes.data),value.nbytes,ct.c_void_p(session.stream))==0
            assert not done
            with monkeypatch.context() as guard:
                guard.setattr(subprocess,"run",forbidden)
                guard.setattr(owner,"_fn",forbidden)
                if bounded:guard.setattr(owner,"_traced_autodiff_module",forbidden)
                output=(owner(**dict(zip(owner.arg_names,args,strict=True))) if index%2 else owner(*args))
            assert done and callback is not None
            np.testing.assert_allclose(output,oracle(values,variant),rtol=.015,atol=.015)
            assert owner._nvidia_lhs_last_program is program
            assert prepared.scratch_stats()==scratch
            assert all(r["native_call_binding"]=="prepared_cpp_ordered_resident_tensor_lhs"
                       for r in owner._nvidia_lhs_last_receipts)
            target=output_capacity.view(0,output.shape,output.dtype)
            profile=prepared.profile_resident(args,target,stream=launch.stream,repeats=8)
            assert profile["program_ms"]>0 and len(profile["grouped_stage_ms"])==len(prepared.component_receipts)
            np.testing.assert_allclose(target.numpy(),oracle(values,variant),rtol=.015,atol=.015)
            # Host/resident transition retains row-major package and owner.
            host=owner(*ordered(values,variant))
            np.testing.assert_array_equal(output,host)
            held.append((output,output.copy()))
        assert len(owner._nvidia_lhs_prepared_calls)==1
    for output,snapshot in held:np.testing.assert_array_equal(output,snapshot)

@pytest.mark.parametrize("two_sided",[False,True])
def test_native_owner_rejects_wrong_single_or_two_sided_mode_without_result_write(two_sided):
    from tessera.compiler.prepared_nvidia_matmul import HostView
    from tessera.compiler.resident_nvidia_tensor import ordered_resident_views
    from tests.device.nvidia.test_native_tensor_dag import public_dag
    if rt._nvidia_device_name()!="sm_120":pytest.skip("owning RTX5070 required")
    owner=ts.jit(target="nvidia_sm120")(public_dag._fn) if two_sided else function("rmsnorm",False)
    with NvidiaDeviceSession() as session:
        roots=[Borrowed(session.upload(np.zeros(shape,np.float16))) for shape in ((17,35),(35,19))]
        owner(*roots)
        try:
            prepared=next(iter(owner._nvidia_lhs_prepared_calls.values()))
            result=np.full((17,19),91,np.float32)
            output=HostView();output.data=result.ctypes.data;output.bytes=result.nbytes
            output.dtype=1;output.rank=2;output.shape[:]=result.shape;output.strides[:]=result.strides
            views,streams=ordered_resident_views(roots,None,writable_from=2)
            declared=(ct.c_uint64*2)(*streams)
            opposite="lhs" if two_sided else "dag"
            fn=getattr(prepared.lib,f"tessera_nvidia_matmul_invoke_{opposite}_resident_to_host_ordered")
            fn.argtypes=[ct.c_uint64,ct.POINTER(HostView),ct.c_size_t,
                         ct.POINTER(ct.c_uint64),ct.c_size_t,ct.POINTER(HostView)]
            fn.restype=ct.c_int
            assert fn(prepared.handle,views,2,declared,2,ct.byref(output))!=0
            np.testing.assert_array_equal(result,np.full_like(result,91))
            np.testing.assert_array_equal(owner(*roots),np.zeros_like(result))
        finally:owner.close_native_storage()


def test_native_single_sided_capacity_checks_actual_allocation_before_read_or_result_write():
    from tessera.compiler.prepared_nvidia_matmul import HostView
    from tessera.compiler.resident_nvidia_tensor import ordered_resident_views
    if rt._nvidia_device_name()!="sm_120":pytest.skip("owning RTX5070 required")
    owner=function("rmsnorm",True)
    with NvidiaDeviceSession() as session:
        roots=[Borrowed(session.upload(np.zeros(shape,np.float16))) for shape in ((17,35),(35,19))]
        owner(*roots)
        try:
            prepared=next(iter(owner._nvidia_lhs_prepared_calls.values()))
            views,streams=ordered_resident_views(roots,None,writable_from=2)
            for view,shape in zip(views,((65,513),(513,33)),strict=True):
                view.bytes=int(np.prod(shape))*2
                view.shape[:]=shape;view.strides[:]=(shape[1]*2,2)
            result=np.full((65,33),91,np.float32)
            output=HostView();output.data=result.ctypes.data;output.bytes=result.nbytes
            output.dtype=1;output.rank=2;output.shape[:]=result.shape;output.strides[:]=result.strides
            declared=(ct.c_uint64*2)(*streams)
            fn=prepared.lib.tessera_nvidia_matmul_invoke_lhs_resident_to_host_ordered
            fn.argtypes=[ct.c_uint64,ct.POINTER(HostView),ct.c_size_t,
                         ct.POINTER(ct.c_uint64),ct.c_size_t,ct.POINTER(HostView)]
            fn.restype=ct.c_int
            assert fn(prepared.handle,views,2,declared,2,ct.byref(output))!=0
            assert b"allocation context or capacity mismatch" in prepared.lib.tessera_nvidia_matmul_last_error()
            np.testing.assert_array_equal(result,np.full_like(result,91))
            np.testing.assert_array_equal(owner(*roots),np.zeros((17,19),np.float32))
        finally:owner.close_native_storage()


def test_host_created_bounded_column_rhs_package_gets_distinct_resident_row_package(monkeypatch):
    if rt._nvidia_device_name()!="sm_120":pytest.skip("owning RTX5070 required")
    owner=function("rmsnorm",True)
    values=[np.ones(shape,np.float16) for shape in ((17,35),(35,19))]
    owner(*values)
    host_program=owner._nvidia_lhs_last_program
    assert host_program.edge.consumer.descriptor.provenance["b_layout"]=="col_major"
    with NvidiaDeviceSession() as session:
        roots=[Borrowed(session.upload(value)) for value in values]
        actual=owner(*roots)
        np.testing.assert_allclose(actual,oracle(values,"rmsnorm"),rtol=.015,atol=.015)
        resident_program=owner._nvidia_lhs_last_program
        assert resident_program is not host_program
        assert resident_program.edge.consumer.descriptor.provenance["b_layout"]=="row_major"
        assert len(owner._bounded_lhs.programs)==2
        def forbidden(*args,**kwargs):pytest.fail("shape/layout cache reuse invoked compiler")
        with monkeypatch.context() as guard:
            guard.setattr(subprocess,"run",forbidden)
            guard.setattr(owner,"_traced_autodiff_module",forbidden)
            smaller=[value[:3,:11].copy() if i==0 else value[:11,:7].copy()
                     for i,value in enumerate(values)]
            roots=[Borrowed(session.upload(value)) for value in smaller]
            np.testing.assert_allclose(owner(*roots),oracle(smaller,"rmsnorm"),rtol=.015,atol=.015)
            assert owner._nvidia_lhs_last_program is resident_program
            np.testing.assert_allclose(owner(*smaller),oracle(smaller,"rmsnorm"),rtol=.015,atol=.015)
            assert owner._nvidia_lhs_last_program is host_program
        owner.close_native_storage()
