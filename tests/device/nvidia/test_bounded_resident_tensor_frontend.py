"""Bounded ordinary JIT resident roots: active shapes, ordering and lifetimes."""
from contextlib import ExitStack
import ctypes as ct
import json
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
CAPACITY={"M":33,"N":25,"K":65}
FRAMES=((17,19,35),(33,25,65),(1,1,1),(31,23,63),(3,7,11),(17,19,35))


def fresh(variant,bounds):
    functions={"ordinary":public_dag,"reordered":public_dag_reordered,
               "deep":public_dag_deep,"fused":public_dag_fused}
    return ts.jit(target="nvidia_sm120",shape_bounds=bounds)(functions[variant]._fn)


def shapes(m,n,k,fused=False):
    return [(m,k),(k,n)]+([(n,),(m,n)] if fused else [])


def expected(values,variant):
    output=oracle(*values[:2],2 if variant=="deep" else 1)
    if variant=="fused":
        output=(np.maximum(output+values[2],0)+values[3]).astype(np.float16)
    return output


def upload(session,buffer,value):
    assert session.lib.tessera_nvidia_device_upload(ct.c_void_p(buffer.ptr),
        ct.c_void_p(value.ctypes.data),value.nbytes,ct.c_void_p(session.stream))==0


@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("variant",["ordinary","reordered","deep","fused"])
@pytest.mark.parametrize("streams",["same","different"])
def test_bounded_resident_reuses_one_package_across_pending_active_frames(dtype,variant,streams,monkeypatch):
    if rt._nvidia_device_name()!="sm_120":pytest.skip("owning RTX5070 required")
    owner=fresh(variant,CAPACITY)
    storage=np.float16 if dtype=="fp16" else ml_dtypes.bfloat16
    rng=np.random.default_rng(19330)
    fused=variant=="fused"
    held=[]
    with ExitStack() as stack:
        left=stack.enter_context(NvidiaDeviceSession())
        right=left if streams=="same" else stack.enter_context(NvidiaDeviceSession())
        sessions=[left,right]+([left,right] if fused else [])
        dtypes=[storage,storage]+([np.float32,np.float32] if fused else [])
        buffers=[session.upload(np.zeros(shape,dtype)) for session,shape,dtype in zip(
            sessions,shapes(33,25,65,fused),dtypes,strict=True)]
        for session in set(sessions):assert session.synchronize()==0
        program=None
        scratch=None
        def forbidden(*args,**kwargs):pytest.fail("warm bounded resident frame traced/compiled/evaluated eager")
        for index,(m,n,k) in enumerate(FRAMES):
            frame_shapes=shapes(m,n,k,fused)
            values=[rng.normal(0,.2,shape).astype(dtype)
                    for shape,dtype in zip(frame_shapes,dtypes,strict=True)]
            views=[buffer.view(0,value.shape,value.dtype) for buffer,value in zip(buffers,values,strict=True)]
            roots=[Borrowed(view) for view in views]
            ordered=[roots[1],roots[0]] if variant=="reordered" else roots
            # Prime cold-owned capacity with a valid compact frame before
            # delayed writes. Later correctness depends on stream events.
            if index==0:
                for session,view,value in zip(sessions,views,values,strict=True):upload(session,view,value)
                first=owner(*ordered)
                np.testing.assert_allclose(first,expected(values,variant),rtol=.015,atol=.015)
                held.append((first,first.copy()))
                program=owner._nvidia_lhs_last_program
                prepared=next(iter(owner._nvidia_lhs_prepared_calls.values()))
                stack.callback(prepared.close)
                scratch=prepared.scratch_stats()
                assert (program.edge.m,program.edge.n,program.edge.k)==(33,25,65)
                plan=json.loads(program.native_plan_json)
                assert plan["dynamic_axes"]==["M","N","K"]
                assert plan["shape_bounds"]==[33,25,65]
                assert plan["original_graph_ir"] and plan["source_graph_ir"]==program.graph_ir
            completion=[]
            callback=queue_delayed_upload(left,views[0],values[0],left.stream,completion)
            for session,view,value in zip(sessions[1:],views[1:],values[1:],strict=True):upload(session,view,value)
            assert not completion
            with monkeypatch.context() as guard:
                guard.setattr(subprocess,"run",forbidden)
                guard.setattr(owner,"_fn",forbidden)
                guard.setattr(owner,"_traced_autodiff_module",forbidden)
                output=(owner(**dict(zip(owner.arg_names,ordered,strict=True))) if index%2 else owner(*ordered))
            assert completion and callback is not None
            np.testing.assert_allclose(output,expected(values,variant),rtol=.015,atol=.015)
            assert owner._nvidia_lhs_last_program is program
            assert len(owner._bounded_lhs.programs)==1 and len(owner._nvidia_lhs_prepared_calls)==1
            assert prepared.scratch_stats()==scratch
            assert all(r["native_call_binding"]=="prepared_cpp_ordered_resident_tensor_dag"
                       for r in owner._nvidia_lhs_last_receipts)
            held.append((output,output.copy()))
    for output,snapshot in held:np.testing.assert_array_equal(output,snapshot)


@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("axes",[("M",),("N",),("K",),("M","N"),("M","K"),("N","K"),("M","N","K")])
def test_resident_independent_bounded_axes_and_host_transition(dtype,axes,monkeypatch):
    if rt._nvidia_device_name()!="sm_120":pytest.skip("owning RTX5070 required")
    storage=np.float16 if dtype=="fp16" else ml_dtypes.bfloat16
    owner=fresh("ordinary",{a:CAPACITY[a] for a in axes})
    initial={"M":17,"N":19,"K":35}
    expanded={a:CAPACITY[a] if a in axes else initial[a] for a in initial}
    with NvidiaDeviceSession() as session:
        roots=[Borrowed(session.upload(np.ones(shape,storage))) for shape in shapes(17,19,35)]
        owner(*roots)
        program=owner._nvidia_lhs_last_program
        def forbidden(*args,**kwargs):pytest.fail("bounded residency transition rebuilt Graph/package")
        monkeypatch.setattr(owner,"_traced_autodiff_module",forbidden)
        values=[(np.ones(shape,storage)*.5).astype(storage) for shape in shapes(expanded["M"],expanded["N"],expanded["K"])]
        roots=[Borrowed(session.upload(value)) for value in values]
        try:
            actual=owner(*roots)
            np.testing.assert_allclose(actual,expected(values,"ordinary"),rtol=.015,atol=.015)
            np.testing.assert_array_equal(actual,owner(*values))
            assert owner._nvidia_lhs_last_program is program
            assert len(owner._bounded_lhs.programs)==1
            assert len(owner._nvidia_lhs_prepared_calls)==1
            assert (program.edge.dynamic_m,program.edge.dynamic_n,program.edge.dynamic_k)==tuple(a in axes for a in ("M","N","K"))
        finally:owner.close_native_storage()


@pytest.mark.parametrize("bad",["m","n","k","contraction","bias","residual","rank","mixed","stream"])
def test_invalid_bounded_resident_frames_reject_before_context_or_compile(bad,monkeypatch):
    if rt._nvidia_device_name()!="sm_120":pytest.skip("owning RTX5070 required")
    owner=fresh("fused",CAPACITY)
    with NvidiaDeviceSession() as session:
        roots=[Borrowed(session.upload(np.zeros(shape,dtype))) for shape,dtype in zip(
            shapes(17,19,35,True),(np.float16,np.float16,np.float32,np.float32),strict=True)]
        owner(*roots)
        invalid_shapes=shapes(17,19,35,True)
        if bad=="m":invalid_shapes[0]=(34,35)
        elif bad=="n":invalid_shapes[1]=(35,26)
        elif bad=="k":invalid_shapes[0]=(17,66)
        elif bad=="contraction":invalid_shapes[1]=(34,19)
        elif bad=="bias":invalid_shapes[2]=(18,)
        elif bad=="residual":invalid_shapes[3]=(16,19)
        elif bad=="rank":invalid_shapes[0]=(35,)
        class Altered(Borrowed):
            @property
            def __cuda_array_interface__(self):
                interface=super().__cuda_array_interface__
                interface["shape"]=invalid_shapes[self.index]
                interface["strides"]=None
                return interface
        values=[]
        for index,root in enumerate(roots):
            value=Altered(root.buffer);value.index=index;values.append(value)
        if bad=="mixed":values[1]=np.zeros((35,19),np.float16)
        if bad=="stream":values[0].stream=0
        def forbidden(*args,**kwargs):pytest.fail("invalid bounded input reached compiler/native context")
        with monkeypatch.context() as guard:
            guard.setattr(owner,"_traced_autodiff_module",forbidden)
            guard.setattr(rt._load_nvidia_ptx_launch(),"tessera_nvidia_matmul_context_identity",forbidden)
            with pytest.raises(ValueError):owner(*values)
            assert owner._nvidia_lhs_last_receipts==()
        np.testing.assert_array_equal(owner(*roots),np.zeros((17,19),np.float16))
        owner.close_native_storage()
