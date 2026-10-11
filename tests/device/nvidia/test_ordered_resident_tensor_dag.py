"""Exact sm_120 borrowed root ordering, native profiles and output lifetime."""
from contextlib import ExitStack, closing
import ctypes as ct
import os
import subprocess
import time

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler import nvidia_tensor_lhs as lhs
from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
from tests.device.nvidia.test_native_tensor_dag import module, oracle

pytestmark=pytest.mark.hardware_nvidia


class Borrowed:
    """External CUDA Array Interface provider that forbids host coercion."""
    def __init__(self,buffer,stream=None):
        self.buffer=buffer
        self.dtype=buffer.dtype
        self.stream=stream

    @property
    def __cuda_array_interface__(self):
        interface=dict(self.buffer.__cuda_array_interface__)
        if self.stream is not None:interface["stream"]=self.stream
        interface["data"]=(interface["data"][0],True)
        return interface

    def __array__(self,*args,**kwargs):
        pytest.fail("borrowed root was coerced into a host tensor")


@pytest.fixture(scope="module",params=[
    (dtype,depth,bounded) for dtype in ("fp16","bf16") for depth in (1,2)
    for bounded in (False,True)])
def compiled(request):
    if rt._nvidia_device_name()!="sm_120":pytest.skip("exact RTX 5070/sm120 required")
    dtype,depth,bounded=request.param
    graph=module(dtype,depth=depth,rhs_first=depth==2)
    if bounded:graph.functions[0].args.reverse()
    program=lhs.package_traced_lhs(graph,**(
        {"shape_bounds":{"M":32,"N":24,"K":64}} if bounded else {}))
    return lhs.from_manifest(program.manifest()),dtype,depth,bounded


def ordered(program,a,b):
    return [a,b] if program.semantics["roles"]["source"]==0 else [b,a]


def queue_delayed_upload(session,buffer,host,stream,completed):
    # The callback deliberately contains no CUDA calls. Its Python object and
    # host source stay live until native synchronous completion.
    callback_type=ct.CFUNCTYPE(None,ct.c_void_p)
    @callback_type
    def delayed(_):
        time.sleep(.15)
        completed.append(True)
    lib=ct.CDLL(os.path.join(os.environ["CUDA_HOME"],"lib64","libcudart.so"))
    launch=lib.cudaLaunchHostFunc
    launch.argtypes=[ct.c_void_p,callback_type,ct.c_void_p];launch.restype=ct.c_int
    assert launch(ct.c_void_p(stream),delayed,None)==0
    assert session.lib.tessera_nvidia_device_upload(ct.c_void_p(buffer.ptr),
        ct.c_void_p(host.ctypes.data),host.nbytes,ct.c_void_p(stream))==0
    return delayed


@pytest.mark.parametrize("mode",["different","shared","legacy","per_thread"])
def test_pending_producers_are_ordered_without_allocation_sync(compiled,mode,monkeypatch):
    program,dtype,depth,bounded=compiled
    storage=np.float16 if dtype=="fp16" else ml_dtypes.bfloat16
    capacity=(32,24,64) if bounded else (17,11,32)
    rng=np.random.default_rng(1924)
    with ExitStack() as stack:
        first=stack.enter_context(NvidiaDeviceSession())
        second=first if mode!="different" else stack.enter_context(NvidiaDeviceSession())
        launch=stack.enter_context(NvidiaDeviceSession())
        a_capacity=first.upload(np.zeros((capacity[0],capacity[2]),storage))
        b_capacity=second.upload(np.zeros((capacity[2],capacity[1]),storage))
        assert first.synchronize()==0 and second.synchronize()==0
        prepared=stack.enter_context(closing(PreparedLhsCall(program)))
        frames=[(17,11,32),capacity] if bounded else [(17,11,32)]*2
        snapshots=[]
        for m,n,k in frames:
            a=rng.normal(0,.2,(m,k)).astype(storage)
            b=rng.normal(0,.2,(k,n)).astype(storage)
            av=a_capacity.view(0,a.shape,a.dtype)
            bv=b_capacity.view(0,b.shape,b.dtype)
            a_stream=1 if mode=="legacy" else 2 if mode=="per_thread" else first.stream
            b_stream=1 if mode=="legacy" else 2 if mode=="per_thread" else second.stream
            ap,bp=Borrowed(av,a_stream),Borrowed(bv,b_stream)
            output=launch.empty((m,n),prepared.output_dtype)
            values=ordered(program,ap,bp)
            # Prime every private allocation before delayed writes. This makes
            # correctness depend on the ordering ABI, not allocation fences.
            prepared.invoke_resident(values,output,stream=launch.stream)
            complete_a=[];complete_b=[]
            callbacks=[queue_delayed_upload(first,av,a,a_stream,complete_a),
                       queue_delayed_upload(second,bv,b,b_stream,complete_b)]
            assert not complete_a and not complete_b
            def forbidden(*args,**kwargs):pytest.fail("resident replay invoked a compiler")
            with monkeypatch.context() as guarded:
                guarded.setattr(subprocess,"run",forbidden)
                receipt=prepared.invoke_resident(values,output,stream=launch.stream)
            assert complete_a and complete_b and len(callbacks)==2
            np.testing.assert_allclose(output.numpy(),oracle(a,b,depth),rtol=.015,atol=.015)
            assert all(r["native_call_binding"]=="prepared_cpp_ordered_resident_tensor_dag"
                       for r in receipt["component_receipts"])
            profile=prepared.profile_resident(values,output,stream=launch.stream,repeats=8)
            assert profile["program_ms"]>0
            assert len(profile["grouped_stage_ms"])==2*depth+1
            assert all(value>0 for value in profile["grouped_stage_ms"])
            np.testing.assert_allclose(output.numpy(),oracle(a,b,depth),rtol=.015,atol=.015)
            with closing(PreparedLhsCall(program)) as host_owner:
                host_output,_=host_owner(ordered(program,a,b))
                np.testing.assert_array_equal(output.numpy(),host_output)
            snapshots.append((output,output.numpy()))
        for output,held in snapshots:np.testing.assert_array_equal(output.numpy(),held)
        # Public route creates only its result allocation; both source owners
        # can be closed after the synchronous call without invalidating output.
        result=program.execute_resident(*values)
        saved=result.output.numpy()
    try:
        np.testing.assert_array_equal(result.output.numpy(),saved)
        np.testing.assert_allclose(saved,oracle(a,b,depth),rtol=.015,atol=.015)
    finally:result.close()
    result.close()
    with pytest.raises(RuntimeError,match="closed"):result.output.numpy()


def test_forged_capacity_rejected_before_any_resident_read(compiled):
    program,dtype,_,bounded=compiled
    if not bounded:pytest.skip("bounded capacity forgery case")
    storage=np.float16 if dtype=="fp16" else ml_dtypes.bfloat16
    with NvidiaDeviceSession() as sources, NvidiaDeviceSession() as launch, closing(PreparedLhsCall(program)) as prepared:
        a=sources.upload(np.zeros((1,1),storage))
        b=sources.upload(np.zeros((64,24),storage))
        proxy=Borrowed(a)
        class Forged(Borrowed):
            @property
            def __cuda_array_interface__(self):
                interface=super().__cuda_array_interface__
                interface["shape"]=(32,64)
                return interface
        forged=Forged(a)
        output=launch.empty((32,24),prepared.output_dtype)
        with pytest.raises(RuntimeError,match="allocation context or capacity"):
            prepared.invoke_resident(ordered(program,forged,Borrowed(b)),output,stream=launch.stream)
        assert proxy.buffer is a


def test_closed_sources_rejected_before_new_native_owner(compiled,monkeypatch):
    program,dtype,_,_=compiled
    storage=np.float16 if dtype=="fp16" else ml_dtypes.bfloat16
    with NvidiaDeviceSession() as source:
        a=source.upload(np.zeros((17,32),storage))
        b=source.upload(np.zeros((32,11),storage))
        a.close()
        from tessera.compiler import prepared_nvidia_lhs
        def forbidden(*args,**kwargs):pytest.fail("closed input reached native prepare")
        monkeypatch.setattr(prepared_nvidia_lhs,"PreparedLhsCall",forbidden)
        with pytest.raises(RuntimeError,match="closed"):
            program.execute_resident(*ordered(program,Borrowed(a),Borrowed(b)))


@pytest.mark.parametrize("kind",["allocation","stream"])
def test_foreign_cuda_context_cannot_enter_owner(compiled,kind):
    program,dtype,_,_=compiled
    storage=np.float16 if dtype=="fp16" else ml_dtypes.bfloat16
    driver=ct.CDLL("libcuda.so.1")
    declarations={
        "cuCtxCreate_v2":[ct.POINTER(ct.c_void_p),ct.c_uint,ct.c_int],
        "cuCtxPopCurrent_v2":[ct.POINTER(ct.c_void_p)],
        "cuCtxPushCurrent_v2":[ct.c_void_p],"cuCtxDestroy_v2":[ct.c_void_p],
        "cuStreamCreate":[ct.POINTER(ct.c_void_p),ct.c_uint],
        "cuStreamDestroy_v2":[ct.c_void_p],
        "cuMemAlloc_v2":[ct.POINTER(ct.c_uint64),ct.c_size_t],
        "cuMemFree_v2":[ct.c_uint64],
    }
    for symbol,types in declarations.items():
        fn=getattr(driver,symbol);fn.argtypes=types;fn.restype=ct.c_int
    with NvidiaDeviceSession() as sources, NvidiaDeviceSession() as launch, closing(PreparedLhsCall(program)) as prepared:
        a=sources.upload(np.zeros((17,32),storage))
        b=sources.upload(np.zeros((32,11),storage))
        output=launch.empty((17,11),prepared.output_dtype)
        assert sources.synchronize()==0
        foreign=ct.c_void_p();stream=ct.c_void_p();pointer=ct.c_uint64()
        assert driver.cuCtxCreate_v2(ct.byref(foreign),0,0)==0
        try:
            assert driver.cuStreamCreate(ct.byref(stream),1)==0
            assert driver.cuMemAlloc_v2(ct.byref(pointer),17*32*2)==0
            popped=ct.c_void_p()
            assert driver.cuCtxPopCurrent_v2(ct.byref(popped))==0
            class Foreign(Borrowed):
                @property
                def __cuda_array_interface__(self):
                    interface=super().__cuda_array_interface__
                    interface["stream"]=int(stream.value)
                    if kind=="allocation":interface["data"]=(pointer.value,True)
                    return interface
            match="allocation context or capacity" if kind=="allocation" else "producer context mismatch"
            with pytest.raises(RuntimeError,match=match):
                prepared.invoke_resident(ordered(program,Foreign(a),Borrowed(b)),output,stream=launch.stream)
            # A refused foreign frame does not poison the valid native owner.
            prepared.invoke_resident(ordered(program,Borrowed(a),Borrowed(b)),output,stream=launch.stream)
            np.testing.assert_array_equal(output.numpy(),np.zeros((17,11),prepared.output_dtype))
        finally:
            assert driver.cuCtxPushCurrent_v2(foreign)==0
            if pointer.value:assert driver.cuMemFree_v2(pointer)==0
            if stream.value:assert driver.cuStreamDestroy_v2(stream)==0
            popped=ct.c_void_p();assert driver.cuCtxPopCurrent_v2(ct.byref(popped))==0
            assert driver.cuCtxDestroy_v2(foreign)==0


@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("bounded",[False,True])
def test_resident_fused_consumer_carries_every_epilogue_root(dtype,bounded,monkeypatch):
    import tessera as ts
    from tests.device.nvidia.test_native_tensor_dag import public_dag_fused
    if rt._nvidia_device_name()!="sm_120":pytest.skip("exact SM120 required")
    storage=np.float16 if dtype=="fp16" else ml_dtypes.bfloat16
    rng=np.random.default_rng(1984)
    def frame(m,n,k):
        a=rng.normal(0,.2,(m,k)).astype(storage)
        b=rng.normal(0,.2,(k,n)).astype(storage)
        bias=rng.normal(0,.2,n).astype(np.float32)
        residual=rng.normal(0,.2,(m,n)).astype(np.float32)
        expected=(np.maximum(oracle(a,b,1)+bias,0)+residual).astype(np.float16)
        return (a,b,bias,residual),expected
    args,_=frame(17,19,35)
    function=ts.jit(target="nvidia_sm120",**(
        {"shape_bounds":{"M":32,"N":24,"K":64}} if bounded else {}))(public_dag_fused._fn)
    program=function.compile_native_lhs_matmul(*args)
    args,expected=frame(*( (32,24,64) if bounded else (17,19,35)))
    with NvidiaDeviceSession() as left, NvidiaDeviceSession() as right, NvidiaDeviceSession() as launch, closing(PreparedLhsCall(program)) as prepared:
        values=[Borrowed((left if index%2==0 else right).upload(value))
                for index,value in enumerate(args)]
        output=launch.empty(expected.shape,prepared.output_dtype)
        def forbidden(*args,**kwargs):pytest.fail("resident fused replay invoked compiler/eager frontend")
        monkeypatch.setattr(subprocess,"run",forbidden)
        monkeypatch.setattr(function,"_fn",forbidden)
        receipt=prepared.invoke_resident(values,output,stream=launch.stream)
        np.testing.assert_allclose(output.numpy(),expected,rtol=.015,atol=.015)
        assert len(receipt["component_receipts"])==3
        profile=prepared.profile_resident(values,output,stream=launch.stream,repeats=8)
        assert len(profile["grouped_stage_ms"])==3 and profile["program_ms"]>0
        np.testing.assert_allclose(output.numpy(),expected,rtol=.015,atol=.015)
        with closing(PreparedLhsCall(program)) as host_owner:
            host_output,_=host_owner(args)
            np.testing.assert_array_equal(output.numpy(),host_output)
        with program.execute_resident(*values) as result:
            np.testing.assert_allclose(result.output.numpy(),expected,rtol=.015,atol=.015)
            np.testing.assert_array_equal(result.output.numpy(),host_output)
