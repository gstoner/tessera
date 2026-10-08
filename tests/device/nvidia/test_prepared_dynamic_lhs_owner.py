"""Native bounded shape, capacity and lifetime proof for two-kernel owners."""
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import numpy as np
import pytest
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lhs_tensor_jit import (
    rms_lhs,layer_lhs,softmax_lhs,rms_lhs_fused,layer_lhs_fused,softmax_lhs_fused,
    _oracle,_storage,
)
from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall

pytestmark=pytest.mark.skipif(not nvidia_cuda_host_ready(),reason="owning NVIDIA host required")


@pytest.mark.parametrize("kind",["rmsnorm","layernorm","softmax"])
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("fused",[False,True])
@pytest.mark.parametrize("axes",[("M",),("N",),("K",),("M","N"),("M","K"),("N","K"),("M","N","K")])
def test_bounded_owner_reuses_capacity_and_preserves_outputs(kind,dtype,fused,axes,monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    storage=_storage(dtype)
    rng=np.random.default_rng(120608)
    m,k,n=128,1024,64
    x=(rng.normal(size=(m,k))*.2).astype(storage)
    b=np.array(rng.normal(size=(k,n))*.2,dtype=storage,order="F")
    bias=(rng.normal(size=n)*.2).astype(np.float32)
    residual=(rng.normal(size=(m,n))*.2).astype(np.float32)
    function=({"rmsnorm":rms_lhs_fused,"layernorm":layer_lhs_fused,"softmax":softmax_lhs_fused}
              if fused else {"rmsnorm":rms_lhs,"layernorm":layer_lhs,"softmax":softmax_lhs})[kind]
    args=(x,b,bias,residual) if fused else (x,b)
    owner=PreparedLhsCall(function.compile_native_lhs_matmul(*args,dynamic_axes=axes))
    def forbidden(*args,**kwargs):
        raise AssertionError("prepared frame escaped to Python descriptor/session execution")
    monkeypatch.setattr(rt,"launch",forbidden)
    monkeypatch.setattr(NvidiaDeviceSession,"__init__",forbidden)
    try:
        first,receipt=owner(args)
        saved=first.copy()
        assert receipt["native_call_binding"]=="prepared_cpp_dynamic_tensor_matmul"
        before=owner.scratch_stats()
        for shape,scale in (((1,1,1),.5),((17,35,19),1.25),((63,511,31),-.75),((m,k,n),1.)):
            am,ak,an=(actual if axis in axes else bound for actual,bound,axis in
                      zip(shape,(m,k,n),("M","K","N"),strict=True))
            source=(x[:am,:ak].astype(np.float32)*scale).astype(storage)
            rhs=b[:ak,:an]
            ab,ar=bias[:an],residual[:am,:an]
            args=(source,rhs,ab,ar) if fused else (source,rhs)
            actual,_=owner(args)
            np.testing.assert_allclose(actual,_oracle(source,rhs,kind,ab if fused else None,
                                                      ar if fused else None),rtol=.015,atol=.015)
            assert actual.shape==(am,an)
            assert owner.scratch_stats()==before
            np.testing.assert_array_equal(first,saved)
            assert not np.shares_memory(first,actual)
    finally:
        owner.close()


def _views(arrays):
    from tessera.compiler.prepared_nvidia_matmul import HostView
    result=(HostView*len(arrays))()
    for view,array in zip(result,arrays,strict=True):
        view.data,view.bytes,view.rank=array.ctypes.data,array.nbytes,array.ndim
        view.dtype={"float32":1,"float16":2,"bfloat16":3}[array.dtype.name]
        for axis in range(array.ndim):
            view.shape[axis],view.strides[axis]=array.shape[axis],array.strides[axis]
    return result


@pytest.mark.parametrize("malformation",["zero","overflow","rhs_k","pitch","bytes","output","dtype"])
def test_native_dynamic_guard_preserves_scratch_and_caller_output(malformation):
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16,order="F")
    owner=PreparedLhsCall(rms_lhs.compile_native_lhs_matmul(source,rhs,dynamic_axes=("M","N","K")))
    try:
        before=owner.scratch_stats()
        output=np.full((17,19),-123,np.float32)
        views=_views((source,rhs,output))
        if malformation=="zero":views[0].shape[0]=0
        elif malformation=="overflow":views[0].shape[0]=18
        elif malformation=="rhs_k":views[1].shape[0]=34
        elif malformation=="pitch":views[0].strides[0]+=2
        elif malformation=="bytes":views[1].bytes-=2
        elif malformation=="output":views[2].shape[1]-=1
        else:views[0].dtype=1
        assert owner.lib.tessera_nvidia_matmul_invoke(owner.handle,views,3)!=0
        np.testing.assert_array_equal(output,-123)
        assert owner.scratch_stats()==before
        actual,_=owner((source[:3,:11],rhs[:11,:7]))
        np.testing.assert_allclose(actual,_oracle(source[:3,:11],rhs[:11,:7],"rmsnorm"),rtol=.015,atol=.015)
    finally:
        owner.close()


@pytest.mark.parametrize("axis",["M","N","K"])
def test_static_axis_cannot_shrink_in_partial_dynamic_owner(axis):
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16,order="F")
    axes=tuple(a for a in ("M","N","K") if a!=axis)
    owner=PreparedLhsCall(rms_lhs.compile_native_lhs_matmul(source,rhs,dynamic_axes=axes))
    try:
        before=owner.scratch_stats()
        m,k,n=(1 if axis=="M" else 17,1 if axis=="K" else 35,1 if axis=="N" else 19)
        with pytest.raises(RuntimeError,match="outside bound"):
            owner((source[:m,:k],rhs[:k,:n]))
        assert owner.scratch_stats()==before
    finally:
        owner.close()


def test_static_kernel_cannot_be_reinterpreted_as_dynamic():
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16,order="F")
    # Prepare a static consumer without attachment so the parameter ABI is the refusal.
    from tessera.compiler.prepared_nvidia_matmul import PreparedMatmulCall
    from tessera.compiler.nvidia_tensor_rhs import NvidiaNormRhsProgram
    program=rms_lhs.compile_native_lhs_matmul(source,rhs)
    owner=PreparedMatmulCall.__new__(PreparedMatmulCall)
    artifact=NvidiaNormRhsProgram.runtime_artifact(program.edge.consumer)
    bindings=sorted(artifact.launch_descriptor.buffers,key=lambda item:item.ordinal)
    owner._initialize(artifact,None,program.graph_ir,
                      binding_names=tuple(item.name for item in bindings[:-1]))
    try:
        setter=owner.lib.tessera_nvidia_matmul_set_dynamic_axes
        setter.argtypes=[ct.c_uint64,ct.c_int]
        setter.restype=ct.c_int
        assert setter(owner.handle,7)!=0
        assert b"parameter ABI" in owner.lib.tessera_nvidia_matmul_last_error()
        actual,_=owner((source,rhs))
        np.testing.assert_array_equal(actual,np.full((17,19),35,np.float32))
    finally:
        owner.close()


def test_concurrent_mixed_shapes_share_one_retired_arena():
    source=np.ones((128,1024),np.float16)
    rhs=np.ones((1024,64),np.float16,order="F")
    calls=[PreparedLhsCall(fn.compile_native_lhs_matmul(source,rhs,dynamic_axes=("M","N","K")))
           for fn in (rms_lhs,layer_lhs)]
    try:
        calls[0]((source,rhs))
        before=calls[0].scratch_stats()
        def invoke(index):
            m,k,n=((1,1,1),(17,35,19),(128,1024,64))[index%3]
            x=source[:m,:k]*np.float16(index+1)
            b=rhs[:k,:n]
            actual,_=calls[index%2]((x,b))
            np.testing.assert_allclose(actual,_oracle(x,b,("rmsnorm","layernorm")[index%2]),rtol=.015,atol=.015)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(invoke,range(18)))
        assert calls[0].scratch_stats()==calls[1].scratch_stats()==before
    finally:
        for call in calls:call.close()


def test_dynamic_context_guard_and_shape_recovery():
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16,order="F")
    owner=PreparedLhsCall(rms_lhs.compile_native_lhs_matmul(source,rhs,dynamic_axes=("M","N","K")))
    cuda=ct.CDLL("libcuda.so.1")
    cuda.cuCtxGetCurrent.argtypes=[ct.POINTER(ct.c_void_p)]
    cuda.cuCtxCreate_v2.argtypes=[ct.POINTER(ct.c_void_p),ct.c_uint,ct.c_int]
    cuda.cuCtxSetCurrent.argtypes=[ct.c_void_p]
    cuda.cuCtxDestroy_v2.argtypes=[ct.c_void_p]
    original,other=ct.c_void_p(),ct.c_void_p()
    assert cuda.cuCtxGetCurrent(ct.byref(original))==0
    assert cuda.cuCtxCreate_v2(ct.byref(other),0,0)==0
    try:
        with pytest.raises(RuntimeError,match="context changed"):
            owner((source[:3,:11],rhs[:11,:7]))
    finally:
        assert cuda.cuCtxSetCurrent(original)==0
        assert cuda.cuCtxDestroy_v2(other)==0
    try:
        actual,_=owner((source[:3,:11],rhs[:11,:7]))
        np.testing.assert_allclose(actual,_oracle(source[:3,:11],rhs[:11,:7],"rmsnorm"),rtol=.015,atol=.015)
    finally:
        owner.close()


def test_dynamic_failed_consumer_retires_real_producer_and_preserves_output():
    source=np.ones((128,1024),np.float16)
    rhs=np.ones((1024,64),np.float16,order="F")
    owner=PreparedLhsCall(rms_lhs.compile_native_lhs_matmul(source,rhs,dynamic_axes=("M","N","K")))
    lib=owner.lib
    entry="nvidia_sm120_scheduled_matmul_dynamic_failure_probe"
    ptx=ct.create_string_buffer(("""
.version 8.8
.target sm_120a
.address_size 64
.visible .entry """+entry+"""(
 .param .u64 a, .param .u64 b, .param .u64 d,
 .param .u64 m, .param .u64 n, .param .u64 k,
 .param .u64 lda, .param .u64 ldb, .param .u64 ldd
)
.maxntid 1, 1, 1
{ ret; }
""").encode())
    dims=(ct.c_int64*3)(128,64,1024)
    handle=ct.c_uint64()
    owner._check(lib.tessera_nvidia_matmul_prepare(
        ptx,len(ptx.value),entry.encode(),dims,2,0,0,0,0,ct.byref(handle)))
    try:
        owner._check(lib.tessera_nvidia_matmul_set_dynamic_axes(handle,7))
        producer=owner.program.edge.producer
        image=ct.create_string_buffer(producer.image.payload)
        owner._check(lib.tessera_nvidia_matmul_attach_producer(
            handle,image,len(producer.image.payload),producer.descriptor.entry_symbol.encode(),
            int(producer.descriptor.provenance["schedule"]=="cooperative_128")))
        x,b=source[:63,:511],np.array(rhs[:511,:31],order="F")
        x=np.ascontiguousarray(x)
        output=np.full((63,31),-123,np.float32)
        assert lib.tessera_nvidia_matmul_invoke(handle,_views((x,b,output)),3)!=0
        assert b"launch prepared matmul" in lib.tessera_nvidia_matmul_last_error()
        np.testing.assert_array_equal(output,-123)
        owner.scratch_stats()
        capacity,allocations=ct.c_size_t(),ct.c_size_t()
        owner._check(lib.tessera_nvidia_matmul_scratch_stats(handle,ct.byref(capacity),ct.byref(allocations)))
        actual,_=owner((x,b))
        np.testing.assert_allclose(actual,_oracle(x,b,"rmsnorm"),rtol=.015,atol=.015)
    finally:
        lib.tessera_nvidia_matmul_close(handle)
        owner.close()


def test_portable_dynamic_owner_reuses_one_handle_without_readmission(monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler import nvidia_tensor_lhs as lhs,prepared_nvidia_lhs as native_owner
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    monkeypatch.setenv("TESSERA_NVIDIA_PREPARED_LHS_REPLAY","1")
    native_owner.clear_portable_lhs_owners()
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16,order="F")
    artifact=lhs.runtime_artifact(rms_lhs.compile_native_lhs_matmul(source,rhs,dynamic_axes=("M","N","K")))
    try:
        receipt=rt.launch(artifact,(source,rhs))
        assert receipt["ok"],receipt
        owner=next(iter(native_owner._portable_owners.values()))
        before=owner.scratch_stats()
        handle=owner.handle
        def forbidden(*args,**kwargs):
            raise AssertionError("warm shape change reconstructed portable/native owners")
        monkeypatch.setattr(lhs,"from_manifest",forbidden)
        monkeypatch.setattr(NvidiaDeviceSession,"__init__",forbidden)
        monkeypatch.setattr(PreparedLhsCall,"__init__",forbidden)
        for m,k,n in ((1,1,1),(3,11,7),(17,35,19)):
            x=source[:m,:k]*np.float16(.5)
            b=rhs[:k,:n]
            receipt=rt.launch(artifact,(x,b))
            assert receipt["ok"],receipt
            assert receipt["component_receipts"][0]["native_call_binding"]=="prepared_cpp_dynamic_tensor_matmul"
            np.testing.assert_allclose(receipt["output"],_oracle(x,b,"rmsnorm"),rtol=.015,atol=.015)
            assert next(iter(native_owner._portable_owners.values())).handle==handle
            assert owner.scratch_stats()==before
    finally:
        native_owner.clear_portable_lhs_owners()


def test_portable_dynamic_owner_uses_distinct_live_contexts(monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler import nvidia_tensor_lhs as lhs,prepared_nvidia_lhs as native_owner
    monkeypatch.setenv("TESSERA_NVIDIA_PREPARED_LHS_REPLAY","1")
    native_owner.clear_portable_lhs_owners()
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16,order="F")
    artifact=lhs.runtime_artifact(rms_lhs.compile_native_lhs_matmul(source,rhs,dynamic_axes=("M","N","K")))
    assert rt.launch(artifact,(source,rhs))["ok"]
    cuda=ct.CDLL("libcuda.so.1")
    cuda.cuCtxGetCurrent.argtypes=[ct.POINTER(ct.c_void_p)]
    cuda.cuCtxCreate_v2.argtypes=[ct.POINTER(ct.c_void_p),ct.c_uint,ct.c_int]
    cuda.cuCtxSetCurrent.argtypes=[ct.c_void_p]
    cuda.cuCtxDestroy_v2.argtypes=[ct.c_void_p]
    original,other=ct.c_void_p(),ct.c_void_p()
    assert cuda.cuCtxGetCurrent(ct.byref(original))==0
    assert cuda.cuCtxCreate_v2(ct.byref(other),0,0)==0
    try:
        receipt=rt.launch(artifact,(source[:3,:11],rhs[:11,:7]))
        assert receipt["ok"],receipt
        np.testing.assert_allclose(receipt["output"],_oracle(source[:3,:11],rhs[:11,:7],"rmsnorm"),rtol=.015,atol=.015)
        assert len(native_owner._portable_owners)==2
        assert len({key[1] for key in native_owner._portable_owners})==2
        assert all(call.dynamic_axes==(True,True,True) for call in native_owner._portable_owners.values())
        native_owner.clear_portable_lhs_owners()
    finally:
        assert cuda.cuCtxSetCurrent(original)==0
        assert cuda.cuCtxDestroy_v2(other)==0
    try:
        assert rt.launch(artifact,(source[:1,:1],rhs[:1,:1]))["ok"]
    finally:
        native_owner.clear_portable_lhs_owners()
