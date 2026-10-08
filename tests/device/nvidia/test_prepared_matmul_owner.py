"""Exact SM120 lifecycle/capacity proof for compiler-owned prepared matmul."""
import ctypes as ct
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import tessera as ts

from tessera.compiler.prepared_nvidia_matmul import HostView
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_permuted_matmul_jit import plain

pytestmark = pytest.mark.skipif(not nvidia_cuda_host_ready(), reason="owning SM120 GPU required")


@pytest.fixture
def prepared(monkeypatch):
    monkeypatch.setenv("TESSERA_NVIDIA_PREPARED_MATMUL", "1")
    fn = ts.jit(target="nvidia_sm120")(plain._fn)
    a = np.ones((17,35), np.float16)
    b = np.ones((35,19), np.float16)
    fn(b,a)
    call = next(iter(fn._native_prepared_matmul_calls.values()))
    yield fn, call, a, b
    fn.close_native_storage()


def native_views(a,b,output):
    views = (HostView * 3)()
    for view,array in zip(views,(a,b,output),strict=True):
        view.data,view.bytes,view.rank=array.ctypes.data,array.nbytes,array.ndim
        view.dtype=1 if array.dtype==np.float32 else 2
        for axis in range(array.ndim):
            view.shape[axis],view.strides[axis]=array.shape[axis],array.strides[axis]
    return views


@pytest.mark.parametrize("field",["null","bytes","rank","dtype","shape","stride","alignment"])
def test_native_rejects_invalid_view_before_output_write(prepared,field):
    _,call,a,b=prepared
    output=np.full((17,19),-123,np.float32)
    views=native_views(a,b,output)
    if field=="null":views[0].data=None
    elif field=="bytes":views[0].bytes-=2
    elif field=="rank":views[0].rank=1
    elif field=="dtype":views[0].dtype=3
    elif field=="shape":views[0].shape[1]+=1
    elif field=="stride":views[0].strides[0]+=2
    elif field=="alignment":views[0].data+=1
    assert call.lib.tessera_nvidia_matmul_invoke(call.handle,views,3)!=0
    assert b"mismatch" in call.lib.tessera_nvidia_matmul_last_error()
    np.testing.assert_array_equal(output,-123)


def test_native_unknown_handle_and_arity(prepared):
    _,call,a,b=prepared
    output=np.empty((17,19),np.float32)
    views=native_views(a,b,output)
    assert call.lib.tessera_nvidia_matmul_invoke(call.handle,views,2)!=0
    assert call.lib.tessera_nvidia_matmul_invoke(0,views,3)!=0


def test_closed_owner_and_public_rebind(prepared):
    fn,call,a,b=prepared
    call.close()
    with pytest.raises(ValueError,match="closed"):call((b,a))
    assert call.lib.tessera_nvidia_matmul_invoke(call.handle,None,0)!=0
    np.testing.assert_array_equal(fn(b,a),35)
    assert next(iter(fn._native_prepared_matmul_calls.values())).handle!=call.handle


def test_pinned_descriptor_cannot_drift(prepared):
    fn,call,a,b=prepared
    shape=call.artifact.launch_descriptor.provenance["shape"]
    old=list(shape)
    try:
        shape[0]+=1
        assert not call.matches(call.frontend_snapshot)
        with pytest.raises(ValueError,match="descriptor changed"):call((b,a))
        with pytest.raises(ValueError,match="descriptor changed"):fn(b,a)
    finally:
        shape[:]=old


def test_native_context_guard(prepared):
    _,call,a,b=prepared
    cuda=ct.CDLL("libcuda.so.1")
    cuda.cuCtxGetCurrent.argtypes=[ct.POINTER(ct.c_void_p)]
    cuda.cuCtxCreate_v2.argtypes=[ct.POINTER(ct.c_void_p),ct.c_uint,ct.c_int]
    cuda.cuCtxSetCurrent.argtypes=[ct.c_void_p]
    cuda.cuCtxDestroy_v2.argtypes=[ct.c_void_p]
    original=ct.c_void_p()
    assert cuda.cuCtxGetCurrent(ct.byref(original))==0
    other=ct.c_void_p()
    assert cuda.cuCtxCreate_v2(ct.byref(other),0,0)==0
    try:
        with pytest.raises(RuntimeError,match="context changed"):call((b,a))
    finally:
        assert cuda.cuCtxSetCurrent(original)==0
        assert cuda.cuCtxDestroy_v2(other)==0
    np.testing.assert_array_equal(call((b,a))[0],35)


def test_native_fork_guard_precedes_mutex_or_cuda(prepared):
    _,call,_,_=prepared
    read,write=os.pipe()
    child=os.fork()
    if child==0:
        os.close(read)
        status=call.lib.tessera_nvidia_matmul_invoke(call.handle,None,0)
        os.write(write,b"ok" if status and b"fork" in call.lib.tessera_nvidia_matmul_last_error() else b"bad")
        os._exit(0)
    os.close(write)
    result=os.read(read,3)
    os.close(read)
    _,status=os.waitpid(child,0)
    assert status==0 and result==b"ok"


def test_native_serializes_concurrent_owner_invocations(prepared):
    _,call,a,b=prepared
    def invoke(scale):
        value=(a*scale).astype(np.float16)
        output,receipt=call((b,value))
        assert receipt["native_call_binding"]=="prepared_cpp_matmul"
        np.testing.assert_array_equal(output,35*scale)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(invoke,range(1,9)))


def test_warm_signature_switch_reuses_each_sealed_native_owner(monkeypatch):
    import ml_dtypes
    from tessera import runtime as rt
    monkeypatch.setenv("TESSERA_NVIDIA_PREPARED_MATMUL","1")
    fn=ts.jit(target="nvidia_sm120")(plain._fn)
    cases=[]
    for m,k,n in ((17,35,19),(16,32,8)):
        for dtype in (np.float16,ml_dtypes.bfloat16):
            for order in ("C","F"):
                a=np.ones((m,k),dtype)
                b=np.ones((k,n),dtype,order=order)
                np.testing.assert_array_equal(fn(b,a),k)
                cases.append((a,b,k))
    handles={key:value.handle for key,value in fn._native_prepared_matmul_calls.items()}
    assert len(handles)==8
    def scratch_stats(call):
        lib=call.lib
        lib.tessera_nvidia_matmul_scratch_stats.argtypes=[
            ct.c_uint64,ct.POINTER(ct.c_size_t),ct.POINTER(ct.c_size_t)]
        lib.tessera_nvidia_matmul_scratch_stats.restype=ct.c_int
        capacity,allocations=ct.c_size_t(),ct.c_size_t()
        assert lib.tessera_nvidia_matmul_scratch_stats(
            call.handle,ct.byref(capacity),ct.byref(allocations))==0
        return capacity.value,allocations.value
    allocations={scratch_stats(call) for call in fn._native_prepared_matmul_calls.values()}
    assert len(allocations)==1 and next(iter(allocations))[0]>0
    def unavailable(*args,**kwargs):
        raise AssertionError("warm native signature must not trace/compile/portable-launch")
    monkeypatch.setattr(fn,"_trace_frontend_capture",unavailable)
    monkeypatch.setattr(rt,"launch",unavailable)
    try:
        for a,b,k in reversed(cases):
            np.testing.assert_array_equal(fn(b,a),k)
        assert handles=={key:value.handle for key,value in fn._native_prepared_matmul_calls.items()}
        assert allocations=={scratch_stats(call) for call in fn._native_prepared_matmul_calls.values()}
    finally:
        fn.close_native_storage()


@pytest.mark.parametrize("shape",[(1,35,19),(17,1,19),(17,35,1),(1,1,1),(1,1,19),(17,1,1)])
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("a_order",["C","F"])
@pytest.mark.parametrize("b_order",["C","F"])
def test_singleton_axis_pitch_is_nonsemantic(monkeypatch,shape,dtype,a_order,b_order):
    import ml_dtypes
    monkeypatch.setenv("TESSERA_NVIDIA_PREPARED_MATMUL","1")
    storage=np.float16 if dtype=="fp16" else ml_dtypes.bfloat16
    m,k,n=shape
    rng=np.random.default_rng(120701)
    a=np.array((rng.normal(size=(m,k))*.2).astype(storage),order=a_order)
    b=np.array((rng.normal(size=(k,n))*.2).astype(storage),order=b_order)
    fn=ts.jit(target="nvidia_sm120")(plain._fn)
    try:
        if not a.flags.c_contiguous:
            # N=1 alone does not make a genuinely column-major A row-major.
            # Preserve the declared A layout gate while accepting unused pitch.
            with pytest.raises(RuntimeError,match="shape/stride/dtype/capacity mismatch"):
                fn(b,a)
            return
        actual=fn(b,a)
        np.testing.assert_allclose(actual,a.astype(np.float64)@b.astype(np.float64),rtol=4e-5,atol=4e-5)
        np.testing.assert_array_equal(actual,fn(b,a))
    finally:
        fn.close_native_storage()


def test_warm_native_owner_preserves_compile_report_emission(prepared):
    from tessera.compiler import compile_report as cr
    fn,_,a,b=prepared
    with cr.capture_compile_reports() as sink:
        np.testing.assert_array_equal(fn(b,a),35)
    assert len(sink)==1
    assert sink[0].target=="nvidia_sm120"
    assert set(sink[0].ir_hashes)=={"graph_ir","schedule_ir","tile_ir","target_ir"}
    assert sink[0].plan_hash==fn.runtime_artifact().artifact_hash
    assert "compile_bundle.executable=True" in sink[0].target_decision["nvidia_sm120"]
    assert sink[0].fallback_reason is None
