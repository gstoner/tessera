"""Ordinary bounded JIT reuse over verified native tensor products."""
import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lhs_tensor_jit import _storage,_oracle

pytestmark=pytest.mark.skipif(not nvidia_cuda_host_ready(),reason="owning NVIDIA host required")


@ts.jit(target="nvidia_sm120",shape_bounds={"M":128,"N":64,"K":1024})
def bounded_rms(source,rhs):
    return ts.ops.matmul(ts.ops.rmsnorm(source,eps=1e-5),rhs,output_dtype="fp32")


@ts.jit(target="nvidia_sm120",shape_bounds={"M":128,"N":64,"K":1024})
def bounded_layer(source,rhs):
    return ts.ops.matmul(ts.ops.layer_norm(source,eps=1e-5),rhs,output_dtype="fp32")


@ts.jit(target="nvidia_sm120",shape_bounds={"M":128,"N":64,"K":1024})
def bounded_softmax(source,rhs):
    return ts.ops.matmul(ts.ops.softmax(source,axis=-1),rhs,output_dtype="fp32")


@ts.jit(target="nvidia_sm120",shape_bounds={"M":128,"N":64,"K":1024})
def bounded_rms_fused(source,rhs,bias,residual):
    return ts.ops.matmul(ts.ops.rmsnorm(source,eps=1e-5),rhs,bias=bias,
                         activation="relu",residual=residual,output_dtype="fp16")


@ts.jit(target="nvidia_sm120",shape_bounds={"M":128,"N":64,"K":1024})
def bounded_layer_fused(source,rhs,bias,residual):
    return ts.ops.matmul(ts.ops.layer_norm(source,eps=1e-5),rhs,bias=bias,
                         activation="relu",residual=residual,output_dtype="fp16")


@ts.jit(target="nvidia_sm120",shape_bounds={"M":128,"N":64,"K":1024})
def bounded_softmax_fused(source,rhs,bias,residual):
    return ts.ops.matmul(ts.ops.softmax(source,axis=-1),rhs,bias=bias,
                         activation="relu",residual=residual,output_dtype="fp16")


@pytest.mark.parametrize("kind",["rmsnorm","layernorm","softmax"])
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("fused",[False,True])
@pytest.mark.parametrize("order",["C","F"])
def test_ordinary_call_reuses_bounded_native_packages(kind,dtype,fused,order,monkeypatch):
    function=({"rmsnorm":bounded_rms_fused,"layernorm":bounded_layer_fused,"softmax":bounded_softmax_fused}
              if fused else {"rmsnorm":bounded_rms,"layernorm":bounded_layer,"softmax":bounded_softmax})[kind]
    # Independent wrappers keep cold trace behavior observable in each test.
    function=ts.jit(target="nvidia_sm120",shape_bounds={"M":128,"N":64,"K":1024})(function._fn)
    rng=np.random.default_rng(120609)
    storage=_storage(dtype)
    x=(rng.normal(size=(128,1024))*.2).astype(storage)
    b=np.array(rng.normal(size=(1024,64))*.2,dtype=storage,order=order)
    bias=(rng.normal(size=64)*.2).astype(np.float32)
    residual=(rng.normal(size=(128,64))*.2).astype(np.float32)
    # The first call is smaller than all declared capacities.
    args=(x[:17,:35],b[:35,:19],bias[:19],residual[:17,:19]) if fused else (x[:17,:35],b[:35,:19])
    first=function(*args)
    saved=first.copy()
    packages=function.native_lhs_packages()
    assert function.execution_kind=="native_gpu"
    assert "?" in function.ir_text() and "shape_bounds = [128, 64, 1024]" in function.ir_text()
    assert function._nvidia_lhs_last_program.edge.m==128
    assert function._nvidia_lhs_last_program.edge.k==1024
    assert function._nvidia_lhs_last_program.edge.n==64
    program=function._nvidia_lhs_last_program
    def forbidden(*args,**kwargs):
        raise AssertionError("warm shape change traced/compiled or executed eager/descriptor code")
    monkeypatch.setattr(function,"_fn",forbidden)
    monkeypatch.setattr(function,"_traced_autodiff_module",forbidden)
    monkeypatch.setattr(function,"compile_native_lhs_matmul",forbidden)
    monkeypatch.setattr(rt,"launch",forbidden)
    for m,k,n in ((128,1024,64),(1,1,1),(63,511,31),(17,35,19)):
        source=(x[:m,:k].astype(np.float32)*.75).astype(storage)
        rhs=np.array(b[:k,:n],order=order)
        args=(source,rhs,bias[:n],residual[:m,:n]) if fused else (source,rhs)
        actual=function(*args)
        np.testing.assert_allclose(actual,_oracle(source,rhs,kind,bias[:n] if fused else None,
                                                 residual[:m,:n] if fused else None),rtol=.015,atol=.015)
        assert function._nvidia_lhs_last_program is program
        assert function.native_lhs_packages()==packages
        assert len(function._bounded_lhs.programs)==1
        assert len(function._nvidia_lhs_prepared_calls)==1
        assert all(r["native_call_binding"]=="prepared_cpp_dynamic_tensor_matmul"
                   for r in function._nvidia_lhs_last_receipts)
        report=function.compile_report()
        assert report.plan_hash==function.runtime_artifact().artifact_hash
        np.testing.assert_array_equal(first,saved)
    function.close_native_storage()


@ts.jit(target="nvidia_sm120",shape_bounds={"M":128,"N":64,"K":1024},deterministic=True)
def bounded_deterministic(source,rhs):
    return ts.ops.matmul(ts.ops.rmsnorm(source,eps=1e-5),rhs,output_dtype="fp32")


def test_deterministic_bounded_calls_keep_call_time_gate_without_retrace(monkeypatch):
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16)
    bounded_deterministic(source,rhs)
    calls=[]
    original=bounded_deterministic._enforce_call_time_stochastic_certificate
    def gate(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(bounded_deterministic,"_enforce_call_time_stochastic_certificate",gate)
    def forbidden(*args,**kwargs):raise AssertionError("warm deterministic call retraced")
    monkeypatch.setattr(bounded_deterministic,"_traced_autodiff_module",forbidden)
    bounded_deterministic(source[:3,:11],rhs[:11,:7])
    assert calls==[1]
    bounded_deterministic.close_native_storage()


@pytest.mark.parametrize("axes",[("M",),("N",),("K",),("M","N"),("M","K"),("N","K"),("M","N","K")])
def test_independently_bounded_axes_reuse_one_program(axes):
    capacities={"M":128,"N":64,"K":1024}
    function=ts.jit(target="nvidia_sm120",shape_bounds={a:capacities[a] for a in axes})(bounded_rms._fn)
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16)
    function(source,rhs)
    program=function._nvidia_lhs_last_program
    m,k,n=(capacities[a] if a in axes else initial for a,initial in (("M",17),("K",35),("N",19)))
    x=np.ones((m,k),np.float16)
    b=np.ones((k,n),np.float16)
    try:
        actual=function(x,b)
        np.testing.assert_allclose(actual,_oracle(x,b,"rmsnorm"),rtol=.015,atol=.015)
        assert function._nvidia_lhs_last_program is program
        assert (program.edge.dynamic_m,program.edge.dynamic_n,program.edge.dynamic_k)==tuple(a in axes for a in ("M","N","K"))
    finally:
        function.close_native_storage()


@pytest.mark.parametrize("malformation",["m_bound","n_bound","k_bound","k_mismatch","zero","rank","dtype"])
def test_invalid_warm_frames_clear_success_before_gpu(malformation,monkeypatch):
    function=ts.jit(target="nvidia_sm120",shape_bounds={"M":17,"N":19,"K":35})(bounded_rms._fn)
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16)
    function(source,rhs)
    lib=rt._load_nvidia_ptx_launch()
    def forbidden(*args,**kwargs):raise AssertionError("invalid warm input reached tracing or CUDA")
    monkeypatch.setattr(function,"_traced_autodiff_module",forbidden)
    monkeypatch.setattr(lib,"tessera_nvidia_matmul_context_identity",forbidden)
    if malformation=="m_bound":source=np.ones((18,35),np.float16)
    elif malformation=="n_bound":rhs=np.ones((35,20),np.float16)
    elif malformation=="k_bound":source=np.ones((17,36),np.float16)
    elif malformation=="k_mismatch":rhs=np.ones((34,19),np.float16)
    elif malformation=="zero":source=source[:0]
    elif malformation=="rank":source=source[0]
    else:
        # dtype changes are a new specialization; reject before its trace by
        # using an unsupported source/RHS storage pair.
        source=source.astype(np.float32)
        monkeypatch.undo()
    try:
        with pytest.raises((ValueError,RuntimeError)):
            function(source,rhs)
        assert function.native_lhs_packages()==()
        assert function._nvidia_lhs_last_receipts==()
    finally:
        function.close_native_storage()


def test_storage_static_axes_and_close_are_real_specialization_boundaries(monkeypatch):
    bounds={"M":128,"K":1024}
    function=ts.jit(target="nvidia_sm120",shape_bounds=bounds)(bounded_rms._fn)
    bounds["M"]=1
    first=None
    for dtype,n in (("fp16",19),("bf16",19),("fp16",23)):
        source=np.ones((17,35),_storage(dtype))
        rhs=np.ones((35,n),_storage(dtype))
        actual=function(source,rhs)
        np.testing.assert_allclose(actual,_oracle(source,rhs,"rmsnorm"),rtol=.015,atol=.015)
        if first is None:first=function._nvidia_lhs_last_program
    assert len(function._bounded_lhs.programs)==3
    function.close_native_storage()
    assert not function._nvidia_lhs_prepared_calls
    def forbidden(*args,**kwargs):raise AssertionError("close/rebind unexpectedly recompiled")
    monkeypatch.setattr(function,"_traced_autodiff_module",forbidden)
    source=np.ones((3,11),np.float16)
    rhs=np.ones((11,19),np.float16)
    function(source,rhs)
    assert function._nvidia_lhs_last_program is first
    function.close_native_storage()


@ts.jit(target="nvidia_sm120",shape_bounds={"M":128,"N":64,"K":1024})
def bounded_permuted(residual,rhs,source,bias):
    return ts.ops.matmul(ts.ops.layer_norm(source,eps=1e-5),rhs,bias=bias,
                         activation="relu",residual=residual,output_dtype="fp16")


def test_permuted_named_call_uses_compiled_roles():
    rng=np.random.default_rng(120610)
    source=(rng.normal(size=(17,35))*.2).astype(np.float16)
    rhs=(rng.normal(size=(35,19))*.2).astype(np.float16)
    bias=(rng.normal(size=19)*.2).astype(np.float32)
    residual=(rng.normal(size=(17,19))*.2).astype(np.float32)
    bounded_permuted(residual,rhs,source,bias)
    try:
        actual=bounded_permuted(source=source[:3,:11],rhs=rhs[:11,:7],bias=bias[:7],residual=residual[:3,:7])
        np.testing.assert_allclose(actual,_oracle(source[:3,:11],rhs[:11,:7],"layernorm",
                                                  bias[:7],residual[:3,:7]),rtol=.015,atol=.015)
        assert len(bounded_permuted._bounded_lhs.programs)==1
    finally:
        bounded_permuted.close_native_storage()


def test_bound_config_is_shared_by_explicit_compile_api():
    source=np.ones((3,11),np.float16)
    rhs=np.ones((11,7),np.float16)
    program=bounded_rms.compile_native_lhs_matmul(source,rhs)
    assert (program.edge.m,program.edge.k,program.edge.n)==(128,1024,64)
    assert program.edge.dynamic_m and program.edge.dynamic_k and program.edge.dynamic_n


def test_public_program_and_native_owner_eviction_retire_together():
    function=ts.jit(target="nvidia_sm120",shape_bounds={"M":4,"K":16})(bounded_rms._fn)
    source=np.ones((3,11),np.float16)
    retired=None
    for n in range(1,26):
        function(source,np.ones((11,n),np.float16))
        if n==1:retired=next(iter(function._nvidia_lhs_prepared_calls.values()))
    try:
        assert len(function._bounded_lhs.programs)==len(function._bounded_lhs.graphs)==24
        assert len(function._nvidia_lhs_prepared_calls)==24
        assert not retired._finalizer.alive
        function(source,np.ones((11,1),np.float16))
        assert len(function._bounded_lhs.programs)==len(function._nvidia_lhs_prepared_calls)==24
    finally:
        function.close_native_storage()


def _cuda_driver():
    import ctypes as ct
    cuda=ct.CDLL("libcuda.so.1")
    cuda.cuCtxGetCurrent.argtypes=[ct.POINTER(ct.c_void_p)]
    cuda.cuCtxCreate_v2.argtypes=[ct.POINTER(ct.c_void_p),ct.c_uint,ct.c_int]
    cuda.cuCtxPopCurrent_v2.argtypes=[ct.POINTER(ct.c_void_p)]
    cuda.cuCtxDestroy_v2.argtypes=[ct.c_void_p]
    return cuda


def _pop_and_destroy(cuda,original,other):
    # cuCtxCreate pushed `other`; pop it rather than replace the stack top with
    # cuCtxSetCurrent, which leaves `original` on the stack twice and lets a
    # later cuCtxSetCurrent(NULL) reveal it instead of unbinding the thread.
    import ctypes as ct
    popped,current=ct.c_void_p(),ct.c_void_p()
    assert cuda.cuCtxPopCurrent_v2(ct.byref(popped))==0 and popped.value==other.value
    assert cuda.cuCtxGetCurrent(ct.byref(current))==0 and current.value==original.value
    assert cuda.cuCtxDestroy_v2(other)==0


def test_bounded_public_owner_is_context_scoped():
    import ctypes as ct
    import gc
    from tessera.compiler.prepared_nvidia_matmul import current_context_identity
    function=ts.jit(target="nvidia_sm120",shape_bounds={"M":17,"N":19,"K":35})(bounded_rms._fn)
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16)
    function(source,rhs)
    cuda=_cuda_driver()
    original,other=ct.c_void_p(),ct.c_void_p()
    assert cuda.cuCtxGetCurrent(ct.byref(original))==0
    original_identity=current_context_identity()
    assert cuda.cuCtxCreate_v2(ct.byref(other),0,0)==0
    try:
        other_identity=current_context_identity()
        assert other_identity!=original_identity
        actual=function(source[:3,:11],rhs[:11,:7])
        np.testing.assert_allclose(actual,_oracle(source[:3,:11],rhs[:11,:7],"rmsnorm"),rtol=.015,atol=.015)
        assert len(function._bounded_lhs.programs)==1
        # A native shape frame owns the first call since 2026-10-21, so the
        # prepared owner created here is the only one; its key carries the
        # context it was prepared in.
        calls=function._nvidia_lhs_prepared_calls
        assert {key[-1] for key in calls}=={other_identity}
        assert all(key[-1]==call.context_identity for key,call in calls.items())
    finally:
        # Close while `other` is alive, even when an assertion failed: a
        # failed test's traceback would otherwise keep these owners past the
        # context's destruction.
        function.close_native_storage()
        _pop_and_destroy(cuda,original,other)
    try:
        actual=function(source[:1,:1],rhs[:1,:1])
        np.testing.assert_allclose(actual,_oracle(source[:1,:1],rhs[:1,:1],"rmsnorm"),rtol=.015,atol=.015)
        assert other_identity not in {key[-1] for key in function._nvidia_lhs_prepared_calls}
    finally:
        function.close_native_storage()
    gc.collect()


def test_owner_of_destroyed_context_is_forgotten_not_closed(monkeypatch):
    import ctypes as ct
    import gc
    from tessera.compiler import prepared_nvidia_matmul as prepared
    lib=rt._load_nvidia_ptx_launch()
    assert hasattr(lib,"tessera_nvidia_matmul_forget"), "matching NVIDIA launch library required"
    native_close=lib.tessera_nvidia_matmul_close
    closed=[]
    def recording_close(handle):
        closed.append(handle)
        return native_close(handle)
    monkeypatch.setattr(lib,"tessera_nvidia_matmul_close",recording_close)
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16)
    kept=ts.jit(target="nvidia_sm120",shape_bounds={"M":17,"N":19,"K":35})(bounded_rms._fn)
    leaked=ts.jit(target="nvidia_sm120",shape_bounds={"M":17,"N":19,"K":35})(bounded_rms._fn)
    kept(source,rhs)
    leaked(source,rhs)
    cuda=_cuda_driver()
    original,other=ct.c_void_p(),ct.c_void_p()
    assert cuda.cuCtxGetCurrent(ct.byref(original))==0
    assert cuda.cuCtxCreate_v2(ct.byref(other),0,0)==0
    try:
        identity=prepared.current_context_identity()
        for function in (kept,leaked):
            function(source[:3,:11],rhs[:11,:7])
        kept_handles={call.handle for call in kept._nvidia_lhs_prepared_calls.values()}
        leaked_handles={call.handle for key,call in leaked._nvidia_lhs_prepared_calls.items()
                        if key[-1]==identity}
        assert kept_handles and leaked_handles
        # A live context still closes natively.
        kept.close_native_storage()
        assert kept_handles<=set(closed)
    finally:
        _pop_and_destroy(cuda,original,other)
    # `leaked` outlived its context, as a failed test's traceback would keep it.
    with pytest.warns(ResourceWarning,match="not provably live"):
        del leaked,function
        gc.collect()
    assert not leaked_handles&set(closed)
    pending={(handle,identity) for handle in leaked_handles}
    assert pending<=set(prepared.deferred_handles())
    # A dead context never becomes live again: a later settle keeps them.
    prepared.close_deferred()
    assert pending<=set(prepared.deferred_handles()) and not leaked_handles&set(closed)
    # What interpreter exit runs: forget them, natively too, so no ~Owner
    # pushes the dead context when the registry is destroyed.
    prepared.close_deferred(final=True)
    assert not pending&set(prepared.deferred_handles())
    assert pending<=set(prepared.forgotten_handles())
    assert all(native_close(handle)!=0 for handle in leaked_handles)
    kept.close_native_storage()


def test_owner_of_live_noncurrent_context_is_deferred_then_closed(monkeypatch):
    import ctypes as ct
    import gc
    from tessera.compiler import prepared_nvidia_matmul as prepared
    lib=rt._load_nvidia_ptx_launch()
    native_close=lib.tessera_nvidia_matmul_close
    closed=[]
    def recording_close(handle):
        closed.append(handle)
        return native_close(handle)
    monkeypatch.setattr(lib,"tessera_nvidia_matmul_close",recording_close)
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16)
    function=ts.jit(target="nvidia_sm120",shape_bounds={"M":17,"N":19,"K":35})(bounded_rms._fn)
    function(source,rhs)
    cuda=_cuda_driver()
    cuda.cuCtxPushCurrent_v2.argtypes=[ct.c_void_p]
    original,other,popped=ct.c_void_p(),ct.c_void_p(),ct.c_void_p()
    assert cuda.cuCtxGetCurrent(ct.byref(original))==0
    assert cuda.cuCtxCreate_v2(ct.byref(other),0,0)==0
    pushed=True
    try:
        identity=prepared.current_context_identity()
        function(source[:3,:11],rhs[:11,:7])
        handles={call.handle for key,call in function._nvidia_lhs_prepared_calls.items()
                 if key[-1]==identity}
        assert handles
        # `other` stays alive but is no longer current when the owner dies.
        assert cuda.cuCtxPopCurrent_v2(ct.byref(popped))==0 and popped.value==other.value
        pushed=False
        with pytest.warns(ResourceWarning,match="not provably live"):
            del function
            gc.collect()
        assert not handles&set(closed)
        assert {(handle,identity) for handle in handles}<=set(prepared.deferred_handles())
        assert cuda.cuCtxPushCurrent_v2(other)==0
        pushed=True
        prepared.close_deferred()
        assert handles<=set(closed)
        assert not {(handle,identity) for handle in handles}&set(prepared.deferred_handles())
        assert not {(handle,identity) for handle in handles}&set(prepared.forgotten_handles())
    finally:
        if pushed:
            _pop_and_destroy(cuda,original,other)
        else:
            assert cuda.cuCtxDestroy_v2(other)==0
