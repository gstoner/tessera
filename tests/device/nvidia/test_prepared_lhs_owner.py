"""Owning-device native lifetime proof for the verified two-kernel edge."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import ctypes as ct
import numpy as np
import pytest
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lhs_tensor_jit import (
    rms_lhs, layer_lhs, softmax_lhs, rms_lhs_fused, _oracle, _storage)
from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall

pytestmark = pytest.mark.skipif(not nvidia_cuda_host_ready(), reason="owning NVIDIA host required")


@pytest.mark.parametrize("dtype", ["fp16","bf16"])
@pytest.mark.parametrize("order", ["C","F"])
@pytest.mark.parametrize("kind", ["rmsnorm","layernorm","softmax"])
@pytest.mark.parametrize("shape", [(17,35,19),(128,1024,64)])
def test_native_owner_changed_values_and_output_lifetime(kind,dtype,order,shape,monkeypatch):
    from tessera import runtime as rt
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    storage = _storage(dtype)
    rng = np.random.default_rng(120507)
    m,k,n = shape
    source = (rng.normal(size=(m,k))*.2).astype(storage)
    rhs = np.array(rng.normal(size=(k,n))*.2,dtype=storage,order=order)
    fn = {"rmsnorm":rms_lhs,"layernorm":layer_lhs,"softmax":softmax_lhs}[kind]
    call = PreparedLhsCall(fn.compile_native_lhs_matmul(source,rhs))
    def unavailable(*args,**kwargs):
        raise AssertionError("prepared invocation escaped native ownership")
    monkeypatch.setattr(rt,"launch",unavailable)
    monkeypatch.setattr(NvidiaDeviceSession,"__init__",unavailable)
    try:
        first,receipt = call((source,rhs))
        saved = first.copy()
        assert receipt["native_call_binding"] == "prepared_cpp_tensor_matmul"
        assert len(receipt["component_receipts"]) == 2
        before = call.scratch_stats()
        for scale in (.5,1.25,-.75):
            changed = (source.astype(np.float32)*scale).astype(storage)
            actual,_ = call((changed,rhs))
            np.testing.assert_allclose(actual,_oracle(changed,rhs,kind),atol=.015,rtol=.015)
        assert call.scratch_stats() == before
        np.testing.assert_array_equal(first,saved)
        assert not np.shares_memory(first,actual)
    finally:
        call.close()
    with pytest.raises(ValueError,match="closed"):
        call((source,rhs))


def test_native_owner_shared_scratch_concurrent_calls():
    storage=np.float16
    source=np.arange(17*35,dtype=storage).reshape(17,35)/1000
    rhs=np.ones((35,19),storage)
    calls=[PreparedLhsCall(fn.compile_native_lhs_matmul(source,rhs))
           for fn in (rms_lhs,layer_lhs)]
    try:
        # First frame grows the shared arena; subsequent owners lease it.
        calls[0]((source,rhs))
        before=calls[0].scratch_stats()
        def invoke(index):
            actual,_=calls[index%2]((source,rhs))
            expected=_oracle(source,rhs,("rmsnorm","layernorm")[index%2])
            np.testing.assert_allclose(actual,expected,atol=.015,rtol=.015)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(invoke,range(16)))
        assert calls[0].scratch_stats()==calls[1].scratch_stats()==before
    finally:
        for call in calls:call.close()


def test_native_owner_bad_host_view_and_semantics_do_not_grow_scratch():
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16)
    call=PreparedLhsCall(rms_lhs.compile_native_lhs_matmul(source,rhs))
    try:
        before=call.scratch_stats()
        with pytest.raises(RuntimeError,match="shape/stride/dtype/capacity"):
            call((source.astype(np.float32),rhs))
        assert call.scratch_stats()==before
        semantics=deepcopy(call.program.semantics)
        call.program.semantics["producer_attrs"]["eps"]=.5
        with pytest.raises(ValueError,match="contract changed"):
            call((source,rhs))
        assert call.scratch_stats()==before
        call.program.semantics.clear()
        call.program.semantics.update(semantics)
        actual,_=call((source,rhs))
        np.testing.assert_allclose(actual,_oracle(source,rhs,"rmsnorm"),atol=.015,rtol=.015)
    finally:
        call.close()


def test_native_owner_cannot_replace_producer():
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16)
    call=PreparedLhsCall(rms_lhs.compile_native_lhs_matmul(source,rhs))
    try:
        p=call.program.edge.producer
        image=ct.create_string_buffer(p.image.payload)
        status=call.lib.tessera_nvidia_matmul_attach_producer(
            call.handle,image,len(p.image.payload),p.descriptor.entry_symbol.encode(),0)
        assert status
        assert b"already attached" in call.lib.tessera_nvidia_matmul_last_error()
        actual,_=call((source,rhs))
        np.testing.assert_allclose(actual,_oracle(source,rhs,"rmsnorm"),atol=.015,rtol=.015)
    finally:
        call.close()


def test_public_warm_call_owns_both_packages_below_python(monkeypatch):
    from tessera import runtime as rt
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16)
    bias=np.ones(19,np.float32)
    residual=np.ones((17,19),np.float32)
    args=source,rhs,bias,residual
    actual=rms_lhs_fused(*args)
    def forbidden(*args,**kwargs):
        raise AssertionError("warm public call used portable launch or compiler")
    monkeypatch.setattr(rt,"launch",forbidden)
    monkeypatch.setattr(rms_lhs_fused,"compile_native_lhs_matmul",forbidden)
    np.testing.assert_array_equal(actual,rms_lhs_fused(*args))
    assert all(r["native_call_binding"]=="prepared_cpp_tensor_matmul"
               for r in rms_lhs_fused._nvidia_lhs_last_receipts)


def test_rejected_consumer_drains_queued_producer_before_scratch_release():
    from tessera.compiler.prepared_nvidia_matmul import HostView
    source=np.ones((1024,1024),np.float16)
    rhs=np.ones((1024,19),np.float16)
    output=np.full((1024,19),-123,np.float32)
    owner=PreparedLhsCall(rms_lhs.compile_native_lhs_matmul(source,rhs))
    lib=owner.lib
    # An isolated private-C-ABI test image with an intentionally incompatible
    # thread bound makes consumer submission fail after the real producer queues.
    entry="nvidia_sm120_scheduled_matmul_failure_probe"
    ptx=ct.create_string_buffer(("""
.version 8.8
.target sm_120a
.address_size 64
.visible .entry """+entry+"""(
 .param .u64 a, .param .u64 b, .param .u64 d,
 .param .u64 m, .param .u64 n, .param .u64 k
)
.maxntid 1, 1, 1
{ ret; }
""").encode())
    dims=(ct.c_int64*3)(1024,19,1024)
    handle=ct.c_uint64()
    owner._check(lib.tessera_nvidia_matmul_prepare(
        ptx,len(ptx.value),entry.encode(),dims,2,0,0,1,0,ct.byref(handle)))
    try:
        producer=owner.program.edge.producer
        image=ct.create_string_buffer(producer.image.payload)
        owner._check(lib.tessera_nvidia_matmul_attach_producer(
            handle,image,len(producer.image.payload),producer.descriptor.entry_symbol.encode(),
            int(producer.descriptor.provenance["schedule"] == "cooperative_128")))
        arrays=(source,rhs,output)
        views=(HostView*3)()
        for view,array in zip(views,arrays,strict=True):
            view.data,view.bytes,view.rank=array.ctypes.data,array.nbytes,2
            view.dtype=2 if array.dtype==np.float16 else 1
            for axis in range(2):
                view.shape[axis],view.strides[axis]=array.shape[axis],array.strides[axis]
        status=lib.tessera_nvidia_matmul_invoke(handle,views,3)
        assert status
        assert b"launch prepared matmul" in lib.tessera_nvidia_matmul_last_error()
        np.testing.assert_array_equal(output,-123)
        # scratch_stats now also queries CUstream readiness. Success proves the
        # queued producer retired even though its consumer was rejected.
        owner.scratch_stats()  # configure the shared ctypes signature
        capacity,allocations=ct.c_size_t(),ct.c_size_t()
        owner._check(lib.tessera_nvidia_matmul_scratch_stats(
            handle,ct.byref(capacity),ct.byref(allocations)))
        actual,_=owner((source,rhs))
        np.testing.assert_allclose(actual,_oracle(source,rhs,"rmsnorm"),atol=.015,rtol=.015)
    finally:
        lib.tessera_nvidia_matmul_close(handle)
        owner.close()


def test_native_edge_context_guard_and_recovery():
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16)
    call=PreparedLhsCall(rms_lhs.compile_native_lhs_matmul(source,rhs))
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
            call((source,rhs))
    finally:
        assert cuda.cuCtxSetCurrent(original)==0
        assert cuda.cuCtxDestroy_v2(other)==0
    try:
        actual,_=call((source,rhs))
        np.testing.assert_allclose(actual,_oracle(source,rhs,"rmsnorm"),atol=.015,rtol=.015)
    finally:
        call.close()


def test_public_closed_edge_rebinds_new_native_handle():
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16)
    rms_lhs(source,rhs)
    closed=[]
    for call in rms_lhs._nvidia_lhs_prepared_calls.values():
        closed.append(call.handle)
    rms_lhs.close_native_storage()
    assert not rms_lhs._nvidia_lhs_prepared_calls
    actual=rms_lhs(source,rhs)
    np.testing.assert_allclose(actual,_oracle(source,rhs,"rmsnorm"),atol=.015,rtol=.015)
    assert any(c._finalizer.alive and c.handle not in closed
               for c in rms_lhs._nvidia_lhs_prepared_calls.values())


def test_warm_paired_native_compile_report_tracks_both_physical_packages():
    from tessera.compiler import compile_report as cr
    source=np.ones((17,35),np.float16)
    row=np.ones((35,19),np.float16,order="C")
    column=np.ones((35,19),np.float16,order="F")
    rms_lhs(source,row)
    rms_lhs(source,column)
    reports=[]
    for rhs in (row,column):
        with cr.capture_compile_reports() as sink:
            rms_lhs(source,rhs)
        assert len(sink)==1
        report=sink[0]
        assert report.target=="nvidia_sm120"
        assert set(report.ir_hashes)=={"graph_ir","schedule_ir","tile_ir","target_ir"}
        assert report.ir_hashes["graph_ir"]==cr.hash_ir_text(rms_lhs._nvidia_lhs_last_program.graph_ir)
        assert report.plan_hash==rms_lhs.runtime_artifact().artifact_hash
        assert "stage_hashes=ordered(producer,consumer)" in report.target_decision["nvidia_sm120"]
        assert "paired_native_packages.executable=True" in report.target_decision["nvidia_sm120"]
        assert report.fallback_reason is None
        reports.append(report)
    assert reports[0].plan_hash!=reports[1].plan_hash
    assert all(reports[0].ir_hashes[layer]!=reports[1].ir_hashes[layer]
               for layer in ("schedule_ir","tile_ir","target_ir"))


def test_failed_named_edge_does_not_report_previous_native_product():
    from tessera.compiler import compile_report as cr
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16)
    rms_lhs(source,rhs)
    assert rms_lhs._nvidia_lhs_last_receipts
    with cr.capture_compile_reports() as sink:
        with pytest.raises(ValueError,match="storage"):
            rms_lhs(source.astype(np.float32),rhs.astype(np.float32))
    assert len(sink)==1
    assert rms_lhs._nvidia_lhs_last_receipts==()
    assert rms_lhs._nvidia_lhs_last_program is None
    assert sink[0].plan_hash is None
    assert "paired_native_packages.executable=True" not in sink[0].target_decision["nvidia_sm120"]


@pytest.mark.parametrize("dtype", ["fp16","bf16"])
@pytest.mark.parametrize("order", ["C","F"])
def test_shared_native_matmul_owner_large_first_frame_stream_order(dtype,order):
    import tessera as ts
    from tests.device.nvidia.test_permuted_matmul_jit import plain
    fn=ts.jit(target="nvidia_sm120")(plain._fn)
    rng=np.random.default_rng(10245070)
    storage=_storage(dtype)
    source=(rng.normal(size=(128,1024))*.2).astype(storage)
    rhs=np.array(rng.normal(size=(1024,64))*.2,dtype=storage,order=order)
    try:
        for scale in (1.,-.75,1.25):
            active=(source.astype(np.float32)*scale).astype(storage)
            actual=fn(rhs,active)
            expected=active.astype(np.float64)@rhs.astype(np.float64)
            np.testing.assert_allclose(actual,expected,atol=.002,rtol=.002)
            assert fn.execution_kind=="native_gpu"
    finally:
        fn.close_native_storage()
