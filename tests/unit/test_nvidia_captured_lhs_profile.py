"""Captured native profiles must preserve compiler-owned frames and leases."""
import math
import numpy as np
import pytest
import ml_dtypes
from tessera import runtime as rt
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tessera.compiler.nvidia_tensor_lhs import package_traced_lhs
from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
from tests.unit.test_sm120_scheduled_activation import _frontend_program

@pytest.fixture
def sm120():
    if find_tessera_opt() is None or rt._nvidia_device_name() != "sm_120":
        pytest.skip("requires owning SM120 and production compiler")

def program(storage,mode,bounded=False):
    dtype=np.float16 if storage=="f16" else ml_dtypes.bfloat16
    _,values,public=_frontend_program(dtype,(17,64,19))
    graph,_=public._trace_frontend_capture(values,{})
    options={"shape_bounds":{"M":33,"K":96,"N":35}} if bounded else {}
    return package_traced_lhs(graph,activation_schedule=mode,**options),values,dtype

def oracle(a,b):
    wide=a.astype(np.float32)
    normalized=(wide/np.sqrt(np.mean(wide*wide,axis=-1,keepdims=True)+np.float32(1e-5))).astype(a.dtype)
    return np.maximum(0,normalized).astype(np.float32) @ b.astype(np.float32)

@pytest.mark.parametrize("storage",["f16","bf16"])
@pytest.mark.parametrize("mode",["serial","flat_128"])
@pytest.mark.parametrize("bounded",[False,True])
def test_captured_profile_reuses_valid_frame_without_retaining_graphs(sm120,storage,mode,bounded):
    p,initial,dtype=program(storage,mode,bounded)
    owner=PreparedLhsCall(p)
    rng=np.random.default_rng(151051)
    retained=[]
    try:
        shapes=((17,64,19),(33,96,35),(7,32,9),(17,64,19)) if bounded else ((17,64,19),)*2
        for shape in shapes:
            m,k,n=shape
            a=rng.uniform(-1,1,(m,k)).astype(dtype)
            b=rng.uniform(-.25,.25,(k,n)).astype(dtype)
            expected=oracle(a,b)
            output,_=owner((a,b))
            before=owner.scratch_stats()
            for repeats in (7,17):
                profile=owner.profile(repeats=repeats,captured=True)
                assert profile["timing_domain"]=="cuda_graph_batch"
                assert len(profile["grouped_stage_ms"])==3
                assert all(math.isfinite(x) and x>=0 for x in [profile["program_ms"],*profile["grouped_stage_ms"]])
                np.testing.assert_allclose(profile["output"],expected,atol=.02 if storage=="bf16" else .003,rtol=.02 if storage=="bf16" else .003)
            assert owner.scratch_stats()==before
            legacy=owner.profile(repeats=7)
            assert legacy["timing_domain"]=="native_enqueue_event"
            np.testing.assert_allclose(legacy["output"],expected,atol=.02 if storage=="bf16" else .003,rtol=.02 if storage=="bf16" else .003)
            retained.append((output,output.copy()))
        for output,snapshot in retained:np.testing.assert_array_equal(output,snapshot)
    finally:owner.close()

@pytest.mark.parametrize("storage",["f16","bf16"])
def test_captured_profile_rejects_stale_lease_and_does_not_break_sibling(sm120,storage):
    p,values,_=program(storage,"flat_128")
    first,second=PreparedLhsCall(p),PreparedLhsCall(p)
    try:
        first(values)
        first.profile(repeats=7,captured=True)
        second(values)
        with pytest.raises(RuntimeError,match="stale"):
            first.profile(repeats=7,captured=True)
        first.close()
        result=second.profile(repeats=7,captured=True)
        np.testing.assert_allclose(result["output"],oracle(*values),atol=.02 if storage=="bf16" else .003,rtol=.02 if storage=="bf16" else .003)
    finally:
        first.close()
        second.close()

def test_capture_profile_bounds_and_old_provider_are_explicit(sm120):
    p,values,_=program("f16","serial")
    owner=PreparedLhsCall(p)
    try:
        owner(values)
        for repeats in (0,-1,True,4097):
            with pytest.raises(ValueError):
                owner.profile(repeats=repeats,captured=True)
        provider=owner.lib
        class OldProvider:
            def __getattr__(self,name):
                if name=="tessera_nvidia_matmul_profile_captured":
                    raise AttributeError(name)
                return getattr(provider,name)
        owner.lib=OldProvider()
        with pytest.raises(RuntimeError,match="captured profiling"):
            owner.profile(repeats=7,captured=True)
        assert owner.profile(repeats=7)["timing_domain"]=="native_enqueue_event"
    finally:owner.close()

@pytest.mark.parametrize("storage",["f16","bf16"])
def test_captured_two_sided_bounded_program(sm120,storage):
    import tessera as ts
    dtype=np.float16 if storage=="f16" else ml_dtypes.bfloat16
    public=ts.from_text("""def both(a,b):
    x = tessera.ops.relu(a)
    y = tessera.ops.relu(b)
    return tessera.ops.matmul(x,y,output_dtype="fp32")
""",target="nvidia_sm120")
    rng=np.random.default_rng(151052)
    def inputs(m,k,n):
        return (rng.uniform(-1,1,(m,k)).astype(dtype),
                rng.uniform(-.25,.25,(k,n)).astype(dtype))
    initial=inputs(17,64,19)
    graph,_=public._trace_frontend_capture(initial,{})
    p=package_traced_lhs(graph,activation_schedule="flat_128",shape_bounds={"M":33,"K":96,"N":35})
    owner=PreparedLhsCall(p)
    retained=[]
    try:
        for shape in ((17,64,19),(33,96,35),(7,32,9)):
            a,b=inputs(*shape)
            expected=np.maximum(0,a).astype(np.float32) @ np.maximum(0,b).astype(np.float32)
            primal,_=owner((a,b))
            profile=owner.profile(repeats=11,captured=True)
            assert len(profile["grouped_stage_ms"])==3
            np.testing.assert_allclose(profile["output"],expected,atol=.02 if storage=="bf16" else .003,rtol=.02 if storage=="bf16" else .003)
            retained.append((primal,primal.copy()))
        for output,snapshot in retained:np.testing.assert_array_equal(output,snapshot)
    finally:owner.close()
