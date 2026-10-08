"""Owning gfx1201 proof for public typed FP8 scale JVP packages."""
import os
import numpy as np
import pytest
import tessera as ts

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="explicit owning gfx1201 execution required")

def scaled(a:ts.Tensor["M","K","fp8_e4m3"], b:ts.Tensor["K","N","fp8_e4m3"],
           sa:ts.Tensor["M","G","fp32"], sb:ts.Tensor["G","C","fp32"]):
    import tessera as ts
    return ts.ops.scaled_matmul(a,b,sa,sb,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})

def oracle(a,b,sa,sb,da,db):
    m,k=a.shape
    n=b.shape[1]
    primal=np.zeros((m,n),np.float64)
    tangent=np.zeros_like(primal)
    for g in range((k+127)//128):
        product=a[:,g*128:(g+1)*128].astype(np.float64) @ b[g*128:(g+1)*128].astype(np.float64)
        columns=np.arange(n)//128
        primal += product*sa[:,g,None]*sb[g,columns][None,:]
        tangent += product*(da[:,g,None]*sb[g,columns][None,:]+sa[:,g,None]*db[g,columns][None,:])
    return primal,tangent

@pytest.mark.parametrize("shape",[(17,19,256),(32,32,256),(200,19,256)])
@pytest.mark.parametrize("wrt",[("sa",),("sb",),("sa","sb")])
def test_public_scale_jvp(shape,wrt,monkeypatch):
    import ml_dtypes
    import tessera as ts
    from tessera import runtime
    assert runtime._rocm_live_arch()=="gfx1201"
    m,n,k=shape
    rng=np.random.default_rng(407)
    a=rng.choice([-.5,0,.25,1],(m,k)).astype(ml_dtypes.float8_e4m3fn)
    b=rng.choice([-1,0,.5,2],(k,n)).astype(ml_dtypes.float8_e4m3fn)
    sa=rng.uniform(.2,1,(m,k//128)).astype(np.float32)
    sb=rng.uniform(.2,1,(k//128,(n+127)//128)).astype(np.float32)
    da=rng.uniform(-.1,.1,sa.shape).astype(np.float32) if "sa" in wrt else np.zeros_like(sa)
    db=rng.uniform(-.1,.1,sb.shape).astype(np.float32) if "sb" in wrt else np.zeros_like(sb)
    compiled=ts.jit(target="rocm",autodiff="forward",wrt=wrt)(scaled)
    seeds=tuple({"sa":da,"sb":db}[name] for name in wrt)
    actual=compiled.native_jvp(a,b,sa,sb,tangents=seeds)
    expected=oracle(a,b,sa,sb,da,db)
    for got,want in zip(actual,expected):
        np.testing.assert_allclose(got,want,rtol=3e-5,atol=3e-5)
    receipt=compiled.last_jvp_execution
    assert receipt["execution_kind"]=="native_gpu"
    assert receipt["evidence_target"]=="rocm_gfx1201"
    assert receipt["family"]=="scaled_product_program"
    assert receipt["frontend_authority"]=="tracer"
    # Reuse the same public package with changed tangent values.
    changed=tuple(seed*-.5 for seed in seeds)
    repeated=compiled.native_jvp(a,b,sa,sb,tangents=changed)
    np.testing.assert_allclose(repeated[0],expected[0],rtol=3e-5,atol=3e-5)
    np.testing.assert_allclose(repeated[1],expected[1]*-.5,rtol=3e-5,atol=3e-5)
    epsilon=1e-3
    plus=oracle(a,b,sa.astype(np.float64)+epsilon*da,sb.astype(np.float64)+epsilon*db,da,db)[0]
    minus=oracle(a,b,sa.astype(np.float64)-epsilon*da,sb.astype(np.float64)-epsilon*db,da,db)[0]
    np.testing.assert_allclose(actual[1],(plus-minus)/(2*epsilon),rtol=3e-5,atol=3e-5)

def test_native_owner_cache_isolation(monkeypatch):
    import ctypes as c
    import ml_dtypes
    from tessera import runtime
    from tessera.compiler.native_scaled_program import NativeScaledProgram, PreparedScaledProgram
    assert runtime._rocm_live_arch()=="gfx1201"
    monkeypatch.setenv("TESSERA_ROCM_PROGRAM_CACHE","1")
    lib=runtime._load_rocm_native_movement_runtime()
    lib.tessera_rocm_program_cache_stats.argtypes=[c.POINTER(c.c_uint64)]*4
    lib.tessera_rocm_program_cache_clear.argtypes=[]
    def stats():
        values=[c.c_uint64() for _ in range(4)]
        assert lib.tessera_rocm_program_cache_stats(*(c.byref(v) for v in values))==0
        return [v.value for v in values]
    assert lib.tessera_rocm_program_cache_clear()==0
    a=np.full((17,256),.25,dtype=ml_dtypes.float8_e4m3fn)
    b=np.full((256,19),.5,dtype=ml_dtypes.float8_e4m3fn)
    sa=np.ones((17,2),np.float32);sb=np.ones((2,1),np.float32)
    da=sa*.1;db=sb*.2
    fn=ts.jit(target="rocm",autodiff="forward",wrt=("sa","sb"))(scaled)
    fn.native_jvp(a,b,sa,sb,tangents=(da,db))
    package=NativeScaledProgram.from_manifest(next(iter(fn._native_jvp_packages.values())).contract["steps"][0]["child_metadata"]["native_scaled_program"])
    assert lib.tessera_rocm_program_cache_clear()==0
    before=stats()
    with PreparedScaledProgram(package,[a,b,sa,sb,da,db],runtime_library=lib._name) as owner:
        old_handle=owner.handle.value
        old_generation,_=owner.invoke()
        expected=owner.read(old_generation)
    assert stats()[2]==1
    with PreparedScaledProgram(package,[a,b,sa,sb,da*2,db*2],runtime_library=lib._name) as owner:
        assert owner.handle.value!=old_handle
        generation,_=owner.invoke()
        assert generation>old_generation
        got=owner.read(generation)
        np.testing.assert_allclose(got[0],expected[0],rtol=1e-5)
        np.testing.assert_allclose(got[1],expected[1]*2,rtol=1e-5)
        # The old handle cannot access the reused allocation.
        assert lib.tessera_rocm_program_close(c.c_uint64(old_handle))==1
        output=np.empty((17,19),np.float32)
        slot=owner.program["outputs"][0]
        assert lib.tessera_rocm_program_read(owner.handle,slot,old_generation,c.c_void_p(output.ctypes.data),output.nbytes)==10
        # A live owner is never lent to another invocation with the same ABI.
        with PreparedScaledProgram(package,[a,b,sa,sb,da*-3,db*-3],runtime_library=lib._name) as other:
            assert other.handle.value != owner.handle.value
            other_generation,_=other.invoke()
            np.testing.assert_allclose(other.read(other_generation)[1],expected[1]*-3,rtol=1e-5)
            np.testing.assert_allclose(owner.read(generation)[1],expected[1]*2,rtol=1e-5)
        # Clearing idle entries leaves active owners valid.
        assert lib.tessera_rocm_program_cache_clear()==0
        np.testing.assert_allclose(owner.read(generation)[0],expected[0],rtol=1e-5)
    after=stats()
    assert after[0]>before[0] and after[2]==1 and after[3]<=128*1024*1024
    assert lib.tessera_rocm_program_cache_clear()==0
    assert stats()[2:]==[0,0]
    # Transfer modes are part of ownership identity and cannot alias in cache.
    monkeypatch.setenv("TESSERA_ROCM_PROGRAM_PINNED","0")
    with PreparedScaledProgram(package,[a,b,sa,sb,da,db],runtime_library=lib._name) as pageable:
        generation,_=pageable.invoke()
        np.testing.assert_allclose(pageable.read(generation)[0],expected[0],rtol=1e-5)
    before_mode=stats()
    monkeypatch.setenv("TESSERA_ROCM_PROGRAM_PINNED","1")
    with PreparedScaledProgram(package,[a,b,sa,sb,da,db],runtime_library=lib._name) as pinned:
        generation,_=pinned.invoke()
        np.testing.assert_allclose(pinned.read(generation)[1],expected[1],rtol=1e-5)
    assert stats()[1]==before_mode[1]+1
    assert stats()[2]==2
    assert lib.tessera_rocm_program_cache_clear()==0

def scaled_nk(a:ts.Tensor["M","K","fp8_e4m3"],b:ts.Tensor["N","K","fp8_e4m3"],
              sa:ts.Tensor["M","G","fp32"],sb:ts.Tensor["G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=True,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})

@pytest.mark.parametrize("shape",[(17,129,256),(200,129,1536),(33,257,2048)])
def test_public_scale_jvp_transposed_rhs_multiple_scale_blocks(shape):
    import ml_dtypes
    from tessera import runtime
    assert runtime._rocm_live_arch()=="gfx1201"
    m,n,k=shape
    rng=np.random.default_rng(411)
    a=rng.choice([-.5,0,.25,1],(m,k)).astype(ml_dtypes.float8_e4m3fn)
    logical_b=rng.choice([-1,0,.5,2],(k,n)).astype(ml_dtypes.float8_e4m3fn)
    b=np.ascontiguousarray(logical_b.T)
    sa=rng.uniform(.2,1,(m,k//128)).astype(np.float32)
    sb=rng.uniform(.2,1,(k//128,(n+127)//128)).astype(np.float32)
    da=rng.uniform(-.1,.1,sa.shape).astype(np.float32)
    db=rng.uniform(-.1,.1,sb.shape).astype(np.float32)
    fn=ts.jit(target="rocm",autodiff="forward",wrt=("sa","sb"))(scaled_nk)
    actual=fn.native_jvp(a,b,sa,sb,tangents=(da,db))
    expected=oracle(a,logical_b,sa,sb,da,db)
    for got,want in zip(actual,expected):
        np.testing.assert_allclose(got,want,rtol=4e-5,atol=1e-4)
    assert fn.last_jvp_execution["family"]=="scaled_product_program"
    assert fn.last_jvp_execution["evidence_target"]=="rocm_gfx1201"
    epsilon=1e-3
    plus=oracle(a,logical_b,sa.astype(np.float64)+epsilon*da,sb.astype(np.float64)+epsilon*db,da,db)[0]
    minus=oracle(a,logical_b,sa.astype(np.float64)-epsilon*da,sb.astype(np.float64)-epsilon*db,da,db)[0]
    np.testing.assert_allclose(actual[1],(plus-minus)/(2*epsilon),rtol=4e-5,atol=1e-4)

@pytest.mark.parametrize("nk",[False,True])
@pytest.mark.parametrize("wrt",[("sa",),("sb",),("sa","sb")])
def test_public_shared_rhs_scale_jvp(nk,wrt,monkeypatch):
    from tessera import runtime
    from tests.unit import test_rocm_shared_scaled_batch as shared
    assert runtime._rocm_live_arch()=="gfx1201"
    values,primal=shared.batch_inputs((3,7,19,256),"fp32",nk)
    a,b,sa,sb=values
    logical_b=b.T if nk else b
    rng=np.random.default_rng(863)
    da=rng.uniform(-.1,.1,sa.shape).astype(np.float32) if "sa" in wrt else np.zeros_like(sa)
    db=rng.uniform(-.1,.1,sb.shape).astype(np.float32) if "sb" in wrt else np.zeros_like(sb)
    tangent=np.zeros_like(primal)
    for g in range(2):
        product=a[...,g*128:(g+1)*128].astype(np.float64)@logical_b[g*128:(g+1)*128].astype(np.float64)
        tangent+=product*(da[...,g,None]*sb[g,0]+sa[...,g,None]*db[g,0])
    source=shared.shared_fp32_nk if nk else shared.shared_fp32_kn
    fn=ts.jit(target="rocm",autodiff="forward",wrt=wrt)(source)
    seeds=tuple({"sa":da,"sb":db}[name] for name in wrt)
    actual=fn.native_jvp(*values,tangents=seeds)
    for got,want in zip(actual,(primal,tangent),strict=True):
        assert got.shape==(3,7,19)
        np.testing.assert_allclose(got,want,rtol=3e-5,atol=3e-5)
    assert fn.last_jvp_execution["execution_kind"]=="native_gpu"
    assert fn.last_jvp_execution["family"]=="scaled_product_program"
    def forbidden(*args,**kwargs):raise AssertionError("warm JVP attempted compiler")
    import subprocess
    monkeypatch.setattr(subprocess,"run",forbidden)
    repeated=fn.native_jvp(*values,tangents=tuple(seed*-.5 for seed in seeds))
    np.testing.assert_allclose(repeated[0],primal,rtol=3e-5,atol=3e-5)
    np.testing.assert_allclose(repeated[1],tangent*-.5,rtol=3e-5,atol=3e-5)
    eps=1e-3
    def independent(sa_value,sb_value):
        output=np.zeros_like(primal)
        for g in range(2):
            product=a[...,g*128:(g+1)*128].astype(np.float64)@logical_b[g*128:(g+1)*128].astype(np.float64)
            output+=product*sa_value[...,g,None]*sb_value[g,0]
        return output
    plus=independent(sa.astype(np.float64)+eps*da,sb.astype(np.float64)+eps*db)
    minus=independent(sa.astype(np.float64)-eps*da,sb.astype(np.float64)-eps*db)
    np.testing.assert_allclose(actual[1],(plus-minus)/(2*eps),rtol=3e-5,atol=3e-5)

@pytest.mark.parametrize("policy",["independent_rhs","shared_lhs"])
@pytest.mark.parametrize("nk",[False,True])
def test_public_independent_batch_scale_jvp(policy,nk,monkeypatch):
    from tests.unit import test_rocm_independent_scaled_batch as batch
    from tessera import runtime
    import subprocess
    assert runtime._rocm_live_arch()=="gfx1201"
    values,primal=batch.batch_inputs(policy=policy,nk=nk)
    a,b,sa,sb=values;logical_b=b.swapaxes(-1,-2) if nk else b
    prefix="independent" if policy=="independent_rhs" else "lhs_shared"
    source=getattr(batch,f"{prefix}_fp32_{'nk' if nk else 'kn'}")
    da=np.full(sa.shape,.05,np.float32);db=np.full(sb.shape,-.03,np.float32)
    def oracle(scale_a,scale_b):
        out=np.zeros_like(primal)
        for g in range(a.shape[-1]//128):
            product=a[...,g*128:(g+1)*128].astype(np.float64)@logical_b[...,g*128:(g+1)*128,:].astype(np.float64)
            out+=product*scale_a[...,g,None]*scale_b[...,g,np.arange(b.shape[-2] if nk else b.shape[-1])//128][...,None,:]
        return out
    sa64,sb64=sa.astype(np.float64),sb.astype(np.float64)
    da64,db64=da.astype(np.float64),db.astype(np.float64)
    tangent=oracle(da64,sb64)+oracle(sa64,db64)
    eps=1e-4
    finite=(oracle(sa64+eps*da64,sb64+eps*db64)-oracle(sa64-eps*da64,sb64-eps*db64))/(2*eps)
    np.testing.assert_allclose(tangent,finite,rtol=3e-5,atol=3e-5)
    fn=ts.jit(target="rocm",autodiff="forward",wrt=("sa","sb"))(source)
    got=fn.native_jvp(*values,tangents=(da,db))
    for actual,expected in zip(got,(primal,tangent),strict=True):
        np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=3e-5)
    def forbidden(*args,**kwargs):raise AssertionError("warm batch JVP compiler subprocess")
    monkeypatch.setattr(subprocess,"run",forbidden)
    got=fn.native_jvp(*values,tangents=(da*-.5,db*-.5))
    for actual,expected in zip(got,(primal,tangent*-.5),strict=True):
        np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=3e-5)
