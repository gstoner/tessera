"""Owning gfx1201 public independent-prefix primal and scale JVP proof."""
import itertools,os,subprocess
import numpy as np
import pytest
from tessera import runtime as rt
from tests.unit.test_public_independent_scaled_primal import case,expected
pytestmark=pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="owning gfx1201 required")

@pytest.mark.parametrize("ta",[False,True])
@pytest.mark.parametrize("mask",range(1,16))
@pytest.mark.parametrize("tb,encoded",tuple(itertools.product((False,True),repeat=2)))
def test_public_independent_primal(mask,tb,encoded,ta,monkeypatch):
    assert rt._rocm_live_arch()=="gfx1201"
    _,owner,values,row=case(mask,tb,encoded,ta=ta)
    got=owner(*values)
    np.testing.assert_allclose(got,expected(values,row)[0],rtol=3e-5,atol=2e-5)
    assert owner.compile_result.executable
    assert owner._native_descriptor_last_receipt["execution_kind"]=="native_gpu"
    manifest=owner.compile_result.launch_descriptor.provenance["native_scaled_primal_program"]
    def forbidden(*args,**kwargs):raise AssertionError("warm call attempted compiler subprocess")
    monkeypatch.setattr(subprocess,"run",forbidden)
    _,_,changed,_=case(mask,tb,encoded,seed=1013,ta=ta)
    repeated=owner(*changed)
    np.testing.assert_allclose(repeated,expected(changed,row)[0],rtol=3e-5,atol=2e-5)
    assert owner.compile_result.launch_descriptor.provenance["native_scaled_primal_program"]==manifest

@pytest.mark.parametrize("ta",[False,True])
@pytest.mark.parametrize("mask",range(1,16))
@pytest.mark.parametrize("tb",[False,True])
def test_public_independent_scale_jvp(mask,tb,ta,monkeypatch):
    assert rt._rocm_live_arch()=="gfx1201"
    _,owner,values,row=case(mask,tb,jvp=True,ta=ta)
    got=owner.native_jvp(*values[:4],tangents=tuple(values[4:]))
    for actual,wanted in zip(got,expected(values,row),strict=True):
        np.testing.assert_allclose(actual,wanted,rtol=3e-5,atol=2e-5)
    assert owner.last_jvp_execution["execution_kind"]=="native_gpu"
    def forbidden(*args,**kwargs):raise AssertionError("warm JVP attempted compiler subprocess")
    monkeypatch.setattr(subprocess,"run",forbidden)
    _,_,changed,_=case(mask,tb,jvp=True,seed=1013,ta=ta)
    repeated=owner.native_jvp(*changed[:4],tangents=tuple(changed[4:]))
    for actual,wanted in zip(repeated,expected(changed,row),strict=True):
        np.testing.assert_allclose(actual,wanted,rtol=3e-5,atol=2e-5)


@pytest.mark.parametrize("shape",[(16,16,64),(17,19,96),(33,35,128)])
@pytest.mark.parametrize("tb,encoded",tuple(itertools.product((False,True),repeat=2)))
def test_transposed_a_interior_and_multiple_tiles(shape,tb,encoded):
    import ml_dtypes
    m,n,k=shape
    _,owner,_,row=case(4,tb,encoded,ta=True)
    rng=np.random.default_rng(1019)
    a=rng.uniform(-.5,.5,(k,m)).astype(ml_dtypes.float8_e4m3fn)
    b=rng.uniform(-.5,.5,(n,k) if tb else (k,n)).astype(ml_dtypes.float8_e4m3fn)
    scale_n=1 if encoded else 3
    if encoded:
        sa=rng.integers(126,129,(2,3,m,k//32),dtype=np.uint8)
        sb=rng.integers(126,129,(k//32,(n+scale_n-1)//scale_n),dtype=np.uint8)
        da=np.exp2(sa.astype(np.float64)-127)
        db=np.exp2(sb.astype(np.float64)-127)
    else:
        sa=rng.uniform(.3,1.3,(2,3,m,k//32)).astype(np.float32)
        sb=rng.uniform(.3,1.3,(k//32,(n+scale_n-1)//scale_n)).astype(np.float32)
        da,db=sa.astype(np.float64),sb.astype(np.float64)
    logical_a=a.astype(np.float64).T
    logical_b=b.astype(np.float64).T if tb else b.astype(np.float64)
    wanted=np.zeros((2,3,m,n),np.float64)
    for g in range(k//32):
        dot=logical_a[:,g*32:(g+1)*32]@logical_b[g*32:(g+1)*32,:]
        wanted+=dot*da[...,g,None]*db[g,np.arange(n)//scale_n]
    np.testing.assert_allclose(owner(a,b,sa,sb),wanted,rtol=4e-5,atol=2e-5)
