"""Exact gfx1201 partial scale-group primal/JVP and interior-tile proof."""
import itertools,os,subprocess
import numpy as np,pytest
from tessera import runtime as rt
from tests.unit.test_public_partial_scaled_groups import partial_case,oracle
pytestmark=pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="owning gfx1201 required")

def prove(mask,ta,tb,encoded,k,jvp,monkeypatch,m=3,n=5):
    assert rt._rocm_live_arch()=="gfx1201"
    _,owner,values,row=partial_case(mask,ta,tb,encoded,k,jvp,m=m,n=n)
    def invoke(v):
        return list(owner.native_jvp(*v[:4],tangents=tuple(v[4:]))) if jvp else [owner(*v)]
    for got,wanted in zip(invoke(values),oracle(values,row),strict=True):
        np.testing.assert_allclose(got,wanted,rtol=4e-5,atol=2e-5)
    def forbidden(*args,**kwargs):raise AssertionError("warm partial package invoked compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    _,_,changed,_=partial_case(mask,ta,tb,encoded,k,jvp,seed=5007,m=m,n=n)
    for got,wanted in zip(invoke(changed),oracle(changed,row),strict=True):
        np.testing.assert_allclose(got,wanted,rtol=4e-5,atol=2e-5)

@pytest.mark.parametrize("mask",range(1,16))
@pytest.mark.parametrize("ta,tb,encoded",tuple(itertools.product((False,True),repeat=3)))
def test_partial_primal_every_independent_map(mask,ta,tb,encoded,monkeypatch):
    prove(mask,ta,tb,encoded,37,False,monkeypatch)

@pytest.mark.parametrize("mask",range(1,16))
@pytest.mark.parametrize("ta,tb",tuple(itertools.product((False,True),repeat=2)))
def test_partial_scale_jvp_every_independent_map(mask,ta,tb,monkeypatch):
    prove(mask,ta,tb,False,37,True,monkeypatch)

@pytest.mark.parametrize("k",[1,15,17,31,33,63,65])
@pytest.mark.parametrize("ta,tb,encoded",tuple(itertools.product((False,True),repeat=3)))
def test_partial_primal_boundary_groups(k,ta,tb,encoded,monkeypatch):
    prove(4,ta,tb,encoded,k,False,monkeypatch)

@pytest.mark.parametrize("ta,tb,encoded",tuple(itertools.product((False,True),repeat=3)))
@pytest.mark.parametrize("m,n",[(16,16),(17,19),(33,35)])
def test_partial_interior_and_multiple_tiles(ta,tb,encoded,m,n,monkeypatch):
    prove(4,ta,tb,encoded,37,False,monkeypatch,m=m,n=n)
