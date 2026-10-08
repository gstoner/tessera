"""Owning gfx1201 numerical proof for composed independent/shared scale adjoints."""
import os
import subprocess
import numpy as np
import pytest
from tessera import runtime as rt
from tests.unit.test_composed_scaled_vjp import case
pytestmark=pytest.mark.skipif(not os.environ.get("TESSERA_GFX1201_DEVICE_PROOF"),reason="owning gfx1201 proof required")

def product_grad(a,b,sa,sb,dy):
    a,b,sa,sb,dy=(np.asarray(v,dtype=np.float64) for v in (a,b,sa,sb,dy))
    lhs=np.zeros_like(sa);rhs=np.zeros_like(sb)
    for g in range(sa.shape[1]):
        lo,hi=g*128,min((g+1)*128,a.shape[1])
        raw=a[:,lo:hi]@b[lo:hi,:]
        for c in range(sb.shape[1]):
            n0,n1=c*128,min((c+1)*128,b.shape[1])
            weighted=raw[:,n0:n1]*dy[:,n0:n1]
            lhs[:,g]+=weighted.sum(axis=1)*sb[g,c]
            rhs[g,c]=(weighted*sa[:,g,None]).sum()
    return lhs,rhs

def oracle(values,dy,shared_scale,wrt):
    if shared_scale:
        a,b,sa,sb0,sb1=values
        da0,db0=product_grad(a,b,sa,sb0,dy)
        da1,db1=product_grad(a,b,sa,sb1,dy)
        gradients={"sa":da0+da1,"sb0":db0,"sb1":db1}
    else:
        a,b,sa0,sb0,sa1,sb1=values
        da0,db0=product_grad(a,b,sa0,sb0,dy)
        da1,db1=product_grad(a,b,sa1,sb1,dy)
        gradients={"sa0":da0,"sb0":db0,"sa1":da1,"sb1":db1}
    return tuple(gradients[name] for name in wrt)

@pytest.mark.parametrize("shape",[(17,19,256),(3,5,37)])
@pytest.mark.parametrize("shared_scale",[False,True])
@pytest.mark.parametrize("selection",["all","single"])
def test_composed_public_reverse_numerics_and_warm_lifetime(shape,shared_scale,selection,monkeypatch):
    assert rt._rocm_live_arch()=="gfx1201"
    wrt=(("sa",) if shared_scale else ("sa1",)) if selection=="single" else None
    owner,values,dy=case(shape,shared_scale,wrt)
    expected=oracle(values,dy,shared_scale,owner.differentiation_request.wrt)
    actual=owner.native_backward(*values,out_cotangents=dy)
    for got,want in zip(actual,expected,strict=True):
        np.testing.assert_allclose(got,want,rtol=3e-4,atol=3e-5)
    receipt=owner.last_backward_execution
    assert receipt["execution_kind"]=="native_gpu"
    assert receipt["evidence_target"]=="rocm_gfx1201"
    assert receipt["family"]=="scaled_product_transpose"
    retained=tuple(v.copy() for v in actual)
    def forbidden(*args,**kwargs):raise AssertionError("warm reverse escaped native package")
    monkeypatch.setattr(subprocess,"run",forbidden)
    from tessera.compiler import reference_typed_scaled_matmul as reference
    monkeypatch.setattr(reference,"reference_typed_scaled_matmul",forbidden)
    changed=owner.native_backward(*values,out_cotangents=-dy)
    for got,want,old,snapshot in zip(changed,expected,actual,retained,strict=True):
        np.testing.assert_allclose(got,-want,rtol=3e-4,atol=3e-5)
        np.testing.assert_array_equal(old,snapshot)
