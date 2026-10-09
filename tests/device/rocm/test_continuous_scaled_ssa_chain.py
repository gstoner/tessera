"""Exact gfx1201 chained Graph products and forward tangents."""
import os
import subprocess

import numpy as np
import pytest
import tessera as ts
from tessera import runtime
from tests.unit.test_continuous_scaled_ssa_chain import case, chained

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="owning gfx1201 required")


def product(a,b,sa,sb):
    m,k=a.shape
    n=b.shape[1]
    result=np.zeros((m,n),dtype=np.float64)
    for row in range(m):
        for col in range(n):
            for group in range((k+3)//4):
                dot=sum(float(a[row,t])*float(b[t,col])
                        for t in range(group*4,min(k,(group+1)*4)))
                result[row,col]+=dot*float(sa[row,group])*float(sb[group,col//4])
    return result.astype(np.float32)


def oracle(values, directions):
    a,b,sa,sb,c,sc,sd=values
    da,db,dsa,dsb,dc,dsc,dsd=directions
    first=product(a,b,sa,sb)
    dfirst=(product(da,b,sa,sb)+product(a,db,sa,sb)+
            product(a,b,dsa,sb)+product(a,b,sa,dsb))
    output=product(first,c,sc,sd)
    tangent=(product(dfirst,c,sc,sd)+product(first,dc,sc,sd)+
             product(first,c,dsc,sd)+product(first,c,sc,dsd))
    return output,tangent


@pytest.mark.parametrize("mode",[None,"forward"])
def test_chained_public_native_matches_oracle_and_reuses_images(mode,monkeypatch):
    assert runtime._rocm_live_arch()=="gfx1201"
    _,_,values=case()
    options={} if mode is None else {"autodiff":"forward","wrt":("a","b","sa","sb","c","sc","sd")}
    owner=ts.jit(target="rocm_gfx1201",**options)(chained)
    directions=tuple(np.full_like(value,.03125) for value in values)
    expected=oracle(values,directions)
    ordinary=owner(*values)
    np.testing.assert_allclose(ordinary,expected[0],rtol=4e-5,atol=3e-6)
    assert owner._native_descriptor_last_receipt["execution_kind"]=="native_gpu"
    actual=owner.native_jvp(*values,tangents=directions) if mode else (ordinary,)
    for got,want in zip(actual,expected[:len(actual)],strict=True):
        np.testing.assert_allclose(got,want,rtol=4e-5,atol=3e-6)
    retained=tuple(value.copy() for value in actual)
    changed=tuple(value*np.float32(1.125) for value in values)
    wanted=oracle(changed,directions)
    def forbidden(*args,**kwargs):
        raise AssertionError("warm chain invoked compiler or eager execution")
    monkeypatch.setattr(subprocess,"run",forbidden)
    monkeypatch.setattr(owner,"_fn",forbidden)
    repeated=(owner.native_jvp(*changed,tangents=directions) if mode else (owner(*changed),))
    for got,want in zip(repeated,wanted[:len(repeated)],strict=True):
        np.testing.assert_allclose(got,want,rtol=4e-5,atol=3e-6)
    for got,old in zip(actual,retained,strict=True):
        np.testing.assert_array_equal(got,old)
