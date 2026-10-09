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


def product_reverse(a,b,sa,sb,seed):
    m,k=a.shape
    n=b.shape[1]
    grads=[np.zeros_like(value,dtype=np.float64) for value in (a,b,sa,sb)]
    da,db,dsa,dsb=grads
    for row in range(m):
        for col in range(n):
            for group in range((k+3)//4):
                dot=sum(float(a[row,t])*float(b[t,col])
                        for t in range(group*4,min(k,(group+1)*4)))
                dy=float(seed[row,col])
                left,right=float(sa[row,group]),float(sb[group,col//4])
                dsa[row,group]+=dy*dot*right
                dsb[group,col//4]+=dy*dot*left
                for t in range(group*4,min(k,(group+1)*4)):
                    da[row,t]+=dy*float(b[t,col])*left*right
                    db[t,col]+=dy*float(a[row,t])*left*right
    return tuple(value.astype(np.float32) for value in grads)


def reverse_oracle(values,seed):
    a,b,sa,sb,c,sc,sd=values
    first=product(a,b,sa,sb)
    dfirst,dc,dsc,dsd=product_reverse(first,c,sc,sd,seed)
    da,db,dsa,dsb=product_reverse(a,b,sa,sb,dfirst)
    return da,db,dsa,dsb,dc,dsc,dsd


@pytest.mark.parametrize("roles",[("a",),("c",),("sb","a","sd"),("a","b","sa","sb","c","sc","sd")])
@pytest.mark.parametrize("shape",[(2,9,5,3),(3,17,7,6),(1,4,4,4)])
def test_chained_public_reverse_executes_residual_and_cotangent_dependencies(roles,shape,monkeypatch):
    assert runtime._rocm_live_arch()=="gfx1201"
    _,_,values=case(shape)
    owner=ts.jit(target="rocm_gfx1201",autodiff="reverse",wrt=roles)(chained)
    seed=np.random.default_rng(19044).uniform(-.5,.5,(shape[0],shape[3])).astype(np.float32)
    wanted=reverse_oracle(values,seed)
    positions={name:index for index,name in enumerate(("a","b","sa","sb","c","sc","sd"))}
    actual=owner.native_backward(*values,out_cotangents=seed)
    assert owner.last_backward_execution["execution_kind"]=="native_gpu"
    for result,name in zip(actual,roles,strict=True):
        np.testing.assert_allclose(result,wanted[positions[name]],rtol=4e-5,atol=3e-6)
    retained=tuple(value.copy() for value in actual)
    changed=tuple(np.ascontiguousarray(value*np.float32(-.875)) for value in values)
    wanted=reverse_oracle(changed,seed*np.float32(-.5))
    def forbidden(*args,**kwargs):
        raise AssertionError("warm reverse chain invoked compiler or eager")
    monkeypatch.setattr(subprocess,"run",forbidden)
    monkeypatch.setattr(owner,"_fn",forbidden)
    repeated=owner.native_backward(*changed,out_cotangents=seed*np.float32(-.5))
    for result,name in zip(repeated,roles,strict=True):
        np.testing.assert_allclose(result,wanted[positions[name]],rtol=4e-5,atol=3e-6)
    for result,old in zip(actual,retained,strict=True):
        np.testing.assert_array_equal(result,old)
