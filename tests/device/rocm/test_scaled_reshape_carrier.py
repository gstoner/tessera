"""Exact gfx1201 public execution of native reshape/product differentiation."""
import os
import subprocess

import numpy as np
import pytest
from tessera import runtime
from tests.unit.test_scaled_reshape_carrier import case

pytestmark=pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",
                             reason="owning gfx1201 required")


def expected(values,directions,output,seed=None):
    a,b,sa,sb=(value.astype(np.float64).reshape(shape) for value,shape in
               zip(values,((3,7),(7,5),(3,2),(2,3)),strict=True))
    da,db,dsa,dsb=(value.astype(np.float64).reshape(shape) for value,shape in
                   zip(directions,((3,7),(7,5),(3,2),(2,3)),strict=True))
    y=np.zeros((3,5));dy=np.zeros((3,5))
    grads=[np.zeros_like(value) for value in (a,b,sa,sb)]
    cotangent=np.asarray(seed,dtype=np.float64).reshape(3,5) if seed is not None else None
    for row in range(3):
        for col in range(5):
            for index in range(7):
                group,column=index//4,col//2
                av,bv,sv,tv=a[row,index],b[index,col],sa[row,group],sb[group,column]
                y[row,col]+=av*bv*sv*tv
                dy[row,col]+=(da[row,index]*bv*sv*tv+av*db[index,col]*sv*tv
                              +av*bv*dsa[row,group]*tv+av*bv*sv*dsb[group,column])
                if cotangent is not None:
                    v=cotangent[row,col]
                    grads[0][row,index]+=v*bv*sv*tv
                    grads[1][index,col]+=v*av*sv*tv
                    grads[2][row,group]+=v*av*bv*tv
                    grads[3][group,column]+=v*av*bv*sv
    return (y.astype(np.float32).reshape(output),dy.astype(np.float32).reshape(output),
            tuple(value.astype(np.float32).reshape(source.shape) for value,source in zip(grads,values,strict=True)))


@pytest.mark.parametrize("output",[(15,),(5,3),(1,3,5)])
@pytest.mark.parametrize("mode",[None,"forward","reverse"])
@pytest.mark.parametrize("layout",["compact","pitched"])
def test_public_reshape_product_roles_and_warm_replay(mode,output,layout,monkeypatch):
    assert runtime._rocm_live_arch()=="gfx1201"
    owner,_,values=case(mode,output)
    if layout=="pitched":
        frames=[]
        for value in values:
            backing=np.zeros(value.size*3+1,dtype=value.dtype)
            view=backing[1:1+value.size*3:3];view[:]=value;frames.append(view)
        values=tuple(frames)
    directions=tuple(np.full_like(value,.03125) for value in values)
    seed=np.random.default_rng(9331).uniform(-.5,.5,output).astype(np.float32)
    def run(frame,cotangent):
        if mode=="reverse":return owner.native_backward(*frame,out_cotangents=cotangent)
        if mode=="forward":return owner.native_jvp(*frame,tangents=directions)
        return (owner(*frame),)
    def check(frame,cotangent,actual):
        y,dy,grads=expected(frame,directions,output,cotangent)
        wanted=grads if mode=="reverse" else (y,dy) if mode=="forward" else (y,)
        for got,want in zip(actual,wanted,strict=True):
            np.testing.assert_allclose(got,want,rtol=4e-5,atol=3e-6)
    actual=run(values,seed);check(values,seed,actual)
    receipt=owner.last_backward_execution if mode=="reverse" else owner.last_jvp_execution if mode=="forward" else owner._native_descriptor_last_receipt
    assert receipt["execution_kind"]=="native_gpu"
    retained=tuple(value.copy() for value in actual)
    def forbidden(*args,**kwargs):raise AssertionError("warm reshape program invoked compiler or eager arithmetic")
    monkeypatch.setattr(subprocess,"run",forbidden);monkeypatch.setattr(owner,"_fn",forbidden)
    changed=tuple(np.ascontiguousarray(value*np.float32(-.875)) for value in values)
    replay=run(changed,seed*np.float32(-.5));check(changed,seed*np.float32(-.5),replay)
    for value,snapshot in zip(actual,retained,strict=True):np.testing.assert_array_equal(value,snapshot)
