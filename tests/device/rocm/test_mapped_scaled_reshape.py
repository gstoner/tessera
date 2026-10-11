"""Exact gfx1201 execution of mapped product/reshape/product SSA."""
import os
import subprocess

import numpy as np
import pytest
from tessera import runtime
from tests.unit.test_mapped_scaled_reshape import case

def product(values, directions, block, seed=None):
    a,b,sa,sb=(np.asarray(value,dtype=np.float64) for value in values)
    da,db,dsa,dsb=(np.asarray(value,dtype=np.float64) for value in directions)
    m,k=a.shape
    n=b.shape[1]
    y=np.zeros((m,n),dtype=np.float64)
    dy=np.zeros_like(y)
    gradients=[np.zeros_like(value) for value in (a,b,sa,sb)]
    for row in range(m):
        for col in range(n):
            for index in range(k):
                group,column=index//block[1],col//block[0]
                av,bv,sv,tv=a[row,index],b[index,col],sa[row,group],sb[group,column]
                y[row,col]+=av*bv*sv*tv
                dy[row,col]+=(da[row,index]*bv*sv*tv+av*db[index,col]*sv*tv
                              +av*bv*dsa[row,group]*tv+av*bv*sv*dsb[group,column])
                if seed is not None:
                    weight=seed[row,col]
                    gradients[0][row,index]+=weight*bv*sv*tv
                    gradients[1][index,col]+=weight*av*sv*tv
                    gradients[2][row,group]+=weight*av*bv*tv
                    gradients[3][group,column]+=weight*av*bv*sv
    return y,dy,tuple(gradients)


def oracle(values,directions):
    first,dfirst,_=product(values[:4],directions[:4],(2,4))
    result,tangent,_=product((first.reshape(6,2),*values[4:]),
                             (dfirst.reshape(6,2),*directions[4:]),(1,2))
    return result,tangent


def reverse_oracle(values,seed):
    zero=tuple(np.zeros_like(value) for value in values)
    first,_,_=product(values[:4],zero[:4],(2,4))
    second_values=(first.reshape(6,2),*values[4:])
    _,_,last_gradients=product(second_values,tuple(np.zeros_like(value) for value in second_values),
                              (1,2),seed)
    _,_,first_gradients=product(values[:4],zero[:4],(2,4),last_gradients[0].reshape(3,4))
    return (*first_gradients,*last_gradients[1:])


pytestmark=pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",
                             reason="owning gfx1201 required")


def expected(values,directions,axes,prefix,out_axes,seed=None):
    outputs=[]
    tangents=[]
    grads=[np.zeros_like(value,dtype=np.float64) for value in values]
    canonical_seed=np.moveaxis(seed,-1,0) if seed is not None and out_axes==-1 else seed
    for plane in np.ndindex(prefix):
        frame=tuple(value[plane] if axis is not None else value
                    for value,axis in zip(values,axes,strict=True))
        seeds=tuple(value[plane] if axis is not None else value
                    for value,axis in zip(directions,axes,strict=True))
        primal,tangent=oracle(frame,seeds)
        outputs.append(primal)
        tangents.append(tangent)
        if seed is not None:
            for destination,gradient,axis in zip(grads,reverse_oracle(frame,canonical_seed[plane]),axes,strict=True):
                if axis is None:destination+=gradient
                else:destination[plane]+=gradient
    output=np.stack(outputs).reshape(*prefix,*outputs[0].shape)
    tangent=np.stack(tangents).reshape(*prefix,*tangents[0].shape)
    if out_axes==-1:
        output=np.moveaxis(output,0,-1)
        tangent=np.moveaxis(tangent,0,-1)
    return output,tangent,tuple(value.astype(np.float32) for value in grads)


@pytest.mark.parametrize("mask",[1,16,64,17,127])
@pytest.mark.parametrize("mode",[None,"forward","reverse"])
@pytest.mark.parametrize("prefix",[(2,),(2,3)])
@pytest.mark.parametrize("out_axes",[0,-1])
def test_mapped_reshape_executes_native_roles_and_reuses_packages(mask,mode,prefix,out_axes,monkeypatch):
    assert runtime._rocm_live_arch()=="gfx1201"
    _,owner,values,axes=case(mask,mode,prefix,out_axes)
    directions=tuple(np.full_like(value,.03125) for value in values)
    primal,tangent,_=expected(values,directions,axes,prefix,out_axes)
    seed=np.random.default_rng(19073).uniform(-.5,.5,primal.shape).astype(np.float32)
    _,_,gradients=expected(values,directions,axes,prefix,out_axes,seed)
    if mode=="reverse":
        actual=owner.native_backward(*values,out_cotangents=seed)
        wanted=gradients
        assert owner.last_backward_execution["execution_kind"]=="native_gpu"
    elif mode=="forward":
        actual=owner.native_jvp(*values,tangents=directions)
        wanted=(primal,tangent)
        assert owner.last_jvp_execution["execution_kind"]=="native_gpu"
    else:
        actual=(owner(*values),)
        wanted=(primal,)
        assert owner._native_descriptor_last_receipt["execution_kind"]=="native_gpu"
    for got,want in zip(actual,wanted,strict=True):
        np.testing.assert_allclose(got,want,rtol=4e-5,atol=3e-6)
    retained=tuple(value.copy() for value in actual)
    changed=tuple(np.ascontiguousarray(value*np.float32(-.875)) for value in values)
    output,doutput,grads=expected(changed,directions,axes,prefix,out_axes,seed*np.float32(-.5))
    def forbidden(*args,**kwargs):
        raise AssertionError("warm mapped chain invoked compiler or eager")
    monkeypatch.setattr(subprocess,"run",forbidden)
    monkeypatch.setattr(owner,"_fn",forbidden)
    if mode=="reverse":
        repeated=owner.native_backward(*changed,out_cotangents=seed*np.float32(-.5))
        wanted=grads
    elif mode=="forward":
        repeated=owner.native_jvp(*changed,tangents=directions)
        wanted=(output,doutput)
    else:
        repeated=(owner(*changed),)
        wanted=(output,)
    for got,want in zip(repeated,wanted,strict=True):
        np.testing.assert_allclose(got,want,rtol=4e-5,atol=3e-6)
    for got,old in zip(actual,retained,strict=True):
        np.testing.assert_array_equal(got,old)
