"""Exact gfx1201 execution of mapped product-to-product SSA."""
import os
import subprocess

import numpy as np
import pytest
from tessera import runtime
from tests.unit.test_continuous_scaled_mapped_ssa import case
from tests.device.rocm.test_continuous_scaled_ssa_chain import oracle,reverse_oracle

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
def test_mapped_chain_executes_native_roles_and_reuses_packages(mask,mode,prefix,out_axes,monkeypatch):
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
