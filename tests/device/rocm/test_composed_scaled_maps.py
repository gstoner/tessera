"""Owning gfx1201 mapped product/sum primal and scale AD execution."""
import os
import subprocess
import numpy as np
import pytest
from tessera import runtime as rt
from tests.unit.test_composed_scaled_maps import case
from tests.device.rocm.test_public_scaled_jvp import oracle as product
from tests.device.rocm.test_composed_scaled_vjp import oracle as reverse

pytestmark=pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="owning gfx1201 required")

def expected(owner,values,seeds,axes,prefix,shared_scale,dy=None):
    primal=np.empty((*prefix,*dy.shape[-2:]),np.float64) if dy is not None else None
    gradients=[np.zeros(values[i].shape,np.float64) for i in (owner.differentiation_request.wrt_indices if owner.differentiation_request else ())]
    outputs=[]
    for index in np.ndindex(prefix):
        plane=tuple(v[index] if axis==0 else v for v,axis in zip(values,axes,strict=True))
        if dy is not None:
            result=reverse(plane,dy[index],shared_scale,owner.differentiation_request.wrt)
            for pos,(got,role) in enumerate(zip(result,owner.differentiation_request.wrt_indices,strict=True)):
                if axes[role]==0:gradients[pos][index]=got
                else:gradients[pos]+=got
            continue
        seed_map={}
        if seeds:
            for role,seed in zip(owner.differentiation_request.wrt_indices,seeds,strict=True):
                seed_map[role]=seed[index] if axes[role]==0 else seed
        a,b=plane[:2]
        roles=((2,3),(2,4)) if shared_scale else ((2,3),(4,5))
        pairs=[product(a,b,plane[l],plane[r],seed_map.get(l,np.zeros_like(plane[l])),seed_map.get(r,np.zeros_like(plane[r]))) for l,r in roles]
        outputs.append(tuple(pairs[0][i]+pairs[1][i] for i in range(2)))
    if dy is not None:return tuple(gradients)
    return tuple(np.stack([v[i] for v in outputs]).reshape((*prefix,*outputs[0][i].shape)) for i in range(2))

@pytest.mark.parametrize("policy",["all","scales","lhs"])
@pytest.mark.parametrize("depth",[1,2])
@pytest.mark.parametrize("shared_scale",[False,True])
@pytest.mark.parametrize("shape",[(3,5,256),(3,5,37)])
@pytest.mark.parametrize("mode",[None,"forward","reverse"])
def test_public_mapped_graph_native_execution(policy,depth,shared_scale,shape,mode,monkeypatch):
    assert rt._rocm_live_arch()=="gfx1201"
    _,owner,values,seeds,axes,prefix=case(policy,depth,mode,shared_scale,shape)
    dy=np.random.default_rng(10897).uniform(-.2,.2,(*prefix,shape[0],shape[1])).astype(np.float32)
    wanted=expected(owner,values,seeds,axes,prefix,shared_scale,dy if mode=="reverse" else None)
    def invoke(frame,tangent,cot):
        if mode=="forward":return owner.native_jvp(*frame,tangents=tangent)
        if mode=="reverse":return owner.native_backward(*frame,out_cotangents=cot)
        return (owner(*frame),)
    actual=invoke(values,seeds,dy)
    if mode is None:
        wanted=wanted[:1]
        receipt=owner._native_descriptor_last_receipt
    else:receipt=owner.last_jvp_execution if mode=="forward" else owner.last_backward_execution
    assert receipt["execution_kind"]=="native_gpu"
    for got,want in zip(actual,wanted,strict=True):np.testing.assert_allclose(got,want,rtol=3e-4,atol=3e-5)
    retained=tuple(x.copy() for x in actual)
    changed=tuple(v if i<2 else v*.9 for i,v in enumerate(values))
    changed_seeds=tuple(-x for x in seeds)
    wanted2=expected(owner,changed,changed_seeds,axes,prefix,shared_scale,-dy if mode=="reverse" else None)
    if mode is None:wanted2=wanted2[:1]
    def forbidden(*args,**kwargs):raise AssertionError("warm mapped execution escaped native package")
    monkeypatch.setattr(subprocess,"run",forbidden)
    from tessera.compiler import reference_typed_scaled_matmul as reference
    monkeypatch.setattr(reference,"reference_typed_scaled_matmul",forbidden)
    repeated=invoke(changed,changed_seeds,-dy)
    for got,want in zip(repeated,wanted2,strict=True):np.testing.assert_allclose(got,want,rtol=3e-4,atol=3e-5)
    for got,saved in zip(actual,retained,strict=True):np.testing.assert_array_equal(got,saved)
