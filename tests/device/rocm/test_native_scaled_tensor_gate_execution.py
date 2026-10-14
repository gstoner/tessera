"""Owning GFX1201 tensor gate integration through native primal/JVP/VJP."""
import copy
import itertools
import os
import subprocess

import numpy as np
import pytest
from tessera import runtime
from tessera.autodiff import jvp, vmap
from tests.device.rocm.test_floating_scaled_product import primal
from tests.unit.test_native_scaled_multiply_carrier import case

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="owning gfx1201 required")


def mapped_case(ta=False,tb=False,prefix=(),mask=31,roles=("a","b","sa","sb","gate")):
    owner,_,values,directions=case("reverse",roles,ta=ta,tb=tb)
    axes=tuple(0 if mask & (1 << slot) else None for slot in range(5))
    for _ in prefix:
        owner=vmap(owner,in_axes=axes)
    rng=np.random.default_rng(19442)
    frames=[]
    for originals in (values,directions):
        frame=[]
        for slot,value in enumerate(originals):
            shape=(*(prefix if axes[slot] is not None else ()),*value.shape)
            frame.append(rng.uniform(-.5,.5,shape).astype(np.float32))
        frames.append(tuple(frame))
    return owner,*frames


def oracle(values,ta,tb):
    return primal(values[:4],ta,tb)*values[4].astype(np.float64)


def tangent_oracle(values,directions,ta,tb,active=range(5)):
    expected=np.zeros_like(oracle(values,ta,tb))
    for role in active:
        frame=list(values)
        frame[role]=directions[role]
        expected+=oracle(frame,ta,tb)
    return expected


def transpose_oracle(values,cotangent,ta,tb,roles):
    expected=[]
    for role in roles:
        gradient=np.empty_like(values[role],dtype=np.float64)
        for index in np.ndindex(gradient.shape):
            plus,minus=list(values),list(values)
            plus[role]=values[role].astype(np.float64)
            minus[role]=values[role].astype(np.float64)
            plus[role][index]+=.0001
            minus[role][index]-=.0001
            gradient[index]=np.sum(cotangent*(oracle(plus,ta,tb)-oracle(minus,ta,tb)))/.0002
        expected.append(gradient)
    return expected


@pytest.mark.parametrize("ta,tb",tuple(itertools.product((False,True),repeat=2)))
@pytest.mark.parametrize("prefix,mask",[((),31)])
def test_native_tensor_gate_primal_public_jvp_and_reverse(ta,tb,prefix,mask,monkeypatch):
    assert runtime._rocm_live_arch()=="gfx1201"
    owner,values,directions=mapped_case(ta,tb,prefix,mask)
    original_request=owner.differentiation_request
    try:
        wanted=oracle(values,ta,tb)
        actual=owner(*values)
        np.testing.assert_allclose(actual,wanted,rtol=4e-5,atol=3e-6)
        assert owner._native_descriptor_last_receipt["execution_kind"]=="native_gpu"
        # Ordinary cold capture may install the canonical traced Graph.
        # Native JVP must preserve that caller state and the source witness.
        original_graph=copy.deepcopy(owner.graph_ir)
        source_witness=copy.deepcopy(owner._legacy_graph_ir or owner.graph_ir)
        forward=jvp(owner,values,directions)
        assert owner.graph_ir==original_graph
        np.testing.assert_allclose(forward[0],wanted,rtol=4e-5,atol=3e-6)
        np.testing.assert_allclose(forward[1],tangent_oracle(values,directions,ta,tb),rtol=4e-5,atol=3e-6)
        cot=np.random.default_rng(19443).uniform(-.5,.5,wanted.shape).astype(np.float32)
        backward=owner.native_backward(*values,out_cotangents=cot)
        expected=transpose_oracle(values,cot,ta,tb,range(5))
        for got,want,value in zip(backward,expected,values,strict=True):
            assert got.shape==value.shape
            np.testing.assert_allclose(got,want,rtol=6e-5,atol=4e-6)
        assert owner.last_backward_execution["execution_kind"]=="native_gpu"
        assert (owner._legacy_graph_ir or owner.graph_ir)==source_witness
        assert owner.differentiation_request is original_request
        # Independent public outputs survive subsequent native generations.
        retained=tuple(value.copy() for value in forward)
        def forbidden(*args,**kwargs):
            raise AssertionError("warm tensor gate execution invoked compiler")
        monkeypatch.setattr(subprocess,"run",forbidden)
        changed=tuple(value*np.float32(1.125) for value in values)
        seeds=tuple(value*np.float32(-.5) for value in directions)
        repeated=jvp(owner,changed,seeds)
        np.testing.assert_allclose(repeated[0],oracle(changed,ta,tb),rtol=4e-5,atol=3e-6)
        np.testing.assert_allclose(repeated[1],tangent_oracle(changed,seeds,ta,tb),rtol=4e-5,atol=3e-6)
        for got,old in zip(forward,retained,strict=True):
            np.testing.assert_array_equal(got,old)
        for got,want in zip(owner.native_backward(*values,out_cotangents=-cot),expected,strict=True):
            np.testing.assert_allclose(got,-want,rtol=6e-5,atol=4e-6)
    finally:
        owner.close_native_storage()


@pytest.mark.parametrize("roles",[(4,),(0,),(4,0),(2,3)])
def test_gate_activity_and_reordered_transpose_roles(roles):
    names=("a","b","sa","sb","gate")
    owner,values,directions=mapped_case(prefix=(),mask=16,roles=tuple(names[i] for i in roles))
    try:
        active=tuple(direction if slot in roles else None for slot,direction in enumerate(directions))
        primal_output,tangent=jvp(owner,values,active)
        np.testing.assert_allclose(primal_output,oracle(values,False,False),rtol=4e-5,atol=3e-6)
        np.testing.assert_allclose(tangent,tangent_oracle(values,directions,False,False,roles),rtol=4e-5,atol=3e-6)
        cot=np.ones_like(primal_output,dtype=np.float32)
        expected=transpose_oracle(values,cot,False,False,roles)
        for role,got,want in zip(roles,owner.native_backward(*values,out_cotangents=cot),expected,strict=True):
            assert got.shape==values[role].shape
            np.testing.assert_allclose(got,want,rtol=6e-5,atol=4e-6)
    finally:
        owner.close_native_storage()
