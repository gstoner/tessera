"""Owning gfx1201 direct broadcast tracing through native primal/scale AD."""
import itertools
import os
import subprocess
import numpy as np
import pytest
from tests.unit.test_direct_broadcast_scaled_frontend import case,oracle
from tessera import runtime as rt

pytestmark=pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",
    reason="explicit owning gfx1201 proof required")

@pytest.mark.parametrize("encoded,ta,tb",tuple(itertools.product((False,True),repeat=3)))
def test_direct_broadcast_primal_and_compiler_free_warm(encoded,ta,tb,monkeypatch):
    assert rt._rocm_live_arch()=="gfx1201"
    owner,values=case(encoded,ta,tb)
    got=owner(*values)
    np.testing.assert_allclose(got,oracle(values,ta,tb,encoded),rtol=3e-5,atol=2e-5)
    assert owner.compile_result.executable
    assert owner._native_descriptor_last_receipt["execution_kind"]=="native_gpu"
    assert 'batching = "broadcast"' in owner.compile_result.graph_ir
    _,changed=case(encoded,ta,tb,seed=9108)
    wanted=oracle(changed,ta,tb,encoded)
    def forbidden(*args,**kwargs):raise AssertionError("warm direct broadcast invoked compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    np.testing.assert_allclose(owner(*changed),wanted,rtol=3e-5,atol=2e-5)

@pytest.mark.parametrize("ta,tb",tuple(itertools.product((False,True),repeat=2)))
def test_direct_broadcast_scale_jvp(ta,tb,monkeypatch):
    assert rt._rocm_live_arch()=="gfx1201"
    owner,values=case(False,ta,tb,mode="forward")
    rng=np.random.default_rng(8918)
    tangents=tuple(rng.uniform(-.2,.2,size=v.shape).astype(np.float32) for v in values[2:])
    wanted=(oracle(values,ta,tb),oracle((*values[:2],tangents[0],values[3]),ta,tb)+
            oracle((*values[:3],tangents[1]),ta,tb))
    actual=owner.native_jvp(*values,tangents=tangents)
    for got,expected in zip(actual,wanted,strict=True):
        np.testing.assert_allclose(got,expected,rtol=3e-5,atol=2e-5)
    assert owner.last_jvp_execution["execution_kind"]=="native_gpu"
    def forbidden(*args,**kwargs):raise AssertionError("warm direct JVP invoked compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    repeated=owner.native_jvp(*values,tangents=tangents)
    for got,expected in zip(repeated,wanted,strict=True):
        np.testing.assert_allclose(got,expected,rtol=3e-5,atol=2e-5)

@pytest.mark.parametrize("ta,tb",tuple(itertools.product((False,True),repeat=2)))
def test_direct_broadcast_scale_vjp_reduces_own_prefixes(ta,tb,monkeypatch):
    assert rt._rocm_live_arch()=="gfx1201"
    owner,values=case(False,ta,tb,mode="reverse")
    cot=np.random.default_rng(8818).uniform(-1,1,size=(2,3,3,5)).astype(np.float32)
    wanted=[]
    for operand in (2,3):
        gradient=np.empty_like(values[operand],dtype=np.float64)
        for index in np.ndindex(gradient.shape):
            plus=list(values);minus=list(values)
            plus[operand]=values[operand].astype(np.float64)
            minus[operand]=values[operand].astype(np.float64)
            plus[operand][index]+=.001;minus[operand][index]-=.001
            gradient[index]=np.sum((oracle(plus,ta,tb)-oracle(minus,ta,tb))*cot)/.002
        wanted.append(gradient)
    actual=owner.native_backward(*values,out_cotangents=cot)
    for got,expected in zip(actual,wanted,strict=True):
        assert got.shape==expected.shape
        np.testing.assert_allclose(got,expected,rtol=3e-5,atol=2e-5)
    assert owner.last_backward_execution["execution_kind"]=="native_gpu"
    def forbidden(*args,**kwargs):raise AssertionError("warm direct VJP invoked compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    repeated=owner.native_backward(*values,out_cotangents=cot)
    for got,expected in zip(repeated,wanted,strict=True):
        np.testing.assert_allclose(got,expected,rtol=3e-5,atol=2e-5)
