"""Owning gfx1201 public independent matrix/scale reverse integration."""
import itertools,os,subprocess
import numpy as np
import pytest
from tests.unit.test_public_independent_scale_vmap import case
from benchmarks.rocm.record_independent_batch_reverse_device import oracle
pytestmark=pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",
    reason="explicit owning gfx1201 proof required")

def expected(values,cot,mask,ta,tb,prefix):
    row={"output_prefix":prefix,"transposeA":ta,"transposeB":tb}
    return oracle((*values,cot),row)

@pytest.mark.parametrize("mask",range(1,16))
@pytest.mark.parametrize("ta,tb",tuple(itertools.product((False,True),repeat=2)))
@pytest.mark.parametrize("prefix",[(2,),(2,3),(2,1,3)])
def test_public_independent_scale_reverse_and_warm_replay(mask,ta,tb,prefix,monkeypatch):
    from tessera import runtime
    assert runtime._rocm_live_arch()=="gfx1201"
    scalar,owner,values,axes=case(mask,ta,tb,prefix)
    cot=np.random.default_rng(1007).uniform(-1,1,size=(*prefix,3,5)).astype(np.float32)
    wanted=expected(values,cot,mask,ta,tb,prefix)
    actual=owner.native_backward(*values,out_cotangents=cot)
    for got,want in zip(actual,wanted,strict=True):
        np.testing.assert_allclose(got,want,rtol=2e-5,atol=1e-5)
    receipt=owner.last_backward_execution
    assert receipt["compiler_path"]=="rocm_scaled_vjp_program_compiled"
    assert receipt["frontend_authority"]=="tracer"
    assert receipt["execution_certificate"]["evidence_scope"]=="exact_device"
    assert receipt["physical_attestation"]["device_arch"]=="gfx1201"
    artifact=owner.native_backward_runtime_artifact()
    assert artifact.target=="rocm_gfx1201"
    _,_,changed,_=case(mask,ta,tb,prefix,seed=5007)
    def forbidden(*args,**kwargs):raise AssertionError("warm public reverse invoked compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    cot=-.75*cot
    repeated=owner.native_backward(*changed,out_cotangents=cot)
    wanted=expected(changed,cot,mask,ta,tb,prefix)
    for got,want in zip(repeated,wanted,strict=True):
        np.testing.assert_allclose(got,want,rtol=2e-5,atol=1e-5)
    assert scalar._frontend_batch_axes is None

@pytest.mark.parametrize("mask",[4,8,12])
@pytest.mark.parametrize("wrt",[("sa",),("sb",),("sb","sa")])
def test_public_independent_gradient_request_order(mask,wrt):
    prefix=(2,3)
    _,owner,values,_=case(mask,False,True,prefix,wrt=wrt)
    cot=np.random.default_rng(3107).uniform(-1,1,size=(*prefix,3,5)).astype(np.float32)
    wanted=expected(values,cot,mask,False,True,prefix)
    actual=owner.native_backward(*values,out_cotangents=cot)
    for got,name in zip(actual,wrt,strict=True):
        np.testing.assert_allclose(got,wanted[0 if name=="sa" else 1],rtol=2e-5,atol=1e-5)
