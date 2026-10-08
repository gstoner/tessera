"""Scalar public scale reverse keeps orientation, roles and compiler-free reuse."""
import itertools,os,subprocess
import numpy as np
import pytest
from tests.unit.test_public_independent_scale_vmap import case
from benchmarks.rocm.record_independent_batch_reverse_device import oracle
pytestmark=pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="owning gfx1201 required")

@pytest.mark.parametrize("ta,tb",tuple(itertools.product((False,True),repeat=2)))
@pytest.mark.parametrize("wrt",[("sa",),("sb",),("sb","sa")])
def test_scalar_reverse(ta,tb,wrt,monkeypatch):
    from tessera import runtime
    assert runtime._rocm_live_arch()=="gfx1201"
    scalar,owner,values,_=case(1,ta,tb,(),wrt=wrt)
    cot=np.random.default_rng(9001).uniform(-1,1,(3,5)).astype(np.float32)
    row={"output_prefix":(),"transposeA":ta,"transposeB":tb}
    wanted=oracle((*values,cot),row)
    actual=owner.native_backward(*values,out_cotangents=cot)
    for got,name in zip(actual,wrt,strict=True):
        np.testing.assert_allclose(got,wanted[0 if name=="sa" else 1],rtol=2e-5,atol=1e-5)
    receipt=owner.last_backward_execution
    assert receipt["compiler_path"]=="rocm_scaled_vjp_program_compiled"
    assert receipt["physical_attestation"]["device_arch"]=="gfx1201"
    assert receipt["execution_certificate"]["evidence_scope"]=="exact_device"
    _,_,changed,_=case(1,ta,tb,(),wrt=wrt,seed=5007)
    def forbidden(*args,**kwargs):raise AssertionError("warm scalar reverse invoked compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    cot=-.75*cot; wanted=oracle((*changed,cot),row)
    actual=owner.native_backward(*changed,out_cotangents=cot)
    for got,name in zip(actual,wrt,strict=True):
        np.testing.assert_allclose(got,wanted[0 if name=="sa" else 1],rtol=2e-5,atol=1e-5)
    assert scalar._frontend_batch_axes is None
