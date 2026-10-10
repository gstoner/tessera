"""Owning gfx1201 composed scale JVP, native lifetimes and warm replay."""
import os
import subprocess
import numpy as np
import pytest
from tests.unit.test_composed_scaled_jvp import case
from tests.device.rocm.test_public_scaled_jvp import oracle

pytestmark=pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="owning gfx1201 required")

@pytest.mark.parametrize("shape",[(17,19,256),(3,5,37)])
@pytest.mark.parametrize("wrt",[("sa0",),("sa1","sb1"),("sb1","sa0","sb0","sa1")])
def test_composed_native_scale_jvp(shape,wrt,monkeypatch):
    from tessera import runtime as rt
    assert rt._rocm_live_arch()=="gfx1201"
    owner,values,seeds=case(shape,wrt)
    mapping=dict(zip(wrt,seeds,strict=True))
    a,b,sa0,sb0,sa1,sb1=values
    def expected(seed_map):
        first=oracle(a,b,sa0,sb0,seed_map.get("sa0",np.zeros_like(sa0)),seed_map.get("sb0",np.zeros_like(sb0)))
        second=oracle(a,b,sa1,sb1,seed_map.get("sa1",np.zeros_like(sa1)),seed_map.get("sb1",np.zeros_like(sb1)))
        return tuple(left+right for left,right in zip(first,second,strict=True))
    wanted=expected(mapping)
    actual=owner.native_jvp(*values,tangents=seeds)
    for got,want in zip(actual,wanted,strict=True):
        np.testing.assert_allclose(got,want,rtol=4e-5,atol=3e-5)
    saved=tuple(x.copy() for x in actual)
    receipt=owner.last_jvp_execution
    assert receipt["execution_kind"]=="native_gpu"
    assert receipt["family"]=="scaled_product_program"
    assert receipt["evidence_target"]=="rocm_gfx1201"
    def forbidden(*args,**kwargs):
        raise AssertionError("warm composed JVP invoked compiler/reference")
    monkeypatch.setattr(subprocess,"run",forbidden)
    from tessera.compiler import reference_typed_scaled_matmul as reference
    monkeypatch.setattr(reference,"reference_typed_scaled_matmul",forbidden)
    repeated=owner.native_jvp(*values,tangents=tuple(-x for x in seeds))
    np.testing.assert_allclose(repeated[0],wanted[0],rtol=4e-5,atol=3e-5)
    np.testing.assert_allclose(repeated[1],-wanted[1],rtol=4e-5,atol=3e-5)
    for got,old in zip(actual,saved,strict=True):np.testing.assert_array_equal(got,old)
