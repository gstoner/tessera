"""Exact gfx1201 scalar bounded-plane execution and compiler-free changed input."""
import itertools,os,subprocess
import numpy as np
import pytest
from tests.unit.test_scalar_scaled_plane import scalar_case,oracle
pytestmark=pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="owning gfx1201 required")

@pytest.mark.parametrize("k",[1,32,37,65])
@pytest.mark.parametrize("ta,tb,encoded",tuple(itertools.product((False,True),repeat=3)))
def test_scalar_primal(ta,tb,encoded,k,monkeypatch):
    from tessera import runtime
    assert runtime._rocm_live_arch()=="gfx1201"
    scalar,values,row=scalar_case(ta,tb,encoded,k)
    expected=oracle(values,row)[0]
    np.testing.assert_allclose(scalar(*values),expected,rtol=4e-5,atol=1e-4)
    assert scalar._native_descriptor_last_receipt["execution_kind"]=="native_gpu"
    def forbidden(*args,**kwargs):raise AssertionError("warm scalar attempted compiler subprocess")
    monkeypatch.setattr(subprocess,"run",forbidden)
    changed=[*values]
    changed[2]=changed[2]-np.uint8(1) if encoded else changed[2]*.5
    np.testing.assert_allclose(scalar(*changed),expected*.5,rtol=4e-5,atol=1e-4)


@pytest.mark.parametrize("k",[1,37])
@pytest.mark.parametrize("ta,tb",tuple(itertools.product((False,True),repeat=2)))
def test_scalar_scale_jvp(k,ta,tb,monkeypatch):
    from tessera import runtime
    assert runtime._rocm_live_arch()=="gfx1201"
    scalar,values,row=scalar_case(ta,tb,False,k,jvp=True)
    expected=oracle(values,row)
    got=scalar.native_jvp(*values[:4],tangents=tuple(values[4:]))
    for actual,wanted in zip(got,expected,strict=True):
        np.testing.assert_allclose(actual,wanted,rtol=4e-5,atol=1e-4)
    def forbidden(*args,**kwargs):raise AssertionError("warm scalar JVP attempted compiler subprocess")
    monkeypatch.setattr(subprocess,"run",forbidden)
    changed=[*values];changed[4]=changed[4]*.5;changed[5]=changed[5]*.5
    got=scalar.native_jvp(*changed[:4],tangents=tuple(changed[4:]))
    np.testing.assert_allclose(got[0],expected[0],rtol=4e-5,atol=1e-4)
    np.testing.assert_allclose(got[1],expected[1]*.5,rtol=4e-5,atol=1e-4)
