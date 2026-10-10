"""Exact gfx1201 multidimensional static native batches."""
import os
import subprocess
import numpy as np
import pytest
from tests.unit.test_rocm_multidimensional_scaled_batch import nested_case
pytestmark=pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="owning gfx1201 required")

@pytest.mark.parametrize("shape",[(2,3,7,19,256),(2,2,200,129,1536)])
@pytest.mark.parametrize("policy",["shared_rhs_rows","independent_rhs","shared_lhs"])
@pytest.mark.parametrize("fmt",["fp32","e8m0"])
@pytest.mark.parametrize("nk",[False,True])
def test_native_two_axis_batch(shape,policy,fmt,nk,monkeypatch):
    from tessera import runtime
    assert runtime._rocm_live_arch()=="gfx1201"
    owner,values,expected=nested_case(policy,fmt,nk,shape)
    got=owner(*values)
    np.testing.assert_allclose(got,expected,rtol=4e-5,atol=1e-4)
    assert got.shape==expected.shape
    assert owner._native_descriptor_last_receipt["execution_kind"]=="native_gpu"
    assert "tile.scaled_matmul_kernel" in owner.compile_result.tile_ir
    def forbidden(*args,**kwargs):raise AssertionError("warm two-axis call invoked compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    a,b,sa,sb=values
    changed=sa*.5 if fmt=="fp32" else sa-np.uint8(1)
    np.testing.assert_allclose(owner(a,b,changed,sb),expected*.5,rtol=4e-5,atol=1e-4)

@pytest.mark.parametrize("policy",["shared_rhs_rows","independent_rhs","shared_lhs"])
@pytest.mark.parametrize("nk",[False,True])
def test_native_two_axis_scale_jvp(policy,nk,monkeypatch):
    import tessera as ts
    owner,values,expected=nested_case(policy,"fp32",nk)
    forward=ts.jit(target="rocm_gfx1201",autodiff="forward",wrt=("sa","sb"))(owner._fn)
    seeds=(values[2]*.125, values[3]*-.0625)
    primal,tangent=forward.native_jvp(*values,tangents=seeds)
    np.testing.assert_allclose(primal,expected,rtol=4e-5,atol=1e-4)
    np.testing.assert_allclose(tangent,expected*.0625,rtol=4e-5,atol=1e-4)
    assert forward.last_jvp_execution["execution_kind"]=="native_gpu"
    def forbidden(*args,**kwargs):raise AssertionError("warm two-axis JVP invoked compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    repeated=forward.native_jvp(*values,tangents=tuple(-s for s in seeds))
    np.testing.assert_allclose(repeated[1],expected*-.0625,rtol=4e-5,atol=1e-4)
