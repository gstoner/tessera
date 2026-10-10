"""Two public leading maps descend into one native gfx1201 primal program."""
import os
import subprocess
import numpy as np
import pytest
from tests.unit.test_native_nested_typed_vmap import nested_case
pytestmark=pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="owning gfx1201 required")

@pytest.mark.parametrize("shape",[(2,3,7,19,256),(2,2,200,129,1536)])
@pytest.mark.parametrize("policy",["shared_rhs_rows","independent_rhs","shared_lhs"])
@pytest.mark.parametrize("fmt",["fp32","e8m0"])
@pytest.mark.parametrize("nk",[False,True])
def test_public_nested_native_primal(shape,policy,fmt,nk,monkeypatch):
    from tessera import runtime
    assert runtime._rocm_live_arch()=="gfx1201"
    scalar,inner,outer,values,expected=nested_case(policy,fmt,nk,shape)
    scalar_result,inner_result=scalar.compile_result,inner.compile_result
    np.testing.assert_allclose(outer(*values),expected,rtol=4e-5,atol=1e-4)
    assert scalar.compile_result is scalar_result and inner.compile_result is inner_result
    assert outer._native_descriptor_last_receipt["execution_kind"]=="native_gpu"
    assert outer._frontend_batch_depth==2
    assert "tile.scaled_matmul_kernel" in outer.compile_result.tile_ir
    def forbidden(*args,**kwargs):raise AssertionError("warm nested primal invoked compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    from tessera.compiler import reference_typed_scaled_matmul as reference
    monkeypatch.setattr(reference,"reference_typed_scaled_matmul",forbidden)
    a,b,sa,sb=values
    changed=sa*.5 if fmt=="fp32" else sa-np.uint8(1)
    np.testing.assert_allclose(outer(a,b,changed,sb),expected*.5,rtol=4e-5,atol=1e-4)

@pytest.mark.parametrize("policy",["shared_rhs_rows","independent_rhs","shared_lhs"])
@pytest.mark.parametrize("nk",[False,True])
def test_public_nested_native_scale_jvp(policy,nk,monkeypatch):
    scalar,inner,outer,values,expected=nested_case(policy,"fp32",nk,jvp=True)
    seeds=(values[2]*.125,values[3]*-.0625)
    result=outer.native_jvp(*values,tangents=seeds)
    np.testing.assert_allclose(result[0],expected,rtol=4e-5,atol=1e-4)
    np.testing.assert_allclose(result[1],expected*.0625,rtol=4e-5,atol=1e-4)
    assert outer.last_jvp_execution["execution_kind"]=="native_gpu"
    assert scalar._frontend_batch_axes is None and inner._frontend_batch_depth==1
    def forbidden(*args,**kwargs):raise AssertionError("warm nested JVP invoked compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    from tessera.compiler import reference_typed_scaled_matmul as reference
    monkeypatch.setattr(reference,"reference_typed_scaled_matmul",forbidden)
    repeated=outer.native_jvp(*values,tangents=tuple(-x for x in seeds))
    np.testing.assert_allclose(repeated[1],expected*-.0625,rtol=4e-5,atol=1e-4)
