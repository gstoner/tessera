"""Exact gfx1201 public vmap execution through native Graph batches."""
import os, subprocess
import numpy as np
import pytest
from tests.unit.test_native_typed_scaled_vmap import case
pytestmark=pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",reason="owning gfx1201 required")

@pytest.mark.parametrize("shape",[(3,7,19,256),(2,200,129,1536),(2,128,4096,256)])
@pytest.mark.parametrize("policy",["shared_rhs_rows","independent_rhs","shared_lhs"])
@pytest.mark.parametrize("fmt",["fp32","e8m0"])
@pytest.mark.parametrize("nk",[False,True])
def test_public_typed_vmap_native(shape,policy,fmt,nk,monkeypatch):
    from tessera import runtime
    assert runtime._rocm_live_arch()=="gfx1201"
    scalar,owner,values,expected=case(policy,fmt,nk,shape)
    scalar_result=scalar.compile_result
    got=owner(*values)
    np.testing.assert_allclose(got,expected,rtol=4e-5,atol=1e-4)
    assert owner._native_descriptor_last_receipt["execution_kind"]=="native_gpu"
    assert "tile.scaled_matmul_kernel" in owner.compile_result.tile_ir
    assert scalar.compile_result is scalar_result
    def forbidden(*args,**kwargs):raise AssertionError("warm mapped call invoked compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    a,b,sa,sb=values
    changed=sa*.5 if fmt=="fp32" else sa-np.uint8(1)
    np.testing.assert_allclose(owner(a,b,changed,sb),expected*.5,rtol=4e-5,atol=1e-4)

@pytest.mark.parametrize("shape",[(3,7,19,256),(2,200,129,1536)])
@pytest.mark.parametrize("policy",["shared_rhs_rows","independent_rhs","shared_lhs"])
@pytest.mark.parametrize("nk",[False,True])
@pytest.mark.parametrize("wrt",[("sa",),("sb",),("sa","sb")])
def test_mapped_native_scale_jvp(shape,policy,nk,wrt,monkeypatch):
    import tessera as ts
    from tessera.autodiff import vmap
    from tessera import runtime
    assert runtime._rocm_live_arch()=="gfx1201"
    scalar,primal,values,expected=case(policy,"fp32",nk,shape)
    forward=ts.jit(target="rocm_gfx1201",autodiff="forward",wrt=wrt)(scalar._fn)
    owner=vmap(forward,in_axes=primal._frontend_batch_axes)
    a,b,sa,sb=values
    logical_b=b.swapaxes(-1,-2) if nk else b
    da=np.full_like(sa,.05) if "sa" in wrt else np.zeros_like(sa)
    db=np.full_like(sb,-.03) if "sb" in wrt else np.zeros_like(sb)
    def independent(scale_a,scale_b):
        result=np.zeros_like(expected)
        for g in range(a.shape[-1]//128):
            product=a[...,g*128:(g+1)*128].astype(np.float64) @ logical_b[...,g*128:(g+1)*128,:].astype(np.float64)
            result+=product*scale_a[...,g,None]*scale_b[...,g,np.arange(expected.shape[-1])//128][...,None,:]
        return result
    tangent=independent(da.astype(np.float64),sb.astype(np.float64))+independent(sa.astype(np.float64),db.astype(np.float64))
    eps=1e-3
    finite=(independent(sa.astype(np.float64)+eps*da,sb.astype(np.float64)+eps*db)-independent(sa.astype(np.float64)-eps*da,sb.astype(np.float64)-eps*db))/(2*eps)
    np.testing.assert_allclose(tangent,finite,rtol=4e-5,atol=1e-4)
    seeds=tuple({"sa":da,"sb":db}[name] for name in wrt)
    got=owner.native_jvp(*values,tangents=seeds)
    for actual,want in zip(got,(expected,tangent),strict=True):
        np.testing.assert_allclose(actual,want,rtol=4e-5,atol=1e-4)
    assert owner.last_jvp_execution["execution_kind"]=="native_gpu"
    assert owner.last_jvp_execution["evidence_target"]=="rocm_gfx1201"
    assert forward._frontend_batch_axes is None
    def forbidden(*args,**kwargs):raise AssertionError("warm mapped JVP invoked compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    repeated=owner.native_jvp(*values,tangents=tuple(seed*-.5 for seed in seeds))
    for actual,want in zip(repeated,(expected,tangent*-.5),strict=True):
        np.testing.assert_allclose(actual,want,rtol=4e-5,atol=1e-4)
