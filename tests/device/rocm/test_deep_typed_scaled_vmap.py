"""Owning gfx1201 arbitrary static leading-prefix execution."""
import os
import subprocess
import numpy as np
import pytest
from tessera import runtime
from tests.unit.test_native_deep_typed_vmap import deep_case
from tests.device.rocm.test_public_scaled_jvp import oracle
from tests.support.scaled_product_transpose_oracle import scale_adjoint

pytestmark=pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",
    reason="owning gfx1201 required")

@pytest.mark.parametrize("prefix",[(2,1,3),(1,2,1,3)])
@pytest.mark.parametrize("policy",["shared_rhs_rows","independent_rhs","shared_lhs"])
@pytest.mark.parametrize("fmt",["fp32","e8m0"])
@pytest.mark.parametrize("nk",[False,True])
def test_deep_native_primal(prefix,policy,fmt,nk,monkeypatch):
    assert runtime._rocm_live_arch()=="gfx1201"
    _,owner,values,expected,_=deep_case(policy,fmt,nk,prefix)
    got=owner(*values)
    np.testing.assert_allclose(got,expected,rtol=4e-5,atol=1e-4)
    assert owner.compile_result.executable
    assert owner._native_descriptor_last_receipt["execution_kind"]=="native_gpu"
    assert "tile.scaled_matmul_kernel" in owner.compile_result.tile_ir
    def forbidden(*a,**kw):raise AssertionError("warm mapped primal invoked compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    if fmt=="fp32":
        changed=(*values[:2],values[2]*.5,values[3])
    else:
        changed=(*values[:2],values[2]-np.uint8(1),values[3])
    np.testing.assert_allclose(owner(*changed),expected*.5,rtol=4e-5,atol=1e-4)

@pytest.mark.parametrize("prefix",[(2,1,3),(1,2,1,3)])
@pytest.mark.parametrize("policy",["shared_rhs_rows","independent_rhs","shared_lhs"])
@pytest.mark.parametrize("nk",[False,True])
def test_deep_native_scale_jvp(prefix,policy,nk,monkeypatch):
    assert runtime._rocm_live_arch()=="gfx1201"
    _,owner,values,expected,_=deep_case(policy,"fp32",nk,prefix,mode="forward")
    axes=owner._frontend_batch_axes
    rng=np.random.default_rng(1907)
    seeds=tuple(rng.uniform(-.1,.1,v.shape).astype(np.float32) for v in values[2:])
    gold=[]
    for plane in np.ndindex(prefix):
        a,b,sa,sb=(v[plane] if axis==0 else v for v,axis in zip(values,axes,strict=True))
        da,db=(v[plane] if axis==0 else v for v,axis in zip(seeds,axes[2:],strict=True))
        gold.append(oracle(a,b.T if nk else b,sa,sb,da,db)[1])
    tangent=np.stack(gold).reshape(expected.shape)
    actual=owner.native_jvp(*values,tangents=seeds)
    np.testing.assert_allclose(actual[0],expected,rtol=4e-5,atol=1e-4)
    np.testing.assert_allclose(actual[1],tangent,rtol=4e-5,atol=1e-4)
    assert owner.last_jvp_execution["execution_kind"]=="native_gpu"
    def forbidden(*a,**kw):raise AssertionError("warm mapped JVP invoked compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    repeated=owner.native_jvp(*values,tangents=tuple(v*-.5 for v in seeds))
    np.testing.assert_allclose(repeated[1],tangent*-.5,rtol=4e-5,atol=1e-4)

@pytest.mark.parametrize("prefix",[(2,1,3),(1,2,1,3)])
@pytest.mark.parametrize("policy",["shared_rhs_rows","independent_rhs","shared_lhs"])
@pytest.mark.parametrize("nk",[False,True])
@pytest.mark.parametrize("schedule",["serial_per_scale_element","wave_per_scale_element"])
def test_deep_native_scale_vjp(prefix,policy,nk,schedule,monkeypatch):
    assert runtime._rocm_live_arch()=="gfx1201"
    monkeypatch.setenv("TESSERA_ROCM_SCALE_VJP_SCHEDULE",schedule)
    _,owner,values,expected,_=deep_case(policy,"fp32",nk,prefix,mode="reverse")
    dy=np.random.default_rng(1908).uniform(-.5,.5,expected.shape).astype(np.float32)
    gold=scale_adjoint(*values,dy,scale_k=128,scale_n=128,batching=policy,transpose_b=nk)
    actual=owner.native_backward(*values,out_cotangents=dy)
    for got,want in zip(actual,gold,strict=True):
        np.testing.assert_allclose(got,want,rtol=4e-5,atol=2e-4)
    assert owner.last_backward_execution["execution_certificate"]["evidence_scope"]=="exact_device"
    assert owner.last_backward_execution["scale_adjoint_schedule"]==schedule
    def forbidden(*a,**kw):raise AssertionError("warm mapped VJP invoked compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    repeated=owner.native_backward(*values,out_cotangents=dy*-.5)
    for got,want in zip(repeated,gold,strict=True):
        np.testing.assert_allclose(got,want*-.5,rtol=4e-5,atol=2e-4)
