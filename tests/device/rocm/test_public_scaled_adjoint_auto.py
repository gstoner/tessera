"""Exact-gfx1201 public reverse through native automatic scale scheduling."""
import json
import os
import subprocess
import numpy as np
import pytest
import tessera as ts
from tessera import runtime
from tessera.autodiff import vmap
from tests.unit.test_native_typed_scaled_vmap import case
from tests.device.rocm.test_public_mapped_scale_vjp import scale_oracle
from tests.unit.test_public_floating_scaled_reverse import floating_owner
from tests.device.rocm.test_floating_scaled_adjoint import inputs,oracle

pytestmark=pytest.mark.skipif(os.environ.get("TESSERA_GFX1201_DEVICE_PROOF")!="1",
                             reason="owning gfx1201 required")


@pytest.mark.parametrize("policy",["independent_rhs","shared_lhs","shared_rhs_rows"])
@pytest.mark.parametrize("nk",[False,True])
@pytest.mark.parametrize("columns",[1,5,19])
@pytest.mark.parametrize("roles",[("sa",),("sb",),("sa","sb"),("sb","sa")])
def test_public_auto_preserves_scale_roles_values_and_warm_ownership(
        policy,nk,columns,roles,monkeypatch):
    assert runtime._rocm_live_arch()=="gfx1201"
    monkeypatch.setenv("TESSERA_ROCM_SCALE_VJP_SCHEDULE","auto")
    scalar,leading,raw,primal=case(policy,"fp32",nk,(2,7,columns,256))
    owner=vmap(ts.jit(target="rocm_gfx1201",autodiff="reverse",wrt=roles)(scalar._fn),
               in_axes=leading._frontend_batch_axes)
    seed=np.random.default_rng(942).uniform(-.5,.5,primal.shape).astype(np.float32)
    oracle_values=list(raw)
    if nk:oracle_values[1]=raw[1].swapaxes(-1,-2)
    expected_all=scale_oracle(oracle_values,leading._frontend_batch_axes,seed)
    expected=tuple(expected_all[{"sa":0,"sb":1}[role]] for role in roles)
    outputs=owner.native_backward(*raw,out_cotangents=seed)
    for actual,wanted in zip(outputs,expected,strict=True):
        np.testing.assert_allclose(actual,wanted,rtol=4e-5,atol=3e-5)
    assert owner.last_backward_execution["execution_kind"]=="native_gpu"
    package=owner._native_backward_artifact
    algorithm="serial_per_scale_element" if columns==1 else "wave_per_scale_element"
    for raw_member in package.members_json:
        member=json.loads(raw_member)
        assert member["scale_adjoint_schedule"]==algorithm
        assert member["geometry"][3:]==([128,1,1] if columns==1 else [32,1,1])
    retained=tuple(output.copy() for output in outputs)
    def forbidden(*args,**kwargs):raise AssertionError("warm automatic reverse invoked compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    changed=owner.native_backward(*raw,out_cotangents=seed*np.float32(-.5))
    for actual,wanted,old,copy in zip(changed,expected,outputs,retained,strict=True):
        np.testing.assert_allclose(actual,wanted*-.5,rtol=4e-5,atol=3e-5)
        np.testing.assert_array_equal(old,copy)


@pytest.mark.parametrize("ta,tb",[(False,False),(True,False),(False,True),(True,True)])
def test_public_auto_keeps_continuous_four_gradient_route(ta,tb,monkeypatch):
    assert runtime._rocm_live_arch()=="gfx1201"
    monkeypatch.setenv("TESSERA_ROCM_SCALE_VJP_SCHEDULE","auto")
    values=inputs(ta,tb)
    owner=floating_owner(ta,tb)
    outputs=owner.native_backward(*values[:4],out_cotangents=values[4])
    for actual,wanted in zip(outputs,oracle(values,ta,tb),strict=True):
        np.testing.assert_allclose(actual,wanted,rtol=4e-5,atol=3e-5)
    for raw_member in owner._native_backward_artifact.members_json:
        assert json.loads(raw_member)["scale_adjoint_schedule"]=="serial_per_scale_element"
