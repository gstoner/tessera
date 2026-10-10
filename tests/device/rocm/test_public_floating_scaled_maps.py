"""Owning f32 mapped adjoints, including matrix unbroadcast and seed inversion."""
import itertools
import json
import os
import subprocess
import numpy as np
import pytest
from tessera import runtime
from tests.unit.test_public_floating_scaled_maps import mapped_case
from tests.device.rocm.test_floating_scaled_adjoint import batch_oracle

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="owning gfx1201 required")

@pytest.mark.parametrize("mask", range(1,16))
@pytest.mark.parametrize("ta,tb", tuple(itertools.product((False,True),repeat=2)))
@pytest.mark.parametrize("prefix", [(2,), (2,3)])
@pytest.mark.parametrize("out_axes", [0,-1])
@pytest.mark.parametrize("roles", [("a","b","sa","sb"), ("sb","b","a","sa")])
def test_mapped_floating_reverse_executes_native_and_retains_outputs(mask,ta,tb,prefix,out_axes,roles,monkeypatch):
    assert runtime._rocm_live_arch() == "gfx1201"
    _,owner,values,seed = mapped_case(mask,ta,tb,prefix,out_axes,roles)
    permutation = owner._frontend_output_permutation
    mapped_seed = np.ascontiguousarray(np.transpose(seed,permutation))
    expected = batch_oracle((*values,seed),ta,tb)
    positions = {"a":0,"b":1,"sa":2,"sb":3}
    actual = owner.native_backward(*values,out_cotangents=mapped_seed)
    for result,name in zip(actual,roles,strict=True):
        np.testing.assert_allclose(result,expected[positions[name]],rtol=4e-5,atol=3e-6)
    receipt = owner.last_backward_execution
    assert receipt["execution_kind"] == "native_gpu"
    assert receipt["evidence_target"] == "rocm_gfx1201"
    assert receipt["frontend_authority"] == "tracer"
    program = json.loads(owner._native_backward_artifact.program_json)
    assert program["gradient_roles"] == [positions[name] for name in roles]
    if out_axes != 0:
        assert any(step["operation"] == "tessera.transpose" for step in program["steps"])
    retained = tuple(value.copy() for value in actual)
    def forbidden(*a,**kw): raise AssertionError("warm mapped reverse invoked compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    repeated = owner.native_backward(*values,out_cotangents=mapped_seed*np.float32(-.5))
    for result,name in zip(repeated,roles,strict=True):
        np.testing.assert_allclose(result,expected[positions[name]]*-.5,rtol=4e-5,atol=3e-6)
    for result,old in zip(actual,retained,strict=True):
        np.testing.assert_array_equal(result,old)



@pytest.mark.parametrize("ta,tb", tuple(itertools.product((False,True),repeat=2)))
@pytest.mark.parametrize("nested",[False,True])
@pytest.mark.parametrize("roles",[("a","b","sa","sb"),("sb","b","a","sa")])
def test_mixed_floating_axes_execute_and_restore_original_gradients(ta,tb,nested,roles,monkeypatch):
    from tests.unit.test_public_floating_scaled_maps import mixed_case
    assert runtime._rocm_live_arch()=="gfx1201"
    _,owner,raw,seed,expected=mixed_case(ta,tb,nested,roles)
    mapped_seed=np.ascontiguousarray(seed.transpose(owner._frontend_output_permutation))
    positions={"a":0,"b":1,"sa":2,"sb":3}
    actual=owner.native_backward(*raw,out_cotangents=mapped_seed)
    retained=tuple(value.copy() for value in actual)
    for result,name in zip(actual,roles,strict=True):
        np.testing.assert_allclose(result,expected[positions[name]],rtol=4e-5,atol=3e-6)
        assert result.shape==raw[positions[name]].shape
    assert owner.last_backward_execution["execution_kind"]=="native_gpu"
    assert owner.last_backward_execution["frontend_authority"]=="tracer"
    def forbidden(*a,**kw): raise AssertionError("warm mixed reverse invoked compiler")
    monkeypatch.setattr(subprocess,"run",forbidden)
    repeated=owner.native_backward(*raw,out_cotangents=mapped_seed*np.float32(-.5))
    for result,name in zip(repeated,roles,strict=True):
        np.testing.assert_allclose(result,expected[positions[name]]*-.5,rtol=4e-5,atol=3e-6)
    for result,old in zip(actual,retained,strict=True):
        np.testing.assert_array_equal(result,old)
