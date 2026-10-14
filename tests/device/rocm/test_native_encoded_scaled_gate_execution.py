"""Owning gfx1201 encoded coefficients and continuous native AD."""
import copy
import os
import subprocess

import numpy as np
import pytest
from tessera import runtime
from tessera.autodiff import jvp

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="owning gfx1201 required")

from tests.unit.test_native_encoded_scaled_gate import typed_case

def typed_oracle(values,fmt):
    a,b,sa,sb,gate=values
    m,k=a.shape
    n=b.shape[1]
    sk,sn=(128,128) if fmt=="fp32" else (32,1)
    left=sa.astype(np.float64) if fmt=="fp32" else np.exp2(sa.astype(np.float64)-127)
    right=sb.astype(np.float64) if fmt=="fp32" else np.exp2(sb.astype(np.float64)-127)
    product=np.zeros((m,n),np.float64)
    for group in range((k+sk-1)//sk):
        product+=(a[:,group*sk:(group+1)*sk].astype(np.float64) @
                  b[group*sk:(group+1)*sk].astype(np.float64))*left[:,group,None]*right[group,np.arange(n)//sn]
    return product*gate.astype(np.float64)

@pytest.mark.parametrize("fmt,roles", [
    ("fp32", (2, 3, 4)), ("fp32", (4,)), ("fp32", (4, 3, 2)), ("e8m0", (4,)),
])
def test_encoded_primal_coefficient_and_continuous_gate_derivatives(fmt,roles,monkeypatch):
    assert runtime._rocm_live_arch()=="gfx1201"
    names = ("a", "b", "sa", "sb", "gate")
    owner,values,directions=typed_case(fmt, roles_override=tuple(names[i] for i in roles))
    request = owner.differentiation_request
    try:
        actual=owner(*values)
        wanted=typed_oracle(values,fmt)
        np.testing.assert_allclose(actual,wanted,rtol=5e-5,atol=6e-5)
        assert owner._native_descriptor_last_receipt['execution_kind'] == 'native_gpu'
        graph = copy.deepcopy(owner.graph_ir)
        witness = copy.deepcopy(owner._legacy_graph_ir or owner.graph_ir)
        forward=jvp(owner,values,directions)
        assert owner.last_jvp_execution['execution_kind'] == 'native_gpu'
        assert owner.graph_ir == graph
        assert owner.differentiation_request is request
        tangent=np.zeros_like(wanted)
        for role in roles:
            frame=list(values)
            frame[role]=directions[role]
            tangent+=typed_oracle(frame,fmt)
        np.testing.assert_allclose(forward[0],wanted,rtol=5e-5,atol=6e-5)
        np.testing.assert_allclose(forward[1],tangent,rtol=5e-5,atol=6e-5)
        cot=np.random.default_rng(81173).uniform(-.5,.5,wanted.shape).astype(np.float32)
        gradients=[]
        for role in roles:
            result=np.empty(values[role].shape,np.float64)
            for index in np.ndindex(result.shape):
                plus,minus=list(values),list(values)
                plus[role]=values[role].astype(np.float64)
                minus[role]=values[role].astype(np.float64)
                plus[role][index]+=.0001
                minus[role][index]-=.0001
                result[index]=np.sum(cot*(typed_oracle(plus,fmt)-typed_oracle(minus,fmt)))/.0002
            gradients.append(result)
        backward=owner.native_backward(*values,out_cotangents=cot)
        for got,want in zip(backward,gradients,strict=True):
            np.testing.assert_allclose(got,want,rtol=6e-5,atol=7e-5)
        assert owner.last_backward_execution["execution_kind"]=="native_gpu"
        retained=tuple(v.copy() for v in forward)
        def forbidden(*args,**kwargs):
            raise AssertionError("warm encoded gate used compiler subprocess")
        monkeypatch.setattr(subprocess,"run",forbidden)
        changed_a = (values[0].astype(np.float32) * np.float32(.5)).astype(values[0].dtype)
        changed_b = (values[1].astype(np.float32) * np.float32(-.5)).astype(values[1].dtype)
        if fmt == "fp32":
            changed_scales = (values[2] * np.float32(1.125), values[3] * np.float32(.75))
        else:
            changed_scales = (values[2] + np.uint8(1), values[3] - np.uint8(1))
        changed=(changed_a,changed_b,*changed_scales,values[4]*np.float32(-.5))
        repeated=jvp(owner,changed,directions)
        np.testing.assert_allclose(repeated[0],typed_oracle(changed,fmt),rtol=5e-5,atol=6e-5)
        changed_tangent = np.zeros_like(wanted)
        for role in roles:
            frame = list(changed)
            frame[role] = directions[role]
            changed_tangent += typed_oracle(frame, fmt)
        np.testing.assert_allclose(repeated[1], changed_tangent, rtol=5e-5, atol=6e-5)
        np.testing.assert_allclose(owner(*changed), typed_oracle(changed, fmt), rtol=5e-5, atol=6e-5)
        assert owner._native_descriptor_last_receipt["execution_kind"] == "native_gpu"
        assert (owner._legacy_graph_ir or owner.graph_ir) == witness
        assert owner.differentiation_request is request
        for got,old in zip(forward,retained,strict=True):
            np.testing.assert_array_equal(got,old)
        for got,want in zip(owner.native_backward(*values,out_cotangents=-cot),gradients,strict=True):
            np.testing.assert_allclose(got,-want,rtol=6e-5,atol=7e-5)
    finally:
        owner.close_native_storage()
