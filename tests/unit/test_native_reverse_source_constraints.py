"""Reverse source bounds precede tracing and physical target selection."""
import numpy as np
import pytest
import tessera as ts
from tessera.compiler.constraints import Range,TesseraConstraintError

def bounded_identity(x:ts.Tensor["M","N","fp32"]):
    return x

@pytest.mark.parametrize("target",["apple_gpu","x86","rocm_gfx1151","rocm_gfx1201","nvidia_sm120"])
@pytest.mark.parametrize("binding",["positional","keyword"])
def test_reverse_invalid_source_bound_precedes_capture(target,binding,monkeypatch):
    owner=ts.jit(target=target,autodiff="reverse",wrt=("x",))(bounded_identity)
    owner.constraints.add(Range("M",1,6))
    owner.last_backward_execution={"stale":True}
    owner._native_backward_artifact=object()
    def forbidden(*a,**kw):raise AssertionError("invalid source bound reached capture")
    monkeypatch.setattr(owner,"_specialized_autodiff_module",forbidden)
    x=np.ones((7,3),np.float32)
    with pytest.raises(TesseraConstraintError,match="M"):
        if binding=="keyword":owner.native_backward(x=x,out_cotangents=x)
        else:owner.native_backward(x,out_cotangents=x)
    assert owner.last_backward_execution is None
    assert owner._native_backward_artifact is None

@pytest.mark.parametrize("target",["apple_gpu","x86","rocm_gfx1151","rocm_gfx1201","nvidia_sm120"])
def test_reverse_valid_source_bound_reaches_capture(target,monkeypatch):
    owner=ts.jit(target=target,autodiff="reverse",wrt=("x",))(bounded_identity)
    owner.constraints.add(Range("M",1,6))
    def control(*a,**kw):raise LookupError("valid source reached capture")
    monkeypatch.setattr(owner,"_specialized_autodiff_module",control)
    x=np.ones((3,3),np.float32)
    with pytest.raises(LookupError,match="valid source"):
        owner.native_backward(x,out_cotangents=x)
