"""Native JVP enforces source shape bounds before frontend or backend work."""
import numpy as np
import pytest
import tessera as ts
from tessera.compiler.constraints import Range, TesseraConstraintError

def sum_rows(x: ts.Tensor["M","K","fp32"]):
    return ts.ops.sum(x,axis=1)

@pytest.mark.parametrize("target",["x86","rocm","nvidia_sm120"])
@pytest.mark.parametrize("keyword",[False,True])
def test_native_jvp_rejects_source_bounds_before_capture(target,keyword,monkeypatch):
    fn=ts.jit(target=target,autodiff="forward",wrt=("x",))(sum_rows)
    fn.constraints.add(Range("M",1,6))
    x=np.ones((7,8),np.float32)
    def forbidden(*args,**kwargs):
        raise AssertionError("invalid source bound reached frontend or backend")
    monkeypatch.setattr(fn,"_traced_autodiff_module",forbidden)
    monkeypatch.setattr(fn,"frontend_differential",forbidden)
    monkeypatch.setattr(fn,"_compile_jvp_module",forbidden)
    if keyword:
        call=lambda:fn.native_jvp(x=x,tangents=np.ones_like(x))
    else:
        call=lambda:fn.native_jvp(x,tangents=np.ones_like(x))
    with pytest.raises(TesseraConstraintError,match="M"):
        call()
