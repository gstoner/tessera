"""Host-free selection tests for the bounded standalone SM120 ReLU route."""
import importlib
import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera.compiler.jit import _JIT_FALLBACK
from tessera.compiler import nvidia_native

def relu(x):
    return ts.ops.relu(x)

def configured(value, monkeypatch):
    fn=ts.jit(target="nvidia_sm120")(relu)
    initial=fn._traced_autodiff_module((np.zeros((7,32),np.float32),),{})
    specialized=fn._traced_autodiff_module((value,),{})
    fn.graph_ir=initial
    monkeypatch.setattr(fn,"_trace_frontend_capture",lambda *args,**kwargs:(specialized,{}))
    return fn,specialized

@pytest.mark.parametrize("shape,dtype",[
    ((7,),np.float16),((7,),ml_dtypes.bfloat16),((7,),np.float32),
    ((2,3,4),np.float32),((7,32),np.float64),((0,32),np.float32),
])
def test_unsupported_specialized_relu_leaves_native_descriptor_path(shape,dtype,monkeypatch):
    value=np.zeros(shape,dtype)
    fn,specialized=configured(value,monkeypatch)
    assert nvidia_native.requests_activation(fn.graph_ir)
    assert nvidia_native.supports_activation(fn.graph_ir)
    assert not nvidia_native.supports_activation(specialized)
    def forbidden(*args,**kwargs):
        pytest.fail("unsupported specialized ReLU committed to native compilation")
    canonical=importlib.import_module("tessera.compiler.canonical_compile")
    monkeypatch.setattr(canonical,"canonical_compile",forbidden)
    assert fn._try_native_descriptor_call((value,),{}) is _JIT_FALLBACK
    assert not fn._native_descriptor_specializations

@pytest.mark.parametrize("dtype",[np.float16,ml_dtypes.bfloat16,np.float32])
def test_supported_specialized_relu_still_selects_canonical_native_compilation(dtype,monkeypatch):
    value=np.zeros((7,32),dtype)
    fn,specialized=configured(value,monkeypatch)
    assert nvidia_native.supports_activation(specialized)
    class SelectedNative(Exception):
        pass
    def selected(module,**kwargs):
        assert module is specialized
        assert kwargs["target"]=="nvidia_sm120"
        raise SelectedNative
    canonical=importlib.import_module("tessera.compiler.canonical_compile")
    monkeypatch.setattr(canonical,"canonical_compile",selected)
    with pytest.raises(SelectedNative):
        fn._try_native_descriptor_call((value,),{})
