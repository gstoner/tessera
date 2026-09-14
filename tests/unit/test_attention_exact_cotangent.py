"""Public reverse capture must enforce the same conversion contract as launch."""
from types import SimpleNamespace
import numpy as np
import pytest
from tessera.compiler.native_attention_vjp import _backward_module


def capture(cotangent):
    arrays = tuple(np.ones(s,np.float16) for s in ((1,2,3,16),(1,1,4,16),(1,1,4,16)))
    return _backward_module(source=SimpleNamespace(op_name='tessera.gqa_attention',operands=['%q','%k','%v'],kwargs={}),
        target='rocm',ordered_inputs=arrays,arg_names=('q','k','v'),source_arg_names=('q','k','v'),out_cotangent=cotangent)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_public_capture_preserves_exact_wider_cotangent(dtype):
    module,_,_ = capture(np.ones((1,2,3,16),dtype))
    assert module.functions[0].args[0].ir_type.dtype == 'fp16'


@pytest.mark.parametrize('value',[.1,float('inf'),float('nan')])
def test_public_capture_refuses_lossy_or_nonfinite_conversion(value):
    with pytest.raises(ValueError,match='information|finite'):
        capture(np.full((1,2,3,16),value,np.float32))
