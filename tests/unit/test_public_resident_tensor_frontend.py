"""Resident CUDA metadata reaches ordinary tracer-owned Graph without copies."""
from copy import deepcopy
import numpy as np
import pytest
import tessera as ts
from tessera.compiler.constraints import TesseraConstraintError
from tessera.compiler.resident_nvidia_tensor import cuda_frontend_specs
from tests.unit.test_ordered_resident_tensor_dag import Buffer

SOURCE = """def native_roots(a: tessera.Tensor["M","K","fp16"], b: tessera.Tensor["K","N","fp16"]):
    left=tessera.ops.rmsnorm(a,eps=1e-5)
    right=tessera.ops.softmax(b,axis=-1)
    return tessera.ops.matmul(left,right,output_dtype="fp32")
"""


def owner():
    return ts.from_text(SOURCE,target="nvidia_sm120")


def test_abstract_resident_trace_keeps_complete_typed_graph_and_cache(monkeypatch):
    fn=owner()
    values=(Buffer((17,32)),Buffer((32,11)))
    module,_=fn._trace_frontend_capture(values,{})
    function=module.functions[0]
    assert [op.op_name for op in function.body]==["tessera.rmsnorm","tessera.softmax","tessera.matmul"]
    assert function.result_types[0].shape==("17","11")
    before=deepcopy(module)
    def forbidden(*args,**kwargs):
        pytest.fail("warm resident signature executed frontend")
    monkeypatch.setattr(fn,"_fn",forbidden)
    assert fn._trace_frontend_capture(values,{})[0]==before


def test_resident_shape_constraints_use_interface_not_missing_provider_shape():
    fn=owner()
    with pytest.raises(TesseraConstraintError,match="Inconsistent binding"):
        fn._enforce_call_time_constraints((Buffer((17,32)),Buffer((31,11))),{})


@pytest.mark.parametrize("field,value",[
    ("version",2),("version",True),("shape",(0,32)),("shape",(True,32)),
    ("shape",(2**62,32)),("shape",(17,32,1)),("typestr","<u2"),
    ("typestr",">f2"),("typestr","bad"),("strides",(66,2)),
    ("strides",(-64,2)),("data",(0,False)),("data",(4096,0)),
    ("stream",None),("stream",0),("stream",False),("stream",-1),
])
def test_bad_resident_metadata_rejected_before_trace(field,value,monkeypatch):
    fn=owner()
    a,b=Buffer((17,32)),Buffer((32,11))
    a.interface[field]=value
    def forbidden(*args,**kwargs):
        pytest.fail("malformed root reached frontend")
    monkeypatch.setattr(fn,"_fn",forbidden)
    with pytest.raises(ValueError):
        fn._trace_frontend_capture((a,b),{})


def test_mixed_roots_rejected_before_host_coercion():
    fn=owner()
    with pytest.raises(ValueError,match="all roots"):
        fn._trace_frontend_capture((Buffer((17,32)),np.zeros((32,11),np.float16)),{})


def test_resident_numerical_certificate_requires_explicit_oracle_inputs():
    with pytest.raises(ValueError,match="explicit host oracle"):
        owner()._trace_frontend_capture((Buffer((17,32)),Buffer((32,11))),{},require_outputs=True)


@pytest.mark.parametrize("dtype",[np.float16,np.float32])
def test_primitive_interface_dtype_is_authoritative(dtype):
    value=Buffer((3,7),dtype)
    value.dtype=object()
    assert cuda_frontend_specs((value,))==(((3,7),np.dtype(dtype)),)

def test_bf16_resident_abstract_trace_uses_canonical_storage():
    import ml_dtypes
    fn=ts.from_text(SOURCE.replace('"fp16"','"bf16"'),target="nvidia_sm120")
    a,b=Buffer((17,32),ml_dtypes.bfloat16),Buffer((32,11),ml_dtypes.bfloat16)
    graph,_=fn._trace_frontend_capture((a,b),{})
    assert all(arg.ir_type.dtype=="bf16" for arg in graph.functions[0].args)
    assert all(op.inferred_type.dtype in {"bf16","fp32"} for op in graph.functions[0].body)


def test_opaque_storage_without_bf16_hint_is_not_interpreted():
    value=Buffer((17,32),np.dtype("V2"))
    with pytest.raises(ValueError,match="dtype is malformed"):
        cuda_frontend_specs((value,))
