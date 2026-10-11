"""Resident reverse signatures and contract guards run without CUDA."""
import numpy as np
import pytest
from tessera.compiler import prepared_attention_vjp as module
from tessera.compiler.native_vjp_plugins import _value_signature
from tests.unit.test_ordered_resident_tensor_dag import Buffer
from tests.unit.test_prepared_attention_vjp import contract,values  # noqa: F401


def roots(owner):
    return tuple(Buffer(shape,np.float32) for shape in owner.shapes)


@pytest.mark.parametrize("index",range(4))
@pytest.mark.parametrize("field,value",[
    ("typestr","<f2"),("shape",(1,1,1,1)),("strides",(100,60,20,4)),
    ("stream",0),("data",(0,True)),("version",2),
])
def test_resident_reverse_contract_precedes_native_prepare(contract,index,field,value):
    owner=module.prepared(contract)
    inputs=list(roots(owner));inputs[index].interface[field]=value
    with pytest.raises(ValueError):owner.invoke(contract,inputs)


def test_mixed_cotangent_rejected_before_prepare(contract):
    owner=module.prepared(contract);inputs=list(roots(owner))
    inputs[-1]=values(owner)[-1]
    with pytest.raises(ValueError,match="all resident"):owner.invoke(contract,inputs)


def test_resident_reverse_signature_never_reads_tensor():
    value=Buffer((1,2,3,4),np.float32)
    assert _value_signature(value)=={"dtype":"float32","shape":[1,2,3,4]}
