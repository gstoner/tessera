"""Bounded resident specialization uses CUDA metadata without host coercion."""
from copy import deepcopy

import numpy as np
import pytest

from tessera.compiler.bounded_nvidia_lhs import specialization_key
from tests.unit.test_ordered_resident_tensor_dag import Buffer


ROLES={"source":0,"rhs":1,"bias":2,"residual":3}
BOUNDS={"M":33,"N":25,"K":65}


def roots(m=17,n=19,k=35):
    result=[Buffer(shape,dtype) for shape,dtype in (
        ((m,k),np.float16),((k,n),np.float16),((n,),np.float32),((m,n),np.float32))]
    return result


def test_key_reuses_bound_dimensions_and_matches_host_inputs():
    first=specialization_key(ROLES,roots(),BOUNDS)
    assert first==specialization_key(ROLES,roots(33,25,65),BOUNDS)
    assert first==specialization_key(ROLES,roots(1,1,1),BOUNDS)
    host=[np.zeros(shape,dtype) for shape,dtype in (
        ((17,35),np.float16),((35,19),np.float16),((19,),np.float32),((17,19),np.float32))]
    assert first==specialization_key(ROLES,host,BOUNDS)


@pytest.mark.parametrize("axis",["M","N","K"])
def test_unbounded_axis_is_a_real_specialization_dimension(axis):
    bounds={key:value for key,value in BOUNDS.items() if key!=axis}
    changed={"M":18,"N":20,"K":36}
    values={"m":17,"n":19,"k":35};values[axis.lower()]=changed[axis]
    assert specialization_key(ROLES,roots(),bounds)!=specialization_key(ROLES,roots(**values),bounds)


@pytest.mark.parametrize("bad",["m","n","k","contraction","bias","residual","rank","mixed","stride","stream"])
def test_invalid_active_frames_rejected_without_tools(bad):
    values=roots()
    if bad in {"m","n","k"}:
        config={"m":17,"n":19,"k":35};config[bad]={"m":34,"n":26,"k":66}[bad]
        values=roots(**config)
    elif bad=="contraction":values[1].interface["shape"]=(34,19)
    elif bad=="bias":values[2].interface["shape"]=(18,)
    elif bad=="residual":values[3].interface["shape"]=(16,19)
    elif bad=="rank":values[0].interface["shape"]=(35,)
    elif bad=="mixed":values[1]=np.zeros((35,19),np.float16)
    elif bad=="stride":values[1].interface["strides"]=(40,2)
    else:values[0].interface["stream"]=0
    with pytest.raises(ValueError):
        specialization_key(ROLES,values,BOUNDS)


def test_argument_reordering_preserves_compiled_role_projection():
    values=roots()
    key=specialization_key(ROLES,values,BOUNDS)
    permutation=[3,1,0,2]
    reordered=[values[i] for i in permutation]
    roles={role:permutation.index(index) for role,index in ROLES.items()}
    other=specialization_key(roles,reordered,BOUNDS)
    assert [(r,d,s) for r,_,d,s in key]==[(r,d,s) for r,_,d,s in other]
    assert all(index==roles[role] for role,index,_,_ in other)


def test_key_revalidates_changed_metadata_without_mutating_inputs():
    values=roots()
    saved=deepcopy(values[0].interface)
    specialization_key(ROLES,values,BOUNDS)
    assert values[0].interface==saved
    values[0].interface["shape"]=(34,35)
    with pytest.raises(ValueError,match="bound"):
        specialization_key(ROLES,values,BOUNDS)
