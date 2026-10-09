"""Native two-sided producer DAG ownership and Schedule/Tile integration."""
import copy
import itertools
import json
import pytest

from tessera.compiler.native_sm120_tensor_program import (
    export_native_sm120_tensor_graph, validate_native_tensor_plan)
from tests.unit.test_native_sm120_tensor_partition import graph

AXES=[tuple(axis for axis,enabled in zip(("M","N","K"),bits,strict=True) if enabled)
      for bits in itertools.product((False,True),repeat=3)]

def dag_module(dtype="fp16", axes=(), rhs_first=False):
    module=graph(dtype=dtype)
    function=module.functions[0]
    lhs,consumer=function.body
    rhs=copy.deepcopy(lhs)
    rhs.op_name="tessera.softmax";rhs.kwargs={"axis":-1}
    rhs.operands=["%arg1"];rhs.result="rhs_edge"
    rhs.operand_types=[str(function.args[1].ir_type)]
    rhs.result_type=str(function.args[1].ir_type)
    consumer.operands[1]="%rhs_edge"
    function.body=[rhs,lhs,consumer] if rhs_first else [lhs,rhs,consumer]
    if axes:
        capacities={"M":32,"N":24,"K":64}
        module.module_attrs["tessera.native.sm120_tensor_bounds"]="{"+", ".join(
            axis+" = "+str(capacities[axis])+" : i64" for axis in axes)+"}"
    return module


def dag_source(dtype="fp16", axes=(), rhs_first=False):
    return dag_module(dtype,axes,rhs_first).to_mlir(target="nvidia_sm120",canonical=True)

@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("rhs_first",[False,True])
@pytest.mark.parametrize("axes",AXES)
def test_two_operand_native_dag_retains_shapes_and_lifetimes(dtype,rhs_first,axes):
    native=export_native_sm120_tensor_graph(dag_source(dtype,axes,rhs_first))
    plan=validate_native_tensor_plan(native.plan_json)
    assert plan["schema"].endswith(".v6" if axes else ".v5")
    assert plan["role_indices"]==[0,1]
    left,right=(3,2) if rhs_first else (2,3)
    assert plan["steps"][-1]["inputs"]==[left,right]
    assert plan["buffers"][left]["shape"]==[32 if "M" in axes else 17,64 if "K" in axes else 32]
    assert plan["buffers"][right]["shape"]==[64 if "K" in axes else 32,24 if "N" in axes else 11]
    assert plan["buffers"][left]["last_read"]==plan["buffers"][right]["last_read"]==2
    members=native.scheduled_members()
    assert len(members)==3
    assert members[0].input_shape!=members[1].input_shape
    assert tuple(members[0].input_shape)==tuple(plan["buffers"][2]["shape"])
    assert tuple(members[1].input_shape)==tuple(plan["buffers"][3]["shape"])
    assert (members[-1].m,members[-1].n,members[-1].k)==(
        32 if "M" in axes else 17,24 if "N" in axes else 11,64 if "K" in axes else 32)
    assert (members[-1].dynamic_m,members[-1].dynamic_n,members[-1].dynamic_k)==tuple(
        axis in axes for axis in ("M","N","K"))
    assert plan["buffers"][4]["ownership"]=="returned_output"

@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("mutation",["future","epilogue_root","swapped_roots","shape","lifetime"])
def test_two_operand_native_dag_rejects_forged_plan(mutation):
    plan=export_native_sm120_tensor_graph(dag_source()).manifest
    if mutation=="future":plan["steps"][0]["inputs"]=[3]
    elif mutation=="epilogue_root":plan["steps"][1]["inputs"]=[2]
    elif mutation=="swapped_roots":plan["steps"][-1]["inputs"]=[3,2]
    elif mutation=="shape":plan["buffers"][3]["shape"]=[17,32]
    else:plan["buffers"][2]["last_read"]=1
    with pytest.raises(ValueError):
        validate_native_tensor_plan(json.dumps(plan))
