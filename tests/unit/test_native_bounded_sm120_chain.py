"""Host-free compiler contract tests preserve every bounded-chain SSA member.

These tests run the native MLIR compiler and validate manifests and IR; they do
not execute CUDA kernels. Exact-device execution lives in tests/device/nvidia.
"""
import copy
import itertools
import json

import pytest

from tessera.compiler.native_sm120_tensor_program import (
    export_native_sm120_tensor_graph, validate_native_tensor_plan)
from tests.unit.test_native_sm120_tensor_partition import graph

AXES = [tuple(axis for axis, bit in zip(("M","N","K"), bits, strict=True) if bit)
        for bits in itertools.product((False,True),repeat=3) if any(bits)]


def chain_graph(dtype="fp16", producers=2):
    module = graph(dtype=dtype)
    function = module.functions[0]
    first, consumer = function.body
    if producers == 3:
        first.op_name = "tessera.layer_norm"
    members = [first]
    for name in (("tessera.softmax",) if producers == 2 else ("tessera.rmsnorm","tessera.softmax")):
        member = copy.deepcopy(first)
        member.op_name = name
        member.kwargs = {"axis":-1} if name == "tessera.softmax" else {"eps":1e-5}
        member.operands = ["%"+members[-1].result]
        member.result = "chain_"+str(len(members))
        members.append(member)
    consumer.operands[0] = "%"+members[-1].result
    function.body = [*members,consumer]
    return module


def bounded_source(dtype="fp16", producers=2, axes=("M","N","K")):
    module = chain_graph(dtype,producers)
    capacity = {"M":32,"N":24,"K":64}
    module.module_attrs["tessera.native.sm120_tensor_bounds"] = "{" + ", ".join(
        axis+" = "+str(capacity[axis])+" : i64" for axis in axes) + "}"
    return module.to_mlir(target="nvidia_sm120",canonical=True)


@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("producers",[2,3])
@pytest.mark.parametrize("axes",AXES)
def test_native_bounded_chain_projects_every_result_and_capacity(dtype,producers,axes):
    source = bounded_source(dtype,producers,axes)
    native = export_native_sm120_tensor_graph(source)
    plan = validate_native_tensor_plan(native.plan_json)
    assert plan["schema"] == "tessera.native.sm120_tensor_program.v4"
    assert plan["dynamic_axes"] == list(axes)
    assert plan["active_shape"] == [17,11,32]
    assert plan["shape_bounds"] == [32 if "M" in axes else 17,
                                  24 if "N" in axes else 11,
                                  64 if "K" in axes else 32]
    assert plan["original_graph_ir"].count("tessera.native.sm120_tensor_bounds") == 1
    assert "tessera.native.sm120_tensor_bounds" not in plan["source_graph_ir"]
    assert export_native_sm120_tensor_graph(plan["original_graph_ir"]).manifest == plan
    assert len(plan["steps"]) == producers+1
    for index in range(producers):
        buffer = plan["buffers"][2+index]
        assert buffer["first_write"] == index and buffer["last_read"] == index+1
        assert buffer["shape"] == [plan["shape_bounds"][0],plan["shape_bounds"][2]]
    members = native.scheduled_members()
    assert len(members) == producers+1
    assert all(member.input_shape == tuple(plan["buffers"][0]["shape"]) for member in members[:-1])
    consumer = members[-1]
    assert (consumer.dynamic_m,consumer.dynamic_n,consumer.dynamic_k) == tuple(axis in axes for axis in ("M","N","K"))
    assert consumer.a_name == "edge"


@pytest.mark.compiler_route
@pytest.mark.usefixtures("production_compiler")
@pytest.mark.parametrize("mutation",["downgrade","axis","capacity","lifetime","lineage"])
def test_bounded_chain_replay_rejects_contract_corruption(mutation):
    native = export_native_sm120_tensor_graph(bounded_source(producers=3))
    plan = native.manifest
    if mutation == "downgrade":
        plan["schema"] = "tessera.native.sm120_tensor_program.v2"
    elif mutation == "axis":
        plan["dynamic_axes"] = ["M","M"]
    elif mutation == "capacity":
        plan["shape_bounds"][2] += 1
    elif mutation == "lifetime":
        plan["buffers"][2]["last_read"] += 1
    else:
        plan["steps"][2]["inputs"] = [2]
    with pytest.raises(ValueError):
        validate_native_tensor_plan(json.dumps(plan))
