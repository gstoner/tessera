"""Host-free: compiler-owned reshape/carrier SSA and differentiation contracts."""
import copy
import json
import os

import numpy as np
import pytest
import tessera as ts
from tessera.compiler.rocm_typed_scaled_native import (
    supports_floating_scaled_primal, supports_floating_scaled_jvp, supports_scaled_reverse)

SOURCE = """def reshaped(a,b,sa,sb):
    aa = tessera.ops.reshape(a, shape=(3,7))
    bb = tessera.ops.reshape(b, shape=(7,5))
    left = tessera.ops.reshape(sa, shape=(3,2))
    right = tessera.ops.reshape(sb, shape=(2,3))
    y = tessera.ops.scaled_matmul(aa,bb,left,right,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[2,4],"format":"fp32"})
    return tessera.ops.reshape(y, shape=OUTPUT)
"""


def case(mode=None, output=(15,)):
    options={} if mode is None else {"autodiff":mode,"wrt":("a","b","sa","sb")}
    owner=ts.from_text(SOURCE.replace("OUTPUT",repr(output)),target="rocm_gfx1201",**options)
    rng=np.random.default_rng(9921)
    values=tuple(rng.uniform(-.4,.4,size).astype(np.float32) for size in (21,35,6,6))
    graph=owner._specialized_autodiff_module(values,{})
    return owner,graph,values


@pytest.mark.parametrize("output",[(15,),(5,3),(1,3,5)])
@pytest.mark.parametrize("mode",[None,"forward","reverse"])
def test_reshape_roots_and_outputs_are_admitted_without_graph_mutation(mode,output):
    _,graph,_=case(mode,output)
    before=copy.deepcopy(graph)
    assert supports_floating_scaled_primal(graph)
    assert supports_floating_scaled_jvp(graph,(0,1,2,3))
    assert supports_scaled_reverse(graph,(0,1,2,3))
    assert graph==before


@pytest.mark.skipif(not os.environ.get("TESSERA_OPT"),reason="matching compiler required")
@pytest.mark.parametrize("output",[(15,),(5,3),(1,3,5)])
@pytest.mark.parametrize("mode",[None,"forward","reverse"])
def test_native_reshape_members_preserve_bytes_and_returned_roles(mode,output):
    from tessera.compiler.native_scaled_program import (
        package_native_scaled_primal,package_native_scaled_jvp,package_native_scaled_vjp)
    _,graph,values=case(mode,output)
    graph.module_attrs.update({"tessera.target":'"rocm"',"tessera.arch":'"gfx1201"'})
    builder=package_native_scaled_vjp if mode=="reverse" else package_native_scaled_jvp if mode=="forward" else package_native_scaled_primal
    package=builder(graph.to_mlir(target="rocm_gfx1201",canonical=True))
    program=json.loads(package.program_json)
    carriers=[step for step in program["steps"] if step["operation"]=="tessera.reshape"]
    assert carriers
    for step in carriers:
        source=program["buffers"][step["inputs"][0]]
        result=program["buffers"][step["output"]]
        assert source["bytes"]==result["bytes"]
        assert step["lowering"]=="structured_f32_carrier"
    expected=[list(value.shape) for value in values] if mode=="reverse" else [list(output)]*(2 if mode=="forward" else 1)
    assert [program["buffers"][slot]["shape"] for slot in program["outputs"]]==expected
    package.validate()


@pytest.mark.parametrize("mutation",["shape","policy","dtype","forward_reference"])
def test_reshape_contract_rejects_changed_semantics(mutation):
    from tessera.compiler.graph_ir import tensor_ir_type
    _,graph,_=case()
    reshape=graph.functions[0].body[0]
    if mutation=="shape":reshape.kwargs["shape"]=(3,8)
    elif mutation=="policy":reshape.numeric_policy={"fastmath":True}
    elif mutation=="dtype":graph.functions[0].args[0].ir_type=tensor_ir_type((21,),"fp16")
    else:reshape.operands=["%unwritten"]
    assert not supports_floating_scaled_primal(graph)


@pytest.fixture(scope="module")
def native_reshape_package():
    if not os.environ.get("TESSERA_OPT"):pytest.skip("matching compiler required")
    from tessera.compiler.native_scaled_program import package_native_scaled_primal
    _,graph,_=case()
    graph.module_attrs.update({"tessera.target":'"rocm"',"tessera.arch":'"gfx1201"'})
    return package_native_scaled_primal(graph.to_mlir(target="rocm_gfx1201",canonical=True))


@pytest.mark.parametrize("mutation",["count","algorithm","geometry"])
def test_reshape_package_rejects_forged_physical_binding(native_reshape_package,mutation):
    from dataclasses import replace
    package=native_reshape_package
    members=[json.loads(raw) for raw in package.members_json]
    if mutation=="count":members[0]["scalars"][0]+=1
    elif mutation=="algorithm":members[0]["scale_adjoint_schedule"]="serial_per_scale_element"
    else:members[0]["geometry"][3]=256
    forged=replace(package,members_json=tuple(json.dumps(member) for member in members))
    with pytest.raises(ValueError,match="native scaled carrier"):forged.validate()
