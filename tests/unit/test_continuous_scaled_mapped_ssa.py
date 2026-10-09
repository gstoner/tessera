"""Mapped product chains retain each intermediate semantic shape."""
import copy
import json
import os

import numpy as np
import pytest
import tessera as ts
from tessera.autodiff import vmap
from tests.unit.test_continuous_scaled_ssa_chain import case as scalar_case,chained
from tessera.compiler.rocm_typed_scaled_native import supports_floating_scaled_primal,supports_scaled_reverse


def case(mask=127,mode=None,prefix=(2,),out_axes=0):
    _,_,scalar_values=scalar_case()
    roles=("a","b","sa","sb","c","sc","sd")
    options={} if mode is None else {"autodiff":mode,"wrt":roles}
    scalar=ts.jit(target="rocm_gfx1201",**options)(chained)
    axes=tuple(0 if mask&(1<<i) else None for i in range(7))
    owner=scalar
    for level in range(len(prefix)):
        owner=vmap(owner,in_axes=axes,out_axes=out_axes if level==len(prefix)-1 else 0)
    rng=np.random.default_rng(19071)
    values=[]
    for value,axis in zip(scalar_values,axes,strict=True):
        if axis is None:
            values.append(value.copy())
        else:
            full=np.broadcast_to(value,(*prefix,*value.shape)).copy()
            values.append(full*rng.uniform(.75,1.25,(*prefix,1,1)).astype(np.float32))
    return scalar,owner,tuple(values),axes


@pytest.mark.parametrize("mask",range(1,128))
@pytest.mark.parametrize("mode",[None,"forward","reverse"])
def test_each_root_map_policy_preserves_computed_operand_shapes(mask,mode):
    scalar,owner,values,axes=case(mask,mode)
    before=copy.deepcopy(scalar.graph_ir)
    graph=owner._specialized_autodiff_module(values,{})
    first,second=graph.functions[0].body
    expected_first=((2,) if any(axis is not None for axis in axes[:4]) else ())+(2,5)
    assert tuple(map(int,first.inferred_type.shape))==expected_first
    assert tuple(map(int,second.inferred_type.shape))==(2,2,3)
    assert supports_floating_scaled_primal(graph)
    assert supports_scaled_reverse(graph,tuple(range(7)))
    assert scalar.graph_ir==before


@pytest.mark.parametrize("mask",[1,16,64,17,127])
@pytest.mark.parametrize("mode",[None,"forward","reverse"])
@pytest.mark.parametrize("out_axes",[0,-1])
def test_nested_chain_certificate_preserves_roles_and_result_axes(mask,mode,out_axes):
    scalar,owner,values,_=case(mask,mode,(2,3),out_axes)
    before=copy.deepcopy(scalar.graph_ir)
    assert owner.frontend_differential(*values)
    graph=owner._specialized_autodiff_module(values,{})
    expected=(2,3,2,3) if out_axes==0 else (3,2,3,2)
    assert tuple(map(int,graph.functions[0].result_types[0].shape))==expected
    assert scalar.graph_ir==before


@pytest.mark.skipif(not os.environ.get("TESSERA_OPT"),reason="matching native compiler required")
@pytest.mark.parametrize("mask",[1,16,64,127])
@pytest.mark.parametrize("mode",[None,"forward","reverse"])
def test_mapped_chain_packages_native_dependency_storage(mask,mode):
    from tessera.compiler.native_scaled_program import (
        package_native_scaled_primal,package_native_scaled_jvp,package_native_scaled_vjp)
    _,owner,values,_=case(mask,mode,(2,3),-1)
    graph=owner._specialized_autodiff_module(values,{})
    graph.module_attrs.update({"tessera.target":'"rocm"',"tessera.arch":'"gfx1201"'})
    builder=package_native_scaled_vjp if mode=="reverse" else package_native_scaled_jvp if mode=="forward" else package_native_scaled_primal
    package=builder(graph.to_mlir(target="rocm_gfx1201",canonical=True))
    program=json.loads(package.program_json)
    if mode=="reverse":
        for role,slot in zip(program["gradient_roles"],program["outputs"],strict=True):
            assert program["buffers"][slot]["shape"]==list(values[role].shape)
    else:
        for slot in program["outputs"]:
            assert program["buffers"][slot]["shape"]==[3,2,3,2]
    assert any(any(slot>=program["argument_count"] for slot in step["inputs"])
               for step in program["steps"])
    package.validate()
