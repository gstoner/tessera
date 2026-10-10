"""Public continuous scaled primal/JVP retain native Graph and map intent."""
import copy
import itertools

import numpy as np
import pytest
import tessera as ts
from tessera.autodiff import vmap
from tessera.compiler.jit import JitFn
from tessera.compiler.rocm_typed_scaled_native import (
    supports_floating_scaled_primal, supports_floating_scaled_jvp)
from tests.unit.test_public_floating_scaled_reverse import floating_python_source
from tests.device.rocm.test_floating_scaled_adjoint import inputs


def case(ta=False, tb=False, mode=None, mask=15, prefix=(2,), out_axes=0,
         roles=("a", "b", "sa", "sb")):
    options = {} if mode is None else {"autodiff": mode, "wrt": roles}
    scalar = ts.from_text(floating_python_source(ta, tb), target="rocm_gfx1201", **options)
    axes = tuple(0 if mask & (1 << slot) else None for slot in range(4))
    owner = scalar
    for level in range(len(prefix)):
        owner = vmap(owner, in_axes=axes, out_axes=out_axes if level == len(prefix)-1 else 0)
    rng = np.random.default_rng(940)
    suffix = ((9,2) if ta else (2,9), (5,9) if tb else (9,5), (2,3), (3,2))
    values = tuple(rng.uniform(.25, 1.25, (*(prefix if axis is not None else ()), *shape)).astype(np.float32)
                   if slot in (2,3) else
                   rng.uniform(-.5, .5, (*(prefix if axis is not None else ()), *shape)).astype(np.float32)
                   for slot, (axis, shape) in enumerate(zip(axes, suffix, strict=True)))
    directions = tuple(rng.uniform(-.5, .5, value.shape).astype(np.float32) for value in values)
    return scalar, owner, values, directions


@pytest.mark.parametrize("ta,tb", tuple(itertools.product((False, True), repeat=2)))
@pytest.mark.parametrize("mode", [None, "forward"])
@pytest.mark.parametrize("mask", range(1,16))
@pytest.mark.parametrize("prefix", [(2,), (2,3)])
@pytest.mark.parametrize("out_axes", [0, -1])
def test_continuous_public_maps_have_native_primal_and_jvp_admission(ta,tb,mode,mask,prefix,out_axes):
    scalar, owner, values, _ = case(ta,tb,mode,mask,prefix,out_axes)
    before = copy.deepcopy(scalar.graph_ir)
    assert isinstance(owner, JitFn) and owner is not scalar
    graph = owner._specialized_autodiff_module(values,{})
    assert supports_floating_scaled_primal(graph)
    assert supports_floating_scaled_jvp(graph, (0,1,2,3))
    assert owner.frontend_differential(*values) is owner.frontend_differential(*values)
    assert scalar.graph_ir == before


@pytest.mark.parametrize("ta,tb", tuple(itertools.product((False, True), repeat=2)))
@pytest.mark.parametrize("mode", [None, "forward"])
def test_direct_continuous_admission_is_separate_from_fp8_wmma(ta,tb,mode):
    scalar, owner, _, _ = case(ta,tb,mode,prefix=())
    values = inputs(ta,tb)[:4]
    assert scalar is owner
    graph = owner._specialized_autodiff_module(values,{})
    assert supports_floating_scaled_primal(graph)
    assert supports_floating_scaled_jvp(graph,(0,1,2,3))


@pytest.mark.parametrize("field,value", [
    ("numeric_policy", {"accum":"fp32","execution_mode":"approximate"}),
    ("scale_layout", {"granularity":"per_tensor","block":[4,4],"format":"fp32"}),
    ("scale_layout", {"granularity":"block","block":[4,4],"format":"e8m0"}),
    ("transposeA", 1),
    ("physical_contract", "nvidia_sm120_nvfp4_blockscale_v1"),
])
def test_continuous_public_primal_rejects_conflicting_semantics(field,value):
    _, owner, values, _ = case()
    graph = copy.deepcopy(owner._specialized_autodiff_module(values,{}))
    graph.functions[0].body[0].kwargs[field] = value
    assert not supports_floating_scaled_primal(graph)
    assert not supports_floating_scaled_jvp(graph,(0,1,2,3))


@pytest.mark.parametrize("roles", [(True,), (4,), (0,0), ()])
def test_continuous_public_jvp_requires_explicit_distinct_roles(roles):
    _, owner, values, _ = case()
    graph = owner._specialized_autodiff_module(values,{})
    assert not supports_floating_scaled_jvp(graph,roles)


@pytest.mark.parametrize("ta,tb", tuple(itertools.product((False, True), repeat=2)))
@pytest.mark.parametrize("mode", ["forward", "reverse"])
def test_ordinary_primal_projection_preserves_the_ad_owner_and_arithmetic(ta,tb,mode):
    from tessera.compiler.rocm_typed_scaled_native import primal_call_module
    _,owner,values,_ = case(ta,tb,mode,prefix=())
    graph = owner._traced_autodiff_module(values,{})
    before = copy.deepcopy(graph)
    projected = primal_call_module(graph)
    assert graph == before and projected is not graph
    assert projected.functions[0].body == graph.functions[0].body
    assert projected.functions[0].args == graph.functions[0].args
    assert "tessera.autodiff" not in projected.module_attrs
    assert "tessera.autodiff" not in projected.functions[0].fn_attrs
    assert "tessera.primal_call.requested_autodiff" in projected.functions[0].fn_attrs
    assert "tessera.autodiff" in graph.functions[0].fn_attrs
    assert supports_floating_scaled_primal(projected)


def mixed_case(ta=False,tb=False,mode=None,nested=False):
    options = {} if mode is None else {"autodiff":mode,"wrt":("a","b","sa","sb")}
    scalar = ts.from_text(floating_python_source(ta,tb),target="rocm_gfx1201",**options)
    rng = np.random.default_rng(942)
    suffix = ((9,2) if ta else (2,9),(5,9) if tb else (9,5),(2,3),(3,2))
    prefixes = ((2,3),(2,),(3,),(2,)) if nested else ((2,),(2,),(),(2,))
    canonical = tuple(rng.uniform(.25,1.25,(*prefix,*shape)).astype(np.float32)
                      if slot in (2,3) else rng.uniform(-.5,.5,(*prefix,*shape)).astype(np.float32)
                      for slot,(prefix,shape) in enumerate(zip(prefixes,suffix,strict=True)))
    directions = tuple(rng.uniform(-.5,.5,value.shape).astype(np.float32) for value in canonical)
    def raw(values):
        return ((values[0],np.moveaxis(values[1],0,1),values[2],np.moveaxis(values[3],0,-1))
                if nested else
                (np.moveaxis(values[0],0,1),values[1],values[2],np.moveaxis(values[3],0,-1)))
    if nested:
        owner = vmap(vmap(scalar,in_axes=(0,None,0,None),out_axes=1),
                     in_axes=(0,1,None,-1),out_axes=-1)
        permutation = (2,1,3,0)
    else:
        owner = vmap(scalar,in_axes=(1,0,None,-1),out_axes=1)
        permutation = (1,0,2)
    return scalar,owner,raw(canonical),raw(directions),canonical,directions,permutation


@pytest.mark.parametrize("ta,tb", tuple(itertools.product((False,True),repeat=2)))
@pytest.mark.parametrize("mode",[None,"forward"])
@pytest.mark.parametrize("nested",[False,True])
def test_continuous_mixed_axes_use_the_same_native_graph_contract(ta,tb,mode,nested):
    scalar,owner,values,_,_,_,permutation = mixed_case(ta,tb,mode,nested)
    before = copy.deepcopy(scalar.graph_ir)
    graph = owner._specialized_autodiff_module(values,{})
    assert supports_floating_scaled_primal(graph)
    assert supports_floating_scaled_jvp(graph,(0,1,2,3))
    assert owner._frontend_output_permutation == permutation
    assert owner.frontend_differential(*values)
    assert scalar.graph_ir == before
