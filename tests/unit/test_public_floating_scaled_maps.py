"""Public continuous maps retain typed native reverse intent."""
import copy
import itertools
import numpy as np
import pytest
from tessera.autodiff import vmap
from tessera.compiler.jit import JitFn
from tessera.compiler.rocm_typed_scaled_native import supports_scaled_reverse, supports_typed_scaled
from tests.unit.test_public_floating_scaled_reverse import floating_owner

def mapped_case(mask=15, ta=False, tb=False, prefix=(2,3), out_axes=0,
                roles=("a","b","sa","sb")):
    scalar = floating_owner(ta, tb, roles)
    axes = tuple(0 if mask & (1 << i) else None for i in range(4))
    owner = scalar
    for level in range(len(prefix)):
        owner = vmap(owner, in_axes=axes,
                     out_axes=out_axes if level == len(prefix)-1 else 0)
    rng = np.random.default_rng(934)
    suffix = ((9,2) if ta else (2,9), (5,9) if tb else (9,5), (2,3), (3,2))
    values = tuple(rng.uniform(.25,1.25,(*(prefix if axis is not None else ()),*shape)).astype(np.float32)
                   if role in (2,3) else
                   rng.uniform(-.5,.5,(*(prefix if axis is not None else ()),*shape)).astype(np.float32)
                   for role,(axis,shape) in enumerate(zip(axes,suffix,strict=True)))
    seed = rng.uniform(-.5,.5,(*prefix,2,5)).astype(np.float32)
    return scalar,owner,values,seed

@pytest.mark.parametrize("mask", range(1,16))
@pytest.mark.parametrize("ta,tb", tuple(itertools.product((False,True),repeat=2)))
@pytest.mark.parametrize("prefix", [(2,), (2,3)])
@pytest.mark.parametrize("out_axes", [0,-1])
def test_floating_map_projects_native_reverse_and_certificate(mask,ta,tb,prefix,out_axes):
    scalar,owner,values,seed = mapped_case(mask,ta,tb,prefix,out_axes)
    assert isinstance(owner,JitFn) and owner is not scalar
    before = copy.deepcopy(scalar.graph_ir)
    graph = owner._specialized_autodiff_module(values,{})
    assert supports_scaled_reverse(graph, owner.differentiation_request.wrt_indices)
    assert not supports_typed_scaled(graph)
    permutation = owner._frontend_output_permutation
    assert graph.functions[0].result_types[0].shape == tuple(str(seed.shape[i]) for i in permutation)
    assert owner.frontend_differential(*values) is owner.frontend_differential(*values)
    assert scalar.graph_ir == before and scalar._frontend_batch_axes is None
    assert owner.differentiation_request is not scalar.differentiation_request

def test_continuous_map_admits_native_primal_and_rejects_encoded_scale():
    import tessera as ts
    from tessera.compiler.rocm_typed_scaled_native import supports_floating_scaled_primal
    from tests.unit.test_public_floating_scaled_reverse import floating_python_source
    from tests.device.rocm.test_floating_scaled_adjoint import inputs
    scalar = ts.from_text(floating_python_source(), target="rocm_gfx1201")
    owner = vmap(scalar)
    assert isinstance(owner, JitFn) and owner is not scalar
    values = tuple(np.stack((value,value)) for value in inputs(False,False)[:4])
    graph = owner._specialized_autodiff_module(values,{})
    assert supports_floating_scaled_primal(graph)
    graph = copy.deepcopy(graph)
    graph.functions[0].body[0].kwargs["scale_layout"]["format"] = "e8m0"
    assert not supports_floating_scaled_primal(graph)



def mixed_case(ta=False,tb=False,nested=False,roles=("a","b","sa","sb")):
    from tests.device.rocm.test_floating_scaled_adjoint import oracle
    rng=np.random.default_rng(935)
    shapes=((9,2) if ta else (2,9),(5,9) if tb else (9,5),(2,3),(3,2))
    prefixes=((2,3),(2,),(3,),(2,)) if nested else ((2,),(2,),(),(2,))
    canonical=tuple(rng.uniform(.25,1.25,(*prefix,*shape)).astype(np.float32)
                    if role in (2,3) else rng.uniform(-.5,.5,(*prefix,*shape)).astype(np.float32)
                    for role,(prefix,shape) in enumerate(zip(prefixes,shapes,strict=True)))
    seed=rng.uniform(-.5,.5,(*((2,3) if nested else (2,)),2,5)).astype(np.float32)
    expected=[np.zeros_like(value,dtype=np.float64) for value in canonical]
    for plane in np.ndindex(seed.shape[:-2]):
        coords=(plane,(plane[0],),(plane[1],),(plane[0],)) if nested else (plane,plane,(),plane)
        selected=tuple(value[coord] for value,coord in zip(canonical,coords,strict=True))
        gradients=oracle((*selected,seed[plane]),ta,tb)
        for result,coord,gradient in zip(expected,coords,gradients,strict=True):
            result[coord]+=gradient
    scalar=floating_owner(ta,tb,roles)
    if nested:
        owner=vmap(vmap(scalar,in_axes=(0,None,0,None),out_axes=1),
                   in_axes=(0,1,None,-1),out_axes=-1)
        raw=(canonical[0],np.moveaxis(canonical[1],0,1),canonical[2],np.moveaxis(canonical[3],0,-1))
        expected=(expected[0],np.moveaxis(expected[1],0,1),expected[2],np.moveaxis(expected[3],0,-1))
    else:
        owner=vmap(scalar,in_axes=(1,0,None,-1),out_axes=1)
        raw=(np.moveaxis(canonical[0],0,1),canonical[1],canonical[2],np.moveaxis(canonical[3],0,-1))
        expected=(np.moveaxis(expected[0],0,1),expected[1],expected[2],np.moveaxis(expected[3],0,-1))
    return scalar,owner,raw,seed,tuple(expected)


@pytest.mark.parametrize("ta,tb", tuple(itertools.product((False,True),repeat=2)))
@pytest.mark.parametrize("nested",[False,True])
def test_mixed_floating_input_axes_preserve_native_graph_and_certificate(ta,tb,nested):
    _,owner,raw,seed,_=mixed_case(ta,tb,nested)
    assert isinstance(owner,JitFn)
    graph=owner._specialized_autodiff_module(raw,{})
    assert supports_scaled_reverse(graph,owner.differentiation_request.wrt_indices)
    assert graph.functions[0].result_types[0].shape==tuple(str(seed.shape[i]) for i in owner._frontend_output_permutation)
    assert owner.frontend_differential(*raw)


@pytest.mark.parametrize("field,value",[
    ("numeric_policy",{"accum":"fp32","execution_mode":"approximate"}),
    ("scale_layout",{"granularity":"per_tensor","block":[4,4],"format":"fp32"}),
    ("scale_layout",{"granularity":"block","block":[4,4],"format":"e8m0"}),
    ("transposeA",1),
    ("physical_contract","nvidia_sm120_nvfp4_blockscale_v1"),
])
def test_continuous_reverse_rejects_conflicting_semantic_contracts(field,value):
    _,owner,values,_=mapped_case()
    graph=copy.deepcopy(owner._specialized_autodiff_module(values,{}))
    graph.functions[0].body[0].kwargs[field]=value
    assert not supports_scaled_reverse(graph,owner.differentiation_request.wrt_indices)



@pytest.mark.parametrize("roles",[(True,),(4,),(0,0)])
def test_continuous_scalar_reverse_roles_are_checked(roles):
    from tests.device.rocm.test_floating_scaled_adjoint import inputs
    owner=floating_owner()
    graph=owner._specialized_autodiff_module(inputs(False,False)[:4],{})
    assert not supports_scaled_reverse(graph,roles)
