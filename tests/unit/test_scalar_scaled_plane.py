"""Scalar Graph types select bounded native planes without invented batching."""
import itertools,json
import pytest
from tests.unit.test_public_partial_scaled_groups import partial_case,oracle
from tessera.compiler.rocm_typed_scaled_native import contract,lower_typed_scaled,package_typed_scaled

def scalar_case(ta,tb,encoded,k,jvp=False,seed=1031):
    scalar,_,values,row=partial_case(1,ta,tb,encoded,k,jvp=jvp,seed=seed)
    values=[value[(0,)*(value.ndim-2)] for value in values]
    row={**row,"output_prefix":[],"prefixes":[[],[],[],[]]}
    return scalar,values,row

@pytest.mark.parametrize("k",[1,32,37,65])
@pytest.mark.parametrize("ta,tb,encoded",tuple(itertools.product((False,True),repeat=3)))
def test_scalar_semantic_projection(ta,tb,encoded,k):
    scalar,values,row=scalar_case(ta,tb,encoded,k)
    graph=scalar._specialized_autodiff_module(values,{})
    before=graph.to_mlir(target="rocm_gfx1201")
    shape,_,_,_=contract(graph)
    assert (shape.m,shape.n,shape.k,shape.groups)==(3,5,k,(k+31)//32)
    assert graph.functions[0].body[0].kwargs.get("batching") is None
    assert graph.to_mlir(target="rocm_gfx1201")==before

@pytest.mark.parametrize("k",[1,32,37,65])
@pytest.mark.parametrize("ta,tb,encoded",tuple(itertools.product((False,True),repeat=3)))
def test_native_scalar_plane_member(ta,tb,encoded,k):
    scalar,values,_=scalar_case(ta,tb,encoded,k)
    graph=scalar._specialized_autodiff_module(values,{})
    package=package_typed_scaled(graph,lower_typed_scaled(graph),pipeline_name="tessera-lower-to-rocm")
    manifest=package.descriptor.provenance["native_scaled_primal_program"]
    step=json.loads(manifest["program_json"])["steps"][0]
    member=json.loads(manifest["members_json"][0])
    assert member["scalars"]==[3,5,k]
    assert member["geometry"][2]==1
    assert package.descriptor.provenance["batching"] is None
    assert step.get("batching")==("broadcast" if ta or k%32 else None)


@pytest.mark.parametrize("k",[1,37])
@pytest.mark.parametrize("ta,tb",tuple(itertools.product((False,True),repeat=2)))
def test_native_scalar_scale_jvp(k,ta,tb):
    from dataclasses import replace
    from tessera.compiler.native_scaled_program import package_native_scaled_jvp
    scalar,values,_=scalar_case(ta,tb,False,k,jvp=True)
    graph=scalar._specialized_autodiff_module(values[:4],{})
    graph=replace(graph,module_attrs={**graph.module_attrs,
        "tessera.target":'"rocm"',"tessera.arch":'"gfx1201"'})
    package=package_native_scaled_jvp(graph.to_mlir(target="rocm_gfx1201"))
    steps=json.loads(package.program_json)["steps"]
    assert len(package.images)==4
    assert [step["operation"] for step in steps].count("tessera.scaled_matmul")==3
    assert all(json.loads(m)["scalars"]==[3,5,k] for m in package.members_json[:3])
