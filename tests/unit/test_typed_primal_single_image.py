"""A typed primal descriptor must describe the image its native owner executes."""
import json
import pytest
from tests.unit.test_native_typed_scaled_vmap import case
from tessera.compiler.rocm_typed_scaled_native import lower_typed_scaled, package_typed_scaled

@pytest.mark.parametrize("policy",[None,"shared_rhs_rows","independent_rhs","shared_lhs"])
@pytest.mark.parametrize("fmt",["fp32","e8m0"])
@pytest.mark.parametrize("nk",[False,True])
def test_descriptor_binds_actual_native_member(policy,fmt,nk,monkeypatch):
    monkeypatch.setenv("TESSERA_ROCM_CHIP","gfx1201")
    scalar,owner,values,_=case(policy or "independent_rhs",fmt,nk)
    if policy is None:
        owner=scalar
        values=tuple(value[0] for value in values)
    graph=owner._specialized_autodiff_module(values,{})
    program=lower_typed_scaled(graph)
    package=package_typed_scaled(graph,program,pipeline_name="tessera-lower-to-rocm")
    manifest=package.descriptor.provenance["native_scaled_primal_program"]
    from tessera.compiler.native_scaled_program import NativeScaledProgram
    native=NativeScaledProgram.from_manifest(manifest)
    member=json.loads(native.members_json[0])
    assert len(native.images)==1
    assert package.image.payload==native.images[0]
    assert package.descriptor.entry_symbol==member["entry"]
    assert package.descriptor.geometry.grid==tuple(member["geometry"][:3])
    assert package.descriptor.geometry.workgroup==tuple(member["geometry"][3:])
    assert package.descriptor.provenance["shape"]==member["scalars"]
    assert package.descriptor.provenance["batching"]==policy
    assert [binding.name for binding in package.descriptor.buffers[:4]]==[
        name.removeprefix("%") for name in graph.functions[0].body[0].operands]
