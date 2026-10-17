"""Bounded native row packages preserve symbolic roles and compiler-free pins."""
from dataclasses import replace
import hashlib
import json
import subprocess

import pytest
from tessera.compiler.native_attention_program import NativeAttentionVJPProgram
from tessera.compiler.native_attention_rows import validate_program
from tessera.compiler.native_attention_vjp_artifact import canonical
from tessera.compiler.attention_shape_contract import DYNAMIC_DIM
from tests.unit.test_native_attention_rows_program import package  # noqa: F401

pytestmark=pytest.mark.compiler_route


@pytest.mark.parametrize("storage",["f16","bf16"])
@pytest.mark.parametrize("wide",[False,True])
@pytest.mark.parametrize("bias,seeded",[(False,False),(False,True),(True,False),(True,True)])
def test_bounded_row_program_has_distinct_schema_and_compiler_free_replay(package,monkeypatch,storage,wide,bias,seeded):
    program=package(storage,wide,bias,seeded,bounded=True)
    text=program.to_json();pin=program.program_digest
    data=json.loads(text)["program"]
    assert data["schema"]=="tessera.native_attention_vjp_program.v4"
    p=program.pair.backward.descriptor.provenance
    assert p["shape"][3:5]==[DYNAMIC_DIM,DYNAMIC_DIM]
    assert p["shape_bounds"]==[1,2,1,9,11,4,3]
    assert program.correction.descriptor.provenance["row_contract"]["row_bytes"]==1*2*9*4
    def forbidden(*args,**kwargs):pytest.fail("bounded restoration invoked compiler/subprocess")
    monkeypatch.setattr(subprocess,"run",forbidden)
    monkeypatch.setattr(subprocess,"check_output",forbidden)
    restored=NativeAttentionVJPProgram.from_json(text,expected_digest=pin)
    assert restored.to_json()==text
    for sq,sk in ((1,1),(3,5),(9,11),(9,1)):
        dims,shapes=validate_program(restored,runtime_dims=(1,2,1,sq,sk,4,3))
        assert dims==(1,2,1,sq,sk,4,3)
        assert shapes==((1,2,sq,4),(1,1,sk,4),(1,1,sk,3),(1,2,sq,3),(1,2,sq))


@pytest.mark.parametrize("dims",[(1,2,1,10,11,4,3),(1,2,1,9,12,4,3),(2,2,1,3,5,4,3),(1,2,1,0,5,4,3)])
def test_runtime_dimensions_are_checked_against_bound_native_contract(package,dims):
    program=package(bounded=True)
    with pytest.raises(ValueError,match="envelope"):
        validate_program(program,runtime_dims=dims)


@pytest.mark.parametrize("member",["correction","backward"])
def test_bounded_member_guards_bind_query_and_key_axes(package,member):
    program=package(bounded=True)
    component=program.correction if member=="correction" else program.pair.backward
    wanted={(g.binding,g.dimension,g.predicate,g.value) for g in component.descriptor.shape_guards}
    for name,axis,cap in (("q",2,9),("k",2,11),("v",2,11),("lse",2,9)):
        assert (name,axis,"min",1) in wanted and (name,axis,"max",cap) in wanted
    assert ("bias",2,"eq",1) in wanted
    assert ("bias",3,"max",11) in wanted
    guards=list(component.descriptor.shape_guards)
    index=next(i for i,g in enumerate(guards) if g.binding=="k" and g.predicate=="max")
    guards[index]=replace(guards[index],value=9)
    altered=replace(component,descriptor=replace(component.descriptor,shape_guards=tuple(guards)))
    broken=replace(program,correction=altered) if member=="correction" else replace(program,pair=replace(program.pair,backward=altered))
    with pytest.raises(ValueError,match="descriptor"):
        broken.validate()


@pytest.mark.parametrize("active",[(2,),(0,),(3,0)])
def test_bounded_compact_roles_and_value_only_elision(package,active):
    program=package(active=active,bounded=True)
    assert (program.correction is None)==(active==(2,))
    assert json.loads(program.to_json())["program"]["schema"].endswith(".v4")


@pytest.mark.parametrize("bounded,schema",[(True,"v3"),(False,"v4")])
def test_static_and_bounded_row_schema_meanings_cannot_be_swapped(package,bounded,schema):
    envelope=json.loads(package(bounded=bounded).to_json())
    envelope["program"]["schema"]="tessera.native_attention_vjp_program."+schema
    pin=hashlib.sha256(canonical(envelope["program"]).encode()).hexdigest()
    envelope["program_digest"]=pin
    with pytest.raises(ValueError,match="schema"):
        NativeAttentionVJPProgram.from_json(canonical(envelope),expected_digest=pin)
