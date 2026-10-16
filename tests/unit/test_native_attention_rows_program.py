"""Native three-member program metadata, no compiler access during restoration."""
from dataclasses import replace
import copy
import hashlib
import json
import subprocess

import pytest

from tessera.compiler import nvidia_native as native
from tessera.compiler.native_attention_program import NativeAttentionVJPProgram, compile_attention_vjp_program
from tessera.compiler.native_attention_vjp_artifact import canonical
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests.unit.test_saved_lse_half_backward import source

pytestmark=pytest.mark.compiler_route


@pytest.fixture
def package(monkeypatch):
    monkeypatch.setattr(native,"_compile_tile_ir",lambda text,entry:(text,"// PTX",{},"compiler","tools",(),"cold"))
    def create(storage="f16",wide=False,bias=True,seeded=True,active=None,threads=128):
        active=active or tuple(range(3+int(bias)))
        text=source(storage,bias=bias,saved=seeded,output="f32" if wide else storage)
        text=text.replace('tessera.autodiff = "reverse"',
            'tessera.autodiff = "reverse", tessera.autodiff.wrt_indices = ['+', '.join(map(str,active))+']')
        tool=find_tessera_opt();assert tool is not None
        return compile_attention_vjp_program(text,active,compiler=tool,compact_gradients=True,
            compact_threads=threads,row_correction=True)
    return create


@pytest.mark.parametrize("storage",["f16","bf16"])
@pytest.mark.parametrize("wide",[False,True])
@pytest.mark.parametrize("bias,seeded",[(False,False),(False,True),(True,False),(True,True)])
def test_native_program_roundtrip_restores_all_members_without_compiler(package,monkeypatch,storage,wide,bias,seeded):
    program=package(storage,wide,bias,seeded)
    text=program.to_json();pin=program.program_digest
    assert json.loads(text)["program"]["schema"]=="tessera.native_attention_vjp_program.v3"
    assert len(program.members)==3
    def forbidden(*args,**kwargs):pytest.fail("portable row validation invoked compiler/subprocess")
    monkeypatch.setattr(subprocess,"run",forbidden)
    monkeypatch.setattr(subprocess,"check_output",forbidden)
    restored=NativeAttentionVJPProgram.from_json(text,expected_digest=pin)
    assert restored.to_json()==text and restored.program_digest==pin


@pytest.mark.parametrize("active",[(2,),(0,),(3,0)])
@pytest.mark.parametrize("threads",[64,128])
def test_requested_roles_and_value_only_member_elision_are_explicit(package,active,threads):
    program=package(active=active,threads=threads)
    assert program.active==active
    assert (program.correction is None)==(active==(2,))
    assert len(program.members)==(2 if active==(2,) else 3)
    program.validate()
    restored=NativeAttentionVJPProgram.from_json(program.to_json(),expected_digest=program.program_digest)
    assert restored.active==active


@pytest.mark.parametrize("member",["forward","backward","correction"])
def test_every_member_contributes_to_pinned_identity(package,member):
    program=package()
    data=json.loads(program.to_json())
    data["program"][member]["backend_ir"]+="changed"
    with pytest.raises(ValueError,match="pinned identity"):
        NativeAttentionVJPProgram.from_json(canonical(data),expected_digest=program.program_digest)


@pytest.mark.parametrize("member,field,value",[
    ("correction","lse_cotangent",False),("correction","scale",1.0),
    ("backward","physical_gradient_roles",[0,1,2]),
    ("backward","frontend_argument_indices",[1,0,2,3]),
    ("correction","source_ad_pair_digest","0"*64),
])
def test_local_provenance_tampering_is_rejected_even_after_envelope_repin(package,member,field,value):
    program=package();data=json.loads(program.to_json())
    data["program"][member]["descriptor"]["provenance"][field]=value
    pin=hashlib.sha256(canonical(data["program"]).encode()).hexdigest();data["program_digest"]=pin
    with pytest.raises(ValueError):NativeAttentionVJPProgram.from_json(canonical(data),expected_digest=pin)


@pytest.mark.parametrize("member",["backward","correction"])
def test_local_geometry_and_width_are_verified_against_native_contract(package,member):
    program=package();component=getattr(program.pair,member) if member=="backward" else program.correction
    buffers=list(component.descriptor.buffers)
    buffers[0]=replace(buffers[0],dtype="fp32")
    changed=replace(component,descriptor=replace(component.descriptor,buffers=tuple(buffers)))
    broken=replace(program,pair=replace(program.pair,backward=changed)) if member=="backward" else replace(program,correction=changed)
    with pytest.raises(ValueError):broken.validate()


def test_cross_program_member_swap_refuses_without_compiler(package):
    left=package(seeded=True);right=package(seeded=False)
    with pytest.raises(ValueError):replace(left,correction=right.correction).validate()


def test_v3_schema_cannot_be_relabelled_as_legacy(package):
    data=json.loads(package().to_json());data["program"]["schema"]="tessera.native_attention_vjp_program.v2"
    pin=hashlib.sha256(canonical(data["program"]).encode()).hexdigest();data["program_digest"]=pin
    with pytest.raises(ValueError,match="schema"):
        NativeAttentionVJPProgram.from_json(canonical(data),expected_digest=pin)
