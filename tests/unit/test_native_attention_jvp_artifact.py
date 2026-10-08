"""Compiler-free portable attention JVP integrity and role validation."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pytest
from tessera.compiler.native_attention_program import NativeAttentionJVPProgram
from tessera.compiler.native_attention_jvp_artifact import canonical

FIXTURE=Path(__file__).resolve().parents[2]/"benchmarks/baselines/nvidia_jvp_portable_20261006/artifacts/qkv_q_5_0.program.json"

@pytest.fixture
def stored():
    return json.loads(FIXTURE.read_text())

def test_portable_program_roundtrip_preserves_all_native_components(stored):
    program=NativeAttentionJVPProgram.from_json(canonical(stored),expected_digest=stored["program_digest"])
    assert json.loads(program.to_json())==stored
    assert program.program_digest==stored["program_digest"]
    assert program.input_indices==(0,1,2) and program.active==(0,)
    assert program.input_names==("q","k","v")

@pytest.mark.parametrize("field",["schema","active","frontend_argument_indices","frontend_parameter_names","checkpoint_digest"])
def test_external_pin_refuses_changed_program(stored,field):
    original=stored["program_digest"]
    stored["program"][field]="corrupt"
    with pytest.raises(ValueError,match="pinned identity"):
        NativeAttentionJVPProgram.from_json(canonical(stored),expected_digest=original)

@pytest.mark.parametrize("field,value,message",[
    ("active",[1],"activity"),("active",[False],"unique physical"),
    ("frontend_argument_indices",[2,1,0],"mapping"),
    ("frontend_argument_indices",[0,0,2],"mapping"),
    ("frontend_parameter_names",["q","q","v"],"parameter names"),
    ("frontend_parameter_names",["q","k","bad-name"],"parameter names"),
])
def test_native_role_checks_survive_a_recomputed_container_digest(stored,field,value,message):
    stored["program"][field]=value
    digest=hashlib.sha256(canonical(stored["program"]).encode()).hexdigest()
    stored["program_digest"]=digest
    with pytest.raises(ValueError,match=message):
        NativeAttentionJVPProgram.from_json(canonical(stored),expected_digest=digest)

@pytest.mark.parametrize("change", [{"active":(1,)},{"input_indices":(2,1,0)}])
def test_capture_validates_program_before_any_cuda_allocation(stored,monkeypatch,change):
    program=NativeAttentionJVPProgram.from_json(canonical(stored),expected_digest=stored["program_digest"])
    def forbidden(*a,**k): raise AssertionError("allocated before validating native program")
    monkeypatch.setattr(type(program.pair),"capture",forbidden)
    with pytest.raises(ValueError,match="portable JVP"):
        replace(program,**change).capture("q","k","v")

def test_forward_product_v2_roundtrip_has_no_reverse_image(stored):
    from tessera.compiler.nvidia_native import AttentionForwardCheckpoint
    program=NativeAttentionJVPProgram.from_json(canonical(stored),expected_digest=stored["program_digest"])
    forward=replace(program,pair=AttentionForwardCheckpoint(program.pair.forward,program.pair.contract_digest))
    data=json.loads(forward.to_json())
    assert data["program"]["schema"]=="tessera.native_attention_jvp_program.v2"
    assert "backward" not in data["program"]
    restored=NativeAttentionJVPProgram.from_json(canonical(data),expected_digest=data["program_digest"])
    assert isinstance(restored.pair,AttentionForwardCheckpoint)
    assert restored.to_json()==forward.to_json()
    assert restored.pair.forward.image.payload==program.pair.forward.image.payload
    assert restored.tangent.image==program.tangent.image

def test_forward_schema_rejects_hidden_reverse_payload(stored):
    from tessera.compiler.nvidia_native import AttentionForwardCheckpoint
    program=NativeAttentionJVPProgram.from_json(canonical(stored),expected_digest=stored["program_digest"])
    forward=replace(program,pair=AttentionForwardCheckpoint(program.pair.forward,program.pair.contract_digest))
    data=json.loads(forward.to_json())
    data["program"]["backward"]=stored["program"]["backward"]
    pin=hashlib.sha256(canonical(data["program"]).encode()).hexdigest()
    data["program_digest"]=pin
    with pytest.raises(ValueError,match="schema"):
        NativeAttentionJVPProgram.from_json(canonical(data),expected_digest=pin)
