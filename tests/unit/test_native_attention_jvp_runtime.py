"""Pinned attention product launch guards run before CUDA allocation."""
import json
from pathlib import Path
import numpy as np
import pytest
from tessera.compiler.native_attention_jvp_runtime import execute

FIXTURE=Path(__file__).resolve().parents[2]/"benchmarks/baselines/nvidia_jvp_portable_20261006/artifacts/qkv_q_5_0.program.json"

@pytest.fixture
def contract(monkeypatch):
    raw=FIXTURE.read_text();program=json.loads(raw)
    def forbidden(*a,**k):
        raise AssertionError("CUDA allocation before host contract validation")
    monkeypatch.setattr("tessera.compiler.native_attention_jvp_runtime.PreparedAttentionJVP._prepare",forbidden)
    return dict(program_json=raw,program_digest=program["program_digest"],
                arg_names=["primal_0","primal_1","primal_2","tangent_0"])

def inputs():
    return tuple(np.zeros(s,np.float32) for s in
        ((1,2,3,4),(1,1,5,4),(1,1,5,3),(1,2,3,4)))

def test_runtime_refuses_changed_pin_before_cuda(contract):
    contract["program_digest"]="0"*64
    with pytest.raises(ValueError,match="pinned identity"):
        execute(contract,inputs())

def test_runtime_refuses_launch_name_activity_mismatch(contract):
    contract["arg_names"][-1]="tangent_1"
    with pytest.raises(ValueError,match="launch names"):
        execute(contract,inputs())

def test_runtime_refuses_missing_tangent(contract):
    with pytest.raises(ValueError,match="arity"):
        execute(contract,inputs()[:3])

@pytest.mark.parametrize("index",range(4))
def test_runtime_refuses_wrong_dtype_before_cuda(contract,index):
    values=list(inputs());values[index]=values[index].astype(np.float16)
    with pytest.raises(ValueError,match="storage"):
        execute(contract,values)

@pytest.mark.parametrize("index",range(4))
def test_runtime_refuses_wrong_shape_before_cuda(contract,index):
    values=list(inputs());values[index]=values[index].reshape(-1)
    with pytest.raises(ValueError,match="storage"):
        execute(contract,values)

def test_retained_owner_refuses_changed_launch_identity(contract):
    from tessera.compiler.native_attention_jvp_runtime import prepared
    owner=prepared(contract)
    changed=dict(contract,program_digest="1"*64)
    with pytest.raises(ValueError,match="pinned identity"):
        owner.invoke(changed,inputs())

def test_closed_owner_refuses_call_before_native_setup(contract):
    from tessera.compiler.native_attention_jvp_runtime import prepared
    owner=prepared(contract);owner.close()
    with pytest.raises(ValueError,match="closed"):
        owner.invoke(contract,inputs())

def test_bounded_registration_retires_old_owner(contract,monkeypatch):
    from tessera.compiler import native_attention_jvp_runtime as module
    module.clear_prepared();monkeypatch.setattr(module,"_CACHE_LIMIT",1)
    old=module.prepared(contract)
    alternate=FIXTURE.with_name("qkv_v_5_0.program.json").read_text()
    data=json.loads(alternate)
    other=dict(program_json=alternate,program_digest=data["program_digest"],
               arg_names=["primal_0","primal_1","primal_2","tangent_2"])
    new=module.prepared(other)
    assert old.closed and not new.closed and len(module._cache)==1
    module.clear_prepared()

def test_registration_is_not_shared_across_different_thread_lifetimes(contract):
    import threading
    from tessera.compiler import native_attention_jvp_runtime as module
    module.clear_prepared()
    owners=[]
    def register():owners.append(module.prepared(contract))
    first=threading.Thread(target=register);first.start();first.join()
    second=threading.Thread(target=register);second.start();second.join()
    assert owners[0] is not owners[1]
    module.clear_prepared()
