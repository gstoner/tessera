"""Prepared reverse registration, pin and lifetime gates before CUDA setup."""
import json
from pathlib import Path
import threading
import numpy as np
import pytest
from tessera.compiler import prepared_attention_vjp as module

ROOT=Path(__file__).resolve().parents[2]
ARTIFACTS=ROOT/"benchmarks/baselines/nvidia_public_attention_vjp_20261006/artifacts"

@pytest.fixture
def contract(monkeypatch):
    module.clear_prepared()
    metadata=json.loads((ARTIFACTS/"qkv_q_5_0.json").read_text())["metadata"]
    def forbidden(*a,**k):
        raise AssertionError("native setup before checked host contract")
    monkeypatch.setattr(module.PreparedAttentionVJP,"_prepare",forbidden)
    yield metadata
    module.clear_prepared()

def values(owner):
    return tuple(np.zeros(shape,np.float32) for shape in owner.shapes)

def test_changed_pin_is_refused_before_registration(contract):
    changed=dict(contract,program_digest="0"*64)
    with pytest.raises(ValueError,match="pinned identity"):
        module.execute(changed,())

@pytest.mark.parametrize("index",range(4))
def test_dtype_shape_gates_precede_native_setup(contract,index):
    owner=module.prepared(contract);inputs=list(values(owner))
    inputs[index]=inputs[index].astype(np.float16)
    with pytest.raises(ValueError,match="storage"):owner.invoke(contract,inputs)
    inputs=list(values(owner));inputs[index]=inputs[index].reshape(-1)
    with pytest.raises(ValueError,match="storage"):owner.invoke(contract,inputs)

def test_closed_owner_and_changed_identity_are_refused(contract):
    owner=module.prepared(contract)
    with pytest.raises(ValueError,match="pinned identity"):
        owner.invoke(dict(contract,program_digest="1"*64),values(owner))
    owner.close()
    with pytest.raises(ValueError,match="closed"):owner.invoke(contract,values(owner))

def test_eviction_retires_old_registration(contract,monkeypatch):
    monkeypatch.setattr(module,"_CACHE_LIMIT",1)
    old=module.prepared(contract)
    alternate=json.loads((ARTIFACTS/"qkv_v_5_0.json").read_text())["metadata"]
    new=module.prepared(alternate)
    assert old.closed and not new.closed and len(module._cache)==1

def test_thread_lifetimes_do_not_share_registration(contract):
    owners=[]
    def register():owners.append(module.prepared(contract))
    for _ in range(2):
        thread=threading.Thread(target=register);thread.start();thread.join()
    assert owners[0] is not owners[1]

def test_inherited_service_is_rejected_before_cache_lock(contract,monkeypatch):
    with monkeypatch.context() as patcher:
        patcher.setattr(module.os,"getpid",lambda:module._PROCESS+1)
        with pytest.raises(ValueError,match="fork"):module.prepared(contract)
