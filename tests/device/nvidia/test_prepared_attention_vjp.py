"""Owning-device reverse ABI, private generation and context guards."""
import ctypes as ct
import json
import os
import select
import signal
import threading
from pathlib import Path
import numpy as np
import pytest
from tessera.compiler.prepared_attention_vjp import PreparedAttentionVJP
from benchmarks.nvidia.benchmark_prepared_attention_vjp import run

ROOT=Path(__file__).resolve().parents[3]
ARTIFACTS=ROOT/"benchmarks/baselines/nvidia_public_attention_vjp_20261006/artifacts"
pytestmark=pytest.mark.skipif(os.environ.get("TESSERA_NVIDIA_DEVICE_PROOF")!="1",
                              reason="requires owning RTX5070 proof lane")

@pytest.mark.parametrize("case",["qkv_q_5_0","vkq_q_k_v_129_1","biasvqk_bias_v_k_q_5_1_1x4x1x1"])
def test_matched_native_reverse(case):
    row=run(ARTIFACTS/(case+".json"),3)
    assert row["correctness"]=="passed_before_timing"
    assert all(f>0 and b>0 for f,b in row["native_forward_backward_event_samples_ms"])

@pytest.fixture
def native():
    metadata=json.loads((ARTIFACTS/"qkv_q_5_0.json").read_text())["metadata"]
    owner=PreparedAttentionVJP(metadata)
    rng=np.random.default_rng(912)
    values=tuple(rng.normal(size=s).astype(np.float32)*.1 for s in owner.shapes)
    expected=owner.invoke(metadata,values)
    yield owner,metadata,values,expected
    owner.close()

def test_bad_native_extent_preserves_next_generation(native):
    owner,metadata,values,expected=native
    pointers=(ct.c_void_p*len(values))(*(x.ctypes.data for x in values))
    sizes=(ct.c_size_t*len(values))(*(x.nbytes for x in values));sizes[-1]-=4
    outputs=tuple(np.empty(s,np.float32) for s in owner.output_shapes)
    destinations=(ct.c_void_p*len(outputs))(*(x.ctypes.data for x in outputs))
    lengths=(ct.c_size_t*len(outputs))(*(x.nbytes for x in outputs))
    rc=owner.lib.tessera_nvidia_attention_vjp_invoke(owner.handle,pointers,sizes,len(values),destinations,lengths,len(outputs),None)
    assert rc and b"extent" in owner.lib.tessera_nvidia_attention_vjp_last_error()
    for x,y in zip(owner.invoke(metadata,values),expected,strict=True):
        np.testing.assert_array_equal(x,y)

def test_closed_handle_is_not_reused(native):
    owner,metadata,values,_=native
    old=owner.handle;owner.close()
    assert owner.lib.tessera_nvidia_attention_vjp_close(old)!=0
    with pytest.raises(ValueError,match="closed"):owner.invoke(metadata,values)

def test_wrong_context_precedes_upload(native):
    owner,metadata,values,expected=native
    driver=ct.CDLL("libcuda.so.1")
    driver.cuCtxGetCurrent.argtypes=[ct.POINTER(ct.c_void_p)]
    driver.cuCtxSetCurrent.argtypes=[ct.c_void_p]
    original=ct.c_void_p();assert driver.cuCtxGetCurrent(ct.byref(original))==0
    assert driver.cuCtxSetCurrent(None)==0
    try:
        with pytest.raises(RuntimeError,match="context generation"):owner.invoke(metadata,values)
    finally:assert driver.cuCtxSetCurrent(original)==0
    for x,y in zip(owner.invoke(metadata,values),expected,strict=True):
        np.testing.assert_array_equal(x,y)

def test_fork_rejects_before_inherited_lock_or_cuda(native):
    owner,metadata,values,_=native
    held=threading.Event();release=threading.Event()
    def hold():
        with owner.lock:held.set();release.wait()
    thread=threading.Thread(target=hold);thread.start();assert held.wait(5)
    read,write=os.pipe();pid=os.fork()
    if pid==0:
        os.close(read)
        try:owner.invoke(metadata,values)
        except ValueError as error:
            assert "fork" in str(error)
        else:os._exit(3)
        rc=owner.lib.tessera_nvidia_attention_vjp_invoke(owner.handle,None,None,0,None,None,0,None)
        os.write(write,str(rc).encode()+b":"+owner.lib.tessera_nvidia_attention_vjp_last_error())
        os._exit(0)
    os.close(write)
    try:
        ready,_,_=select.select([read],[],[],5)
        if not ready:
            os.kill(pid,signal.SIGKILL);pytest.fail("inherited reverse lock blocked PID guard")
        message=os.read(read,4096);_,status=os.waitpid(pid,0)
        assert status==0 and message.startswith(b"1:") and b"fork" in message
    finally:release.set();thread.join(5);os.close(read)

@pytest.mark.parametrize("case",["qkv_q_5_0","biasvqk_bias_5_0_1x4x1x1"])
def test_native_logical_64_thread_geometry(tmp_path,case):
    import re
    from tessera.compiler.native_attention_program import compile_attention_vjp_program, NativeAttentionVJPProgram
    from tessera.compiler.scheduled_matmul import find_tessera_opt
    from tessera.runtime import RuntimeArtifact
    original=RuntimeArtifact.from_json((ARTIFACTS/(case+".json")).read_text())
    prior=NativeAttentionVJPProgram.from_json(
        original.metadata["program_json"],expected_digest=original.metadata["program_digest"])
    source=re.sub(r'=\s+(tessera\.[A-Za-z0-9_.]+)\(',r'= "\1"(',original.graph_ir)
    active=tuple(prior.input_indices[i] for i in prior.active)
    program=compile_attention_vjp_program(source,active,compiler=find_tessera_opt(),
        compact_gradients=True,compact_launch="logical_v1",compact_threads=64)
    assert program.pair.backward.descriptor.provenance["gradient_block_threads"]==64
    metadata=dict(original.metadata,program_json=program.to_json(),program_digest=program.program_digest)
    artifact=RuntimeArtifact(graph_ir=source,target_ir=program.pair.backward.target_ir,metadata=metadata)
    file=tmp_path/(case+".json");file.write_text(artifact.to_json())
    file.with_suffix(".npz").write_bytes((ARTIFACTS/(case+".npz")).read_bytes())
    row=run(file,3)
    assert row["correctness"]=="passed_before_timing"

def test_native_reverse_preserves_existing_wide_key_extent():
    import tessera as ts
    from benchmarks.nvidia.benchmark_public_attention_vjp import oracle
    @ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=("v",))
    def attention(q,k,v):
        return ts.ops.flash_attn(q,k,v,causal=False)
    rng=np.random.default_rng(918)
    values={n:rng.normal(size=s).astype(np.float32)*.1 for n,s in
            (("q",(1,1,1,1)),("k",(1,1,65537,1)),("v",(1,1,65537,1)))}
    cot=np.full((1,1,1,1),.1,np.float32)
    expected=oracle(values,cot,False)["v"]
    (actual,)=attention.native_backward(**values,out_cotangents=cot)
    np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=3e-5)
