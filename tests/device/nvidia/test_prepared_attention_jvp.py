"""Owning SM120 native prepared ABI, lifetime and context guards."""
import ctypes as ct
import json
import os
import select
import signal
import threading
from pathlib import Path
import numpy as np
import pytest
from tessera.compiler.native_attention_jvp_runtime import PreparedAttentionJVP
from benchmarks.nvidia.benchmark_prepared_attention_jvp import run

ROOT=Path(__file__).resolve().parents[3]
ARTIFACTS=ROOT/"benchmarks/baselines/nvidia_public_attention_jvp_20261006/artifacts"
pytestmark=pytest.mark.skipif(os.environ.get("TESSERA_NVIDIA_DEVICE_PROOF")!="1",
                              reason="requires owning RTX5070 proof lane")

@pytest.mark.parametrize("name",["qkv_q_5_0","vqk_v_129_1","kvq_k_q_5_0","vkq_q_k_v_129_1"])
def test_matched_prepared_product_oracle(name):
    row=run(ARTIFACTS/(name+".json"),3)
    assert row["correctness"]=="passed_before_timing"
    assert all(f>0 and t>0 for f,t in row["native_forward_tangent_event_samples_ms"])

@pytest.fixture(params=[
    "qkv_q_5_0",
    "k5_c0_1x4x1x5_bias_v_k_q",
    "k129_c1_2x4x3x129_bias",
    "k129_c1_1x4x1x129_v",
])
def native(request):
    directory=(ARTIFACTS if request.param=="qkv_q_5_0" else
        ROOT/"benchmarks/baselines/nvidia_bias_jvp_20261006/public-artifacts")
    raw=json.loads((directory/(request.param+".json")).read_text())
    metadata=raw["native_jvp"]["steps"][0]["child_metadata"]
    owner=PreparedAttentionJVP(metadata)
    rng=np.random.default_rng(916)
    values=tuple(rng.normal(size=s).astype(np.float32)*.1 for s in owner.shapes)
    expected=owner.invoke(metadata,values)
    yield owner,metadata,values,expected
    owner.close()

def test_native_extent_failure_does_not_corrupt_retained_state(native):
    owner,metadata,values,expected=native
    pointers=(ct.c_void_p*len(values))(*(x.ctypes.data for x in values))
    lengths=(ct.c_size_t*len(values))(*(x.nbytes for x in values));lengths[0]-=4
    outputs=tuple(np.empty(owner.output_shape,np.float32) for _ in range(2))
    destinations=(ct.c_void_p*2)(*(x.ctypes.data for x in outputs))
    sizes=(ct.c_size_t*2)(*(x.nbytes for x in outputs))
    rc=owner.lib.tessera_nvidia_attention_jvp_invoke(
        owner.handle,pointers,lengths,len(values),destinations,sizes,None)
    assert rc!=0 and b"extent" in owner.lib.tessera_nvidia_attention_jvp_last_error()
    actual=owner.invoke(metadata,values)
    for x,y in zip(actual,expected,strict=True):np.testing.assert_array_equal(x,y)

def test_native_closed_handle_is_not_reused(native):
    owner,metadata,values,_=native
    handle=owner.handle;owner.close()
    assert owner.lib.tessera_nvidia_attention_jvp_close(handle)!=0
    assert b"closed" in owner.lib.tessera_nvidia_attention_jvp_last_error()
    with pytest.raises(ValueError,match="closed"):owner.invoke(metadata,values)

def test_native_wrong_context_is_rejected_before_upload(native):
    owner,metadata,values,expected=native
    driver=ct.CDLL("libcuda.so.1")
    driver.cuCtxGetCurrent.argtypes=[ct.POINTER(ct.c_void_p)]
    driver.cuCtxSetCurrent.argtypes=[ct.c_void_p]
    original=ct.c_void_p();assert driver.cuCtxGetCurrent(ct.byref(original))==0
    assert driver.cuCtxSetCurrent(None)==0
    try:
        with pytest.raises(RuntimeError,match="context disagrees"):owner.invoke(metadata,values)
    finally:assert driver.cuCtxSetCurrent(original)==0
    for x,y in zip(owner.invoke(metadata,values),expected,strict=True):np.testing.assert_array_equal(x,y)


def test_native_fork_refuses_inherited_handle_before_driver_access(native):
    owner,metadata,values,_=native
    held=threading.Event();release=threading.Event()
    def hold():
        with owner.lock:
            held.set();release.wait()
    thread=threading.Thread(target=hold);thread.start()
    assert held.wait(5)
    read,write=os.pipe();pid=os.fork()
    if pid==0:
        os.close(read)
        try:
            owner.invoke(metadata,values)
        except ValueError as error:
            assert "fork" in str(error)
        else:os._exit(3)
        rc=owner.lib.tessera_nvidia_attention_jvp_invoke(owner.handle,None,None,0,None,None,None)
        os.write(write,str(rc).encode()+b":"+owner.lib.tessera_nvidia_attention_jvp_last_error())
        os._exit(0)
    os.close(write)
    try:
        ready,_,_=select.select([read],[],[],5)
        if not ready:
            os.kill(pid,signal.SIGKILL)
            pytest.fail("fork guard blocked on an inherited lock")
        message=os.read(read,4096)
        _,status=os.waitpid(pid,0)
        assert status==0 and message.startswith(b"1:") and b"fork" in message
    finally:
        release.set();thread.join(5);os.close(read)


def test_biased_native_tangent_extent_failure_preserves_next_generation(native):
    owner,metadata,values,expected=native
    if not owner.biased:
        return
    pointers=(ct.c_void_p*len(values))(*(x.ctypes.data for x in values))
    lengths=(ct.c_size_t*len(values))(*(x.nbytes for x in values));lengths[-1]-=4
    outputs=tuple(np.empty(owner.output_shape,np.float32) for _ in range(2))
    destinations=(ct.c_void_p*2)(*(x.ctypes.data for x in outputs))
    sizes=(ct.c_size_t*2)(*(x.nbytes for x in outputs))
    rc=owner.lib.tessera_nvidia_attention_jvp_invoke(
        owner.handle,pointers,lengths,len(values),destinations,sizes,None)
    assert rc and b"extent" in owner.lib.tessera_nvidia_attention_jvp_last_error()
    for actual,prior in zip(owner.invoke(metadata,values),expected,strict=True):
        np.testing.assert_array_equal(actual,prior)
