"""Portable native-owner admission, replay, context and retirement proof."""
from copy import deepcopy
from dataclasses import replace
import ctypes as ct
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler import prepared_nvidia_lhs as owner
from tessera.compiler.nvidia_tensor_lhs import runtime_artifact
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lhs_tensor_jit import (
    rms_lhs,layer_lhs,softmax_lhs,rms_lhs_fused,layer_lhs_fused,softmax_lhs_fused,
    _storage,_oracle)

pytestmark=pytest.mark.skipif(not nvidia_cuda_host_ready(),reason="owning NVIDIA host required")


@pytest.fixture(autouse=True)
def cache(monkeypatch):
    monkeypatch.setenv("TESSERA_NVIDIA_PREPARED_LHS_REPLAY","1")
    owner.clear_portable_lhs_owners()
    yield
    owner.clear_portable_lhs_owners()


@pytest.mark.parametrize("kind",["rmsnorm","layernorm","softmax"])
@pytest.mark.parametrize("dtype",["fp16","bf16"])
@pytest.mark.parametrize("order",["C","F"])
@pytest.mark.parametrize("fused",[False,True])
def test_portable_cached_clone_and_named_inputs(kind,dtype,order,fused,monkeypatch):
    from tessera.compiler import nvidia_tensor_lhs
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    storage=_storage(dtype)
    rng=np.random.default_rng(50701024)
    source=(rng.normal(size=(128,1024))*.2).astype(storage)
    rhs=np.array(rng.normal(size=(1024,64))*.2,dtype=storage,order=order)
    bias=(rng.normal(size=64)*.1).astype(np.float32)
    residual=(rng.normal(size=(128,64))*.1).astype(np.float32)
    functions=({"rmsnorm":rms_lhs_fused,"layernorm":layer_lhs_fused,"softmax":softmax_lhs_fused}
               if fused else {"rmsnorm":rms_lhs,"layernorm":layer_lhs,"softmax":softmax_lhs})
    args=(source,rhs,bias,residual) if fused else (source,rhs)
    artifact=rt.RuntimeArtifact.from_json(runtime_artifact(
        functions[kind].compile_native_lhs_matmul(*args)).to_json())
    first=rt.launch(artifact,args)
    assert first["ok"],first
    expected=_oracle(source,rhs,kind,bias if fused else None,residual if fused else None)
    np.testing.assert_allclose(first["output"],expected,atol=.015,rtol=.015)
    call=next(iter(owner._portable_owners.values()))
    stats=call.scratch_stats()
    def forbidden(*args,**kwargs):
        raise AssertionError("cached portable replay restored/compiled or used Python CUDA session")
    monkeypatch.setattr(nvidia_tensor_lhs,"from_manifest",forbidden)
    monkeypatch.setattr(NvidiaDeviceSession,"__init__",forbidden)
    saved=first["output"].copy()
    clone=rt.RuntimeArtifact.from_json(artifact.to_json())
    for scale in (.5,-.75,1.25):
        changed=(source.astype(np.float32)*scale).astype(storage)
        values=(changed,*args[1:])
        receipt=rt.launch(clone,dict(zip(clone.metadata["arg_names"],values,strict=True)))
        assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
        assert all(r["native_call_binding"]=="prepared_cpp_tensor_matmul"
                   for r in receipt["component_receipts"])
        np.testing.assert_allclose(receipt["output"],_oracle(changed,rhs,kind,
            bias if fused else None,residual if fused else None),atol=.015,rtol=.015)
    assert call.scratch_stats()==stats
    np.testing.assert_array_equal(first["output"],saved)


@pytest.mark.parametrize("field",["graph","names","target","manifest","bool_shape"])
def test_invalid_parent_or_manifest_cannot_use_warm_owner(field,monkeypatch):
    source=np.ones((1,35),np.float16)
    rhs=np.ones((35,19),np.float16)
    artifact=runtime_artifact(rms_lhs.compile_native_lhs_matmul(source,rhs))
    assert rt.launch(artifact,(source,rhs))["ok"]
    bad=rt.RuntimeArtifact.from_json(artifact.to_json())
    if field=="graph":bad=replace(bad,graph_ir=bad.graph_ir+"\n// different parent")
    elif field=="names":bad.metadata["arg_names"].reverse()
    elif field=="target":bad.metadata["target"]="rocm_gfx1201"
    elif field=="manifest":bad.metadata["native_program"]["semantics"]["producer_attrs"]["eps"]=.5
    else:bad.metadata["native_program"]["edge"]["m"]=True
    def forbidden(*args,**kwargs):
        raise AssertionError("invalid seal reached CUDA context/owner")
    monkeypatch.setattr(owner,"PreparedLhsCall",forbidden)
    monkeypatch.setattr(rt._load_nvidia_ptx_launch(),"tessera_nvidia_matmul_context_identity",forbidden)
    # Direct executor proves admission irrespective of outer target selection.
    with pytest.raises((ValueError,rt.ArtifactContractError)):
        rt._execute_nvidia_lhs_program_artifact(bad,(source,rhs))
    assert len(owner._portable_owners)==1


def test_invalid_tensor_shape_is_checked_before_native_context(monkeypatch):
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16)
    artifact=runtime_artifact(rms_lhs.compile_native_lhs_matmul(source,rhs))
    lib=rt._load_nvidia_ptx_launch()
    def forbidden(*args,**kwargs):
        raise AssertionError("invalid shape reached CUDA")
    monkeypatch.setattr(lib,"tessera_nvidia_matmul_context_identity",forbidden)
    with pytest.raises(ValueError,match="sealed"):
        rt._execute_nvidia_lhs_program_artifact(artifact,(source[:1],rhs))


def test_cache_retirement_and_lazy_rebinding(monkeypatch):
    monkeypatch.setattr(owner,"_PORTABLE_LIMIT",2)
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16)
    program=rms_lhs.compile_native_lhs_matmul(source,rhs)
    retired=None
    for i in range(3):
        renamed=replace(program,argument_names=(f"source{i}",f"rhs{i}"))
        receipt=rt.launch(runtime_artifact(renamed),(source,rhs))
        assert receipt["ok"],receipt
        if i==0:retired=next(iter(owner._portable_owners.values()))
    assert len(owner._portable_programs)==len(owner._portable_owners)==2
    assert not retired._finalizer.alive
    live=list(owner._portable_owners.values())
    owner.clear_portable_lhs_owners()
    assert all(not call._finalizer.alive for call in live)
    receipt=rt.launch(runtime_artifact(program),(source,rhs))
    assert receipt["ok"],receipt


def test_portable_owner_is_scoped_to_live_context_identity():
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16)
    artifact=runtime_artifact(rms_lhs.compile_native_lhs_matmul(source,rhs))
    assert rt.launch(artifact,(source,rhs))["ok"]
    original_call=next(iter(owner._portable_owners.values()))
    cuda=ct.CDLL("libcuda.so.1")
    cuda.cuCtxGetCurrent.argtypes=[ct.POINTER(ct.c_void_p)]
    cuda.cuCtxCreate_v2.argtypes=[ct.POINTER(ct.c_void_p),ct.c_uint,ct.c_int]
    cuda.cuCtxSetCurrent.argtypes=[ct.c_void_p]
    cuda.cuCtxDestroy_v2.argtypes=[ct.c_void_p]
    original,other=ct.c_void_p(),ct.c_void_p()
    assert cuda.cuCtxGetCurrent(ct.byref(original))==0
    assert cuda.cuCtxCreate_v2(ct.byref(other),0,0)==0
    try:
        receipt=rt.launch(artifact,(source,rhs))
        assert receipt["ok"],receipt
        assert len(owner._portable_owners)==2
        assert len({key[1] for key in owner._portable_owners})==2
        assert original_call in owner._portable_owners.values()
        # Close the owning native leases while both contexts remain live.
        owner.clear_portable_lhs_owners()
    finally:
        assert cuda.cuCtxSetCurrent(original)==0
        assert cuda.cuCtxDestroy_v2(other)==0
    assert rt.launch(artifact,(source,rhs))["ok"]


def test_ordered_manifest_uses_its_declared_identity():
    from collections import OrderedDict
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16)
    program=rms_lhs.compile_native_lhs_matmul(source,rhs)
    for i in range(2):
        artifact=runtime_artifact(replace(program,argument_names=(f"source{i}",f"rhs{i}")))
        artifact.metadata["native_program"]=OrderedDict(artifact.metadata["native_program"])
        receipt=rt.launch(artifact,(source,rhs))
        assert receipt["ok"],receipt
    assert len(owner._portable_owners)==2
    assert all(isinstance(key[0],str) for key in owner._portable_owners)


def test_concurrent_portable_replays_share_one_native_context_owner():
    from concurrent.futures import ThreadPoolExecutor
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16)
    artifact=runtime_artifact(rms_lhs.compile_native_lhs_matmul(source,rhs))
    assert rt.launch(artifact,(source,rhs))["ok"]
    def replay(index):
        active=source*(1+index/10)
        receipt=rt.launch(artifact,(active,rhs))
        assert receipt["ok"],receipt
        np.testing.assert_allclose(receipt["output"],_oracle(active,rhs,"rmsnorm"),
                                   atol=.015,rtol=.015)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(replay,range(16)))
    assert len(owner._portable_owners)==1


def test_inherited_cache_pid_guard_precedes_held_python_lock():
    import os
    import select
    import threading
    source=np.ones((17,35),np.float16)
    rhs=np.ones((35,19),np.float16)
    artifact=runtime_artifact(rms_lhs.compile_native_lhs_matmul(source,rhs))
    assert rt.launch(artifact,(source,rhs))["ok"]
    entered,release=threading.Event(),threading.Event()
    def hold():
        with owner._portable_lock:
            entered.set()
            release.wait(10)
    thread=threading.Thread(target=hold)
    thread.start()
    assert entered.wait(5)
    read,write=os.pipe()
    child=os.fork()
    if child==0:
        os.close(read)
        try:
            rt._execute_nvidia_lhs_program_artifact(artifact,(source,rhs))
        except ValueError as error:
            os.write(write,b"ok" if "fork" in str(error) else b"bad")
        else:os.write(write,b"bad")
        os._exit(0)
    os.close(write)
    try:
        ready,_,_=select.select([read],[],[],5)
        assert ready,"child inherited a locked cache instead of rejecting fork"
        assert os.read(read,3)==b"ok"
    finally:
        release.set()
        thread.join(5)
        os.close(read)
        if not ready:
            import signal
            os.kill(child,signal.SIGKILL)
        os.waitpid(child,0)
