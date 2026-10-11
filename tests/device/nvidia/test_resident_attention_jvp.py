"""Exact SM120 resident public attention JVP and private snapshot lifetime."""
from contextlib import ExitStack
import ctypes as ct
import itertools
import subprocess

import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt

from benchmarks.nvidia.benchmark_jvp_argument_order import function
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_attention_jvp_runtime import prepared
from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed,queue_delayed_upload

pytestmark=pytest.mark.hardware_nvidia


def oracle(values,causal):
    q,k,v=(values[name].astype(np.float64) for name in ("q","k","v"))
    sk=k.shape[2];sq=q.shape[2];repeat=q.shape[1]//k.shape[1]
    scores=.5*(q@np.swapaxes(np.repeat(k,repeat,axis=1),-1,-2))
    if causal:
        scores=np.where(np.arange(sk)[None,:]<=np.arange(sq)[:,None]+max(sk-sq,0),
                        scores,-np.inf)
    probabilities=np.exp(scores-scores.max(axis=-1,keepdims=True))
    probabilities/=probabilities.sum(axis=-1,keepdims=True)
    return probabilities@np.repeat(v,repeat,axis=1)


@pytest.mark.parametrize("order",list(itertools.permutations(("q","k","v"))))
@pytest.mark.parametrize("sk,causal,wrt",[(5,False,("q",)),(129,True,("v","k","q"))])
def test_public_resident_jvp_orders_pending_roots_and_retains_outputs(order,sk,causal,wrt,monkeypatch):
    if rt._nvidia_device_name()!="sm_120":pytest.skip("owning RTX5070 required")
    fn=ts.jit(target="nvidia_sm120",autodiff="forward",wrt=wrt)(function(order,causal))
    rng=np.random.default_rng(20261009)
    shapes={"q":(1,2,3,4),"k":(1,1,sk,4),"v":(1,1,sk,3)}
    values={name:rng.normal(0,.2,shape).astype(np.float32) for name,shape in shapes.items()}
    directions={name:rng.normal(0,.1,shape).astype(np.float32) if name in wrt
                else np.zeros(shape,np.float32) for name,shape in shapes.items()}
    with ExitStack() as stack:
        sessions={name:stack.enter_context(NvidiaDeviceSession()) for name in shapes}
        buffers={name:sessions[name].upload(np.zeros(shape,np.float32)) for name,shape in shapes.items()}
        seeds={name:sessions[name].upload(directions[name]) for name in wrt}
        roots={name:Borrowed(buffer) for name,buffer in buffers.items()}
        tangents=tuple(Borrowed(seeds[name]) for name in wrt)
        for session in sessions.values():assert session.synchronize()==0
        first=fn.native_jvp(**roots,tangents=tangents)
        package=next(iter(fn._native_jvp_packages.values()))
        metadata=package.contract["steps"][0]["child_metadata"]
        owner=prepared(metadata);stack.callback(owner.close)
        handle=owner.handle
        held=tuple(value.copy() for value in first)
        def forbidden(*args,**kwargs):pytest.fail("warm resident JVP traced/compiled/evaluated")
        for generation in (1,2):
            current={name:value*generation for name,value in values.items()}
            completed=[]
            # The owner arena is already allocated: no allocation synchronization
            # may hide a missing producer-stream dependency.
            callback=queue_delayed_upload(sessions["k"],buffers["k"],current["k"],
                                           sessions["k"].stream,completed)
            for name in ("q","v"):
                value=current[name];session=sessions[name]
                assert session.lib.tessera_nvidia_device_upload(ct.c_void_p(buffers[name].ptr),
                    ct.c_void_p(value.ctypes.data),value.nbytes,ct.c_void_p(session.stream))==0
            assert not completed
            with monkeypatch.context() as guard:
                guard.setattr(subprocess,"run",forbidden)
                guard.setattr(fn,"_fn",forbidden)
                guard.setattr(fn,"_trace_frontend_capture",forbidden)
                result=fn.native_jvp(*(roots[name] for name in order),tangents=tangents)
            assert completed and callback is not None
            h=1e-4
            expected=oracle(current,causal)
            plus={name:value.astype(np.float64)+h*directions[name] for name,value in current.items()}
            minus={name:value.astype(np.float64)-h*directions[name] for name,value in current.items()}
            derivative=(oracle(plus,causal)-oracle(minus,causal))/(2*h)
            np.testing.assert_allclose(result[0],expected,rtol=3e-5,atol=3e-5)
            np.testing.assert_allclose(result[1],derivative,rtol=3e-5,atol=3e-5)
            for actual,saved in zip(first,held,strict=True):np.testing.assert_array_equal(actual,saved)
            assert owner.handle==handle
            assert all(time>0 for time in owner.last_device_ms)
            receipt=fn.last_jvp_execution
            assert receipt["execution_kind"]=="native_gpu"
            assert receipt["compiler_path"]=="nvidia_sm120_jvp_compiled"
            assert receipt["host_preparation"]=="native_ordered_resident_snapshot"
            assert receipt["frontend_certificate"]["concrete_executions"]==0
            assert receipt["frontend_certificate"]["numerical_authority"]=="physical_package_required"


@pytest.mark.parametrize("violation",["capacity","alignment","stream_count"])
def test_native_resident_rejection_leaves_output_and_owner_usable(violation):
    if rt._nvidia_device_name()!="sm_120":pytest.skip("owning RTX5070 required")
    import json
    from pathlib import Path
    from tessera.compiler.native_attention_jvp_runtime import PreparedAttentionJVP
    file=Path(__file__).resolve().parents[3]/"benchmarks/baselines/nvidia_jvp_portable_20261006/artifacts/qkv_q_5_0.program.json"
    raw=file.read_text()
    metadata={"program_json":raw,"program_digest":json.loads(raw)["program_digest"],
              "arg_names":["primal_0","primal_1","primal_2","tangent_0"]}
    owner=PreparedAttentionJVP(metadata)
    values=tuple(np.full(shape,.1,np.float32) for shape in owner.shapes)
    expected=owner.invoke(metadata,values)
    with NvidiaDeviceSession() as session:
        buffers=[session.upload(value) for value in values]
        tiny=session.empty((1,),np.float32)
        roots=tuple(Borrowed(buffer) for buffer in buffers)
        actual=owner.invoke(metadata,roots)
        for x,y in zip(actual,expected,strict=True):np.testing.assert_array_equal(x,y)
        lib=owner.lib
        pointers=(ct.c_void_p*len(buffers))(*(buffer.ptr for buffer in buffers))
        sizes=(ct.c_size_t*len(buffers))(*(value.nbytes for value in values))
        streams=(ct.c_uint64*len(buffers))(*(session.stream for _ in buffers))
        outputs=tuple(np.full(owner.output_shape,123,np.float32) for _ in range(2))
        destinations=(ct.c_void_p*2)(*(value.ctypes.data for value in outputs))
        lengths=(ct.c_size_t*2)(*(value.nbytes for value in outputs))
        if violation=="capacity":pointers[0]=tiny.ptr
        if violation=="alignment":pointers[0]=buffers[0].ptr+1
        count=len(buffers)-1 if violation=="stream_count" else len(buffers)
        rc=lib.tessera_nvidia_attention_jvp_invoke_resident_ordered(
            owner.handle,pointers,sizes,len(buffers),streams,count,destinations,lengths,None)
        assert rc!=0
        for output in outputs:np.testing.assert_array_equal(output,np.full(output.shape,123,np.float32))
        for x,y in zip(owner.invoke(metadata,roots),expected,strict=True):np.testing.assert_array_equal(x,y)
    owner.close()
