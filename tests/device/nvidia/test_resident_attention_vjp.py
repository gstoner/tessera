"""Owning SM120 resident public saved-LSE reverse products."""
from contextlib import ExitStack
import ctypes as ct
import itertools
import subprocess

import numpy as np
import pytest
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.prepared_attention_vjp import prepared
from benchmarks.nvidia.benchmark_jvp_argument_order import function
from benchmarks.nvidia.benchmark_public_attention_vjp import oracle,biased,biased_causal
from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed,queue_delayed_upload

pytestmark=pytest.mark.hardware_nvidia


@pytest.mark.parametrize("order",list(itertools.permutations(("q","k","v"))))
@pytest.mark.parametrize("sk,causal,wrt",[(5,False,("q",)),(129,True,("v","k","q"))])
def test_resident_reverse_orders_primal_and_cotangent_streams(order,sk,causal,wrt,monkeypatch):
    prove(order,sk,causal,wrt,None,monkeypatch)


@pytest.mark.parametrize("causal",[False,True])
def test_resident_broadcast_bias_gradient(causal,monkeypatch):
    prove(("bias","v","q","k"),5,causal,("bias","k","q","v"),(1,4,1,1),monkeypatch)


def prove(order,sk,causal,wrt,bias_shape,monkeypatch):
    if rt._nvidia_device_name()!="sm_120":pytest.skip("owning RTX5070 required")
    b,hq,hkv=(2,4,2) if bias_shape else (1,2,1)
    shapes={"q":(b,hq,3,4),"k":(b,hkv,sk,4),"v":(b,hkv,sk,3)}
    if bias_shape:shapes["bias"]=bias_shape
    body=(biased_causal if causal else biased) if bias_shape else function(order,causal)
    fn=ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=wrt)(body)
    rng=np.random.default_rng(202610092)
    values={name:rng.normal(0,.2,shape).astype(np.float32) for name,shape in shapes.items()}
    seed=rng.normal(0,.1,(b,hq,3,3)).astype(np.float32)
    with ExitStack() as stack:
        sessions={name:stack.enter_context(NvidiaDeviceSession()) for name in shapes}
        seed_session=stack.enter_context(NvidiaDeviceSession())
        buffers={name:sessions[name].upload(np.zeros(shape,np.float32)) for name,shape in shapes.items()}
        seed_buffer=seed_session.upload(np.zeros_like(seed))
        roots={name:Borrowed(buffer) for name,buffer in buffers.items()}
        cotangent=Borrowed(seed_buffer)
        for session in (*sessions.values(),seed_session):assert session.synchronize()==0
        first=fn.native_backward(**roots,out_cotangents=cotangent)
        artifact=fn.native_backward_runtime_artifact()
        owner=prepared(artifact.metadata);stack.callback(owner.close)
        handle=owner.handle;held=tuple(value.copy() for value in first)
        def forbidden(*args,**kwargs):pytest.fail("warm resident reverse compiled/traced/evaluated")
        for generation in (1,2):
            current={name:value*generation for name,value in values.items()}
            current_seed=seed*generation
            completed=[];callbacks=[]
            callbacks.append(queue_delayed_upload(sessions["k"],buffers["k"],current["k"],
                sessions["k"].stream,completed))
            callbacks.append(queue_delayed_upload(seed_session,seed_buffer,current_seed,
                seed_session.stream,completed))
            for name in shapes:
                if name=="k":continue
                value=current[name];session=sessions[name]
                assert session.lib.tessera_nvidia_device_upload(ct.c_void_p(buffers[name].ptr),
                    ct.c_void_p(value.ctypes.data),value.nbytes,ct.c_void_p(session.stream))==0
            assert len(completed)<2
            with monkeypatch.context() as guard:
                guard.setattr(subprocess,"run",forbidden)
                guard.setattr(fn,"_fn",forbidden)
                guard.setattr(fn,"_trace_frontend_capture",forbidden)
                gradients=fn.native_backward(*(roots[name] for name in order),out_cotangents=cotangent)
            assert len(completed)==2 and all(callbacks)
            expected=oracle(current,current_seed,causal)
            for name,gradient in zip(wrt,gradients,strict=True):
                np.testing.assert_allclose(gradient,expected[name],atol=3e-5,rtol=3e-5)
            for actual,saved in zip(first,held,strict=True):np.testing.assert_array_equal(actual,saved)
            assert owner.handle==handle and all(time>0 for time in owner.last_device_ms)
            receipt=fn.last_backward_execution
            assert receipt["compiler_path"]=="nvidia_sm120_attention_vjp_compiled"
            assert receipt["host_preparation"]=="native_ordered_resident_snapshot"
            assert receipt["execution_certificate"]["evidence_scope"]=="exact_device"
            assert receipt["execution_certificate"]["source_reexecution"]=="prohibited"
            assert fn.last_frontend_differential.contract["concrete_executions"]==0
            assert fn.last_frontend_differential.contract["typed_graph_digest"]==receipt["source_graph_ir_digest"]
            assert len(receipt["execution_certificate"]["input_signature"])==len(shapes)+1


@pytest.mark.parametrize("violation",["capacity","alignment","stream_count"])
def test_native_reverse_resident_failure_preserves_gradients(violation):
    import json
    from pathlib import Path
    from tessera.compiler.prepared_attention_vjp import PreparedAttentionVJP
    if rt._nvidia_device_name()!="sm_120":pytest.skip("owning RTX5070 required")
    file=Path(__file__).resolve().parents[3]/"benchmarks/baselines/nvidia_public_attention_vjp_20261006/artifacts/qkv_q_5_0.json"
    owner=PreparedAttentionVJP(json.loads(file.read_text())["metadata"])
    values=tuple(np.full(shape,.1,np.float32) for shape in owner.shapes)
    metadata=json.loads(file.read_text())["metadata"]
    expected=owner.invoke(metadata,values)
    with NvidiaDeviceSession() as session:
        buffers=[session.upload(value) for value in values]
        tiny=session.empty((1,),np.float32)
        roots=tuple(Borrowed(buffer) for buffer in buffers)
        owner.invoke(metadata,roots)
        pointers=(ct.c_void_p*len(buffers))(*(buffer.ptr for buffer in buffers))
        sizes=(ct.c_size_t*len(buffers))(*(value.nbytes for value in values))
        streams=(ct.c_uint64*len(buffers))(*(session.stream for _ in buffers))
        outputs=tuple(np.full(shape,123,np.float32) for shape in owner.output_shapes)
        destinations=(ct.c_void_p*len(outputs))(*(value.ctypes.data for value in outputs))
        lengths=(ct.c_size_t*len(outputs))(*(value.nbytes for value in outputs))
        if violation=="capacity":pointers[0]=tiny.ptr
        if violation=="alignment":pointers[0]=buffers[0].ptr+1
        count=len(buffers)-1 if violation=="stream_count" else len(buffers)
        rc=owner.lib.tessera_nvidia_attention_vjp_invoke_resident_ordered(owner.handle,
            pointers,sizes,len(buffers),streams,count,destinations,lengths,len(outputs),None)
        assert rc!=0
        for output in outputs:np.testing.assert_array_equal(output,np.full(output.shape,123,np.float32))
        for x,y in zip(owner.invoke(metadata,roots),expected,strict=True):np.testing.assert_array_equal(x,y)
    owner.close()


@pytest.mark.parametrize("bias",[False,True])
def test_resident_reverse_matches_scalar_finite_difference(bias):
    if rt._nvidia_device_name()!="sm_120":pytest.skip("owning RTX5070 required")
    rng=np.random.default_rng(202610094)
    shapes={"q":(1,2,3,4),"k":(1,1,5,4),"v":(1,1,5,3)}
    if bias:shapes["bias"]=(1,2,1,1)
    values={name:rng.normal(0,.2,shape).astype(np.float32) for name,shape in shapes.items()}
    directions={name:rng.normal(0,.1,shape).astype(np.float64) for name,shape in shapes.items()}
    cot=rng.normal(0,.1,(1,2,3,3)).astype(np.float32)
    wrt=tuple(shapes)
    fn=ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=wrt)(
        biased_causal if bias else function(("q","k","v"),True))
    with ExitStack() as stack:
        session=stack.enter_context(NvidiaDeviceSession())
        roots={name:Borrowed(session.upload(value)) for name,value in values.items()}
        seed=Borrowed(session.upload(cot));assert session.synchronize()==0
        gradients=fn.native_backward(**roots,out_cotangents=seed)
        owner=prepared(fn.native_backward_runtime_artifact().metadata);stack.callback(owner.close)
        def loss(xs):
            q,k,v=(xs[name] for name in ("q","k","v"))
            scores=.5*(q@np.swapaxes(np.repeat(k,2,axis=1),-1,-2))
            if bias:scores=scores+xs["bias"]
            scores=np.where(np.arange(5)[None,:]<=np.arange(3)[:,None]+2,scores,-np.inf)
            p=np.exp(scores-scores.max(axis=-1,keepdims=True));p/=p.sum(axis=-1,keepdims=True)
            return float(np.sum((p@np.repeat(v,2,axis=1))*cot.astype(np.float64)))
        h=1e-4
        plus={name:value.astype(np.float64)+h*directions[name] for name,value in values.items()}
        minus={name:value.astype(np.float64)-h*directions[name] for name,value in values.items()}
        finite=(loss(plus)-loss(minus))/(2*h)
        adjoint=sum(float(np.sum(gradient.astype(np.float64)*directions[name]))
                    for name,gradient in zip(wrt,gradients,strict=True))
        np.testing.assert_allclose(adjoint,finite,atol=1e-7,rtol=1e-5)
