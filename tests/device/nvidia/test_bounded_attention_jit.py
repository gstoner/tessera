"""Public frontend/native AD bounded attention, compiler-free exact-device replay."""
import os
import subprocess
import numpy as np
import pytest
import tessera as ts
from tessera.compiler.native_attention_program import NativeAttentionVJPProgram
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lse_cotangent_native import oracle
from benchmarks.nvidia.benchmark_jit_attention_vjp import download

@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("bias",[False,True,"key","query","both"])
@pytest.mark.parametrize("compact",[False,True])
@pytest.mark.parametrize("asynchronous",[False,True])
def test_public_bounded_ad_same_images_and_private_generations(bias,compact,asynchronous,monkeypatch):
    if not nvidia_cuda_host_ready():pytest.skip("exact SM120 CUDA host required")
    if bias:
        @ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=("bias","v","q"))
        def attention(q,k,v,bias):
            return ts.ops.flash_attn(q,k,v,attn_bias=bias,causal=True,lse_checkpoint="saved")
    else:
        @ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=("v","q"))
        def attention(q,k,v):
            return ts.ops.flash_attn(q,k,v,causal=True,lse_checkpoint="saved")
    def values(sq,sk,index):
        rng=np.random.default_rng(1800+index)
        inputs=[rng.normal(0,.2,shape).astype("f4") for shape in
            ((1,2,sq,8),(1,1,sk,8),(1,1,sk,6))]
        if bias:
            shape={"key":(1,1,1,sk),"query":(1,1,sq,1),"both":(1,1,sq,sk)}.get(bias,(1,2,sq,sk))
            inputs.append(rng.normal(0,.1,shape).astype("f4"))
        seeds=[rng.normal(0,.2,shape).astype("f4") for shape in ((1,2,sq,6),(1,2,sq))]
        return inputs,seeds
    def reference(inputs,seeds):
        output,lse,grads=oracle(*inputs[:3],*seeds,True,inputs[3] if bias else None)
        if bias:
            axes=tuple(i for i,n in enumerate(inputs[3].shape) if n==1)
            grads=(*grads[:3],grads[3].sum(axis=axes,keepdims=True))
        return output,lse,grads
    traced,_=values(3,4,0)
    program=attention.compile_native_attention_vjp(*traced,compiler=os.environ["TESSERA_OPT"],
        sequence_bounds=(9,11),compact_gradients=compact)
    program=NativeAttentionVJPProgram.from_json(program.to_json(),expected_digest=program.program_digest)
    # Strong replay boundary: no subprocess can reconstruct or replace a package.
    monkeypatch.setattr(subprocess,"run",lambda *a,**k:pytest.fail("compiler called during bounded public replay"))
    with NvidiaDeviceSession() as session:
        held=[]
        try:
            for index,(sq,sk) in enumerate(((1,1),(3,4),(9,11),(9,1),(1,11),(3,4))):
                inputs,seeds=values(sq,sk,index)
                expected=reference(inputs,seeds)
                resident=[session.upload(x) for x in inputs]
                rs=tuple(session.upload(x) for x in seeds)
                assert session.synchronize()==0
                frame=program.capture(*resident,asynchronous=asynchronous);held.append(frame)
                frame.wait_on(session.stream);assert session.synchronize()==0
                for got,want in zip(frame.primal,expected[:2],strict=True):
                    np.testing.assert_allclose(download(session,got),want,atol=4e-5,rtol=4e-5)
                first=frame.backward(rs)
                frame.wait_on(session.stream);assert session.synchronize()==0
                for got,role in zip(first,program.active,strict=True):
                    np.testing.assert_allclose(download(session,got),expected[2][role],atol=4e-5,rtol=4e-5)
                changed=[x*-.7 for x in seeds]
                second_expected=reference(inputs,changed)[2]
                second=frame.backward(tuple(session.upload(x) for x in changed))
                frame.wait_on(session.stream);assert session.synchronize()==0
                for got,role in zip(second,program.active,strict=True):
                    np.testing.assert_allclose(download(session,got),second_expected[role],atol=4e-5,rtol=4e-5)
                for got,role in zip(first,program.active,strict=True):
                    np.testing.assert_allclose(download(session,got),expected[2][role],atol=4e-5,rtol=4e-5)
                if index==0:prior=(frame,first,expected[2])
                else:
                    prior[0].wait_on(session.stream);assert session.synchronize()==0
                    for got,role in zip(prior[1],program.active,strict=True):
                        np.testing.assert_allclose(download(session,got),prior[2][role],atol=4e-5,rtol=4e-5)
        finally:
            for frame in reversed(held):frame.close()
