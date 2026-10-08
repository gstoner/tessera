"""Public JIT O/LSE reverse AD with private CUDA generation ownership."""
from __future__ import annotations
import ctypes as ct
import numpy as np
import pytest
import tessera as ts
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_attention_program import NativeAttentionVJPProgram
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lse_cotangent_native import oracle
from benchmarks.nvidia.benchmark_jit_attention_vjp import download


def function(bias=False,causal=True):
    wrt=('bias','v','q') if bias else ('v','q')
    if bias:
        @ts.jit(target='nvidia_sm120',autodiff='reverse',wrt=wrt)
        def attention(q,k,v,bias):
            return ts.ops.flash_attn(q,k,v,attn_bias=bias,causal=causal,lse_checkpoint='saved')
    else:
        @ts.jit(target='nvidia_sm120',autodiff='reverse',wrt=wrt)
        def attention(q,k,v):
            return ts.ops.flash_attn(q,k,v,causal=causal,lse_checkpoint='saved')
    return attention


def values(shape,bias,seed_mode):
    b,hq,hkv,sq,sk,d,dv=shape;rng=np.random.default_rng(507024)
    q,k,v,do=[(rng.normal(size=z)*.2).astype(np.float32) for z in
             [(b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),(b,hq,sq,dv)]]
    seed=(rng.normal(size=(b,hq,sq))*.3).astype(np.float32)
    if seed_mode=='output_only':seed.fill(0)
    if seed_mode=='lse_only':do.fill(0)
    inputs=[q,k,v]
    if bias:inputs.append((rng.normal(size=(1,1,1,sk))*.2).astype(np.float32))
    return inputs,[do,seed]


def reference(inputs,seeds,causal):
    q,k,v=inputs[:3];b,hq,sq,_=q.shape;sk=k.shape[2]
    bias=np.broadcast_to(inputs[3],(b,hq,sq,sk)) if len(inputs)==4 else None
    output,lse,grads=oracle(q,k,v,*seeds,causal,bias)
    if bias is not None:grads=(*grads[:3],grads[3].sum(axis=(0,1,2),keepdims=True))
    return (output,lse),grads


@pytest.mark.parametrize('shape',[(1,2,1,3,5,4,3),(1,2,1,5,3,4,3),(2,4,2,7,9,8,6)])
@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('seed_mode',['output_only','lse_only','mixed'])
@pytest.mark.parametrize('compact',[False,True])
@pytest.mark.parametrize('causal',[False,True])
def test_public_jit_tuple_reverse_private_replay(shape,bias,seed_mode,compact,causal,monkeypatch):
    if not nvidia_cuda_host_ready():pytest.skip('requires exact SM120 host and toolchain')
    import os
    inputs,seeds=values(shape,bias,seed_mode)
    program=function(bias,causal).compile_native_attention_vjp(*inputs,compiler=os.environ['TESSERA_OPT'],compact_gradients=compact)
    digest=program.program_digest
    program=NativeAttentionVJPProgram.from_json(program.to_json(),expected_digest=digest)
    monkeypatch.setenv('TESSERA_OPT','/no/compiler');monkeypatch.setenv('TESSERA_NVIDIA_OPT','/no/compiler')
    expected,grads=reference(inputs,seeds,causal)
    with NvidiaDeviceSession() as session:
        resident=[session.upload(x) for x in inputs]
        seed_buffers=tuple(session.upload(x) for x in seeds)
        frame=program.capture(*resident)
        borrowed=frame.primal
        try:
            assert isinstance(borrowed,tuple) and len(borrowed)==2
            for value,want in zip(borrowed,expected,strict=True):
                assert value.__cuda_array_interface__['data'][1] is True
                np.testing.assert_allclose(download(session,value),want,rtol=4e-5,atol=4e-5)
            # Caller writes cannot change the captured Q/K/V/bias generation.
            for value in resident:
                changed=np.full(value.shape,19,np.float32)
                assert session.lib.tessera_nvidia_device_upload(ct.c_void_p(value.ptr),ct.c_void_p(changed.ctypes.data),changed.nbytes,ct.c_void_p(session.stream))==0
            session.synchronize()
            first=frame.backward(seed_buffers)
            for got,role in zip(first,program.active,strict=True):
                np.testing.assert_allclose(download(session,got),grads[role],rtol=4e-5,atol=4e-5)
            second_seeds=[x*-.7 for x in seeds]
            _,second_expected=reference(inputs,second_seeds,causal)
            second=frame.backward(tuple(session.upload(x) for x in second_seeds))
            for got,role in zip(second,program.active,strict=True):
                np.testing.assert_allclose(download(session,got),second_expected[role],rtol=4e-5,atol=4e-5)
            for got,role in zip(first,program.active,strict=True):
                np.testing.assert_allclose(download(session,got),grads[role],rtol=4e-5,atol=4e-5)
            with pytest.raises(ValueError,match='two result cotangents'):frame.backward(seed_buffers[0])
            small=session.empty((1,),np.float32)
            class ShortSeed:
                @property
                def __cuda_array_interface__(self):
                    return {**small.__cuda_array_interface__,'shape':seeds[1].shape,'strides':None}
            before=len(frame._frame.buffers)
            with pytest.raises(ValueError,match='exceeds device allocation'):frame.backward((seed_buffers[0],ShortSeed()))
            assert len(frame._frame.buffers)==before
        finally:frame.close()
        with pytest.raises(ValueError,match='closed'):frame.backward(seed_buffers)
        for value in borrowed:
            with pytest.raises(ValueError,match='closed'):_=value.__cuda_array_interface__
