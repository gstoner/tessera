"""Exact-device bounded package replay and generation-specific residual owners."""
from dataclasses import replace
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler import nvidia_native as native
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from benchmarks.nvidia.benchmark_jit_attention_vjp import download
from tests.unit.test_bounded_attention_packages import scheduled
from tests.device.nvidia.test_lse_checkpoint_native import _reference
from tests._support.nvidia import nvidia_cuda_host_ready

def artifact(package):
    obj=rt.RuntimeArtifact(tile_ir=package.tile_ir,target_ir=package.target_ir,
        metadata={"target":"nvidia_sm120"},native_image=package.image,
        launch_descriptor=package.descriptor)
    return rt.RuntimeArtifact.from_json(obj.to_json())

def restore(package):
    a=artifact(package)
    return replace(package,tile_ir=a.tile_ir,target_ir=a.target_ir,
        image=a.native_image,descriptor=a.launch_descriptor)

@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("bias",[False,True])
@pytest.mark.parametrize("asynchronous",[False,True])
def test_bounded_serialized_packages_and_private_residuals(bias,asynchronous,monkeypatch):
    if not nvidia_cuda_host_ready():pytest.skip("exact SM120 CUDA host required")
    pair=native.package_scheduled_checkpoint_pair(scheduled(bias=bias),scheduled(True,bias),
        pipeline_name="tessera-nvidia-pipeline-sm120")
    pair=replace(pair,forward=restore(pair.forward),backward=restore(pair.backward))
    forward,backward=artifact(pair.forward),artifact(pair.backward)
    hashes=(pair.forward.image.image_digest,pair.backward.image.image_digest)
    with NvidiaDeviceSession() as session:
        import subprocess
        monkeypatch.setattr(subprocess,"run",lambda *a,**k:pytest.fail("compiler recovery during serialized reuse"))
        held=[]
        try:
            for index,(sq,sk) in enumerate(((1,1),(3,4),(9,11),(9,1),(1,11),(3,4))):
                dims=(1,2,1,sq,sk,8,6);rng=np.random.default_rng(500+index)
                q,k,v,seed=[rng.normal(0,.2,shape).astype("f4") for shape in
                    ((1,2,sq,8),(1,1,sk,8),(1,1,sk,6),(1,2,sq,6))]
                score_bias=rng.normal(0,.1,(1,2,sq,sk)).astype("f4") if bias else None
                want,lse,grad=_reference(q,k,v,seed,score_bias,scale=.5)
                scalars=dict(zip(("B","Hq","Hkv","Sq","Sk","D","Dv"),dims,strict=True))
                args=dict(q=q,k=k,v=v,o=np.empty_like(want,dtype="f4"),lse=np.empty_like(lse,dtype="f4"),**scalars)
                if bias:args["bias"]=score_bias
                receipt=rt.launch(forward,args);assert receipt["ok"],receipt
                assert receipt["execution_kind"]=="native_gpu"
                np.testing.assert_allclose(args["o"],want,atol=4e-5,rtol=4e-5)
                np.testing.assert_allclose(args["lse"],lse,atol=4e-5,rtol=4e-5)
                reverse={**args,"do":seed,"dq":np.empty_like(q),"dk":np.empty_like(k),"dv":np.empty_like(v)}
                receipt=rt.launch(backward,reverse);assert receipt["ok"],receipt
                for name,wanted in zip(("dq","dk","dv"),grad,strict=True):
                    np.testing.assert_allclose(reverse[name],wanted,atol=4e-5,rtol=4e-5)
                resident=[session.upload(x) for x in (q,k,v)]
                rb=session.upload(score_bias) if bias else None
                rs=session.upload(seed);assert session.synchronize()==0
                frame=pair.capture(*resident,bias=rb,asynchronous=asynchronous);held.append(frame)
                assert frame.dims==dims
                frame.wait_on(session.stream);assert session.synchronize()==0
                np.testing.assert_allclose(download(session,frame.primal),want,atol=4e-5,rtol=4e-5)
                old=frame.backward(rs);frame.wait_on(session.stream);assert session.synchronize()==0
                for got,wanted in zip(old,grad,strict=True):
                    np.testing.assert_allclose(download(session,got),wanted,atol=4e-5,rtol=4e-5)
                seed2=seed*-.7;rs2=session.upload(seed2);assert session.synchronize()==0
                _,_,grad2=_reference(q,k,v,seed2,score_bias,scale=.5)
                new=frame.backward(rs2);frame.wait_on(session.stream);assert session.synchronize()==0
                for got,wanted,prior,oldwant in zip(new,grad2,old,grad,strict=True):
                    np.testing.assert_allclose(download(session,got),wanted,atol=4e-5,rtol=4e-5)
                    np.testing.assert_allclose(download(session,prior),oldwant,atol=4e-5,rtol=4e-5)
                # Different-shape generations remain live together.
                if index==0:first=(frame,old,grad)
                else:
                    first[0].wait_on(session.stream);assert session.synchronize()==0
                    for got,wanted in zip(first[1],first[2],strict=True):
                        np.testing.assert_allclose(download(session,got),wanted,atol=4e-5,rtol=4e-5)
                assert hashes==(pair.forward.image.image_digest,pair.backward.image.image_digest)
            class Beyond:
                __cuda_array_interface__={"shape":(1,2,10,8),"typestr":"<f4","data":(256,False),"version":3,"stream":session.stream}
            with pytest.raises(ValueError,match="envelope"):pair.capture(Beyond(),resident[1],resident[2],bias=rb)
        finally:
            for frame in reversed(held):frame.close()

@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("bias",[False,True])
@pytest.mark.parametrize("seeded",[False,True])
@pytest.mark.parametrize("compact",[False,True])
def test_bounded_seeded_compact_residual_numerics(bias,seeded,compact):
    if not nvidia_cuda_host_ready():pytest.skip("exact SM120 CUDA host required")
    pair=native.package_scheduled_checkpoint_pair(
        scheduled(bias=bias,seeded=seeded,compact=compact),
        scheduled(True,bias,seeded=seeded,compact=compact),
        pipeline_name="tessera-nvidia-pipeline-sm120")
    pair=replace(pair,forward=restore(pair.forward),backward=restore(pair.backward))
    roles=(0,2) if compact else (0,1,2)
    with NvidiaDeviceSession() as session:
      for sq,sk in ((3,4),(9,11)):
        rng=np.random.default_rng(120+sq+sk)
        q,k,v,seed,row_seed=[rng.normal(0,.2,shape).astype("f4") for shape in
            ((1,2,sq,8),(1,1,sk,8),(1,1,sk,6),(1,2,sq,6),(1,2,sq))]
        if not seeded:row_seed.fill(0)
        score_bias=rng.normal(0,.1,(1,2,sq,sk)).astype("f4") if bias else None
        q64,k64,v64,seed64,row64=[x.astype(np.float64) for x in (q,k,v,seed,row_seed)]
        kr=np.repeat(k64,2,axis=1);vr=np.repeat(v64,2,axis=1)
        scores=.5*(q64@kr.swapaxes(-1,-2))
        if bias:scores+=score_bias.astype(np.float64)
        mask=np.arange(sk)[None,:]<=np.arange(sq)[:,None]+max(sk-sq,0)
        scores=np.where(mask,scores,-np.inf)
        maximum=scores.max(axis=-1,keepdims=True);weight=np.exp(scores-maximum)
        denom=weight.sum(axis=-1,keepdims=True);prob=weight/denom
        wanted=prob@vr;lse=(maximum+np.log(denom))[...,0]
        dp=seed64@vr.swapaxes(-1,-2)
        ds=prob*(dp-(prob*dp).sum(axis=-1,keepdims=True)+row64[...,None])
        gradients=(.5*(ds@kr),(.5*(ds.swapaxes(-1,-2)@q64)).sum(axis=1,keepdims=True),
            (prob.swapaxes(-1,-2)@seed64).sum(axis=1,keepdims=True))
        inputs=[session.upload(x) for x in (q,k,v)]
        rb=session.upload(score_bias) if bias else None
        rs=session.upload(seed);rl=session.upload(row_seed);assert session.synchronize()==0
        with pair.capture(*inputs,bias=rb,asynchronous=True) as frame:
            frame.wait_on(session.stream);assert session.synchronize()==0
            primal=frame.primal
            if seeded:
                np.testing.assert_allclose(download(session,primal[0]),wanted,rtol=4e-5,atol=4e-5)
                np.testing.assert_allclose(download(session,primal[1]),lse,rtol=4e-5,atol=4e-5)
            else:np.testing.assert_allclose(download(session,primal),wanted,rtol=4e-5,atol=4e-5)
            actual=frame.backward((rs,rl) if seeded else rs)
            frame.wait_on(session.stream);assert session.synchronize()==0
            for got,role in zip(actual,roles,strict=True):
                np.testing.assert_allclose(download(session,got),gradients[role],rtol=4e-5,atol=4e-5)
