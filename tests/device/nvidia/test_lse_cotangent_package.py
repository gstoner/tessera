"""Exact-device seeded Graph package, portable replay and resident proof."""
from __future__ import annotations
import math
from dataclasses import replace
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler.driver import compile_graph_module
from tessera.compiler.canonical_compile import compile_result_from_bundle
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.lse_cotangent_contract import lse_cotangent_contract, validate_lse_cotangent_invocation
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.unit.test_lse_cotangent_schedule import module
from tests.device.nvidia.test_lse_cotangent_native import oracle


def fixture(bias=False,causal=True,shape=(1,2,1,3,5,4,3),seed_mode="mixed"):
    graph=module(bias,shape,causal)
    bundle=compile_graph_module(graph,source_origin="NVIDIA-LSE-1",target="nvidia_sm120",
                                options={"package_native":True},enable_tool_validation=False)
    artifact=compile_result_from_bundle(bundle,module=graph).to_runtime_artifact()
    artifact=rt.RuntimeArtifact.from_json(artifact.to_json())
    b,hq,hkv,sq,sk,d,dv=shape;rng=np.random.default_rng(507012)
    q,k,v,do=[(rng.normal(size=s)*.2).astype(np.float32) for s in
              ((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),(b,hq,sq,dv))]
    seed=(rng.normal(size=(b,hq,sq))*.3).astype(np.float32)
    if seed_mode=="output_only":seed.fill(0)
    if seed_mode=="lse_only":do.fill(0)
    score_bias=(rng.normal(size=(b,hq,sq,sk))*.2).astype(np.float32) if bias else None
    output,lse,expected=oracle(q,k,v,do,seed,causal,score_bias)
    arrays=[do,q,k,v,output]+([score_bias] if bias else [])+[lse,seed]+[np.full(x.shape,np.nan,np.float32) for x in expected[:3]]
    descriptor=artifact.launch_descriptor
    assert descriptor is not None
    _,_,input_count=lse_cotangent_contract(descriptor)
    args={x.name:value for x,value in zip(descriptor.buffers,arrays,strict=True)}
    args.update(dict(zip(("B","Hq","Hkv","Sq","Sk","D","Dv"),shape,strict=True)))
    return artifact,args,expected[:3],input_count


@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('causal',[False,True])
@pytest.mark.parametrize('shape',[(1,2,1,3,5,4,3),(1,2,1,5,3,4,3),(2,4,2,7,9,8,6)])
@pytest.mark.parametrize('seed_mode',['output_only','lse_only','mixed'])
def test_seeded_package_replay_host_resident_and_timing(bias,causal,shape,seed_mode,monkeypatch):
    if not nvidia_cuda_host_ready():pytest.skip('requires exact SM120 host and toolchain')
    artifact,args,expected,input_count=fixture(bias,causal,shape,seed_mode)
    # Replay must use the serialized image, without compiler recovery.
    monkeypatch.setenv('TESSERA_OPT','/nonexistent/tessera-opt')
    monkeypatch.setenv('TESSERA_NVIDIA_OPT','/nonexistent/tessera-nvidia-opt')
    result=rt.launch(artifact,args)
    assert result['ok'],result
    assert result['execution_kind']=='native_gpu'
    descriptor=artifact.launch_descriptor;image=artifact.native_image
    assert descriptor is not None and image is not None
    outputs=descriptor.buffers[input_count:]
    for x,want in zip(outputs,expected,strict=True):
        np.testing.assert_allclose(args[x.name],want,rtol=4e-5,atol=4e-5)
        args[x.name].fill(np.nan)
    latency=rt._nvidia_native_descriptor_device_latency(image,descriptor,args,warmup=2,reps=10)
    assert math.isfinite(latency) and latency>0
    for x,want in zip(outputs,expected,strict=True):
        np.testing.assert_allclose(args[x.name],want,rtol=4e-5,atol=4e-5)
    with NvidiaDeviceSession() as session:
        resident={x.name:session.upload(args[x.name]) for x in descriptor.buffers}
        for x in outputs:resident[x.name]=session.upload(np.full(args[x.name].shape,np.nan,np.float32))
        resident.update({x.name:args[x.name] for x in descriptor.scalars})
        result=rt.launch(artifact,resident,stream=session.stream)
        assert result['ok'],result
        latency=rt._nvidia_native_descriptor_resident_device_latency(image,descriptor,resident,stream=session.stream,warmup=2,reps=10)
        assert math.isfinite(latency) and latency>0
        session.synchronize()
        for x,want in zip(outputs,expected,strict=True):
            np.testing.assert_allclose(session.download(resident[x.name]),want,rtol=4e-5,atol=4e-5)


@pytest.mark.parametrize('defect',['shape','dtype','strides','alias','scalar','guards','entry','seed_policy'])
def test_seeded_package_invalid_contract_before_cuda(defect,monkeypatch):
    if not nvidia_cuda_host_ready():pytest.skip('requires matching SM120 compiler')
    artifact,args,_,input_count=fixture()
    descriptor=artifact.launch_descriptor;assert descriptor is not None
    seed=descriptor.buffers[input_count-1].name
    if defect=='shape':args[seed]=np.zeros((1,2,4),np.float32)
    elif defect=='dtype':args[seed]=args[seed].astype(np.float16)
    elif defect=='strides':args[seed]=np.zeros((1,2,6),np.float32)[:,:,::2]
    elif defect=='alias':args[descriptor.buffers[input_count].name]=args[descriptor.buffers[1].name]
    elif defect=='scalar':args['Sk']+=1
    elif defect=='guards':descriptor=replace(descriptor,shape_guards=descriptor.shape_guards[:-1])
    elif defect=='entry':descriptor=replace(descriptor,entry_symbol=descriptor.entry_symbol+'x')
    elif defect=='seed_policy':descriptor=replace(descriptor,provenance={**descriptor.provenance,'lse_cotangent':False})
    monkeypatch.setattr(rt,'_load_nvidia_ptx_launch',lambda:pytest.fail('invalid seed loaded CUDA'))
    with pytest.raises(ValueError):validate_lse_cotangent_invocation(descriptor,args,args)
    with pytest.raises(ValueError):rt._submit_nvidia_sm120_native(artifact.native_image,descriptor,args,args,None)
