"""Seeded compact/physical-bias native Schedule package proofs."""
from __future__ import annotations
import json
import math
from dataclasses import replace
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler import scheduled_checkpoint as checkpoint
from tessera.compiler.nvidia_native import package_scheduled_checkpoint
from tessera.compiler.scheduled_matmul import find_tessera_opt,run_tessera_opt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.lse_cotangent_contract import lse_cotangent_contract
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lse_cotangent_native import oracle


def source(shape,physical,causal,mask=None,launch='packed_v1',threads=128):
    # Diagnostic authored MLIR fixture. Production still consumes native IR.
    names=('dO','q','k','v','output','bias','lse','dq','dk','dv','dbias')
    text=checkpoint._graph_text(names,shape,1/math.sqrt(shape[5]),causal,True,True,True,physical)
    b,hq,_,sq,_,_,_=shape;lse=f'tensor<{b}x{hq}x{sq}xf32>'
    text=text.replace(f'%arg6: {lse}',f'%arg6: {lse}, %arg7: {lse}',1)
    text=text.replace('"tessera_attn.checkpoint_backward"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6)',
                      '"tessera_attn.checkpoint_backward"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7)')
    text=text.replace('{scale =','{lse_cotangent = true, scale =')
    text=text.replace(json.dumps(names[:7]),json.dumps((*names[:7],'row_seed')))
    lines=text.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('      : '):lines[i]=line.replace(') ->',f', {lse}) ->',1)
    text='\n'.join(lines)+'\n'
    if mask is not None:
        activity=', '.join(str((mask>>i)&1) for i in range(4))
        text=text.replace('attributes {\n    tessera.argument_bindings',
            f'attributes {{\n    tessera.checkpoint_gradient_activity = array<i64: {activity}>, '
            f'tessera.checkpoint_gradient_output = "compact_v1", '
            f'tessera.checkpoint_gradient_launch = "{launch}", '
            f'tessera.checkpoint_gradient_threads = {threads} : i64,\n    tessera.argument_bindings')
    tool=find_tessera_opt();assert tool is not None
    schedule=run_tessera_opt(tool,text,'--tessera-graph-to-schedule')
    tile=run_tessera_opt(tool,schedule,'--tessera-schedule-to-tile')
    return checkpoint._decode_checkpoint(text,schedule,tile,backward=True,
        compact_gradients=mask is not None,compact_launch=launch if mask is not None else 'packed_v1',
        compact_threads=threads if mask is not None else 128)


def fixture(mask=None,launch='packed_v1',threads=128,physical=None,causal=True):
    shape=(2,4,2,5,7,4,3)
    physical=physical or (1,1,1,7)
    scheduled=source(shape,physical,causal,mask,launch,threads)
    package=package_scheduled_checkpoint(scheduled,pipeline_name='tessera-nvidia-pipeline-sm120')
    artifact=rt.RuntimeArtifact(target_ir=package.target_ir,native_image=package.image,
                                launch_descriptor=package.descriptor,metadata={'target':'nvidia_sm120'})
    artifact=rt.RuntimeArtifact.from_json(artifact.to_json())
    b,hq,hkv,sq,sk,d,dv=shape;rng=np.random.default_rng(507015)
    q,k,v,do=[(rng.normal(size=s)*.2).astype(np.float32) for s in
              ((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),(b,hq,sq,dv))]
    seed=(rng.normal(size=(b,hq,sq))*.3).astype(np.float32)
    if mask in (4,8):do.fill(0)  # Pure row-seed: dV must remain zero.
    bias=(rng.normal(size=physical)*.2).astype(np.float32)
    output,lse,logical=oracle(q,k,v,do,seed,causal,np.broadcast_to(bias,(b,hq,sq,sk)))
    dbias=logical[3]
    for axis,extent in enumerate(physical):
        if extent==1 and dbias.shape[axis]!=1:dbias=dbias.sum(axis=axis,keepdims=True)
    expected=(*logical[:3],dbias)
    roles=list(range(4)) if mask is None else [i for i in range(4) if mask&(1<<i)]
    expected=tuple(expected[i] for i in roles)
    descriptor=artifact.launch_descriptor;assert descriptor is not None
    dims,shapes,input_count=lse_cotangent_contract(descriptor)
    values=[do,q,k,v,output,bias,lse,seed]+[np.full(x.shape,np.nan,np.float32) for x in expected]
    assert tuple(x.shape for x in values)==shapes
    args={x.name:value for x,value in zip(descriptor.buffers,values,strict=True)}
    args.update(dict(zip((x.name for x in descriptor.scalars),dims,strict=True)))
    return artifact,args,expected,input_count


@pytest.mark.parametrize('mask',range(1,16))
@pytest.mark.parametrize('launch',['packed_v1','logical_v1'])
@pytest.mark.parametrize('threads',[64,128])
def test_seeded_compact_all_activity_masks(mask,launch,threads):
    if not nvidia_cuda_host_ready():pytest.skip('requires exact SM120 host and toolchain')
    run_case(mask,launch,threads,causal=bool(mask&1))


@pytest.mark.parametrize('physical',[(2,4,5,7),(1,1,1,7),(2,4,5,1)])
@pytest.mark.parametrize('causal',[False,True])
def test_seeded_complete_bias_gradient(physical,causal):
    if not nvidia_cuda_host_ready():pytest.skip('requires exact SM120 host and toolchain')
    run_case(physical=physical,causal=causal)


def run_case(mask=None,launch='packed_v1',threads=128,physical=None,causal=True):
    artifact,args,expected,input_count=fixture(mask,launch,threads,physical,causal)
    d=artifact.launch_descriptor;i=artifact.native_image
    assert d is not None and i is not None
    outputs=d.buffers[input_count:]
    result=rt.launch(artifact,args);assert result['ok'],result
    assert result['execution_kind']=='native_gpu'
    for binding,want in zip(outputs,expected,strict=True):
        np.testing.assert_allclose(args[binding.name],want,rtol=4e-5,atol=4e-5)
        args[binding.name].fill(np.nan)
    latency=rt._nvidia_native_descriptor_device_latency(i,d,args,warmup=2,reps=10)
    assert math.isfinite(latency) and latency>0
    for binding,want in zip(outputs,expected,strict=True):
        np.testing.assert_allclose(args[binding.name],want,rtol=4e-5,atol=4e-5)
    with NvidiaDeviceSession() as session:
        resident={x.name:session.upload(args[x.name]) for x in d.buffers}
        resident.update({x.name:args[x.name] for x in d.scalars})
        for x in outputs:resident[x.name]=session.upload(np.full(args[x.name].shape,np.nan,np.float32))
        result=rt.launch(artifact,resident,stream=session.stream);assert result['ok'],result
        latency=rt._nvidia_native_descriptor_resident_device_latency(i,d,resident,stream=session.stream,warmup=2,reps=10)
        assert math.isfinite(latency) and latency>0
        session.synchronize()
        for x,want in zip(outputs,expected,strict=True):
            np.testing.assert_allclose(session.download(resident[x.name]),want,rtol=4e-5,atol=4e-5)


@pytest.mark.parametrize('defect',['entry','activity','roles','seed','threads','guard','scale','causal','bias_reduction'])
def test_seeded_compact_rejects_changed_physical_contract(defect):
    if not nvidia_cuda_host_ready():pytest.skip('requires matching SM120 compiler')
    artifact,_,_,_=fixture(9)
    d=artifact.launch_descriptor;p=dict(d.provenance)
    if defect=='entry':d=replace(d,entry_symbol=d.entry_symbol+'x')
    elif defect=='activity':p['gradient_activity']=[0,1,0,1]
    elif defect=='roles':p['physical_gradient_roles']=[1,3]
    elif defect=='seed':p['lse_cotangent']=False
    elif defect=='threads':p['gradient_block_threads']=32
    elif defect=='guard':d=replace(d,shape_guards=d.shape_guards[:-1])
    elif defect=='scale':p['scale']*=2
    elif defect=='causal':p['causal']=not p['causal']
    elif defect=='bias_reduction':p['bias_gradient_reduction']='unordered'
    d=replace(d,provenance=p)
    with pytest.raises(ValueError):lse_cotangent_contract(d)
