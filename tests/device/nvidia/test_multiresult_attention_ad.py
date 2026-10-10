"""Native reverse differentiation of semantic Graph attention O/LSE tuples."""
from __future__ import annotations
import math
import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler.scheduled_checkpoint import lower_generated_checkpoint
from tessera.compiler.nvidia_native import package_scheduled_checkpoint
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lse_cotangent_native import oracle


def source(shape=(1,2,1,3,5,4,3),bias=False,causal=True,wrt=None):
    b,hq,hkv,sq,sk,d,dv=shape
    def tensor(z):return 'tensor<'+'x'.join(map(str,z))+'xf32>'
    q,k,v,o,lse=[tensor(z) for z in [(b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),(b,hq,sq,dv),(b,hq,sq)]]
    names=['q','k','v'];types=[q,k,v]
    if bias:names.append('bias');types.append(tensor((1,1,1,sk)))
    args=', '.join('%'+n+': '+t for n,t in zip(names,types,strict=True))
    operands=', '.join('%'+n for n in names)
    activity='' if wrt is None else ', tessera.autodiff.wrt_indices = ['+', '.join(str(x)+' : i64' for x in wrt)+']'
    return f'''module attributes {{tessera.target = "nvidia_sm120", tessera.arch = "sm_120"}} {{
  func.func @attention({args}) -> ({o}, {lse}) attributes {{tessera.autodiff = "reverse"{activity}}} {{
    %o, %lse = "tessera.flash_attn"({operands}) {{head_dim = {d} : i64,
      causal = {str(causal).lower()}, lse_checkpoint = "saved",
      operandSegmentSizes = array<i32: 1, 1, 1, {int(bias)}>}} : ({', '.join(types)}) -> ({o}, {lse})
    return %o, %lse : {o}, {lse}
  }}
}}
'''


def compile_pair(shape,bias,causal,compact=False):
    wrt=(0,3) if bias else (0,2)
    text=source(shape,bias,causal,wrt if compact else None)
    forward=lower_generated_checkpoint(text)
    backward=lower_generated_checkpoint(text,backward=True,prune_inactive=compact,compact_gradients=compact)
    assert backward.lse_cotangent
    assert backward.names[6+int(bias)]=='row_seed'
    assert 'tessera.flash_attn' in backward.tile_ir  # retained paired semantic lineage
    assert 'lse_cotangent = true' in backward.tile_ir
    result=[]
    for scheduled in (forward,backward):
        package=package_scheduled_checkpoint(scheduled,pipeline_name='tessera-nvidia-pipeline-sm120')
        artifact=rt.RuntimeArtifact(graph_ir=scheduled.graph_ir,schedule_ir=scheduled.schedule_ir,
            tile_ir=scheduled.tile_ir,target_ir=package.target_ir,native_image=package.image,
            launch_descriptor=package.descriptor,metadata={'target':'nvidia_sm120'})
        result.append(rt.RuntimeArtifact.from_json(artifact.to_json()))
    return result,backward


@pytest.mark.parametrize('shape',[(1,2,1,3,5,4,3),(1,2,1,5,3,4,3),(2,4,2,7,9,8,6)])
@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('causal',[False,True])
@pytest.mark.parametrize('seed_mode',['output_only','lse_only','mixed'])
@pytest.mark.parametrize('compact',[False,True])
def test_automatic_multiresult_reverse_package(shape,bias,causal,seed_mode,compact):
    if not nvidia_cuda_host_ready():pytest.skip('requires exact SM120 host and matching compiler')
    (forward,backward),scheduled=compile_pair(shape,bias,causal,compact)
    b,hq,hkv,sq,sk,d,dv=shape;rng=np.random.default_rng(507020)
    q,k,v,do=[(rng.normal(size=z)*.2).astype(np.float32) for z in
        [(b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),(b,hq,sq,dv)]]
    seed=(rng.normal(size=(b,hq,sq))*.3).astype(np.float32)
    if seed_mode=='output_only':seed.fill(0)
    if seed_mode=='lse_only':do.fill(0)
    score_bias=(rng.normal(size=(1,1,1,sk))*.2).astype(np.float32) if bias else None
    expected_o,expected_lse,gradients=oracle(q,k,v,do,seed,causal,
        np.broadcast_to(score_bias,(b,hq,sq,sk)) if bias else None)
    expected=list(gradients[:3])
    if bias:expected.append(gradients[3].sum(axis=(0,1,2),keepdims=True))
    roles=[i for i,a in enumerate(scheduled.gradient_activity) if a] if compact else list(range(3+int(bias)))
    expected=[expected[i] for i in roles]
    scalars=dict(zip(('B','Hq','Hkv','Sq','Sk','D','Dv'),shape,strict=True))
    if bias:scalars.update(dict(zip(('BiasB','BiasH','BiasQ','BiasK'),(1,1,1,sk),strict=True)))
    with NvidiaDeviceSession() as session:
        resident={name:session.upload(value) for name,value in [('q',q),('k',k),('v',v)]}
        if bias:resident['bias']=session.upload(score_bias)
        resident['output']=session.upload(np.full(expected_o.shape,np.nan,np.float32))
        resident['lse']=session.upload(np.full(expected_lse.shape,np.nan,np.float32))
        result=rt.launch(forward,{**resident,**scalars},stream=session.stream)
        assert result['ok'],result
        np.testing.assert_allclose(session.download(resident['output']),expected_o,rtol=4e-5,atol=4e-5)
        np.testing.assert_allclose(session.download(resident['lse']),expected_lse,rtol=4e-5,atol=4e-5)
        resident['dO']=session.upload(do);resident['row_seed']=session.upload(seed)
        descriptor=backward.launch_descriptor;assert descriptor is not None
        outputs=[x for x in descriptor.buffers if x.direction=='output']
        for binding,want in zip(outputs,expected,strict=True):
            resident[binding.name]=session.upload(np.full(want.shape,np.nan,np.float32))
        result=rt.launch(backward,{**resident,**scalars},stream=session.stream)
        assert result['ok'],result
        assert result['execution_kind']=='native_gpu'
        latency=rt._nvidia_native_descriptor_resident_device_latency(backward.native_image,descriptor,
            {**resident,**scalars},stream=session.stream,warmup=2,reps=10)
        assert math.isfinite(latency) and latency>0
        for binding,want in zip(outputs,expected,strict=True):
            np.testing.assert_allclose(session.download(resident[binding.name]),want,rtol=4e-5,atol=4e-5)


def fixture(shape=(1,2,1,3,5,4,3),bias=False,causal=True,seed_mode='mixed',compact=False,stage='backward'):
    (forward,backward),scheduled=compile_pair(shape,bias,causal,compact)
    b,hq,hkv,sq,sk,d,dv=shape;rng=np.random.default_rng(507020)
    q,k,v,do=[(rng.normal(size=z)*.2).astype(np.float32) for z in
        [(b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),(b,hq,sq,dv)]]
    seed=(rng.normal(size=(b,hq,sq))*.3).astype(np.float32)
    if seed_mode=='output_only':seed.fill(0)
    if seed_mode=='lse_only':do.fill(0)
    score_bias=(rng.normal(size=(1,1,1,sk))*.2).astype(np.float32) if bias else None
    expected_o,expected_lse,gradients=oracle(q,k,v,do,seed,causal,
        np.broadcast_to(score_bias,(b,hq,sq,sk)) if bias else None)
    scalars=dict(zip(('B','Hq','Hkv','Sq','Sk','D','Dv'),shape,strict=True))
    if bias:scalars.update(dict(zip(('BiasB','BiasH','BiasQ','BiasK'),(1,1,1,sk),strict=True)))
    args={'q':q,'k':k,'v':v,'output':np.full(expected_o.shape,np.nan,np.float32),
          'lse':np.full(expected_lse.shape,np.nan,np.float32)}
    if bias:args['bias']=score_bias
    args.update(scalars)
    if stage=='forward':return forward,args,(expected_o,expected_lse),3+int(bias)
    if stage!='backward':raise ValueError('unknown generated attention stage')
    result=rt.launch(forward,args);assert result['ok'],result
    np.testing.assert_allclose(args['output'],expected_o,rtol=4e-5,atol=4e-5)
    np.testing.assert_allclose(args['lse'],expected_lse,rtol=4e-5,atol=4e-5)
    expected=list(gradients[:3])
    if bias:expected.append(gradients[3].sum(axis=(0,1,2),keepdims=True))
    roles=[i for i,a in enumerate(scheduled.gradient_activity) if a] if compact else list(range(3+int(bias)))
    expected=tuple(expected[i] for i in roles)
    args.update({'dO':do,'row_seed':seed})
    outputs=[x for x in backward.launch_descriptor.buffers if x.direction=='output']
    for binding,want in zip(outputs,expected,strict=True):args[binding.name]=np.full(want.shape,np.nan,np.float32)
    return backward,args,expected,7+int(bias)
