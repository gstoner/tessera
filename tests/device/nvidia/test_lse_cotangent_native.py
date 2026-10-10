"""Numerical proof for the native Tile LSE-cotangent consumer, not public AD closure."""
from __future__ import annotations
import ctypes as ct
import math
import numpy as np
import pytest
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.nvidia_native import _compile_tile_ir
from tests._support.nvidia import nvidia_cuda_host_ready


def tile_fixture(scale, causal, bias, *, lse_cotangent=True):
    names=['do','q','k','v','output']+(['bias'] if bias else [])+['row_lse']+(['row_seed'] if lse_cotangent else [])+['dq','dk','dv']+(['db'] if bias else [])
    dimensions=['b','hq','hkv','sq','sk','d','vwidth']
    signature=', '.join('%'+n+': !llvm.ptr' for n in names)+', '+', '.join('%'+n+': i64' for n in dimensions)
    operands=', '.join('%'+n for n in names+dimensions)
    types=', '.join(['!llvm.ptr']*len(names)+['i64']*7)
    return f'''module {{
      llvm.func @test_lse_cotangent({signature}) attributes {{nvvm.kernel}} {{
        tile.attention_backward_kernel {operands} {{
          storage = "f32", accum = "f32", scale = {scale} : f32,
          causal = {str(causal).lower()}, bias = {str(bias).lower()},
          window_left = -1 : i64, window_right = -1 : i64,
          softcap = 0.0 : f32, dropout_p = 0.0 : f32, dropout_seed = 0 : i64,
          route = "deterministic_direct", deterministic = true,
          workspace_bytes = 0 : i64, workspace_owner = "output_element",
          lse_checkpoint = "saved", saved_output = true, lse_cotangent = {str(lse_cotangent).lower()},
          bias_gradient = {str(bias).lower()}
        }} : {types}
        llvm.return
      }}
    }}'''


def oracle(q,k,v,do,row_seed,causal,bias):
    q,k,v,do,row_seed=[x.astype(np.float64) for x in (q,k,v,do,row_seed)]
    b,hq,sq,d=q.shape;hkv=k.shape[1];sk=k.shape[2];group=hq//hkv
    kr=np.repeat(k,group,axis=1);vr=np.repeat(v,group,axis=1)
    scores=q@kr.swapaxes(-1,-2)/math.sqrt(d)
    if bias is not None:scores+=bias
    if causal:scores=np.where(np.arange(sk)[None,:]<=np.arange(sq)[:,None]+max(sk-sq,0),scores,-np.inf)
    maximum=scores.max(axis=-1,keepdims=True)
    weight=np.exp(scores-maximum);denom=weight.sum(axis=-1,keepdims=True)
    p=weight/denom;output=p@vr;lse=(maximum+np.log(denom))[...,0]
    dp=do@vr.swapaxes(-1,-2)
    ds=p*(dp-(p*dp).sum(axis=-1,keepdims=True)+row_seed[...,None])
    dq=ds@kr/math.sqrt(d)
    dk=(ds.swapaxes(-1,-2)@q/math.sqrt(d)).reshape(b,hkv,group,sk,d).sum(axis=2)
    dv=(p.swapaxes(-1,-2)@do).reshape(b,hkv,group,sk,v.shape[-1]).sum(axis=2)
    return output.astype(np.float32),lse.astype(np.float32),(dq,dk,dv,*((ds,) if bias is not None else ()))


def execute(ptx,values,shapes,dims, *, timed_expected=None, samples=5, reps=100, entry='test_lse_cotangent'):
    P,U=ct.c_void_p,ct.c_uint
    driver=ct.CDLL('libcuda.so.1')
    def bind(name,types):
        fn=getattr(driver,name);fn.argtypes=types;fn.restype=ct.c_int;return fn
    load=bind('cuModuleLoadData',[ct.POINTER(P),P])
    find=bind('cuModuleGetFunction',[ct.POINTER(P),P,ct.c_char_p])
    launch=bind('cuLaunchKernel',[P]+[U]*7+[P,ct.POINTER(P),ct.POINTER(P)])
    unload=bind('cuModuleUnload',[P])
    with NvidiaDeviceSession() as session:
        inputs=[session.upload(x) for x in values]
        outputs=[session.upload(np.full(s,np.nan,np.float32)) for s in shapes]
        session.synchronize()
        module,function=P(),P();blob=ct.create_string_buffer(ptx.encode())
        assert load(ct.byref(module),ct.cast(blob,P))==0
        try:
            assert find(ct.byref(function),module,entry.encode())==0
            args=[P(x.ptr) for x in inputs+outputs]+[ct.c_int64(x) for x in dims]
            pointers=(P*len(args))(*(ct.cast(ct.pointer(x),P) for x in args))
            count=sum(math.prod(s) for s in shapes)
            assert launch(function,(count+127)//128,1,1,128,1,1,0,P(session.stream),pointers,None)==0
            session.synchronize()
            if timed_expected is None:
                return [session.download(x) for x in outputs]
            create=bind('cuEventCreate',[ct.POINTER(P),U])
            record=bind('cuEventRecord',[P,P])
            synchronize=bind('cuEventSynchronize',[P])
            elapsed=bind('cuEventElapsedTime',[ct.POINTER(ct.c_float),P,P])
            destroy=bind('cuEventDestroy_v2',[P])
            start,stop=P(),P()
            assert create(ct.byref(start),0)==0
            try:
                assert create(ct.byref(stop),0)==0
                try:
                    timings=[];errors=[]
                    for _ in range(samples):
                        for output in outputs:
                            poison=np.full(output.shape,np.nan,np.float32)
                            assert session.lib.tessera_nvidia_device_upload(
                                P(output.ptr),P(poison.ctypes.data),poison.nbytes,P(session.stream))==0
                        session.synchronize()
                        assert record(start,P(session.stream))==0
                        for _ in range(reps):
                            assert launch(function,(count+127)//128,1,1,128,1,1,0,P(session.stream),pointers,None)==0
                        assert record(stop,P(session.stream))==0
                        assert synchronize(stop)==0
                        duration=ct.c_float()
                        assert elapsed(ct.byref(duration),start,stop)==0
                        timings.append(duration.value/reps)
                        actual=[session.download(x) for x in outputs]
                        for result,reference in zip(actual,timed_expected,strict=True):
                            np.testing.assert_allclose(result,reference,rtol=4e-5,atol=4e-5)
                        errors.append(max(float(np.max(np.abs(x-y))) for x,y in zip(actual,timed_expected,strict=True)))
                    return dict(device_event_samples_ms=timings,max_abs_errors=errors,reps=reps,
                                timing_scope="resident CUDA-event windows around Python diagnostic driver launches; includes driver gaps; allocation/upload/download/oracle excluded")
                finally:
                    assert destroy(stop)==0
            finally:
                assert destroy(start)==0
        finally:
            assert unload(module)==0


@pytest.mark.parametrize('shape',[(1,2,1,3,5,4,3),(1,2,1,5,3,4,3),(2,4,2,7,9,8,6)])
@pytest.mark.parametrize('causal',[False,True])
@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('seed_mode',['output_only','lse_only','mixed'])
def test_native_lse_cotangent(shape,causal,bias,seed_mode):
    if not nvidia_cuda_host_ready():pytest.skip('requires exact SM120 host and toolchain')
    b,hq,hkv,sq,sk,d,dv=shape;rng=np.random.default_rng(121203)
    q,k,v,do=[(rng.normal(size=s)*.2).astype(np.float32) for s in ((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),(b,hq,sq,dv))]
    row_seed=(rng.normal(size=(b,hq,sq))*.3).astype(np.float32)
    if seed_mode=='output_only':row_seed.fill(0)
    if seed_mode=='lse_only':do.fill(0)
    score_bias=(rng.normal(size=(b,hq,sq,sk))*.2).astype(np.float32) if bias else None
    output,lse,expected=oracle(q,k,v,do,row_seed,causal,score_bias)
    _,ptx,*_=_compile_tile_ir(tile_fixture(1/math.sqrt(d),causal,bias),'test_lse_cotangent')
    values=[do,q,k,v,output]+([score_bias] if bias else [])+[lse,row_seed]
    actual=execute(ptx,values,[x.shape for x in expected],shape)
    for result,reference in zip(actual,expected,strict=True):
        np.testing.assert_allclose(result,reference,rtol=4e-5,atol=4e-5)


@pytest.mark.parametrize('before,after,message',[
    ('lse_cotangent = true','lse_cotangent = "yes"','bool'),
    ('saved_output = true','saved_output = false','requires saved output'),
    ('lse_checkpoint = "saved"','lse_checkpoint = "recompute"','requires saved output'),
    ('storage = "f32"','storage = "f16"','requires f32'),
    ('route = "deterministic_direct"','route = "deterministic_split_reduced"','deterministic direct'),
])
def test_lse_cotangent_tile_contract(before,after,message):
    import subprocess
    from tessera.compiler.nvidia_native import _tool
    tool=_tool('tessera-nvidia-opt')
    if tool is None:pytest.skip('requires matching NVIDIA compiler')
    result=subprocess.run([str(tool),'--tessera-lower-to-nvidia-sm120'],
                          input=tile_fixture(.5,False,False).replace(before,after),
                          text=True,capture_output=True)
    assert result.returncode!=0
    assert message in result.stderr,result.stderr


@pytest.mark.parametrize('shape',[(1,2,1,3,5,4,3),(1,2,1,5,3,4,3),(2,4,2,7,9,8,6)])
@pytest.mark.parametrize('causal',[False,True])
@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('seed_mode',['output_only','lse_only','mixed'])
def test_graph_scheduled_lse_cotangent(shape,causal,bias,seed_mode):
    if not nvidia_cuda_host_ready():pytest.skip('requires exact SM120 host and toolchain')
    from tessera.compiler.scheduled_checkpoint import lower_checkpoint_graph
    from tests.unit.test_lse_cotangent_schedule import module
    artifact=lower_checkpoint_graph(module(bias,shape,causal),backward=True)
    b,hq,hkv,sq,sk,d,dv=shape;rng=np.random.default_rng(121203)
    q,k,v,do=[(rng.normal(size=s)*.2).astype(np.float32) for s in ((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),(b,hq,sq,dv))]
    row_seed=(rng.normal(size=(b,hq,sq))*.3).astype(np.float32)
    if seed_mode=='output_only':row_seed.fill(0)
    if seed_mode=='lse_only':do.fill(0)
    score_bias=(rng.normal(size=(b,hq,sq,sk))*.2).astype(np.float32) if bias else None
    output,lse,expected=oracle(q,k,v,do,row_seed,causal,score_bias)
    _,ptx,*_=_compile_tile_ir(artifact.tile_ir,artifact.entry)
    values=[do,q,k,v,output]+([score_bias] if bias else [])+[lse,row_seed]
    actual=execute(ptx,values,[x.shape for x in expected[:3]],shape,entry=artifact.entry)
    for result,reference in zip(actual,expected[:3],strict=True):
        np.testing.assert_allclose(result,reference,rtol=4e-5,atol=4e-5)


@pytest.mark.parametrize('bias',[False,True])
@pytest.mark.parametrize('causal',[False,True])
def test_seeded_native_bridge_host_resident_and_event(bias,causal):
    if not nvidia_cuda_host_ready():pytest.skip('requires exact SM120 host and toolchain')
    from tessera import runtime as rt
    from tessera.compiler.scheduled_checkpoint import lower_checkpoint_graph
    from tests.unit.test_lse_cotangent_schedule import module
    shape=(1,2,1,3,5,4,3)
    artifact=lower_checkpoint_graph(module(bias,shape,causal),backward=True)
    _,ptx,*_=_compile_tile_ir(artifact.tile_ir,artifact.entry)
    b,hq,hkv,sq,sk,d,dv=shape;rng=np.random.default_rng(5070)
    q,k,v,do=[(rng.normal(size=z)*.2).astype(np.float32) for z in
               ((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),(b,hq,sq,dv))]
    seed=(rng.normal(size=(b,hq,sq))*.3).astype(np.float32)
    score_bias=(rng.normal(size=(b,hq,sq,sk))*.2).astype(np.float32) if bias else None
    output,lse,expected=oracle(q,k,v,do,seed,causal,score_bias)
    inputs=[do,q,k,v,output]+([score_bias] if bias else [])+[lse,seed]
    outputs=[np.full(x.shape,np.nan,np.float32) for x in expected[:3]]
    values=inputs+outputs
    pointers=(ct.c_void_p*len(values))(*(x.ctypes.data for x in values))
    dims=(ct.c_int64*7)(*shape)
    lib=rt._load_nvidia_ptx_launch();assert lib is not None
    assert rt._register_nvidia_ptx(lib,artifact.entry,ptx)==0
    for bad_count in [len(values)-2,len(values)+3]:
        assert lib.tessera_nvidia_ptx_invoke(artifact.entry.encode(),pointers,bad_count,dims,7)==5
    assert lib.tessera_nvidia_ptx_invoke(artifact.entry.encode(),pointers,len(values),dims,7)==0
    for got,want in zip(outputs,expected[:3],strict=True):
        np.testing.assert_allclose(got,want,rtol=4e-5,atol=4e-5)
    for x in outputs:x.fill(np.nan)
    latency=ct.c_float()
    assert lib.tessera_nvidia_ptx_benchmark(artifact.entry.encode(),pointers,len(values),dims,7,2,10,ct.byref(latency))==0
    assert np.isfinite(latency.value) and latency.value>0
    for got,want in zip(outputs,expected[:3],strict=True):
        np.testing.assert_allclose(got,want,rtol=4e-5,atol=4e-5)
    with NvidiaDeviceSession() as session:
        resident=[session.upload(x) for x in inputs]+[session.upload(np.full(x.shape,np.nan,np.float32)) for x in outputs]
        session.synchronize()
        addresses=(ct.c_void_p*len(resident))(*(x.ptr for x in resident))
        assert lib.tessera_nvidia_ptx_invoke_resident(artifact.entry.encode(),addresses,len(resident),dims,7,ct.c_void_p(session.stream))==0
        session.synchronize()
        for got,want in zip(resident[len(inputs):],expected[:3],strict=True):
            np.testing.assert_allclose(session.download(got),want,rtol=4e-5,atol=4e-5)
