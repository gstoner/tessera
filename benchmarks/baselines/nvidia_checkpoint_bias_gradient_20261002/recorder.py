"""Exact SM120 checked checkpoint bias-gradient package proof and timings."""
from __future__ import annotations
import argparse, hashlib, json, os, statistics, subprocess, time
from pathlib import Path
import numpy as np
from tessera import runtime as rt
from tessera.compiler import nvidia_native as native
from tessera.compiler.scheduled_checkpoint import lower_scheduled_checkpoint
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests._support.nvidia import nvidia_cuda_host_ready


def reference(q, k, v, bias, seed, causal):
    q,k,v,bias,seed=(x.astype(np.float64) for x in (q,k,v,bias,seed))
    b,hq,sq,d=q.shape;hkv=k.shape[1];sk=k.shape[2];group=hq//hkv
    kr=np.repeat(k,group,axis=1);vr=np.repeat(v,group,axis=1)
    score=(q@np.swapaxes(kr,-1,-2))/np.sqrt(d)+bias
    mask=(np.arange(sk)[None,:]<=np.arange(sq)[:,None]+max(sk-sq,0)
          if causal else np.ones((sq,sk),bool))
    score=np.where(mask,score,-np.inf)
    maximum=score.max(axis=-1,keepdims=True)
    weight=np.exp(score-maximum);denominator=weight.sum(axis=-1,keepdims=True)
    p=weight/denominator
    output=p@vr;lse=(maximum+np.log(denominator))[...,0]
    dp=seed@np.swapaxes(vr,-1,-2)
    db=p*(dp-(p*dp).sum(axis=-1,keepdims=True))
    dq=(db@kr)/np.sqrt(d)
    dk=(np.swapaxes(db,-1,-2)@q)/np.sqrt(d)
    dv=np.swapaxes(p,-1,-2)@seed
    dk=dk.reshape(b,hkv,group,sk,d).sum(axis=2)
    dv=dv.reshape(b,hkv,group,sk,v.shape[-1]).sum(axis=2)
    return output,lse,(dq,dk,dv,db)


def package(names, dims, causal, backward):
    scheduled=lower_scheduled_checkpoint(names,dims,float(1/np.sqrt(dims[5])),causal,
        backward=backward,bias=True,bias_gradient=backward)
    return native.package_scheduled_checkpoint(scheduled,pipeline_name="tessera-nvidia-pipeline-sm120")


def runtime(package):
    return rt.RuntimeArtifact(metadata={"target":"nvidia_sm120"},
        native_image=package.image,launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir,target_ir=package.target_ir)


def run_case(dims, causal, *, samples=3, reps=100):
    b,hq,hkv,sq,sk,d,dv=dims
    rng=np.random.default_rng(120_404)
    q,k,v,bias,seed=[(rng.normal(size=shape)*.3).astype(np.float32) for shape in
        ((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),(b,hq,sq,sk),(b,hq,sq,dv))]
    expected,lse,grad=reference(q,k,v,bias,seed,causal)
    forward=package(("q","k","v","bias","output","lse"),dims,causal,False)
    backward=package(("do","q","k","v","output","bias","lse","dq","dk","dv","dbias"),dims,causal,True)
    output=np.full(expected.shape,np.nan,np.float32);saved=np.full(lse.shape,np.nan,np.float32)
    gradients=[np.full(x.shape,np.nan,np.float32) for x in grad]
    scalars=dict(zip(("B","Hq","Hkv","Sq","Sk","D","Dv"),dims,strict=True))
    fargs={**scalars,**dict(zip([x.name for x in forward.descriptor.buffers],
        (q,k,v,bias,output,saved),strict=True))}
    bargs={**scalars,**dict(zip([x.name for x in backward.descriptor.buffers],
        (seed,q,k,v,output,bias,saved,*gradients),strict=True))}
    for pkg,args in ((forward,fargs),(backward,bargs)):
        receipt=rt.launch(runtime(pkg),args)
        if not receipt.get("ok") or receipt.get("execution_kind")!="native_gpu":
            raise RuntimeError(receipt)
    np.testing.assert_allclose(output,expected,atol=4e-5,rtol=4e-5)
    np.testing.assert_allclose(saved,lse,atol=4e-5,rtol=4e-5)
    for actual,oracle in zip(gradients,grad,strict=True):
        np.testing.assert_allclose(actual,oracle,atol=4e-5,rtol=4e-5)
    host=[]
    for _ in range(samples):
        start=time.perf_counter_ns()
        receipt=rt.launch(runtime(backward),bargs)
        if not receipt.get("ok"):raise RuntimeError(receipt)
        host.append((time.perf_counter_ns()-start)/1e6)
        for actual,oracle in zip(gradients,grad,strict=True):
            np.testing.assert_allclose(actual,oracle,atol=4e-5,rtol=4e-5)
    with NvidiaDeviceSession() as session:
        resident={**scalars}
        for binding in backward.descriptor.buffers:
            value=bargs[binding.name]
            resident[binding.name]=session.upload(
                np.full(value.shape,np.nan,np.float32) if binding.direction=="output" else value)
        device=[rt._nvidia_native_descriptor_resident_device_latency(
            backward.image,backward.descriptor,resident,stream=session.stream,
            warmup=20,reps=reps) for _ in range(samples)]
        session.synchronize()
        for binding,oracle in zip(backward.descriptor.buffers[-4:],grad,strict=True):
            np.testing.assert_allclose(session.download(resident[binding.name]),oracle,atol=4e-5,rtol=4e-5)
    return dict(shape=list(dims),causal=causal,abi=backward.descriptor.abi_id,
        max_abs_errors=[float(np.max(np.abs(x-y))) for x,y in zip(gradients,grad,strict=True)],
        device_window_samples_ms=device,device_window_median_ms=statistics.median(device),
        end_to_end_samples_ms=host,end_to_end_median_ms=statistics.median(host),
        schedule_digest=backward.descriptor.provenance["schedule_digest"],
        image_sha256=hashlib.sha256(backward.image.payload).hexdigest(),
        route="checkpoint Graph -> Schedule -> Tile -> NVIDIA Target/NVVM/PTX -> checked runtime ABI")


def record(samples,reps):
    if not nvidia_cuda_host_ready():raise RuntimeError("SM120 native host required")
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or gpu.split(",")[-1].strip()!="12.0":
        raise RuntimeError("requires one exact SM120 device")
    rows=[run_case(shape,causal,samples=samples,reps=reps)
        for shape in ((1,2,1,3,5,4,3),(1,2,1,5,3,4,3),(2,4,2,16,19,8,6))
        for causal in (False,True)]
    root=Path(__file__).resolve().parents[2]
    sources=("src/compiler/programming_model/lib/NativeCheckpoint.h",
        "src/compiler/ir/TileOps.cpp","src/compiler/tile_opt_fa4/lib/Dialect/Attn/AttnOps.cpp",
        "src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp",
        "python/tessera/compiler/scheduled_checkpoint.py","python/tessera/compiler/nvidia_native.py",
        "python/tessera/runtime.py","benchmarks/nvidia/benchmark_checkpoint_bias_gradient.py")
    return dict(schema="tessera.nvidia.checkpoint_bias_gradient.v1",architecture="sm_120",gpu=gpu,
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        bridge_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
        rows=rows,samples=samples,reps=reps,
        timing_scope="resident CUDA-event launch windows include driver gaps; end-to-end includes host staging and synchronization; no speedup claim; automatic AD export remains separate")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=3);parser.add_argument("--reps",type=int,default=100)
    args=parser.parse_args()
    if args.samples<3 or args.reps<20:raise ValueError("requires >=3 samples and >=20 reps")
    result=record(args.samples,args.reps);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("verified",len(result["rows"]),"checked bias-gradient rows")
