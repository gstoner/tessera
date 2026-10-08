"""JIT reverse attention -> paired AD -> Schedule/Tile -> native CUDA proof."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"python")]
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests._support.nvidia import nvidia_cuda_host_ready

def function(wrt, causal):
    if causal:
        @ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=wrt)
        def attention(q,k,v):
            return ts.ops.flash_attn(q,k,v,causal=True)
    else:
        @ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=wrt)
        def attention(q,k,v):
            return ts.ops.flash_attn(q,k,v,causal=False)
    return attention

def reference(q,k,v,seed,causal):
    q,k,v,seed=(x.astype(np.float64) for x in (q,k,v,seed))
    b,hq,sq,d=q.shape;hkv=k.shape[1];sk=k.shape[2];group=hq//hkv
    kr=np.repeat(k,group,axis=1);vr=np.repeat(v,group,axis=1)
    scores=(q@np.swapaxes(kr,-1,-2))/np.sqrt(d)
    mask=(np.arange(sk)[None,:]<=np.arange(sq)[:,None]+max(sk-sq,0)
          if causal else np.ones((sq,sk),bool))
    scores=np.where(mask,scores,-np.inf)
    maximum=scores.max(axis=-1,keepdims=True)
    weight=np.exp(scores-maximum);normal=weight.sum(axis=-1,keepdims=True)
    prob=weight/normal
    output=prob@vr;lse=(maximum+np.log(normal))[...,0]
    dp=seed@np.swapaxes(vr,-1,-2)
    ds=prob*(dp-(prob*dp).sum(axis=-1,keepdims=True))
    dq=(ds@kr)/np.sqrt(d)
    dk=(np.swapaxes(ds,-1,-2)@q)/np.sqrt(d)
    dv=np.swapaxes(prob,-1,-2)@seed
    dk=dk.reshape(b,hkv,group,sk,d).sum(axis=2)
    dv=dv.reshape(b,hkv,group,sk,v.shape[-1]).sum(axis=2)
    return output,lse,(dq,dk,dv)

def download(session,value):
    spec=value.__cuda_array_interface__
    output=np.empty(spec["shape"],np.dtype(spec["typestr"]))
    copy=ctypes.CDLL("libcuda.so.1").cuMemcpyDtoH_v2
    copy.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
    copy.restype=ctypes.c_int
    session.synchronize()
    status=copy(output.ctypes.data,spec["data"][0],output.nbytes)
    if status:raise RuntimeError("CUDA download failed: "+str(status))
    return output

def shape_argument(text):
    values=tuple(int(v) for v in text.split("x"))
    if len(values)!=7 or min(values)<1 or values[1]%values[2]:
        raise argparse.ArgumentTypeError("requires positive BxHqxHkvxSqxSkxDxDv with Hq divisible by Hkv")
    return values

def record(samples,reps,shapes=None):
    if not nvidia_cuda_host_ready():raise RuntimeError("exact SM120 host/toolchain required")
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or gpu.split(",")[-1].strip()!="12.0":
        raise RuntimeError("requires one selected SM120 GPU: "+gpu)
    rows=[]
    for shape in (shapes or ((1,2,1,3,5,4,3),(1,2,1,5,3,4,3),(2,4,2,16,19,8,6))):
      for causal in (False,True):
       for wrt in (("q",),("k","q"),("v","q","k")):
        print(json.dumps(dict(shape=shape,causal=causal,wrt=wrt,phase="compile")),file=sys.stderr,flush=True)
        b,hq,hkv,sq,sk,d,dv=shape
        rng=np.random.default_rng(120_303)
        values=[(rng.normal(size=s)*.2).astype(np.float32) for s in
            ((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),(b,hq,sq,dv))]
        expected,lse,grad=reference(*values,causal)
        start=time.perf_counter_ns()
        program=function(wrt,causal).compile_native_attention_vjp(
            *values[:3],compiler=Path(os.environ["TESSERA_OPT"]))
        compile_ms=(time.perf_counter_ns()-start)/1e6
        pair=program.pair;active=program.active
        with NvidiaDeviceSession() as session:
            resident=[session.upload(x) for x in values]
            with program.capture(*resident[:3]) as frame:
                np.testing.assert_allclose(download(session,frame.primal),expected,atol=4e-5,rtol=4e-5)
                for value in resident[:3]:
                    changed=np.full(value.shape,19,np.float32)
                    status=session.lib.tessera_nvidia_device_upload(
                        ctypes.c_void_p(value.ptr),ctypes.c_void_p(changed.ctypes.data),
                        changed.nbytes,ctypes.c_void_p(session.stream))
                    if status:raise RuntimeError("CUDA mutation failed")
                session.synchronize()
                backward_wall=[];errors=[]
                for _ in range(samples):
                    start=time.perf_counter_ns();actual=frame.backward(resident[3])
                    backward_wall.append((time.perf_counter_ns()-start)/1e6)
                    for result,i in zip(actual,active,strict=True):
                        host=download(session,result)
                        np.testing.assert_allclose(host,grad[i],atol=4e-5,rtol=4e-5)
                        errors.append(float(np.max(np.abs(host-grad[i]))))
            fresh=[session.upload(x) for x in values]
            output=session.empty(expected.shape,np.float32)
            saved_lse=session.empty(lse.shape,np.float32)
            gradients=[session.empty(x.shape,np.float32) for x in values[:3]]
            scalars=dict(zip(("B","Hq","Hkv","Sq","Sk","D","Dv"),shape,strict=True))
            fargs={**scalars,**dict(zip(
                [x.name for x in pair.forward.descriptor.buffers],
                [*fresh[:3],output,saved_lse],strict=True))}
            bargs={**scalars,**dict(zip(
                [x.name for x in pair.backward.descriptor.buffers],
                [fresh[3],*fresh[:3],output,saved_lse,*gradients],strict=True))}
            ftime=[rt._nvidia_native_descriptor_resident_device_latency(
                pair.forward.image,pair.forward.descriptor,fargs,
                stream=session.stream,reps=reps,warmup=20) for _ in range(samples)]
            np.testing.assert_allclose(download(session,output),expected,atol=4e-5,rtol=4e-5)
            np.testing.assert_allclose(download(session,saved_lse),lse,atol=4e-5,rtol=4e-5)
            btime=[rt._nvidia_native_descriptor_resident_device_latency(
                pair.backward.image,pair.backward.descriptor,bargs,
                stream=session.stream,reps=reps,warmup=20) for _ in range(samples)]
            for i in active:
                np.testing.assert_allclose(download(session,gradients[i]),grad[i],atol=4e-5,rtol=4e-5)
            capture_wall=[]
            for _ in range(samples):
                start=time.perf_counter_ns();frame=program.capture(*fresh[:3])
                capture_wall.append((time.perf_counter_ns()-start)/1e6)
                np.testing.assert_allclose(download(session,frame.primal),expected,atol=4e-5,rtol=4e-5)
                frame.close()
            paired_wall=[]
            paired_errors=[]
            for trial in range(samples):
                # Residual views are frame-owned: validate before release.
                # Downloads/oracles and release are outside capture/backward wall.
                print(json.dumps(dict(shape=shape,causal=causal,wrt=wrt,
                                      phase="paired_wall",sample=trial)),file=sys.stderr,flush=True)
                start=time.perf_counter_ns()
                with program.capture(*fresh[:3]) as paired_frame:
                    paired_gradients=paired_frame.backward(fresh[3])
                    paired_wall.append((time.perf_counter_ns()-start)/1e6)
                    np.testing.assert_allclose(download(session,paired_frame.primal),expected,atol=4e-5,rtol=4e-5)
                    for result,i in zip(paired_gradients,active,strict=True):
                        host=download(session,result)
                        np.testing.assert_allclose(host,grad[i],atol=4e-5,rtol=4e-5)
                        paired_errors.append(float(np.max(np.abs(host-grad[i]))))
        rows.append(dict(shape_b_hq_hkv_sq_sk_d_dv=shape,causal=causal,wrt=wrt,
            max_abs_gradient_error=max(errors),compile_wall_ms=compile_ms,
            forward_device_window_samples_ms=ftime,backward_device_window_samples_ms=btime,
            forward_device_window_median_ms=statistics.median(ftime),
            backward_device_window_median_ms=statistics.median(btime),
            capture_wall_samples_ms=capture_wall,backward_wall_samples_ms=backward_wall,
            capture_wall_median_ms=statistics.median(capture_wall),
            backward_wall_median_ms=statistics.median(backward_wall),
            paired_wall_samples_ms=paired_wall,
            paired_wall_median_ms=statistics.median(paired_wall),
            paired_max_abs_gradient_error=max(paired_errors),
            paired_ownership="primal and gradients validated while frame owns their storage",
            compiler_provenance={stage:{key:package.descriptor.provenance.get(key)
                for key in ("graph_ir_digest","schedule_ir_digest","tile_ir_digest","target_ir_digest")}
                for stage,package in (("forward",pair.forward),("backward",pair.backward))},
            checkpoint_identity=pair.contract_digest,
            forward_image_sha256=hashlib.sha256(pair.forward.image.payload).hexdigest(),
            backward_image_sha256=hashlib.sha256(pair.backward.image.payload).hexdigest(),
            forward_abi=pair.forward.descriptor.abi_id,backward_abi=pair.backward.descriptor.abi_id,
            ownership="private saved generation survives resident caller mutation and repeated backward",
            route="JIT trace -> native paired AD -> Graph/Schedule/Tile -> NVIDIA Target/NVVM/PTX -> checked resident ABI"))
    sources=("python/tessera/__init__.py","python/tessera/compiler/op_catalog.py",
        "python/tessera/compiler/graph_ir.py","python/tessera/compiler/trace.py",
        "python/tessera/compiler/matmul_pipeline.py","src/transforms/lib/AutodiffPairedPass.cpp","src/compiler/ir/AdjointInterface.cpp",
        "src/compiler/ir/AttentionADContract.h","python/tessera/compiler/jit.py",
        "python/tessera/compiler/native_attention_program.py",
        "python/tessera/compiler/scheduled_checkpoint.py","python/tessera/compiler/nvidia_native.py",
        "python/tessera/compiler/resident_attention.py","python/tessera/runtime.py",
        "benchmarks/nvidia/benchmark_jit_attention_vjp.py")
    return dict(schema="tessera.nvidia.jit_attention_vjp.v2",gpu=gpu,architecture="sm_120",
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
        nvidia_compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_OPT"]).read_bytes()).hexdigest(),
        runtime_library_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
        samples=samples,reps=reps,rows=rows,
        timing_scope="resident CUDA-event C++ launch windows include driver gaps; capture wall includes private copies, module load and forward; backward wall includes allocation and synchronization; paired wall includes capture and backward with downloads/oracles and frame release outside; native gradient activity follows requested wrt; no speedup claim")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=5);parser.add_argument("--reps",type=int,default=200)
    parser.add_argument("--shapes",type=shape_argument,nargs="+")
    args=parser.parse_args()
    if args.samples<3 or args.reps<20:raise ValueError("requires >=3 samples and >=20 reps")
    result=record(args.samples,args.reps,args.shapes)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("verified",len(result["rows"]),"JIT reverse attention rows")
