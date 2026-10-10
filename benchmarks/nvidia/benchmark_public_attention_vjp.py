"""Owning SM120 native_backward, saved-LSE, independent grouped-query gradients."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import time
from unittest.mock import patch
import numpy as np
import tessera as ts
from benchmarks.nvidia.benchmark_jvp_argument_order import function

def biased(bias,v,q,k):
    return ts.ops.flash_attn(q,k,v,attn_bias=bias,causal=False)

def biased_causal(bias,v,q,k):
    return ts.ops.flash_attn(q,k,v,attn_bias=bias,causal=True)

def oracle(values,cot,causal):
    q,k,v=(values[n].astype(np.float64) for n in ("q","k","v"))
    group=q.shape[1]//k.shape[1]
    kr=np.repeat(k,group,axis=1);vr=np.repeat(v,group,axis=1)
    scale=1/np.sqrt(q.shape[-1])
    score=scale*(q@np.swapaxes(kr,-1,-2))
    if "bias" in values:score+=values["bias"]
    if causal:
        sq,sk=q.shape[-2],k.shape[-2]
        score=np.where(np.arange(sk)[None,:]<=np.arange(sq)[:,None]+max(sk-sq,0),score,-np.inf)
    p=np.exp(score-score.max(axis=-1,keepdims=True));p/=p.sum(axis=-1,keepdims=True)
    dy=cot.astype(np.float64)
    dp=dy@np.swapaxes(vr,-1,-2)
    ds=p*(dp-(dp*p).sum(axis=-1,keepdims=True))
    dq=scale*(ds@kr)
    dk=scale*(np.swapaxes(ds,-1,-2)@q)
    dv=np.swapaxes(p,-1,-2)@dy
    b,hkv,sk,d=k.shape
    dk=dk.reshape(b,hkv,group,sk,d).sum(axis=2)
    dv=dv.reshape(b,hkv,group,sk,v.shape[-1]).sum(axis=2)
    out=dict(q=dq,k=dk,v=dv)
    if "bias" in values:
        axes=tuple(i for i,extent in enumerate(values["bias"].shape) if extent==1 and ds.shape[i]!=1)
        out["bias"]=ds.sum(axis=axes,keepdims=True)
    return out

def run(order,wrt,sk,causal,artifacts,bias_shape=None):
    rng=np.random.default_rng(914)
    b,hq,hkv=(2,4,2) if bias_shape is not None else (1,2,1)
    values={n:rng.normal(size=s).astype(np.float32)*.2 for n,s in zip(
        ("q","k","v"),((b,hq,3,4),(b,hkv,sk,4),(b,hkv,sk,3)),strict=True)}
    if bias_shape is not None:values["bias"]=rng.normal(size=bias_shape).astype(np.float32)*.1
    cot=rng.normal(size=(b,hq,3,3)).astype(np.float32)*.1
    expected=oracle(values,cot,causal)
    fn=ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=wrt)(
        (biased_causal if causal else biased) if bias_shape is not None else function(order,causal))
    start=time.perf_counter()
    result=fn.native_backward(**values,out_cotangents=cot)
    cold_ms=(time.perf_counter()-start)*1e3
    assert len(result)==len(wrt)
    for name,x in zip(wrt,result,strict=True):np.testing.assert_allclose(x,expected[name],atol=3e-5,rtol=3e-5)
    receipt=fn.last_backward_execution
    assert receipt["family"]=="attention_backward" and receipt["execution_kind"]=="native_gpu"
    assert receipt["compiler_path"]=="nvidia_sm120_attention_vjp_compiled"
    assert receipt["execution_certificate"]["evidence_scope"]=="exact_device"
    artifact=fn.native_backward_runtime_artifact()
    assert artifact.artifact_hash==receipt["artifact_hash"]
    def forbidden(*a,**k):raise AssertionError("compiler subprocess in warm native_backward")
    retained=tuple(x.copy() for x in result);samples=[]
    with patch("subprocess.run",forbidden),patch("subprocess.Popen",forbidden),patch("subprocess.check_output",forbidden):
        for scale in (2.,1.):
            start=time.perf_counter()
            gradients=fn.native_backward(*(values[n] for n in order),out_cotangents=scale*cot)
            samples.append((time.perf_counter()-start)*1e3)
            for name,x in zip(wrt,gradients,strict=True):
                np.testing.assert_allclose(x,scale*expected[name],atol=3e-5,rtol=3e-5)
            for x,prior in zip(result,retained,strict=True):np.testing.assert_array_equal(x,prior)
    label="".join(order)+"_"+"_".join(wrt)+f"_{sk}_{int(causal)}"+(
        "_"+"x".join(map(str,bias_shape)) if bias_shape is not None else "")
    (artifacts/(label+".json")).write_text(artifact.to_json())
    inputfile=artifacts/(label+".npz")
    np.savez(inputfile,**{f"primal_{i}":values[n] for i,n in enumerate(order)},cotangent=cot,
             **{f"expected_{i}":expected[n] for i,n in enumerate(wrt)})
    return dict(order=list(order),wrt=list(wrt),sk=sk,causal=causal,bias_shape=bias_shape,
        max_abs_error=max(float(np.max(np.abs(x-expected[n]))) for n,x in zip(wrt,result,strict=True)),
        first_compile_launch_ms=cold_ms,warm_wall_samples_ms=samples,warm_wall_median_ms=statistics.median(samples),
        receipt=receipt,artifact_hash=artifact.artifact_hash,
        program_digest=artifact.metadata["program_digest"],artifact=label+".json",inputs=label+".npz",
        warm_compiler_subprocesses="forbidden",correctness="independent_fp64_gradients")

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--limit",type=int);args=parser.parse_args()
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,compute_cap,driver_version",
                                 "--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or "RTX 5070" not in gpu or gpu.split(",")[2].strip()!="12.0":
        raise RuntimeError("owning RTX 5070 / SM120 required")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    artifacts=args.output.parent/"artifacts";artifacts.mkdir(exist_ok=True)
    cases=[]
    for order in itertools.permutations(("q","k","v")):
        for sk,causal in ((5,False),(129,True)):
            for wrt in (("q",),("k",),("v",),("k","q"),("v","q"),("q","k","v")):
                cases.append((order,wrt,sk,causal,None))
    for bias in ((1,4,1,1),(2,4,3,5)):
        for causal in (False,True):
            for wrt in (("bias",),("bias","q","v"),("k","bias"),("bias","v","k","q")):
                cases.append((("bias","v","q","k"),wrt,5,causal,bias))
    if args.limit is not None:cases=cases[:args.limit]
    rows=[]
    for order,wrt,sk,causal,bias in cases:
        rows.append(run(order,wrt,sk,causal,artifacts,bias))
        print("verified",order,wrt,sk,causal,bias,flush=True)
    files=("python/tessera/compiler/native_vjp_plugins.py","python/tessera/compiler/native_attention_vjp_artifact.py",
           "python/tessera/compiler/native_attention_vjp_runtime.py","python/tessera/compiler/native_attention_program.py",
           "python/tessera/compiler/jit.py","python/tessera/compiler/execution_matrix.py","python/tessera/runtime.py",
           "benchmarks/nvidia/benchmark_public_attention_vjp.py",
           "python/tessera/compiler/prepared_attention_vjp.py",
           "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp",
           "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.h")
    packet=dict(device=gpu,architecture="sm120",rows=rows,
                fingerprints={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
                timing_scope="first compilation+execution and warm synchronous public host calls; no isolated kernel claim")
    args.output.write_text(json.dumps(packet,indent=2)+"\n")
if __name__=="__main__":main()
