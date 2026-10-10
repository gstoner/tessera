"""Exact SM120 public native_jvp route, independent finite-difference oracle."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"python")]
import tessera as ts  # noqa: E402
from benchmarks.nvidia.benchmark_jvp_argument_order import function  # noqa: E402

def run(order,wrt,sk,causal,artifacts=None):
    rng=np.random.default_rng(913)
    values={n:rng.normal(size=s).astype(np.float32)*.2 for n,s in zip(
        ("q","k","v"),((1,2,3,4),(1,1,sk,4),(1,1,sk,3)),strict=True)}
    directions={n:rng.normal(size=x.shape).astype(np.float32)*.1
                if n in wrt else np.zeros_like(x) for n,x in values.items()}
    def oracle(xs):
        q,k,v=(xs[n] for n in ("q","k","v"))
        score=.5*(q@np.swapaxes(np.repeat(k,2,axis=1),-1,-2))
        if causal:
            score=np.where(np.arange(sk)[None,:]<=np.arange(3)[:,None]+max(sk-3,0),score,-np.inf)
        p=np.exp(score-score.max(axis=-1,keepdims=True));p/=p.sum(axis=-1,keepdims=True)
        return p@np.repeat(v,2,axis=1)
    h=1e-4
    expected=oracle({n:x.astype(np.float64) for n,x in values.items()})
    tangent=(oracle({n:x.astype(np.float64)+h*directions[n] for n,x in values.items()})-
             oracle({n:x.astype(np.float64)-h*directions[n] for n,x in values.items()}))/(2*h)
    fn=ts.jit(target="nvidia_sm120",autodiff="forward",wrt=wrt)(function(order,causal))
    start=time.perf_counter()
    primal,result=fn.native_jvp(**values,tangents=tuple(directions[n] for n in wrt))
    compile_launch_ms=(time.perf_counter()-start)*1e3
    np.testing.assert_allclose(primal,expected,atol=3e-5,rtol=3e-5)
    np.testing.assert_allclose(result,tangent,atol=3e-5,rtol=3e-5)
    receipt=dict(fn.last_jvp_execution)
    assert receipt["family"]=="attention_checkpoint"
    assert receipt["compiler_path"]=="nvidia_sm120_jvp_compiled"
    assert receipt["execution_kind"]=="native_gpu"
    packages=list(fn._native_jvp_packages.values())
    assert len(packages)==1
    package=packages[0]
    def forbidden(*a,**k):
        raise AssertionError("compiler subprocess during warm native_jvp")
    samples=[]
    retained=result.copy()
    with patch("subprocess.run",forbidden),patch("subprocess.Popen",forbidden),patch("subprocess.check_output",forbidden):
        for scale in (2.0,1.0):
            start=time.perf_counter()
            p,t=fn.native_jvp(*(values[n] for n in order),
                tangents=tuple(scale*directions[n] for n in wrt))
            samples.append((time.perf_counter()-start)*1e3)
            np.testing.assert_allclose(p,expected,atol=3e-5,rtol=3e-5)
            np.testing.assert_allclose(t,scale*tangent,atol=3e-5,rtol=3e-5)
            np.testing.assert_array_equal(result,retained)
    if artifacts is not None:
        name="".join(order)+"_"+"_".join(wrt)+f"_{sk}_{int(causal)}"
        (artifacts/(name+".json")).write_text(json.dumps(package.runtime_metadata(),sort_keys=True))
    return dict(order=order,wrt=wrt,sk=sk,causal=causal,
        max_abs_error=float(np.max(np.abs(result-tangent))),
        compile_and_first_launch_ms=compile_launch_ms,
        warm_checked_end_to_end_samples_ms=samples,
        warm_checked_end_to_end_median_ms=statistics.median(samples),
        compiler_subprocesses_on_warm_call="forbidden",receipt=receipt,
        correctness="passed_before_timing")

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,driver_version","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or "RTX 5070" not in gpu or gpu.split(",")[2].strip()!="12.0":
        raise RuntimeError("requires owning RTX 5070 / SM120")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    artifacts=args.output.parent/"artifacts";artifacts.mkdir(exist_ok=True)
    rows=[]
    for order in itertools.permutations(("q","k","v")):
        for sk,causal in ((5,False),(129,True)):
            for wrt in (("q",),("k",),("v",),("k","q"),("v","q"),("q","k","v")):
                row=run(order,wrt,sk,causal,artifacts)
                rows.append(row);print("verified",order,wrt,sk,causal,flush=True)
    files=("python/tessera/compiler/graph_ir_cache.py","python/tessera/compiler/jit.py","python/tessera/compiler/native_jvp_plugins.py",
           "python/tessera/compiler/native_attention_jvp_runtime.py",
           "python/tessera/compiler/native_attention_program.py",
           "python/tessera/compiler/native_attention_jvp_artifact.py",
           "python/tessera/compiler/execution_matrix.py","python/tessera/runtime.py",
           "benchmarks/nvidia/benchmark_public_attention_jvp.py")
    packet=dict(device=gpu,architecture="sm120",rows=rows,
        fingerprints={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files},
        timing_scope="first compile+launch and warm checked synchronous host end-to-end; no isolated kernel claim")
    args.output.write_text(json.dumps(packet,indent=2)+"\n")
    print("wrote",args.output,flush=True)
if __name__=="__main__": main()
