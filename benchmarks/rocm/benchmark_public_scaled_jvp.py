"""Separate public-call and native-owner costs for typed FP8 scale JVP."""
import argparse
import cProfile
import io
import pstats
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess
import time
import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime
from tessera.compiler.native_scaled_program import NativeScaledProgram, PreparedScaledProgram

def scaled(a:ts.Tensor["M","K","fp8_e4m3"],b:ts.Tensor["K","N","fp8_e4m3"],
           sa:ts.Tensor["M","G","fp32"],sb:ts.Tensor["G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})

def scaled_nk(a:ts.Tensor["M","K","fp8_e4m3"],b:ts.Tensor["N","K","fp8_e4m3"],
              sa:ts.Tensor["M","G","fp32"],sb:ts.Tensor["G","C","fp32"]):
    return ts.ops.scaled_matmul(a,b,sa,sb,transposeB=True,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[128,128],"format":"fp32"})

def oracle(a,b,sa,sb,da,db):
    primal=np.zeros((a.shape[0],b.shape[1]),np.float64)
    tangent=np.zeros_like(primal)
    columns=np.arange(b.shape[1])//128
    for g in range(a.shape[1]//128):
        product=a[:,g*128:(g+1)*128].astype(np.float64)@b[g*128:(g+1)*128].astype(np.float64)
        primal+=product*sa[:,g,None]*sb[g,columns][None,:]
        tangent+=product*(da[:,g,None]*sb[g,columns][None,:]+sa[:,g,None]*db[g,columns][None,:])
    return primal,tangent

def run(shape,transposed_rhs=False):
    m,n,k=shape
    rng=np.random.default_rng(407)
    a=rng.choice([-.5,0,.25,1],(m,k)).astype(ml_dtypes.float8_e4m3fn)
    b=rng.choice([-1,0,.5,2],(k,n)).astype(ml_dtypes.float8_e4m3fn)
    logical_b=b
    if transposed_rhs: b=np.ascontiguousarray(b.T)
    sa=rng.uniform(.2,1,(m,k//128)).astype(np.float32)
    sb=rng.uniform(.2,1,(k//128,(n+127)//128)).astype(np.float32)
    da=rng.uniform(-.1,.1,sa.shape).astype(np.float32)
    db=rng.uniform(-.1,.1,sb.shape).astype(np.float32)
    fn=ts.jit(target="rocm",autodiff="forward",wrt=("sa","sb"))(scaled_nk if transposed_rhs else scaled)
    start=time.perf_counter()
    actual=fn.native_jvp(a,b,sa,sb,tangents=(da,db))
    cold=(time.perf_counter()-start)*1000
    expected=oracle(a,logical_b,sa,sb,da,db)
    for got,want in zip(actual,expected):
        np.testing.assert_allclose(got,want,rtol=3e-5,atol=3e-5)
    artifact=next(iter(fn._native_jvp_packages.values()))
    child=artifact.contract["steps"][0]["child_metadata"]
    package=NativeScaledProgram.from_manifest(child["native_scaled_program"])
    public=[]
    old_run=subprocess.run
    def forbidden(*args,**kwargs):
        raise AssertionError("warm public call attempted compiler subprocess")
    subprocess.run=forbidden
    try:
        for _ in range(7):
            start=time.perf_counter()
            for _ in range(10):
                fn.native_jvp(a,b,sa,sb,tangents=(da,db))
            public.append((time.perf_counter()-start)*100)
    finally:
        subprocess.run=old_run
    profile=cProfile.Profile()
    profile.enable()
    for _ in range(20):
        fn.native_jvp(a,b,sa,sb,tangents=(da,db))
    profile.disable()
    report=io.StringIO()
    pstats.Stats(profile,stream=report).sort_stats("cumulative").print_stats(25)
    library=runtime._load_rocm_native_movement_runtime()._name
    events=[]; readback=[]; update=[]
    with PreparedScaledProgram(package,[a,b,sa,sb,da,db],runtime_library=library) as owner:
        generation,_=owner.invoke()
        for got,want in zip(owner.read(generation),expected):
            np.testing.assert_allclose(got,want,rtol=3e-5,atol=3e-5)
        for _ in range(11):
            generation,elapsed=owner.invoke(repeats=100,timed=True)
            events.append(elapsed)  # Native ABI already returns milliseconds per sequence.
        for _ in range(11):
            start=time.perf_counter();generation,_=owner.invoke();owner.read(generation)
            readback.append((time.perf_counter()-start)*1000)
            start=time.perf_counter();owner.update([a,b,sa,sb,da,db])
            generation,_=owner.invoke();owner.read(generation)
            update.append((time.perf_counter()-start)*1000)
    return {"shape_mnk":list(shape),"rhs_storage":"NK" if transposed_rhs else "KN","cold_public_ms":cold,
        "max_abs_error":[float(np.max(np.abs(x-y))) for x,y in zip(actual,expected)],
        "public_warm_ms":public,"public_warm_median_ms":median(public),
        "native_event_sequence_ms":events,"native_event_sequence_median_ms":median(events),
        "prepared_invoke_two_readbacks_ms":readback,
        "prepared_invoke_two_readbacks_median_ms":median(readback),
        "prepared_update_invoke_two_readbacks_ms":update,
        "prepared_update_invoke_two_readbacks_median_ms":median(update),
        "warm_compiler_subprocess_forbidden":True,
        "warm_public_profile_20_calls":report.getvalue(),
        "execution":fn.last_jvp_execution,
        "image_sha256":[hashlib.sha256(i).hexdigest() for i in package.images]}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--transposed-rhs",action="store_true")
    args=parser.parse_args()
    arch=runtime._rocm_live_arch()
    if arch!="gfx1201":raise RuntimeError("owning gfx1201 device required: "+arch)
    info=subprocess.run(["rocminfo"],capture_output=True,text=True,check=True).stdout
    shapes=[(17,129,256),(200,129,1536),(33,257,2048)] if args.transposed_rhs else [(17,19,256),(32,32,256),(200,19,256)]
    rows=[run(shape,args.transposed_rhs) for shape in shapes]
    result={"architecture":arch,"rocminfo":info,"rows":rows,
        "timing_note":"Native events cover the four-member sequence including native enqueue gaps. Public warm calls include frontend/descriptor validation, native preparation/allocation, launch, readback and cleanup. No isolated-kernel or speedup claim."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"architecture":arch,"rows":[{k:r[k] for k in ["shape_mnk","public_warm_median_ms","native_event_sequence_median_ms","prepared_update_invoke_two_readbacks_median_ms"]} for r in rows]},indent=2))

if __name__=="__main__":main()
