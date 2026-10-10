"""Ordinary typed FP8 primal JIT and its existing native image timings."""
import argparse
import os
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess
import time
from types import SimpleNamespace
import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime
from tests.device.rocm.test_public_scaled_jvp import scaled,scaled_nk,oracle
from tests.device.rocm.test_public_typed_scaled_primal import mxfp8,mxfp8_nk
from benchmarks.rocm.benchmark_native_program_format_staging import DiagnosticOwner,library

def run(shape,nk,scale_format="fp32"):
    m,n,k=shape;rng=np.random.default_rng(719)
    a=rng.choice([-.5,0,.25,1],(m,k)).astype(ml_dtypes.float8_e4m3fn)
    logical_b=rng.choice([-1,0,.5,2],(k,n)).astype(ml_dtypes.float8_e4m3fn)
    b=np.ascontiguousarray(logical_b.T) if nk else logical_b
    if scale_format=="e8m0":
        sa=rng.integers(125,130,(m,k//32),dtype=np.uint8)
        sb=rng.integers(125,130,(k//32,n),dtype=np.uint8)
        source=mxfp8_nk if nk else mxfp8
    else:
        sa=rng.uniform(.2,1,(m,k//128)).astype(np.float32)
        sb=rng.uniform(.2,1,(k//128,(n+127)//128)).astype(np.float32)
        source=scaled_nk if nk else scaled
    fn=ts.jit(target="rocm_gfx1201")(source)
    start=time.perf_counter();actual=fn(a,b,sa,sb);cold=(time.perf_counter()-start)*1000
    if scale_format=="e8m0":
        expected=np.zeros((m,n),np.float64)
        for g in range(k//32):
            expected+=(a[:,g*32:(g+1)*32].astype(np.float64) @ logical_b[g*32:(g+1)*32].astype(np.float64))*np.exp2(sa[:,g,None].astype(np.float64)-127)*np.exp2(sb[g,None,:].astype(np.float64)-127)
    else:
        expected=oracle(a,logical_b,sa,sb,np.zeros_like(sa),np.zeros_like(sb))[0]
    np.testing.assert_allclose(actual,expected,rtol=4e-5,atol=1e-4)
    warm=[];old=subprocess.run
    def forbidden(*args,**kwargs):raise AssertionError("warm primal attempted compiler subprocess")
    subprocess.run=forbidden
    try:
        for _ in range(7):
            start=time.perf_counter()
            for _ in range(10):fn(a,b,sa,sb)
            warm.append((time.perf_counter()-start)*100)
    finally:subprocess.run=old
    compiled=fn.compile_result;d=compiled.launch_descriptor
    bm,bn=d.provenance["macro_tile"]
    lib=library();package=SimpleNamespace(image=compiled.native_image,descriptor=d)
    from tessera.compiler.native_scaled_program import NativeScaledProgram,PreparedScaledProgram
    native_package=NativeScaledProgram.from_manifest(d.provenance["native_scaled_primal_program"])
    owner=PreparedScaledProgram(native_package,[a,b,sa,sb],runtime_library=lib._name)
    events=[]
    try:
        generation,_=owner.invoke()
        np.testing.assert_array_equal(owner.read(generation)[0],actual)
        for _ in range(11):
            generation,elapsed=owner.invoke(repeats=100,timed=True);events.append(elapsed)
    finally:owner.close()
    return {"native_owner_image_sha256":[hashlib.sha256(image).hexdigest() for image in native_package.images],"scale_format":scale_format,"shape_mnk":list(shape),"rhs_layout":"NK" if nk else "KN","cold_public_ms":cold,
        "public_warm_ms":warm,"public_warm_median_ms":median(warm),
        "native_image_event_ms":events,"native_image_event_median_ms":median(events),
        "execution":fn._native_descriptor_last_receipt,"image_sha256":hashlib.sha256(compiled.native_image.payload).hexdigest(),
        "ir_sha256":{name:hashlib.sha256(getattr(compiled,name+"_ir").encode()).hexdigest() for name in ["graph","schedule","tile","target"]},
        "correctness":"independent_float64_block_oracle_and_bitwise_native_owner_parity",
        "warm_compiler_subprocess_forbidden":True}

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--scale-format",choices=["fp32","e8m0"],default="fp32");args=p.parse_args()
    if runtime._rocm_live_arch()!="gfx1201":raise RuntimeError("owning gfx1201 required")
    rows=[run(shape,nk,args.scale_format) for shape in [(17,19,256),(200,129,1536)] for nk in [False,True]]
    info=subprocess.run(["rocminfo"],text=True,capture_output=True,check=True).stdout
    packet={"native_pinned_override":os.environ.get("TESSERA_ROCM_PROGRAM_PINNED","automatic"),"architecture":"gfx1201","rocminfo":info,"rows":rows,
        "timing_note":"Public JIT includes checked descriptor launch/staging/readback. Native event uses the actual compiler-projected primal member package and includes enqueue gaps; no isolated kernel speedup claim."}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(packet,indent=2,default=str)+"\n")
    print(json.dumps([{"shape":r["shape_mnk"],"layout":r["rhs_layout"],"public_ms":r["public_warm_median_ms"],"native_ms":r["native_image_event_median_ms"]} for r in rows],indent=2))
if __name__=="__main__":main()
