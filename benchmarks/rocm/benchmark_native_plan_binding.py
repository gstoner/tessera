"""Paired exact-gfx1201 public cost of immutable native ABI binding."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
from statistics import median
import numpy as np
import tessera as ts
from tessera import runtime
from tests.unit import test_rocm_independent_scaled_batch as batch

def measure(shape,fmt,nk,jvp=False):
    values,expected=batch.batch_inputs(shape,fmt,nk)
    source=getattr(batch,f"independent_{fmt}_{'nk' if nk else 'kn'}")
    if jvp:
        a,b,sa,sb=values;logical_b=b.swapaxes(-1,-2) if nk else b
        da=np.full(sa.shape,.05,np.float32);db=np.full(sb.shape,-.03,np.float32)
        tangent=np.zeros_like(expected)
        for g in range(shape[3]//128):
            product=a[...,g*128:(g+1)*128].astype(np.float64)@logical_b[...,g*128:(g+1)*128,:].astype(np.float64)
            columns=np.arange(shape[2])//128
            tangent+=product*(da[...,g,None]*sb[...,g,columns][...,None,:]+sa[...,g,None]*db[...,g,columns][...,None,:])
        fn=ts.jit(target="rocm",autodiff="forward",wrt=("sa","sb"))(source)
        call=lambda:fn.native_jvp(*values,tangents=(da,db))
        oracle=(expected,tangent)
    else:
        fn=ts.jit(target="rocm_gfx1201")(source)
        call=lambda:fn(*values)
        oracle=(expected,)
    samples={"0":[],"1":[]};calls=10
    from tessera.compiler.native_scaled_program import NativeScaledProgram
    original=NativeScaledProgram.from_manifest.__func__
    image_proof={}
    try:
        def capture(cls,value):
            package=original(cls,value)
            image_proof[os.environ["TESSERA_ROCM_PROGRAM_BINDING_CACHE"]]=[
                hashlib.sha256(image).hexdigest() for image in package.images]
            return package
        NativeScaledProgram.from_manifest=classmethod(capture)
        for enabled in ("0","1"):
            os.environ["TESSERA_ROCM_PROGRAM_BINDING_CACHE"]=enabled
            actual=call()
            actual=actual if jvp else (actual,)
            for got,want in zip(actual,oracle,strict=True):
                np.testing.assert_allclose(got,want,rtol=4e-5,atol=1e-4)
            call()
    finally:NativeScaledProgram.from_manifest=classmethod(original)
    assert image_proof["0"]==image_proof["1"]
    old=subprocess.run
    try:
        def forbidden(*args,**kwargs):raise AssertionError("warm compiler subprocess")
        subprocess.run=forbidden
        for window in range(21):
            order=("0","1") if window%2==0 else ("1","0")
            for enabled in order:
                os.environ["TESSERA_ROCM_PROGRAM_BINDING_CACHE"]=enabled
                start=time.perf_counter()
                for _ in range(calls):call()
                samples[enabled].append((time.perf_counter()-start)*1000/calls)
    finally:subprocess.run=old
    return {"shape_bmnk":shape,"format":fmt,"rhs_layout":"NK" if nk else "KN",
        "kind":"scale_jvp" if jvp else "primal","samples_ms":samples,
        "reference_median_ms":median(samples["0"]),"binding_median_ms":median(samples["1"]),
        "paired_binding_over_reference":[b/a for a,b in zip(samples["0"],samples["1"],strict=True)],
        "median_ratio":median([b/a for a,b in zip(samples["0"],samples["1"],strict=True)]),
        "correctness":"independent_float64_before_timing","compiler_subprocess_forbidden":True,
        "owner_image_sha256_by_mode":image_proof}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if runtime._rocm_live_arch()!="gfx1201":raise RuntimeError("owning gfx1201 required")
    cases=[((3,7,19,256),"fp32",False,False),((3,7,19,256),"e8m0",True,False),
        ((2,128,4096,256),"fp32",True,False),((2,100,129,1536),"e8m0",False,False),
        ((3,7,19,256),"fp32",False,True),((2,100,129,1536),"fp32",True,True)]
    rows=[measure(*case) for case in cases]
    lib=runtime._load_rocm_native_movement_runtime()
    packet={"architecture":"gfx1201","rocminfo":subprocess.run(["rocminfo"],text=True,capture_output=True,check=True).stdout,
        "runtime_sha256":hashlib.sha256(Path(lib._name).read_bytes()).hexdigest(),
        "scope":"public wall-clock binding attribution, identical compiler images and native execution; not isolated kernel performance",
        "rows":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2)+"\n")
    print(json.dumps([{k:r[k] for k in ("shape_bmnk","format","rhs_layout","kind","reference_median_ms","binding_median_ms","median_ratio")} for r in rows],indent=2))
if __name__=="__main__":main()
