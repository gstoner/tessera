"""Paired public-call transfer attribution with identical-mode controls."""
import argparse,base64,hashlib,json,os,subprocess,time
from pathlib import Path
from statistics import median
import ml_dtypes,numpy as np
import tessera as ts
from tessera import runtime
from tests.device.rocm.test_public_scaled_jvp import scaled,scaled_nk,oracle
from tests.device.rocm.test_public_typed_scaled_primal import mxfp8,mxfp8_nk

def run(shape,nk,fmt):
    m,n,k=shape;rng=np.random.default_rng(921)
    a=rng.choice([-.5,0,.25,1],(m,k)).astype(ml_dtypes.float8_e4m3fn)
    logical_b=rng.choice([-1,0,.5,2],(k,n)).astype(ml_dtypes.float8_e4m3fn)
    b=np.ascontiguousarray(logical_b.T) if nk else logical_b
    if fmt=="e8m0":
        sa=rng.integers(125,130,(m,k//32),dtype=np.uint8)
        sb=rng.integers(125,130,(k//32,n),dtype=np.uint8)
        source=mxfp8_nk if nk else mxfp8
        expected=np.zeros((m,n),np.float64)
        for g in range(k//32):
            expected+=(a[:,g*32:(g+1)*32].astype(np.float64) @ logical_b[g*32:(g+1)*32].astype(np.float64))*np.exp2(sa[:,g,None].astype(np.float64)-127)*np.exp2(sb[g,None,:].astype(np.float64)-127)
    else:
        sa=rng.uniform(.2,1,(m,k//128)).astype(np.float32)
        sb=rng.uniform(.2,1,(k//128,(n+127)//128)).astype(np.float32)
        source=scaled_nk if nk else scaled
        expected=oracle(a,logical_b,sa,sb,np.zeros_like(sa),np.zeros_like(sb))[0]
    fn=ts.jit(target="rocm_gfx1201")(source)
    modes={"pageable":"0","pinned":"1","automatic":None,"automatic_control":None,"pageable_control":"0"}
    samples={name:[] for name in modes}
    def select(mode):
        if mode is None:os.environ.pop("TESSERA_ROCM_PROGRAM_PINNED",None)
        else:os.environ["TESSERA_ROCM_PROGRAM_PINNED"]=mode
    old=subprocess.run
    try:
        for mode in modes.values():
            select(mode)
            np.testing.assert_allclose(fn(a,b,sa,sb),expected,rtol=4e-5,atol=1e-4)
            changed=sa-np.uint8(1) if fmt=="e8m0" else sa*.5
            np.testing.assert_allclose(fn(a,b,changed,sb),expected*.5,rtol=4e-5,atol=1e-4)
            fn(a,b,sa,sb)
        def forbidden(*args,**kwargs):raise AssertionError("warm compiler subprocess")
        subprocess.run=forbidden
        names=list(modes)
        for trial in range(21):
            for name in (names if trial%2==0 else names[::-1]):
                select(modes[name])
                start=time.perf_counter()
                for _ in range(10):fn(a,b,sa,sb)
                samples[name].append((time.perf_counter()-start)*100)
    finally:subprocess.run=old
    def ratio(a,b):return median([x/y for x,y in zip(samples[a],samples[b],strict=True)])
    return {"shape_mnk":list(shape),"rhs_layout":"NK" if nk else "KN","scale_format":fmt,
        "public_samples_ms":samples,"public_medians_ms":{n:median(s) for n,s in samples.items()},
        "automatic_over_pageable":ratio("automatic","pageable"),
        "automatic_over_pinned":ratio("automatic","pinned"),
        "automatic_control_ratio":ratio("automatic","automatic_control"),
        "pageable_control_ratio":ratio("pageable","pageable_control"),
        "correctness":"float64_block_oracle_and_changed_scale_before_timing",
        "compiler_subprocess_forbidden":True,
        "owner_image_sha256":[hashlib.sha256(base64.b64decode(image,validate=True)).hexdigest() for image in fn.compile_result.launch_descriptor.provenance["native_scaled_primal_program"]["images"]]}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if runtime._rocm_live_arch()!="gfx1201":raise RuntimeError("owning gfx1201 required")
    rows=[run(shape,nk,fmt) for shape in [(17,19,256),(200,129,1536)] for fmt in ["fp32","e8m0"] for nk in [False,True]]
    lib=runtime._load_rocm_native_movement_runtime()
    packet={"architecture":"gfx1201","rocminfo":subprocess.run(["rocminfo"],capture_output=True,text=True,check=True).stdout,
        "runtime_sha256":hashlib.sha256(Path(lib._name).read_bytes()).hexdigest(),"rows":rows,
        "timing":"alternating public-call wall-clock windows; includes frontend/validation/transfers/native launch/readback"}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(packet,indent=2)+"\n")
    print(json.dumps([{k:r[k] for k in ["shape_mnk","scale_format","rhs_layout","public_medians_ms","automatic_over_pageable","automatic_control_ratio","pageable_control_ratio"]} for r in rows],indent=2))
if __name__=="__main__":main()
