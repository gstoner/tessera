"""Paired native-owner A/B and identical-image controls for primal projection."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from statistics import median
import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime
from tessera.compiler.native_scaled_program import NativeScaledProgram,PreparedScaledProgram,package_native_scaled_primal
from tests.device.rocm.test_public_scaled_jvp import scaled,scaled_nk,oracle
from tests.device.rocm.test_public_typed_scaled_primal import mxfp8,mxfp8_nk
from benchmarks.rocm.benchmark_native_program_format_staging import library

def run(shape,nk,fmt):
    m,n,k=shape;rng=np.random.default_rng(900)
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
    public=fn(a,b,sa,sb)
    np.testing.assert_allclose(public,expected,rtol=4e-5,atol=1e-4)
    policy=NativeScaledProgram.from_manifest(fn.compile_result.launch_descriptor.provenance["native_scaled_primal_program"])
    projected=package_native_scaled_primal(fn.compile_result.graph_ir,profile_policy=False)
    static=package_native_scaled_primal(fn.compile_result.graph_ir,project_image_identity=False)
    pm,sm=json.loads(projected.members_json[0]),json.loads(static.members_json[0])
    assert pm["geometry"]==sm["geometry"] and pm["scalars"]==sm["scalars"]
    packages={"policy":policy,"projected":projected,"static":static,"policy_control":policy,"projected_control":projected,"static_control":static}
    lib=library();owners={};samples={name:[] for name in packages}
    try:
        for name,p in packages.items():
            owner=PreparedScaledProgram(p,[a,b,sa,sb],runtime_library=lib._name)
            owners[name]=owner
            generation,_=owner.invoke(repeats=3)
            result=owner.read(generation)[0]
            np.testing.assert_allclose(result,expected,rtol=4e-5,atol=1e-4)
            if name.startswith("policy"):np.testing.assert_array_equal(result,public)
        order=list(owners)
        for trial in range(21):
            for name in (order if trial%2==0 else order[::-1]):
                _,elapsed=owners[name].invoke(repeats=100,timed=True)
                samples[name].append(elapsed)
    finally:
        for owner in owners.values():owner.close()
    return {"shape_mnk":list(shape),"rhs_layout":"NK" if nk else "KN","scale_format":fmt,
        "samples_ms":samples,"medians_ms":{n:median(s) for n,s in samples.items()},
        "paired_policy_over_static":median([a/b for a,b in zip(samples["policy"],samples["static"],strict=True)]),
        "paired_policy_over_projected":median([a/b for a,b in zip(samples["policy"],samples["projected"],strict=True)]),
        "paired_policy_control":median([a/b for a,b in zip(samples["policy"],samples["policy_control"],strict=True)]),
        "paired_projected_over_static":median([a/b for a,b in zip(samples["projected"],samples["static"],strict=True)]),
        "paired_projected_control":median([a/b for a,b in zip(samples["projected"],samples["projected_control"],strict=True)]),
        "paired_static_control":median([a/b for a,b in zip(samples["static"],samples["static_control"],strict=True)]),
        "members":{"policy":json.loads(policy.members_json[0]),"projected":pm,"static":sm},"image_sha256":{n:hashlib.sha256(p.images[0]).hexdigest() for n,p in packages.items()},
        "correctness":"float64_block_oracle_before_timing_and_bitwise_projected_public_parity",
        "timing":"native sequence event windows include enqueue gaps; alternating order and identical-image controls"}

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if runtime._rocm_live_arch()!="gfx1201":raise RuntimeError("owning gfx1201 required")
    rows=[run(shape,nk,fmt) for shape in [(17,19,256),(200,129,1536),(64,64,512),(256,256,2048)] for fmt in ["fp32","e8m0"] for nk in [False,True]]
    packet={"architecture":"gfx1201","rocminfo":subprocess.run(["rocminfo"],capture_output=True,text=True,check=True).stdout,"rows":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(packet,indent=2)+"\n")
    print(json.dumps([{k:r[k] for k in ["shape_mnk","rhs_layout","scale_format","medians_ms","paired_projected_over_static","paired_projected_control","paired_static_control"]} for r in rows],indent=2))
if __name__=="__main__":main()
