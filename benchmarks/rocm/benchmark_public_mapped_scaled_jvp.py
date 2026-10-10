"""Exact gfx1201 mapped native scale-JVP timing, with independent numerical proof."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
from statistics import median
import numpy as np
import tessera as ts
from tessera.autodiff import vmap
from tessera import runtime
from tessera.compiler.native_scaled_program import NativeScaledProgram,PreparedScaledProgram
from benchmarks.rocm.benchmark_independent_scaled_batch import samples
from tests.unit.test_native_typed_scaled_vmap import case

def measure(shape,policy,nk):
    scalar,primal,values,expected=case(policy,"fp32",nk,shape)
    forward=ts.jit(target="rocm_gfx1201",autodiff="forward",wrt=("sa","sb"))(scalar._fn)
    fn=vmap(forward,in_axes=primal._frontend_batch_axes)
    a,b,sa,sb=values
    da=np.full_like(sa,.05);db=np.full_like(sb,-.03)
    logical_b=b.swapaxes(-1,-2) if nk else b
    tangent=np.zeros_like(expected)
    for g in range(a.shape[-1]//128):
        product=a[...,g*128:(g+1)*128].astype(np.float64)@logical_b[...,g*128:(g+1)*128,:].astype(np.float64)
        columns=np.arange(expected.shape[-1])//128
        tangent+=product*(da[...,g,None].astype(np.float64)*sb[...,g,columns][...,None,:].astype(np.float64)+
                          sa[...,g,None].astype(np.float64)*db[...,g,columns][...,None,:].astype(np.float64))
    public=lambda:fn.native_jvp(*values,tangents=(da,db))
    start=time.perf_counter();actual=public();cold=(time.perf_counter()-start)*1000
    for got,want in zip(actual,(expected,tangent),strict=True):
        np.testing.assert_allclose(got,want,rtol=4e-5,atol=1e-4)
    artifact=next(iter(fn._native_jvp_packages.values()))
    child=artifact.contract["steps"][0]["child_metadata"]
    package=NativeScaledProgram.from_manifest(child["native_scaled_program"])
    inputs=(*values,da,db)
    library=runtime._load_rocm_native_movement_runtime()
    with PreparedScaledProgram(package,inputs,runtime_library=library._name) as owner:
        generation,_=owner.invoke()
        for got,want in zip(owner.read(generation),(expected,tangent),strict=True):
            np.testing.assert_allclose(got,want,rtol=4e-5,atol=1e-4)
        old=subprocess.run
        try:
            def forbidden(*args,**kwargs):raise AssertionError("warm mapped JVP compiler invocation")
            subprocess.run=forbidden
            public_ms=samples(public)
            def host():
                owner.update(inputs)
                generation,_=owner.invoke()
                owner.read(generation)
            host_ms=samples(host)
            events=[owner.invoke(repeats=30,timed=True)[1] for _ in range(21)]
        finally:subprocess.run=old
    return {"shape_bmnk":list(shape),"batching":policy,"rhs_layout":"NK" if nk else "KN",
        "cold_public_ms":cold,"correctness":"independent_float64_block_primal_and_scale_derivative_before_timing",
        "compiler_subprocess_forbidden":True,
        "public_samples_ms":public_ms,"public_median_ms":median(public_ms),
        "prepared_host_samples_ms":host_ms,"prepared_host_median_ms":median(host_ms),
        "native_sequence_event_samples_ms":events,"native_sequence_event_median_ms":median(events),
        "owner_image_sha256":[hashlib.sha256(i).hexdigest() for i in package.images],
        "program_json_sha256":hashlib.sha256(package.program_json.encode()).hexdigest(),
        "member_geometry":[json.loads(m)["geometry"] for m in package.members_json],
        "native_steps":len(json.loads(package.program_json)["steps"]),
        "receipt":fn.last_jvp_execution}

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path);args=p.parse_args()
    if runtime._rocm_live_arch()!="gfx1201":raise RuntimeError("owning gfx1201 required")
    rows=[measure(shape,policy,nk) for shape in [(3,7,19,256),(2,200,129,1536)]
        for policy in ["shared_rhs_rows","independent_rhs","shared_lhs"] for nk in [False,True]]
    library=runtime._load_rocm_native_movement_runtime()
    packet={"architecture":"gfx1201","rocminfo":subprocess.run(["rocminfo"],text=True,capture_output=True,check=True).stdout,
        "runtime_sha256":hashlib.sha256(Path(library._name).read_bytes()).hexdigest(),"rows":rows,
        "timing":"separate public-call wall, prepared update/invoke/read wall, native HIP sequence-event windows; events include native enqueue gaps, not isolated kernel time"}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(packet,indent=2)+"\n")
    print("Recorded",len(rows),"mapped native scale-JVP rows")
if __name__=="__main__":main()
