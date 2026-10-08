"""Exact gfx1201 mapped product/sum program characterization."""
import argparse,ctypes as ct,hashlib,json,os,subprocess,time
from pathlib import Path
from statistics import median
import numpy as np
from tessera import runtime as rt
from tessera.compiler.native_scaled_program import NativeScaledProgram,PreparedScaledProgram
from tests.unit.test_composed_scaled_maps import case
from tests.device.rocm.test_composed_scaled_maps import expected
from benchmarks.rocm.benchmark_composed_scaled_vjp import _validation_active

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if _validation_active():raise RuntimeError("validation/graph extraction active")
    if rt._rocm_live_arch()!="gfx1201":raise RuntimeError("owning gfx1201 required")
    hip=rt._load_hip_from_toolkit()
    index=ct.c_int();hip.hipGetDevice.argtypes=[ct.POINTER(ct.c_int)]
    if hip.hipGetDevice(ct.byref(index)):raise RuntimeError("HIP device query failed")
    name=ct.create_string_buffer(256);hip.hipDeviceGetName.argtypes=[ct.c_void_p,ct.c_int,ct.c_int]
    if hip.hipDeviceGetName(name,256,index.value):raise RuntimeError("HIP name query failed")
    uuid=(ct.c_ubyte*16)();hip.hipDeviceGetUuid.argtypes=[ct.c_void_p,ct.c_int]
    if hip.hipDeviceGetUuid(ct.byref(uuid),index.value):raise RuntimeError("HIP UUID query failed")
    lib=rt._load_rocm_native_movement_runtime()
    rows=[]
    for mode in (None,"forward","reverse"):
        for policy,depth,shared,shape in (("all",1,False,(17,19,256)),("scales",2,True,(3,5,256)),("lhs",2,False,(3,5,37))):
            _,owner,values,seeds,axes,prefix=case(policy,depth,mode,shared,shape)
            dy=np.random.default_rng(10901).uniform(-.2,.2,(*prefix,shape[0],shape[1])).astype(np.float32)
            wanted=expected(owner,values,seeds,axes,prefix,shared,dy if mode=="reverse" else None)
            def invoke():
                if mode=="forward":return owner.native_jvp(*values,tangents=seeds)
                if mode=="reverse":return owner.native_backward(*values,out_cotangents=dy)
                return (owner(*values),)
            actual=invoke()
            if mode is None:
                wanted=wanted[:1];package=owner._native_composed_scaled_last_program;frame=list(values)
            elif mode=="forward":
                artifact=next(iter(owner._native_jvp_packages.values()))
                package=NativeScaledProgram.from_manifest(artifact.contract["steps"][0]["child_metadata"]["native_scaled_program"])
                frame=[*values,*seeds]
            else:package=owner.native_backward_runtime_artifact();frame=[*values,dy]
            for got,want in zip(actual,wanted,strict=True):np.testing.assert_allclose(got,want,rtol=3e-4,atol=3e-5)
            device=[]
            with PreparedScaledProgram(package,frame,runtime_library=lib._name) as prepared:
                prepared.invoke(repeats=3)
                for _ in range(5):
                    generation,ms=prepared.invoke(repeats=100,timed=True);device.append(ms)
                    for got,want in zip(prepared.read(generation),wanted,strict=True):np.testing.assert_allclose(got,want,rtol=3e-4,atol=3e-5)
            from tessera.compiler import reference_typed_scaled_matmul as reference
            old_run,old_ref=subprocess.run,reference.reference_typed_scaled_matmul
            def forbidden(*a,**k):raise AssertionError("warm native program invoked compiler/reference")
            host=[]
            subprocess.run=forbidden;reference.reference_typed_scaled_matmul=forbidden
            try:
                for _ in range(5):
                    start=time.perf_counter()
                    for _ in range(11):actual=invoke()
                    host.append((time.perf_counter()-start)*1000/11)
            finally:subprocess.run=old_run;reference.reference_typed_scaled_matmul=old_ref
            for got,want in zip(actual,wanted,strict=True):np.testing.assert_allclose(got,want,rtol=3e-4,atol=3e-5)
            row=dict(mode=mode or "primal",policy=policy,depth=depth,shared_scale=shared,shape_mnk=list(shape),batch_prefix=list(prefix),members=len(package.images),native_program_event_samples_ms=device,native_program_event_median_ms=median(device),public_host_samples_ms=host,public_host_median_ms=median(host),max_abs_error=max(float(np.max(np.abs(g-w))) for g,w in zip(actual,wanted,strict=True)),image_sha256=[hashlib.sha256(i).hexdigest() for i in package.images],program_sha256=hashlib.sha256(package.program_json.encode()).hexdigest(),correctness="before_and_after_timing")
            rows.append(row);print(json.dumps(row),flush=True)
    root=Path(__file__).resolve().parents[2]
    sources=("benchmarks/rocm/benchmark_composed_scaled_maps.py","tests/device/rocm/test_composed_scaled_maps.py","tests/unit/test_composed_scaled_maps.py","tests/device/rocm/test_public_scaled_jvp.py","tests/device/rocm/test_composed_scaled_vjp.py","python/tessera/compiler/native_vmap.py","python/tessera/compiler/rocm_typed_scaled_native.py","python/tessera/compiler/jit.py","python/tessera/runtime.py","python/tessera/compiler/native_scaled_program.py","python/tessera/compiler/capabilities.py","python/tessera/compiler/execution_matrix.py")
    packet=dict(schema="tessera.gfx1201.composed_scaled_maps.v1",architecture=rt._rocm_live_arch(),device=name.value.decode(),hip_uuid_hex=bytes(uuid).hex(),compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),runtime_path=lib._name,runtime_sha256=hashlib.sha256(Path(lib._name).read_bytes()).hexdigest(),sources={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},rows=rows,timing="Complete prepared native HIP program events; separate warm public calls including copies/completion. Characterization only; no speedup claim.")
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(packet,indent=2)+"\n")
if __name__=="__main__":main()
