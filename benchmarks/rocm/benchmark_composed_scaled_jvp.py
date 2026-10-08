"""Exact gfx1201 composed scale-JVP characterization, with separate timings."""
import argparse,ctypes as ct,hashlib,json,os,subprocess,time
from pathlib import Path
from statistics import median
import numpy as np
from tessera import runtime as rt
from tessera.compiler.native_scaled_program import NativeScaledProgram,PreparedScaledProgram
from tests.unit.test_composed_scaled_jvp import case
from tests.device.rocm.test_public_scaled_jvp import oracle

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    busy=subprocess.run(["pgrep","-af","[p]ytest|[g]raphify update"],capture_output=True,text=True)
    if busy.returncode==0 and busy.stdout.strip():raise RuntimeError("validation/graph extraction is active")
    if rt._rocm_live_arch()!="gfx1201":raise RuntimeError("owning gfx1201 required")
    hip=rt._load_hip_from_toolkit()
    if hip is None:raise RuntimeError("HIP runtime unavailable")
    hip.hipGetDevice.argtypes=[ct.POINTER(ct.c_int)]
    index=ct.c_int()
    if hip.hipGetDevice(ct.byref(index))!=0:raise RuntimeError("HIP device query failed")
    name=ct.create_string_buffer(256)
    hip.hipDeviceGetName.argtypes=[ct.c_void_p,ct.c_int,ct.c_int]
    if hip.hipDeviceGetName(name,256,index.value)!=0:raise RuntimeError("HIP device name query failed")
    uuid=(ct.c_ubyte*16)()
    hip.hipDeviceGetUuid.argtypes=[ct.c_void_p,ct.c_int]
    if hip.hipDeviceGetUuid(ct.byref(uuid),index.value)!=0:raise RuntimeError("HIP UUID query failed")
    lib=rt._load_rocm_native_movement_runtime()
    rows=[]
    for shape in ((17,19,256),(3,5,37)):
        owner,values,seeds=case(shape)
        expected0=oracle(*values[:2],*values[2:4],*seeds[:2])
        expected1=oracle(*values[:2],*values[4:6],*seeds[2:4])
        expected=tuple(x+y for x,y in zip(expected0,expected1,strict=True))
        actual=owner.native_jvp(*values,tangents=seeds)
        for got,want in zip(actual,expected,strict=True):np.testing.assert_allclose(got,want,rtol=4e-5,atol=3e-5)
        artifact=next(iter(owner._native_jvp_packages.values()))
        package=NativeScaledProgram.from_manifest(artifact.contract["steps"][0]["child_metadata"]["native_scaled_program"])
        device=[]
        with PreparedScaledProgram(package,[*values,*seeds],runtime_library=lib._name) as prepared:
            prepared.invoke(repeats=3)
            for _ in range(5):
                generation,ms=prepared.invoke(repeats=100,timed=True)
                device.append(ms)
                for got,want in zip(prepared.read(generation),expected,strict=True):np.testing.assert_allclose(got,want,rtol=4e-5,atol=3e-5)
        host=[]
        from tessera.compiler import reference_typed_scaled_matmul as reference
        old_run,old_ref=subprocess.run,reference.reference_typed_scaled_matmul
        def forbidden(*a,**k):raise AssertionError("warm compiled program invoked compiler/reference")
        subprocess.run=forbidden;reference.reference_typed_scaled_matmul=forbidden
        try:
            for _ in range(5):
                start=time.perf_counter()
                for _ in range(11):actual=owner.native_jvp(*values,tangents=seeds)
                host.append((time.perf_counter()-start)*1000/11)
        finally:subprocess.run=old_run;reference.reference_typed_scaled_matmul=old_ref
        for got,want in zip(actual,expected,strict=True):np.testing.assert_allclose(got,want,rtol=4e-5,atol=3e-5)
        row=dict(shape_mnk=list(shape),members=len(package.images),native_program_event_samples_ms=device,native_program_event_median_ms=median(device),public_host_samples_ms=host,public_host_median_ms=median(host),max_abs_error=max(float(np.max(np.abs(got-want))) for got,want in zip(actual,expected,strict=True)),artifact_hash=artifact.artifact_hash,image_sha256=[hashlib.sha256(image).hexdigest() for image in package.images],correctness="before_and_after_timing")
        rows.append(row);print(json.dumps(row),flush=True)
    root=Path(__file__).resolve().parents[2]
    sources=("python/tessera/compiler/jit.py","python/tessera/compiler/native_jvp_plugins.py","python/tessera/compiler/rocm_typed_scaled_native.py","tests/unit/test_composed_scaled_jvp.py")
    packet=dict(schema="tessera.gfx1201.composed_scaled_jvp.v1",architecture=rt._rocm_live_arch(),device=name.value.decode(),hip_device=index.value,hip_uuid_hex=bytes(uuid).hex(),compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),runtime_path=lib._name,runtime_sha256=hashlib.sha256(Path(lib._name).read_bytes()).hexdigest(),sources={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},rows=rows,timing="native HIP program event windows over ten members; separate warm public host calls including copies/completion; no speedup claim")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2)+"\n")
if __name__=="__main__":main()
