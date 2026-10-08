"""Exact gfx1201 composed scale-VJP characterization, with separate timings."""
import argparse,ctypes as ct,hashlib,json,os,subprocess,time
from pathlib import Path
from statistics import median
import numpy as np
from tessera import runtime as rt
from tessera.compiler.native_scaled_program import NativeScaledProgram,PreparedScaledProgram
from tests.unit.test_composed_scaled_vjp import case
from tests.device.rocm.test_composed_scaled_vjp import oracle

def _is_validation_argv(argv):
    if not argv:return False
    index=0
    if Path(argv[0]).name.startswith("python"):
        index=1
        while index<len(argv) and argv[index] in {"-u","-B","-E","-s","-S"}:index+=1
        if index<len(argv) and argv[index]=="-m":index+=1
    if index>=len(argv):return False
    tool=Path(argv[index]).name
    if tool in {"pytest","py.test"}:return True
    return tool=="graphify" and index+1<len(argv) and argv[index+1]=="update"

def _validation_active():
    busy=subprocess.run(["pgrep","-af","[p]ytest|[g]raphify update"],capture_output=True,text=True)
    for line in busy.stdout.splitlines():
        pid=line.split(maxsplit=1)[0]
        try:
            argv=[value.decode() for value in Path("/proc",pid,"cmdline").read_bytes().split(b"\0") if value]
        except (OSError,UnicodeError):continue
        if _is_validation_argv(argv):return True
    return False

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if _validation_active():raise RuntimeError("validation/graph extraction is active")
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
    for shape,shared_scale in (((17,19,256),False),((17,19,256),True),((3,5,37),False),((3,5,37),True)):
        owner,values,dy=case(shape,shared_scale)
        expected=oracle(values,dy,shared_scale,owner.differentiation_request.wrt)
        actual=owner.native_backward(*values,out_cotangents=dy)
        for got,want in zip(actual,expected,strict=True):np.testing.assert_allclose(got,want,rtol=3e-4,atol=3e-5)
        package=owner.native_backward_runtime_artifact()
        device=[]
        with PreparedScaledProgram(package,[*values,dy],runtime_library=lib._name) as prepared:
            prepared.invoke(repeats=3)
            for _ in range(5):
                generation,ms=prepared.invoke(repeats=100,timed=True)
                device.append(ms)
                for got,want in zip(prepared.read(generation),expected,strict=True):np.testing.assert_allclose(got,want,rtol=3e-4,atol=3e-5)
        host=[]
        from tessera.compiler import reference_typed_scaled_matmul as reference
        old_run,old_ref=subprocess.run,reference.reference_typed_scaled_matmul
        def forbidden(*a,**k):raise AssertionError("warm compiled program invoked compiler/reference")
        subprocess.run=forbidden;reference.reference_typed_scaled_matmul=forbidden
        try:
            for _ in range(5):
                start=time.perf_counter()
                for _ in range(11):actual=owner.native_backward(*values,out_cotangents=dy)
                host.append((time.perf_counter()-start)*1000/11)
        finally:subprocess.run=old_run;reference.reference_typed_scaled_matmul=old_ref
        for got,want in zip(actual,expected,strict=True):np.testing.assert_allclose(got,want,rtol=3e-4,atol=3e-5)
        row=dict(shape_mnk=list(shape),shared_scale=shared_scale,scale_adjoint_schedule=owner.last_backward_execution["scale_adjoint_schedule"],gradient_roles=list(owner.differentiation_request.wrt),members=len(package.images),native_program_event_samples_ms=device,native_program_event_median_ms=median(device),public_host_samples_ms=host,public_host_median_ms=median(host),max_abs_error=max(float(np.max(np.abs(got-want))) for got,want in zip(actual,expected,strict=True)),artifact_hash=owner.last_backward_execution["artifact_hash"],image_sha256=[hashlib.sha256(image).hexdigest() for image in package.images],correctness="before_and_after_timing")
        rows.append(row);print(json.dumps(row),flush=True)
    root=Path(__file__).resolve().parents[2]
    sources=("benchmarks/rocm/benchmark_composed_scaled_vjp.py","tests/device/rocm/test_composed_scaled_vjp.py","src/transforms/lib/NativeScaledMatmulProgram.h","src/compiler/programming_model/lib/NativeScaleTranspose.h","python/tessera/compiler/native_scaled_program.py","python/tessera/compiler/jit.py","python/tessera/compiler/native_vjp_plugins.py","python/tessera/compiler/rocm_typed_scaled_native.py","tests/unit/test_composed_scaled_vjp.py")
    packet=dict(schema="tessera.gfx1201.composed_scaled_vjp.v1",architecture=rt._rocm_live_arch(),device=name.value.decode(),hip_device=index.value,hip_uuid_hex=bytes(uuid).hex(),compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),runtime_path=lib._name,runtime_sha256=hashlib.sha256(Path(lib._name).read_bytes()).hexdigest(),sources={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},rows=rows,timing="native HIP program event windows over four reductions and any shared-gradient sum; separate warm public host calls including copies/completion; no speedup claim")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2)+"\n")
if __name__=="__main__":main()
