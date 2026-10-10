"""Public independent primal/JVP wall and averaged native event baselines."""
import argparse,ctypes as c,hashlib,itertools,json,os,subprocess,time
from pathlib import Path
from statistics import median
import numpy as np
from tessera import runtime as rt
from tessera.compiler.native_scaled_program import NativeScaledProgram,PreparedScaledProgram
from tests.unit.test_public_independent_scaled_primal import case,expected

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(actual,wanted):
    error=0.
    for a,b in zip(actual,wanted,strict=True):
        np.testing.assert_allclose(a,b,rtol=3e-5,atol=2e-5)
        error=max(error,float(np.max(np.abs(a-b))))
    return error

def measure():
    arch=rt._rocm_live_arch()
    if arch!="gfx1201":raise RuntimeError("actual gfx1201 required")
    library=Path(rt._load_rocm_native_movement_runtime()._name)
    hip=c.CDLL("/opt/rocm/lib/libamdhip64.so")
    if hip.hipInit(0):raise RuntimeError("HIP init failed")
    name=c.create_string_buffer(256);uuid=(c.c_ubyte*16)()
    if hip.hipDeviceGetName(name,256,0) or hip.hipDeviceGetUuid(c.byref(uuid),0):
        raise RuntimeError("GPU identity query failed")
    rows=[]
    profiles=[(mask,tb,encoded,False,ta) for mask,tb,encoded,ta in itertools.product(range(1,16),(False,True),(False,True),(False,True))]
    profiles += [(mask,tb,False,True,ta) for mask,tb,ta in itertools.product(range(1,16),(False,True),(False,True))]
    for mask,tb,encoded,jvp,ta in profiles:
        _,owner,values,row=case(mask,tb,encoded,jvp=jvp,ta=ta)
        def invoke(v):
            return list(owner.native_jvp(*v[:4],tangents=tuple(v[4:]))) if jvp else [owner(*v)]
        maximum=check(invoke(values),expected(values,row))
        if jvp:
            cached=next(iter(owner._native_jvp_packages.values()))
            manifest=cached.contract["steps"][0]["child_metadata"]["native_scaled_program"]
        else:manifest=owner.compile_result.launch_descriptor.provenance["native_scaled_primal_program"]
        package=NativeScaledProgram.from_manifest(manifest)
        _,_,changed,_=case(mask,tb,encoded,jvp=jvp,seed=5007,ta=ta)
        wanted=expected(changed,row)
        original=subprocess.run
        def forbidden(*args,**kwargs):raise AssertionError("warm replay invoked compiler")
        subprocess.run=forbidden
        try:
            maximum=max(maximum,check(invoke(changed),wanted))
            public=[];events=[];host=[]
            for _ in range(5):
                start=time.perf_counter()
                for _ in range(5):actual=invoke(changed)
                public.append((time.perf_counter()-start)*1e3/5)
                maximum=max(maximum,check(actual,wanted))
            with PreparedScaledProgram(package,changed,runtime_library=str(library)) as prepared:
                old,_=prepared.invoke()
                maximum=max(maximum,check(prepared.read(old),wanted))
                for _ in range(5):
                    generation,elapsed=prepared.invoke(repeats=20,timed=True)
                    # HIP ABI already divides its event window by repeats.
                    events.append(elapsed)
                    maximum=max(maximum,check(prepared.read(generation),wanted))
                    start=time.perf_counter()
                    for _ in range(5):
                        prepared.update(changed)
                        generation,_=prepared.invoke()
                        actual=prepared.read(generation)
                    host.append((time.perf_counter()-start)*1e3/5)
                    maximum=max(maximum,check(actual,wanted))
                try:prepared.read(old)
                except RuntimeError:pass
                else:raise AssertionError("stale generation accepted")
        finally:subprocess.run=original
        rows.append({"mask":mask,"transposeA":ta,"transposeB":tb,"encoded":encoded,"kind":row["kind"],
            "max_abs_error":maximum,"compiler_free_changed_input_replay":True,
            "stale_generation":"rejected","public_warm_samples_ms":public,
            "public_warm_median_ms":median(public),"native_launch_window_samples_ms":events,
            "native_launch_window_median_ms":median(events),
            "prepared_update_invoke_read_samples_ms":host,
            "prepared_update_invoke_read_median_ms":median(host),
            "program_sha256":hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
            "images_sha256":[hashlib.sha256(x).hexdigest() for x in package.images]})
        print("passed",len(rows),"/",len(profiles),flush=True)
    from tessera.compiler import native_vmap,rocm_typed_scaled_native,native_scaled_program
    return {"architecture":arch,"device":name.value.decode(),"device_uuid_raw_hex":bytes(uuid).hex(),
        "compiler_sha256":digest(os.environ["TESSERA_OPT"]),"runtime_sha256":digest(library),
        "recorder_sha256":digest(__file__),"adapter_source_sha256":{m.__name__:digest(m.__file__) for m in
        (rt,native_vmap,rocm_typed_scaled_native,native_scaled_program)},
        "timing_boundaries":{"public":"compiler-free warm host call including frontend, upload, execution and readback",
            "native":"prepared native program HIP launch window averaged by runtime over 20 repeats",
            "prepared":"update/invoke/read host wall; preparation excluded"},
        "limitations":["static M3 N5 K64 and prefix 2x3","FP32 scale derivatives only",
            "no speedup or selector promotion","no sibling-device proof"],"rows":rows}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=measure()
    args.output.write_text(json.dumps(result,indent=2)+"\n")
