"""Matched native idle-allocation reuse on exact gfx1201; identical images."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
from statistics import median
import time
import numpy as np
from tessera import runtime as rt
from tessera.compiler.rocm_nvfp4_resident import package_resident_nvfp4_matmul
from tessera.compiler.rocm_nvfp4_ingest import nvfp4_requantization_policy
from tests.unit.test_rocm_nvfp4_resident import inputs_and_oracle
from tests.device.rocm.test_native_nvfp4_allocation_reuse import changed_inputs, clear_cache


def record(shape, samples):
    args,offsets,_,_,expected=inputs_and_oracle(*shape)
    changed,_,_,changed_expected=changed_inputs(args,offsets)
    program=package_resident_nvfp4_matmul(*shape,offsets,
        numeric_policy=nvfp4_requantization_policy(),approximate_policy="explicit_allow")
    variants=(args,changed)
    references=(expected,changed_expected)
    values=[x.copy() for x in args] # identical pointers, different values each trial
    walls={"uncached":[], "cached":[]}
    hits=[]
    clear_cache()
    def invoke(name):
        with program.native_session(*values,reuse=name=="cached") as active:
            hit=active.native_cache_hit
            active.run_combined()
            out=active.read_output()
        return out,hit
    for _ in range(3):
        for name in walls:
            out,_=invoke(name)
            np.testing.assert_allclose(out.astype(np.float32),expected,rtol=.008,atol=.015625)
    for trial in range(samples):
        index=trial%2
        for dest,src in zip(values,variants[index],strict=True):
            np.copyto(dest,src)
        names=("cached","uncached") if trial%2 else ("uncached","cached")
        for name in names:
            start=time.perf_counter_ns()
            out,hit=invoke(name)
            walls[name].append((time.perf_counter_ns()-start)/1e6)
            if name=="cached":
                assert hit
                hits.append(hit)
            np.testing.assert_allclose(out.astype(np.float32),references[index],rtol=.008,atol=.015625)
    events={}
    for name in ("uncached","cached"):
        with program.native_session(*values,reuse=name=="cached") as active:
            active.run_combined()
            np.testing.assert_allclose(active.read_output().astype(np.float32),
                references[(samples-1)%2],rtol=.008,atol=.015625)
            events[name]=active.measure("combined",samples=3,repeats=64)
            np.testing.assert_allclose(active.read_output().astype(np.float32),
                references[(samples-1)%2],rtol=.008,atol=.015625)
    clear_cache()
    return {"shape_mnk":list(shape),"component_image_digests":program.receipt["component_image_digests"],
            "wall_samples_ms":walls,"wall_medians_ms":{key:median(v) for key,v in walls.items()},
            "cached_over_uncached_wall":median(walls["cached"])/median(walls["uncached"]),
            "resident_combined_event_samples_ms":events,
            "timed_cached_hits":hits,
            "correctness":"independent conversion/folded storage/matmul oracle after every call; same input pointers rebound to alternating full-input values"}


def record_public(shape, samples):
    from tessera.compiler.rocm_nvfp4_resident import NVFP4ResidentProgram
    from tests.device.rocm.test_nvfp4_resident_jit import make_function
    arrays,offsets,_,_,expected=inputs_and_oracle(*shape)
    changed,_,_,changed_expected=changed_inputs(arrays,offsets)
    owned=[value.copy() for value in arrays]
    named=dict(zip(("codes","scales","projection_globals","a","a_scale"),owned))
    function=make_function(shape[1],shape[2],True)
    mode={"reuse":True,"hit":False}
    original=NVFP4ResidentProgram.native_session
    def controlled(self,*args,**kwargs):
        active=original(self,*args,reuse=mode["reuse"])
        mode["hit"]=active.native_cache_hit
        return active
    NVFP4ResidentProgram.native_session=controlled
    clear_cache()
    try:
        function(**named)
        images=[p.image.image_digest for p in function.native_nvfp4_packages()]
        walls={"uncached":[], "cached":[]}
        for trial in range(samples):
            index=trial%2
            for dest,src in zip(owned,(arrays,changed)[index],strict=True):
                np.copyto(dest,src)
            names=("cached","uncached") if trial%2 else ("uncached","cached")
            for arm in names:
                mode["reuse"]=arm=="cached"
                start=time.perf_counter_ns()
                out=function(**named)
                walls[arm].append((time.perf_counter_ns()-start)/1e6)
                assert mode["hit"]==(arm=="cached")
                assert function.execution_kind=="native_gpu"
                np.testing.assert_allclose(out.astype(np.float32),(expected,changed_expected)[index],
                                           rtol=.008,atol=.015625)
                assert images==[p.image.image_digest for p in function.native_nvfp4_packages()]
        return {"shape_mnk":list(shape),"component_image_digests":images,
                "wall_samples_ms":walls,"wall_medians_ms":{key:median(v) for key,v in walls.items()},
                "cached_over_uncached_wall":median(walls["cached"])/median(walls["uncached"]),
                "correctness":"ordinary reordered @jit, identical images, same addresses with alternating full-input values; independent oracle after each call",
                "control":"same public JIT and checked artifact path; only native_session reuse flag is changed"}
    finally:
        NVFP4ResidentProgram.native_session=original
        clear_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=11)
    args=parser.parse_args()
    if args.samples<3:raise ValueError("at least three alternating samples required")
    if rt._rocm_live_arch()!="gfx1201":raise RuntimeError("exact gfx1201 required")
    hip=rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):raise RuntimeError("HIP unavailable")
    device=C.c_int();name=C.create_string_buffer(256)
    if hip.hipGetDevice(C.byref(device)) or hip.hipDeviceGetName(name,len(name),device.value):
        raise RuntimeError("live device identity unavailable")
    uuid=C.create_string_buffer(16)
    hip.hipDeviceGetUuid.argtypes=[C.c_void_p,C.c_int]
    hip.hipDeviceGetUuid.restype=C.c_int
    if hip.hipDeviceGetUuid(uuid,device.value):raise RuntimeError("live device UUID unavailable")
    rows=[record(shape,args.samples) for shape in ((128,32,256),(257,80,1024),(256,64,64))]
    root=Path(__file__).resolve().parents[2]
    source_paths=("src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_nvfp4_runtime.cpp",
                  "python/tessera/compiler/native_resident_nvfp4.py",
                  "python/tessera/compiler/rocm_nvfp4_program.py",
                  "benchmarks/rocm/benchmark_native_nvfp4_allocation_reuse.py")
    packet={"schema":"tessera.rocm.nvfp4.native-allocation-reuse.v1",
            "architecture":"gfx1201","device":name.value.decode(),"device_ordinal":device.value,
            "device_uuid":uuid.raw.hex(),
            "work_item":"ROCM-NVFP4-INGEST-1","rows":rows,
            "source_sha256":{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source_paths},
            "runtime_sha256":hashlib.sha256(Path(rt._load_rocm_native_movement_runtime()._name).read_bytes()).hexdigest(),
            "wall_domain":"native_session construction, full input upload, three kernel launches, readback and close/release; oracle outside window",
            "event_domain":"native HIP event windows around combined kernels on resident buffers; dispatch gaps included, transfers excluded",
            "cache_policy":"four idle owners; 64 MiB accounted device/staging/image/readback bytes; complete image/entry/shape/geometry/context keys; fresh tokens; full rebinding",
            "ordinary_jit_promoted":True,
            "ordinary_jit_rows":[record_public(shape,args.samples)
                for shape in ((128,32,256),(257,80,1024),(256,64,64))]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+"\n")
    print(json.dumps([{"shape":r["shape_mnk"],"ratio":r["cached_over_uncached_wall"],
                      "wall_ms":r["wall_medians_ms"]} for r in rows],indent=2))
if __name__=="__main__":main()
