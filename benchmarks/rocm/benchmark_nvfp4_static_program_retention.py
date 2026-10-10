"""Matched warm static program retention; native allocation reuse in both arms."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
from statistics import median
import time
import numpy as np
from tessera import runtime as rt
from tessera.compiler import prepared_rocm_nvfp4_program as cache
from tessera.compiler import rocm_nvfp4_program as source
from tests.device.rocm.test_nvfp4_resident_jit import make_function
from tests.unit.test_rocm_nvfp4_resident import inputs_and_oracle
from tests.device.rocm.test_native_nvfp4_allocation_reuse import changed_inputs,clear_cache


def record(shape,samples):
    arrays,offsets,_,_,expected=inputs_and_oracle(*shape)
    changed,_,_,changed_expected=changed_inputs(arrays,offsets)
    owned=[value.copy() for value in arrays]
    named=dict(zip(("codes","scales","projection_globals","a","a_scale"),owned))
    function=make_function(shape[1],shape[2],True)
    original=cache.resolve_program
    mode={"retained":True}
    def controlled(artifact):
        if mode["retained"]:
            return original(artifact)
        metadata=artifact.metadata
        program=source.program_from_manifest(metadata["native_program"])
        if artifact.graph_ir!=program.graph_ir or metadata["arg_names"]!=list(program.argument_names):
            raise AssertionError("control parent contract differs")
        return program
    cache.resolve_program=controlled
    clear_cache()
    try:
        for _ in range(3):
            function(**named)
        images=[p.image.image_digest for p in function.native_nvfp4_packages()]
        walls={"retained_static_contract":[], "reconstructed_contract_control":[]}
        for trial in range(samples):
            index=trial%2
            for dest,value in zip(owned,(arrays,changed)[index],strict=True):
                np.copyto(dest,value)
            labels=list(walls)
            if trial%2:labels.reverse()
            for label in labels:
                mode["retained"]=label=="retained_static_contract"
                start=time.perf_counter_ns()
                actual=function(**named)
                walls[label].append((time.perf_counter_ns()-start)/1e6)
                assert function.execution_kind=="native_gpu"
                np.testing.assert_allclose(actual.astype(np.float32),(expected,changed_expected)[index],
                                           rtol=.008,atol=.015625)
                assert images==[p.image.image_digest for p in function.native_nvfp4_packages()]
        medians={key:median(value) for key,value in walls.items()}
        return {"shape_mnk":list(shape),"component_image_digests":images,
                "wall_samples_ms":walls,"wall_medians_ms":medians,
                "retained_over_reconstructed":medians["retained_static_contract"]/medians["reconstructed_contract_control"],
                "correctness":"independent folded oracle after every call; unchanged addresses with alternating five-input values"}
    finally:
        cache.resolve_program=original
        clear_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=11)
    args=parser.parse_args()
    if args.samples<3:raise ValueError("requires at least three balanced trials")
    if rt._rocm_live_arch()!="gfx1201":raise RuntimeError("exact gfx1201 required")
    hip=rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):raise RuntimeError("HIP unavailable")
    ordinal=C.c_int();name=C.create_string_buffer(256)
    if hip.hipGetDevice(C.byref(ordinal)) or hip.hipDeviceGetName(name,len(name),ordinal.value):
        raise RuntimeError("live device name unavailable")
    uuid=C.create_string_buffer(16)
    hip.hipDeviceGetUuid.argtypes=[C.c_void_p,C.c_int]
    hip.hipDeviceGetUuid.restype=C.c_int
    if hip.hipDeviceGetUuid(uuid,ordinal.value):raise RuntimeError("live HIP UUID unavailable")
    rows=[record(shape,args.samples) for shape in ((128,32,256),(257,80,1024),(256,64,64))]
    root=Path(__file__).resolve().parents[2]
    paths=("python/tessera/runtime.py","python/tessera/compiler/prepared_rocm_nvfp4_program.py",
           "python/tessera/compiler/rocm_nvfp4_program.py",
           "benchmarks/rocm/benchmark_nvfp4_static_program_retention.py")
    packet={"schema":"tessera.rocm.nvfp4.static-program-retention.v1",
            "architecture":"gfx1201","device":name.value.decode(),"device_ordinal":ordinal.value,
            "opaque_hip_uuid":uuid.raw.hex(),"rows":rows,
            "source_sha256":{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
            "runtime_sha256":hashlib.sha256(Path(rt._load_rocm_native_movement_runtime()._name).read_bytes()).hexdigest(),
            "timing_domain":"ordinary warm host-array JIT wall: frontend dispatch, checked contract, native preparation/rebinding, uploads, three kernels, readback and release",
            "control":"same images and native allocation reuse; only validated static-program retention is disabled",
            "kernel_schedule_changed":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+"\n")
    print(json.dumps([{"shape":r["shape_mnk"],"wall_ms":r["wall_medians_ms"],
                      "ratio":r["retained_over_reconstructed"]} for r in rows],indent=2))
if __name__=="__main__":main()
