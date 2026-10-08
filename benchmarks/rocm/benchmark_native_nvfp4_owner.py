"""Matched native versus Python NVFP4 resident ownership on exact gfx1201."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
from statistics import median
import time
import numpy as np
from tessera import runtime as rt
from tessera.compiler.rocm_nvfp4_ingest import nvfp4_requantization_policy
from tessera.compiler.rocm_nvfp4_resident import package_resident_nvfp4_matmul
from tests.unit.test_rocm_nvfp4_resident import inputs_and_oracle

def record(shape):
    arrays,offsets,converted,stored,expected=inputs_and_oracle(*shape)
    program=package_resident_nvfp4_matmul(*shape,offsets,
        numeric_policy=nvfp4_requantization_policy(),approximate_policy="explicit_allow")
    factories={"python":program.session,"native":program.native_session,
               "native_graph":program.native_session}
    def combined(name,session):
        if name=="native_graph":session.run_combined_graph()
        else:session.run_combined()
    def consumer(name,session):
        if name=="native_graph":session.launch_matmul_graph()
        else:session.launch_matmul()
    sessions={};records={}
    def verify(session):
        np.testing.assert_allclose(session.read_output().astype(np.float32),expected,rtol=.008,atol=.015625)
        d=session.diagnostics()
        for name,wanted in zip(("packed","exponents","stats"),converted):
            if name=="stats":np.testing.assert_allclose(d[name],wanted,rtol=1e-13,atol=1e-30)
            else:np.testing.assert_array_equal(d[name],wanted)
        np.testing.assert_array_equal(d["fragment"],stored[0])
        np.testing.assert_array_equal(d["plane"],stored[1])
    try:
        for name,factory in factories.items():
            session=factory(*arrays);sessions[name]=session
            combined(name,session);verify(session)
            records[name]={"warm_combined_wall_ms":[],"weight_reuse_wall_ms":[],
                           "allocating_combined_wall_ms":[],
                           "stage_event_samples_ms":{}, "stage_graph_windows":{}}
        for trial in range(7):
            names=list(sessions)
            shift=trial%len(names)
            names=names[shift:]+names[:shift]
            if trial%2:names.reverse()
            for name in names:
                session=sessions[name]
                start=time.perf_counter_ns()
                combined(name,session);out=session.read_output()
                records[name]["warm_combined_wall_ms"].append((time.perf_counter_ns()-start)/1e6)
                np.testing.assert_allclose(out.astype(np.float32),expected,rtol=.008,atol=.015625)
                start=time.perf_counter_ns()
                consumer(name,session);out=session.read_output()
                records[name]["weight_reuse_wall_ms"].append((time.perf_counter_ns()-start)/1e6)
                np.testing.assert_allclose(out.astype(np.float32),expected,rtol=.008,atol=.015625)
                start=time.perf_counter_ns()
                with factories[name](*arrays) as active:
                    combined(name,active);out=active.read_output()
                records[name]["allocating_combined_wall_ms"].append((time.perf_counter_ns()-start)/1e6)
                np.testing.assert_allclose(out.astype(np.float32),expected,rtol=.008,atol=.015625)
        for name,session in sessions.items():
            for stage in ("converter","storage","consumer","combined"):
                session.run_combined()
                records[name]["stage_event_samples_ms"][stage]=session.measure(stage,samples=3,repeats=16)
                session.run_combined()
                records[name]["stage_graph_windows"][stage]=session.measure_graph(stage,samples=3,repeats=16)
            combined(name,session);verify(session)
            for key in ("warm_combined_wall_ms","weight_reuse_wall_ms","allocating_combined_wall_ms"):
                records[name][key.replace("_ms","_median_ms")]=median(records[name][key])
        ratios={key:records["native"][key]/records["python"][key]
                for key in ("warm_combined_wall_median_ms","weight_reuse_wall_median_ms",
                            "allocating_combined_wall_median_ms")}
        return {"shape_mnk":shape,"component_image_digests":program.receipt["component_image_digests"],
                "correctness":"independent conversion, lossless storage and folded output before/after timing",
                "timing_scope":"same compiler images; resident/reuse walls include output readback; allocating walls also own images, storage and upload; stage events include dispatch gaps",
                "arms":records,"native_over_python":ratios,
                "native_graph_over_native":{key:records["native_graph"][key]/records["native"][key]
                    for key in ratios}}
    finally:
        for session in sessions.values():session.close()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if rt._rocm_live_arch()!="gfx1201":raise RuntimeError("exact gfx1201 required")
    hip=rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):raise RuntimeError("HIP unavailable")
    name=C.create_string_buffer(256);ordinal=C.c_int()
    if hip.hipGetDevice(C.byref(ordinal)) or hip.hipDeviceGetName(name,len(name),ordinal.value):
        raise RuntimeError("device identity unavailable")
    rows=[record(shape) for shape in ((128,32,256),(257,80,1024),(256,64,64))]
    root=Path(__file__).resolve().parents[2]
    sources=("python/tessera/compiler/native_resident_nvfp4.py",
             "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_nvfp4_runtime.cpp",
             "benchmarks/rocm/benchmark_native_nvfp4_owner.py")
    packet={"architecture":"gfx1201","device":name.value.decode(),"device_ordinal":ordinal.value,
            "source_sha256":{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
            "native_library_sha256":hashlib.sha256(Path(rt._load_rocm_native_movement_runtime()._name).read_bytes()).hexdigest(),
            "ownership":"native private stream, 11 allocations, three image leases, native argument binding/copy/readiness/events/close",
            "scope":"matched explicit resident ownership/graph APIs; ordinary JIT separately measured; no default promotion",
            "rows":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+"\n")
    print(json.dumps([{"shape":r["shape_mnk"],"ratios":r["native_over_python"]} for r in rows],indent=2))
if __name__=="__main__":main()
