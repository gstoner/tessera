#!/usr/bin/env python3
"""Exact gfx1201 ordinary JIT checkpoint chain, cache and portable timing."""
from pathlib import Path
import argparse
import ctypes as C
import hashlib
import json
import os
import time
from statistics import median

import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.rocm_nvfp4_ingest import nvfp4_requantization_policy
from tessera.compiler.rocm_mxfp4_storage import MXFP4_STORAGE_CONTRACT
from tessera.compiler.rocm_nvfp4_program import packed_consumer_attrs
from tests.unit.test_rocm_nvfp4_resident import inputs_and_oracle

ROOT=Path(__file__).resolve().parents[2]


def frontend_function(n,k,reordered=False,*,row_offsets=None):
    offsets=list(row_offsets) if row_offsets is not None else [0,n//2,n]
    policy=nvfp4_requantization_policy()
    attributes=packed_consumer_attrs(k)
    if reordered:
        @ts.jit(target="rocm_gfx1201")
        def chain(a_scale,a,codes,projection_globals,scales):
            packed,exponents,stats=ts.ops.nvfp4_requantize(codes,scales,projection_globals,
                row_offsets=offsets,numeric_policy=policy)
            fragment,plane=ts.ops.mxfp4_folded_storage(packed,exponents,
                storage_contract=MXFP4_STORAGE_CONTRACT)
            return ts.ops.scaled_matmul(a,fragment,a_scale,plane,
                physical_contract=attributes["physical_contract"],
                numeric_policy=attributes["numeric_policy"],scale_layout=attributes["scale_layout"])
    else:
        @ts.jit(target="rocm_gfx1201")
        def chain(codes,scales,projection_globals,a,a_scale):
            packed,exponents,stats=ts.ops.nvfp4_requantize(codes,scales,projection_globals,
                row_offsets=offsets,numeric_policy=policy)
            fragment,plane=ts.ops.mxfp4_folded_storage(packed,exponents,
                storage_contract=MXFP4_STORAGE_CONTRACT)
            return ts.ops.scaled_matmul(a,fragment,a_scale,plane,
                physical_contract=attributes["physical_contract"],
                numeric_policy=attributes["numeric_policy"],scale_layout=attributes["scale_layout"])
    return chain


def record_case(shape,reordered,windows):
    m,n,k=shape
    args,offsets,converted,stored,expected=inputs_and_oracle(m,n,k)
    return record_arrays(args,expected,reordered=reordered,windows=windows,
                         converted=converted,stored=stored)


def record_arrays(args,expected,*,reordered,windows,converted=None,stored=None,verify=None,row_offsets=None):
    m,k=args[3].shape
    n=args[0].shape[0]
    shape=(m,n,k)
    def check(output):
        if verify is not None:
            return verify(output)
        np.testing.assert_allclose(output.astype(np.float32),expected,rtol=.008,atol=.015625)
    named=dict(zip(("codes","scales","projection_globals","a","a_scale"),args))
    function=frontend_function(n,k,reordered,row_offsets=row_offsets)
    start=time.perf_counter_ns()
    output=function(**named)
    cold_ms=(time.perf_counter_ns()-start)/1e6
    check(output)
    if function.execution_kind!="native_gpu":
        raise AssertionError("JIT did not execute the native program")
    packages=function.native_nvfp4_packages()
    start=time.perf_counter_ns()
    encoded=function.runtime_artifact().to_json()
    serialize_ms=(time.perf_counter_ns()-start)/1e6
    warm,replay,restore=[],[],[]
    for _ in range(windows):
        start=time.perf_counter_ns()
        output=function(**named)
        warm.append((time.perf_counter_ns()-start)/1e6)
        check(output)
        assert packages==function.native_nvfp4_packages()
        start=time.perf_counter_ns()
        artifact=rt.RuntimeArtifact.from_json(encoded)
        restore.append((time.perf_counter_ns()-start)/1e6)
        start=time.perf_counter_ns()
        receipt=rt.launch(artifact,named)
        replay.append((time.perf_counter_ns()-start)/1e6)
        if receipt.get("ok") is not True or receipt.get("execution_kind")!="native_gpu":
            raise AssertionError(receipt)
        np.testing.assert_array_equal(receipt["output"],output)
    program=function._rocm_nvfp4_last_program
    with program.native.native_session(*args) as session:
        session.run_combined()
        if converted is not None:
            np.testing.assert_array_equal(session.diagnostics()["packed"],converted[0])
        if stored is not None:
            np.testing.assert_array_equal(session.diagnostics()["plane"],stored[1])
        device={}
        events={}
        for stage in ("converter","storage","consumer","combined"):
            session.run_combined()
            events[stage]=session.measure(stage,samples=windows,repeats=32)
            device[stage]=session.measure_graph(stage,samples=windows,repeats=32)
        check(session.read_output())
    return {
        "shape_mnk":list(shape),"frontend_argument_order":list(program.argument_names),
        "role_indices":list(program.role_indices),"cold_jit_wall_ms":cold_ms,
        "warm_jit_wall_samples_ms":warm,"warm_jit_wall_median_ms":median(warm),
        "serialization_wall_ms":serialize_ms,"serialized_artifact_bytes":len(encoded.encode()),
        "artifact_restore_wall_samples_ms":restore,"artifact_restore_wall_median_ms":median(restore),
        "portable_checked_launch_wall_samples_ms":replay,"portable_checked_launch_wall_median_ms":median(replay),
        "device_event_samples_ms":events,
        "device_event_median_ms":{stage:median(values) for stage,values in events.items()},
        "device_graph_windows":device,
        "device_graph_dispatch_median_ms":{stage:median(x["per_iteration_ms"] for x in values)
                                          for stage,values in device.items()},
        "numeric_correctness":"independent conversion/storage/folded float64 oracle before and after timing",
        "max_abs_error":float(np.max(np.abs(output.astype(np.float64)-expected))),
        "component_image_digests":[p.image.image_digest for p in packages],
        "component_descriptors":[p.descriptor.to_dict() for p in packages],
        "frontend_graph_sha256":hashlib.sha256(program.graph_ir.encode()).hexdigest(),
        "input_sha256":{name:hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                        for name,value in named.items()},
        "native_call_binding":"native_cpp_nvfp4",
        "route":"Python frontend -> verified Graph -> native Schedule/Tile/Target/LLVM -> three HSACO images -> checked resident HIP program",
    }


def record(arguments):
    if arguments.windows<3:
        raise ValueError("at least three independent windows required")
    os.environ["TESSERA_OPT"]=str(arguments.compiler.resolve())
    if rt._rocm_live_arch()!="gfx1201":
        raise RuntimeError("benchmark requires exact live gfx1201")
    hip=rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):
        raise RuntimeError("HIP unavailable")
    hip.hipGetDevice.argtypes=[C.POINTER(C.c_int)]
    hip.hipDeviceGetName.argtypes=[C.c_char_p,C.c_int,C.c_int]
    device=C.c_int();name=C.create_string_buffer(256)
    if hip.hipGetDevice(C.byref(device)) or hip.hipDeviceGetName(name,len(name),device.value) or not name.value:
        raise RuntimeError("actual GPU identity required")
    packet={"architecture":"gfx1201","device":name.value.decode(),"device_ordinal":device.value,
        "compiler_sha256":hashlib.sha256(arguments.compiler.read_bytes()).hexdigest(),
        "scope":"named static primal ordinary JIT program; general AD/dynamic/layout/model quality remain open",
        "timing_scope":"cold includes compilation and checked full launch; warm uses image cache and bounded native owner reuse; portable launch includes allocation/upload/readback/cleanup; graph windows include GPU dispatch",
        "selector_promotion":False,"rows":[]}
    paths=("python/tessera/compiler/rocm_nvfp4_program.py","python/tessera/compiler/rocm_nvfp4_resident.py",
        "python/tessera/compiler/jit.py","python/tessera/compiler/graph_ir.py",
        "python/tessera/compiler/op_catalog.py","python/tessera/compiler/trace.py",
        "python/tessera/compiler/native_nvfp4_program.py","src/transforms/lib/NativeNVFP4Program.h",
        "python/tessera/runtime.py","src/compiler/programming_model/lib/PMPasses.cpp",
        "benchmarks/rocm/benchmark_jit_nvfp4_program.py","tests/unit/test_rocm_nvfp4_resident.py")
    packet["source_sha256"]={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    arguments.output.parent.mkdir(parents=True,exist_ok=True)
    for shape in ((128,32,256),(257,80,1024),(256,64,64)):
        for reordered in (False,True):
            row=record_case(shape,reordered,arguments.windows)
            packet["rows"].append(row)
            arguments.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+"\n")
            print("verified JIT/portable",shape,reordered,row["warm_jit_wall_median_ms"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--windows",type=int,default=3)
    record(parser.parse_args())
