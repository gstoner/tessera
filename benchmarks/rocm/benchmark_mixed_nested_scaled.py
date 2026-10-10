"""Exact gfx1201 mixed nested maps with native scale primal/JVP/VJP timing."""
import argparse
from dataclasses import replace
import ctypes as c
import hashlib
import itertools
import json
import os
import time
from pathlib import Path
from statistics import median
import numpy as np
from tessera import runtime as rt
from tessera.compiler.native_scaled_program import (
    package_native_scaled_primal,package_native_scaled_jvp,
    package_native_scaled_vjp,PreparedScaledProgram)
from tests.unit.test_native_mixed_scaled_maps import mixed_case, mixed_oracle
from tessera.compiler.native_vmap import normalize_mixed_batch_inputs

def gradients(values,cot,policies,nk):
    result=[]
    for operand in (2,3):
        gradient=np.empty_like(values[operand],dtype=np.float64)
        for index in np.ndindex(gradient.shape):
            plus=list(values);minus=list(values)
            plus[operand]=values[operand].astype(np.float64)
            minus[operand]=values[operand].astype(np.float64)
            plus[operand][index]+=.001;minus[operand][index]-=.001
            gradient[index]=np.sum((mixed_oracle(plus,policies,"fp32",nk)-mixed_oracle(minus,policies,"fp32",nk))*cot)/.002
        result.append(gradient)
    return tuple(result)

def record(fmt,nk,cartesian,kind,*,warm_samples=3):
    mode={"primal":None,"jvp":"forward","vjp":"reverse"}[kind]
    _,_,fn,values,expected_primal=mixed_case(fmt,nk,cartesian,mode=mode)
    policies=fn._frontend_batch_policies
    normalized=tuple(fn._ordered_inputs(values,{}))
    rng=np.random.default_rng(10108)
    tangents=tuple(rng.uniform(-.2,.2,size=v.shape).astype(np.float32) for v in values[2:])
    cot=rng.uniform(-1,1,size=(2,3,3,5)).astype(np.float32)
    if kind=="primal":
        expected=(expected_primal,)
        call=lambda:(fn(*values),)
        inputs=normalized
    elif kind=="jvp":
        expected=(expected_primal,
                  mixed_oracle((*values[:2],tangents[0],values[3]),policies,"fp32",nk)+
                  mixed_oracle((*values[:3],tangents[1]),policies,"fp32",nk))
        call=lambda:fn.native_jvp(*values,tangents=tangents)
        seed_values=(*values[:2],*tangents)
        seed_views=normalize_mixed_batch_inputs(seed_values,policies)
        inputs=(*normalized,*seed_views[2:])
    else:
        expected=gradients(values,cot,policies,nk)
        call=lambda:fn.native_backward(*values,out_cotangents=cot)
        inputs=(*normalized,cot)
    def check(outputs, *, prepared=False):
        assert len(outputs)==len(expected)
        for got,wanted in zip(outputs,expected,strict=True):
            if prepared and kind=="vjp":
                got=got.reshape(wanted.shape)
            assert got.shape==wanted.shape
            np.testing.assert_allclose(got,wanted,rtol=3e-5,atol=2e-5)
    start=time.perf_counter_ns()
    actual=call();cold=(time.perf_counter_ns()-start)/1e6
    check(actual)
    warm=[]
    for _ in range(warm_samples):
        start=time.perf_counter_ns();actual=call()
        warm.append((time.perf_counter_ns()-start)/1e6);check(actual)
    traced=fn._specialized_autodiff_module(values,{})
    projected=replace(traced,module_attrs={**traced.module_attrs,
        "tessera.target":'"rocm"',"tessera.arch":'"gfx1201"'})
    graph=projected.to_mlir(target="rocm_gfx1201",canonical=True)
    package={"primal":package_native_scaled_primal,"jvp":package_native_scaled_jvp,
             "vjp":package_native_scaled_vjp}[kind](graph)
    package.validate()
    lib=rt._load_rocm_native_movement_runtime()
    if lib is None:raise RuntimeError("native owner unavailable")
    with PreparedScaledProgram(package,inputs,runtime_library=lib._name) as owner:
        generation,_=owner.invoke();check(owner.read(generation),prepared=True)
        events=[]
        for _ in range(3):
            generation,elapsed=owner.invoke(repeats=32,timed=True)
            events.append(elapsed);check(owner.read(generation),prepared=True)
    return {"kind":kind,"scale_format":fmt,"transposeB":nk,"cartesian_maps":cartesian,"map_level_axes":[list(p) for p in policies],
        "input_shapes":[list(v.shape) for v in values],"normalized_input_shapes":[list(v.shape) for v in normalized],"output_shapes":[list(v.shape) for v in expected],
        "max_abs_error":max(float(np.max(np.abs(a-b))) for a,b in zip(actual,expected,strict=True)),
        "cold_public_wall_ms":cold,"warm_public_wall_samples_ms":warm,
        "warm_public_wall_median_ms":median(warm),"prepared_program_event_samples_ms":events,
        "prepared_program_event_median_ms":median(events),
        "native_program_sha256":hashlib.sha256(package.program_json.encode()).hexdigest(),
        "member_image_sha256":[hashlib.sha256(v).hexdigest() for v in package.images],
        "frontend_graph_sha256":hashlib.sha256(graph.encode()).hexdigest(),
        "correctness":"independent float64 primal/JVP and finite-difference scale VJP before/after timing"}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if rt._rocm_live_arch()!="gfx1201":raise RuntimeError("exact gfx1201 required")
    hip=rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):raise RuntimeError("HIP unavailable")
    ordinal=c.c_int();name=c.create_string_buffer(256);uuid=(c.c_ubyte*16)()
    if hip.hipGetDevice(c.byref(ordinal)) or hip.hipDeviceGetName(name,256,ordinal.value) or hip.hipDeviceGetUuid(c.byref(uuid),ordinal.value):
        raise RuntimeError("live GPU identity required")
    compiler=Path(os.environ["TESSERA_OPT"])
    root=Path(__file__).resolve().parents[2]
    paths=("python/tessera/compiler/jit.py","python/tessera/compiler/native_vmap.py","python/tessera/compiler/graph_ir.py","python/tessera/compiler/native_scaled_program.py",
           "python/tessera/compiler/native_jvp_plugins.py","python/tessera/compiler/native_vjp_plugins.py",
           "src/compiler/ir/include/Tessera/IR/ScaledBatchContract.h","src/compiler/ir/LinearTransposeInterface.cpp",
           "src/compiler/programming_model/lib/PMPasses.cpp",
           "tests/unit/test_native_mixed_scaled_maps.py","tests/device/rocm/test_mixed_nested_scaled_execution.py",
           "benchmarks/rocm/benchmark_mixed_nested_scaled.py")
    runtime=Path(rt._load_rocm_native_movement_runtime()._name)
    packet={"schema":"tessera.mixed_nested_scaled.v1","architecture":"gfx1201",
        "device":name.value.decode(),"device_ordinal":ordinal.value,"device_uuid":bytes(uuid).hex(),
        "compiler_sha256":hashlib.sha256(compiler.read_bytes()).hexdigest(),
        "runtime_sha256":hashlib.sha256(runtime.read_bytes()).hexdigest(),
        "source_sha256":{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        "timing_scope":"cold/warm public wall includes checked package/host transfers; prepared HIP event window covers native program dispatch and member kernels; not an isolated single-kernel metric",
        "selector_promotion":False,"rows":[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    profiles=[(fmt,nk,cartesian,kind) for fmt in ("fp32","e8m0")
              for nk in (False,True) for cartesian in (False,True)
              for kind in (("primal","jvp","vjp") if fmt=="fp32" else ("primal",))]
    for profile in profiles:
        row=record(*profile);packet["rows"].append(row)
        args.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+"\n")
        print("verified",profile,row["max_abs_error"],flush=True)

if __name__=="__main__":main()
