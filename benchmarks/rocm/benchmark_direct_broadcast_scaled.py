"""Exact gfx1201 direct broadcast JIT/AD and prepared-program timing."""
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
from tests.unit.test_direct_broadcast_scaled_frontend import case,oracle

def gradients(values,cot,ta,tb):
    result=[]
    for operand in (2,3):
        gradient=np.empty_like(values[operand],dtype=np.float64)
        for index in np.ndindex(gradient.shape):
            plus=list(values);minus=list(values)
            plus[operand]=values[operand].astype(np.float64)
            minus[operand]=values[operand].astype(np.float64)
            plus[operand][index]+=.001;minus[operand][index]-=.001
            gradient[index]=np.sum((oracle(plus,ta,tb)-oracle(minus,ta,tb))*cot)/.002
        result.append(gradient)
    return tuple(result)

def record(encoded,ta,tb,kind):
    mode={"primal":None,"jvp":"forward","vjp":"reverse"}[kind]
    fn,values=case(encoded,ta,tb,mode=mode)
    rng=np.random.default_rng(10108)
    tangents=tuple(rng.uniform(-.2,.2,size=v.shape).astype(np.float32) for v in values[2:])
    cot=rng.uniform(-1,1,size=(2,3,3,5)).astype(np.float32)
    if kind=="primal":
        expected=(oracle(values,ta,tb,encoded),)
        call=lambda:(fn(*values),)
        inputs=values
    elif kind=="jvp":
        expected=(oracle(values,ta,tb),oracle((*values[:2],tangents[0],values[3]),ta,tb)+
                  oracle((*values[:3],tangents[1]),ta,tb))
        call=lambda:fn.native_jvp(*values,tangents=tangents)
        inputs=(*values,*tangents)
    else:
        expected=gradients(values,cot,ta,tb)
        call=lambda:fn.native_backward(*values,out_cotangents=cot)
        inputs=(*values,cot)
    def check(outputs):
        assert len(outputs)==len(expected)
        for got,wanted in zip(outputs,expected,strict=True):
            assert got.shape==wanted.shape
            np.testing.assert_allclose(got,wanted,rtol=3e-5,atol=2e-5)
    start=time.perf_counter_ns()
    actual=call();cold=(time.perf_counter_ns()-start)/1e6
    check(actual)
    warm=[]
    for _ in range(3):
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
        generation,_=owner.invoke();check(owner.read(generation))
        events=[]
        for _ in range(3):
            generation,elapsed=owner.invoke(repeats=32,timed=True)
            events.append(elapsed);check(owner.read(generation))
    return {"kind":kind,"encoded_e8m0":encoded,"transposeA":ta,"transposeB":tb,
        "input_shapes":[list(v.shape) for v in values],"output_shapes":[list(v.shape) for v in expected],
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
    paths=("python/tessera/compiler/graph_ir.py","python/tessera/compiler/native_scaled_program.py",
           "python/tessera/compiler/native_jvp_plugins.py","python/tessera/compiler/native_vjp_plugins.py",
           "src/compiler/ir/include/Tessera/IR/ScaledBatchContract.h","src/compiler/ir/LinearTransposeInterface.cpp",
           "src/compiler/programming_model/lib/PMPasses.cpp",
           "tests/unit/test_direct_broadcast_scaled_frontend.py","tests/device/rocm/test_direct_broadcast_scaled_frontend.py",
           "benchmarks/rocm/benchmark_direct_broadcast_scaled.py")
    runtime=Path(rt._load_rocm_native_movement_runtime()._name)
    packet={"schema":"tessera.direct_broadcast_scaled.v1","architecture":"gfx1201",
        "device":name.value.decode(),"device_ordinal":ordinal.value,"device_uuid":bytes(uuid).hex(),
        "compiler_sha256":hashlib.sha256(compiler.read_bytes()).hexdigest(),
        "runtime_sha256":hashlib.sha256(runtime.read_bytes()).hexdigest(),
        "source_sha256":{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        "timing_scope":"cold/warm public wall includes checked package/host transfers; prepared HIP event window covers native program dispatch and member kernels; not an isolated single-kernel metric",
        "selector_promotion":False,"rows":[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    profiles=[(encoded,ta,tb,"primal") for encoded,ta,tb in itertools.product((False,True),repeat=3)]
    profiles += [(False,ta,tb,kind) for ta,tb in itertools.product((False,True),repeat=2) for kind in ("jvp","vjp")]
    for profile in profiles:
        row=record(*profile);packet["rows"].append(row)
        args.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+"\n")
        print("verified",profile,row["max_abs_error"],flush=True)

if __name__=="__main__":main()
