"""Correctness-gated exact SM120 two-sided tensor DAG dispatch profiles."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time

import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
from tests.device.nvidia.test_native_tensor_dag import public_dag, public_dag_deep, oracle


def record(dtype, shape, depth):
    storage=np.float16
    if dtype=="bf16":
        import ml_dtypes
        storage=ml_dtypes.bfloat16
    m,n,k=shape
    rng=np.random.default_rng(9123+depth)
    a=rng.normal(0,.2,(m,k)).astype(storage);b=rng.normal(0,.2,(k,n)).astype(storage)
    function=ts.jit(target="nvidia_sm120")((public_dag if depth==1 else public_dag_deep)._fn)
    started=time.perf_counter();output=function(a,b);cold=(time.perf_counter()-started)*1000
    expected=oracle(a,b,depth)
    def check(value, wanted):
        np.testing.assert_allclose(value,wanted,rtol=.015,atol=.015)
        return float(np.max(np.abs(value.astype(np.float64)-wanted)))
    error=check(output,expected)
    program=function._nvidia_lhs_last_program
    program.validate()
    prepared=PreparedLhsCall(program)
    native=[];stages=[];host=[];public=[]
    try:
        for index in range(7):
            factor=1 if index%2==0 else -.875
            left=(a*factor).astype(storage);right=(b*(1 if index%2==0 else .75)).astype(storage)
            wanted=oracle(left,right,depth)
            started=time.perf_counter();out,_=prepared([left,right]);host.append((time.perf_counter()-started)*1000)
            error=max(error,check(out,wanted))
            measured=prepared.profile(128)
            error=max(error,check(measured["output"],wanted))
            native.append(measured["program_ms"]);stages.append(measured["grouped_stage_ms"])
            started=time.perf_counter();out=function(left,right);public.append((time.perf_counter()-started)*1000)
            error=max(error,check(out,wanted))
        allocations=prepared.scratch_stats()
    finally:prepared.close()
    packages=function.native_lhs_packages()
    return {"dtype":dtype,"shape_mnk":[m,n,k],"chain_depth":depth,"correctness":"passed_before_and_after_every_window",
            "max_abs_error":error,"cold_compile_and_call_ms":cold,
            "device_program_samples_ms":native,"device_program_median_ms":median(native),
            "grouped_stage_samples_ms":stages,
            "grouped_stage_medians_ms":[median(column) for column in zip(*stages,strict=True)],
            "prepared_host_samples_ms":host,"prepared_host_median_ms":median(host),
            "public_warm_samples_ms":public,"public_warm_median_ms":median(public),
            "arena_capacity_bytes":allocations[0],"arena_allocation_count":allocations[1],
            "physical_stage_order":[*["lhs."+p.descriptor.provenance["kind"] for p in (
                program.producer_chain or (program.edge.producer,))],
                *["rhs."+p.descriptor.provenance["kind"] for p in program.rhs_chain],"matmul"],
            "native_plan":json.loads(program.native_plan_json),
            "images":[{"image_digest":p.image.image_digest,"abi_id":p.descriptor.abi_id,
                       "entry":p.descriptor.entry_symbol,"schedule_digest":p.descriptor.provenance.get("schedule_digest"),
                       "tile_ir_digest":p.descriptor.provenance.get("tile_ir_digest")} for p in packages]}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if rt._nvidia_device_name()!="sm_120":raise RuntimeError("exact SM120 required")
    cuda=ct.CDLL("libcuda.so.1");device=ct.c_int();name=ct.create_string_buffer(256);uuid=(ct.c_ubyte*16)()
    if cuda.cuInit(0) or cuda.cuDeviceGet(ct.byref(device),0) or cuda.cuDeviceGetName(name,256,device) or cuda.cuDeviceGetUuid(ct.byref(uuid),device):
        raise RuntimeError("CUDA device identity unavailable")
    paths=["src/transforms/lib/NativeSM120TensorProgram.h",
           "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/matmul_prepared.cpp",
           "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.h",
           "python/tessera/compiler/native_sm120_tensor_program.py",
           "python/tessera/compiler/nvidia_tensor_dag.py","python/tessera/compiler/nvidia_tensor_lhs.py",
           "python/tessera/compiler/prepared_nvidia_lhs.py","python/tessera/compiler/bounded_nvidia_lhs.py",
           "python/tessera/compiler/resident_nvidia_tensor.py",
           "python/tessera/compiler/emit/nvidia_cuda.py",
           "python/tessera/compiler/jit.py","python/tessera/runtime.py",
           "tests/device/nvidia/test_native_tensor_dag.py",
           "benchmarks/nvidia/record_two_sided_tensor_dag.py"]
    packet={"architecture":"sm_120","device":name.value.decode(),"uuid_hex":bytes(uuid).hex(),
            "route":"Python frontend->typed Graph->native SSA export->Schedule->Tile views/fragments->NVIDIA Target->LLVM NVPTX->PTX->checked C++ owner",
            "timing_scope":"CUDA event dispatch windows include ordered stream submission gaps; grouped stage repeats are not additive program time",
            "public_scope":"Compilation warm wall time includes frontend preparation, host copies and synchronization",
            "sources":{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
            "tools":{key:hashlib.sha256(Path(os.environ[key]).read_bytes()).hexdigest() for key in (
                "TESSERA_OPT","TESSERA_NVIDIA_OPT","TESSERA_NVIDIA_PTX_LAUNCH_LIB")},
            "rows":[record(dtype,shape,depth) for dtype in ("fp16","bf16") for shape in (
                (17,19,35),(129,65,513)) for depth in (1,2)]}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(packet,indent=2)+"\n")
    print(json.dumps([{key:row[key] for key in ("dtype","shape_mnk","chain_depth","max_abs_error",
                     "device_program_median_ms","public_warm_median_ms")} for row in packet["rows"]]))


if __name__=="__main__":main()
