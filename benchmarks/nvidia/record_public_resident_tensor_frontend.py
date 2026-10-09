"""Public JIT borrowed CUDA roots: exact-device native/host timing domains."""
from contextlib import ExitStack
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time
from unittest.mock import patch

import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_native_tensor_dag import public_dag,public_dag_deep,oracle
from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed
from benchmarks.nvidia.record_ordered_resident_tensor_dag import identity


def record(dtype,depth,shape):
    storage=np.float16 if dtype=="fp16" else ml_dtypes.bfloat16
    m,n,k=shape
    rng=np.random.default_rng(19328+depth)
    owner=ts.jit(target="nvidia_sm120")((public_dag if depth==1 else public_dag_deep)._fn)
    errors=[];events=[];resident=[];host=[];retained=[]
    def check(output,a,b):
        expected=oracle(a,b,depth)
        np.testing.assert_allclose(output,expected,rtol=.015,atol=.015)
        errors.append(float(np.max(np.abs(output.astype(np.float64)-expected))))
    with ExitStack() as stack:
        left=stack.enter_context(NvidiaDeviceSession())
        right=stack.enter_context(NvidiaDeviceSession())
        launch=stack.enter_context(NvidiaDeviceSession())
        a=np.zeros((m,k),storage);b=np.zeros((k,n),storage)
        av=left.upload(a);bv=right.upload(b)
        roots=[Borrowed(av),Borrowed(bv)]
        start=time.perf_counter();output=owner(*roots)
        cold=(time.perf_counter()-start)*1000
        check(output,a,b)
        program=owner._nvidia_lhs_last_program
        prepared=next(iter(owner._nvidia_lhs_prepared_calls.values()))
        stack.callback(prepared.close)
        target=launch.empty((m,n),prepared.output_dtype)
        # Prime every native allocation before the repeated timing windows.
        prepared.invoke_resident(roots,target,stream=launch.stream)
        def forbidden(*args,**kwargs):
            raise RuntimeError("warm public resident benchmark invoked compiler/frontend/eager")
        with patch.object(subprocess,"run",forbidden),patch.object(owner,"_fn",forbidden):
            for index in range(7):
                a=rng.normal(0,.2,(m,k)).astype(storage)
                b=rng.normal(0,.2,(k,n)).astype(storage)
                for session,buffer,value in ((left,av,a),(right,bv,b)):
                    if session.lib.tessera_nvidia_device_upload(ct.c_void_p(buffer.ptr),
                        ct.c_void_p(value.ctypes.data),value.nbytes,ct.c_void_p(session.stream)):
                        raise RuntimeError("resident root update failed")
                output=owner(*roots);check(output,a,b)
                profile=prepared.profile_resident(roots,target,stream=launch.stream,repeats=128)
                check(target.numpy(),a,b);events.append(profile)
                arms=("resident","host") if index%2==0 else ("host","resident")
                results={}
                for arm in arms:
                    start=time.perf_counter()
                    results[arm]=owner(*(roots if arm=="resident" else (a,b)))
                    (resident if arm=="resident" else host).append((time.perf_counter()-start)*1000)
                    check(results[arm],a,b)
                np.testing.assert_array_equal(results["resident"],results["host"])
                retained.append((results["resident"],results["resident"].copy()))
                if owner._nvidia_lhs_last_program is not program:
                    raise RuntimeError("resident/host transition changed native program identity")
    for output,snapshot in retained:
        np.testing.assert_array_equal(output,snapshot)
    return {"dtype":dtype,"depth_per_operand":depth,"shape_mnk":list(shape),
            "cold_compile_and_public_call_ms":cold,"max_abs_error":max(errors),
            "native_program_event_samples_ms":[v["program_ms"] for v in events],
            "native_program_event_median_ms":median(v["program_ms"] for v in events),
            "grouped_stage_event_samples_ms":[v["grouped_stage_ms"] for v in events],
            "public_resident_completed_samples_ms":resident,
            "public_resident_completed_median_ms":median(resident),
            "public_host_completed_samples_ms":host,"public_host_completed_median_ms":median(host),
            "resident_over_host_median_ratio":median(resident)/median(host),
            "program_contract_digest":program.manifest()["contract_digest"],
            "source_graph_ir":program.graph_ir,
            "image_digests":[p.image.image_digest for p in (*program.producer_chain,*program.rhs_chain,program.edge.consumer)],
            "correctness":"independent float64 oracle and bitwise resident/host equality; retained outputs; compiler/frontend disabled"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    options=parser.parse_args()
    if rt._nvidia_device_name()!="sm_120":
        raise RuntimeError("owning RTX5070/sm120 required")
    rows=[record(dtype,depth,shape) for dtype in ("fp16","bf16") for depth in (1,2)
          for shape in ((17,19,35),(129,65,513))]
    sources=["python/tessera/compiler/jit.py","python/tessera/compiler/resident_nvidia_tensor.py",
             "python/tessera/compiler/prepared_nvidia_lhs.py","python/tessera/compiler/nvidia_tensor_dag.py",
             "tests/unit/test_public_resident_tensor_frontend.py","tests/device/nvidia/test_public_resident_tensor_frontend.py",
             "benchmarks/nvidia/record_public_resident_tensor_frontend.py",
             "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/matmul_prepared.cpp",
             "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.h",
             "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp",
             "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp"]
    packet={"architecture":"sm_120","device":identity(),"rows":rows,
            "route":"Python/text frontend abstract metadata->typed Graph->Schedule->Tile->NVIDIA Target->LLVM/PTX->ordered native CUDA ABI",
            "timing_scope":"Public completed wall time includes result allocation/download and input-stream waits; native program/grouped members are distinct CUDA event domains",
            "sources":{path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sources},
            "tools":{key:hashlib.sha256(Path(os.environ[key]).read_bytes()).hexdigest()
                     for key in ("TESSERA_OPT","TESSERA_NVIDIA_OPT","TESSERA_NVIDIA_PTX_LAUNCH_LIB")}}
    options.output.parent.mkdir(parents=True,exist_ok=True)
    options.output.write_text(json.dumps(packet,indent=2)+"\n")
    print(json.dumps([{key:row[key] for key in ("dtype","depth_per_operand","shape_mnk","max_abs_error",
        "native_program_event_median_ms","public_resident_completed_median_ms","public_host_completed_median_ms")} for row in rows]))


if __name__=="__main__":
    main()
