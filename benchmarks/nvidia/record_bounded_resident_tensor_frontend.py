"""Bounded public resident inputs: native events and completed-call wall time."""
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

BOUNDS={"M":129,"N":65,"K":513}
FRAMES=((17,19,35),(129,65,513),(1,1,1),(63,31,255),(17,19,35))


def record(dtype,depth):
    storage=np.float16 if dtype=="fp16" else ml_dtypes.bfloat16
    owner=ts.jit(target="nvidia_sm120",shape_bounds=BOUNDS)(
        (public_dag if depth==1 else public_dag_deep)._fn)
    rng=np.random.default_rng(19331+depth)
    retained=[];rows=[];errors=[]
    def check(output,a,b):
        wanted=oracle(a,b,depth)
        np.testing.assert_allclose(output,wanted,rtol=.015,atol=.015)
        errors.append(float(np.max(np.abs(output.astype(np.float64)-wanted))))
    with ExitStack() as stack:
        left=stack.enter_context(NvidiaDeviceSession())
        right=stack.enter_context(NvidiaDeviceSession())
        launch=stack.enter_context(NvidiaDeviceSession())
        ac=left.upload(np.zeros((BOUNDS["M"],BOUNDS["K"]),storage))
        bc=right.upload(np.zeros((BOUNDS["K"],BOUNDS["N"]),storage))
        oc=launch.empty((BOUNDS["M"],BOUNDS["N"]),np.float32)
        for session in (left,right,launch):
            if session.synchronize():raise RuntimeError("initial CUDA synchronization failed")
        views=[ac.view(0,(17,35),storage),bc.view(0,(35,19),storage)]
        roots=[Borrowed(view) for view in views]
        start=time.perf_counter();out=owner(*roots)
        cold=(time.perf_counter()-start)*1000
        check(out,np.zeros((17,35),storage),np.zeros((35,19),storage))
        program=owner._nvidia_lhs_last_program
        prepared=next(iter(owner._nvidia_lhs_prepared_calls.values()))
        stack.callback(prepared.close)
        # Prime both public input routes at full capacity before measuring.
        full_a=np.zeros(ac.shape,storage);full_b=np.zeros(bc.shape,storage)
        check(owner(full_a,full_b),full_a,full_b)
        prepared.invoke_resident([Borrowed(ac),Borrowed(bc)],oc,stream=launch.stream)
        scratch=prepared.scratch_stats()
        digest=program.manifest()["contract_digest"]
        def forbidden(*args,**kwargs):
            raise RuntimeError("warm bounded benchmark invoked compiler/frontend/eager")
        with patch.object(subprocess,"run",forbidden),patch.object(owner,"_fn",forbidden),patch.object(
            owner,"_traced_autodiff_module",forbidden):
            for ordinal,(m,n,k) in enumerate(FRAMES):
                events=[];resident=[];host=[];errors.clear()
                av=ac.view(0,(m,k),storage);bv=bc.view(0,(k,n),storage)
                target=oc.view(0,(m,n),np.float32)
                roots=[Borrowed(av),Borrowed(bv)]
                for index in range(7):
                    a=rng.normal(0,.2,(m,k)).astype(storage)
                    b=rng.normal(0,.2,(k,n)).astype(storage)
                    for session,buffer,value in ((left,av,a),(right,bv,b)):
                        if session.lib.tessera_nvidia_device_upload(ct.c_void_p(buffer.ptr),
                            ct.c_void_p(value.ctypes.data),value.nbytes,ct.c_void_p(session.stream)):
                            raise RuntimeError("active root upload failed")
                    check(owner(*roots),a,b)
                    profile=prepared.profile_resident(roots,target,stream=launch.stream,repeats=128)
                    check(target.numpy(),a,b);events.append(profile)
                    arms=("resident","host") if index%2==0 else ("host","resident")
                    outputs={}
                    for arm in arms:
                        start=time.perf_counter()
                        outputs[arm]=owner(*(roots if arm=="resident" else (a,b)))
                        (resident if arm=="resident" else host).append((time.perf_counter()-start)*1000)
                        check(outputs[arm],a,b)
                    np.testing.assert_array_equal(outputs["resident"],outputs["host"])
                    retained.append((outputs["resident"],outputs["resident"].copy()))
                    if (owner._nvidia_lhs_last_program is not program or
                        len(owner._bounded_lhs.programs)!=1 or prepared.scratch_stats()!=scratch):
                        raise RuntimeError("active frame changed package/allocation identity")
                rows.append({"dtype":dtype,"depth_per_operand":depth,"frame_ordinal":ordinal,
                    "shape_mnk":[m,n,k],"shape_bounds":BOUNDS,"max_abs_error":max(errors),
                    "native_program_event_samples_ms":[e["program_ms"] for e in events],
                    "native_program_event_median_ms":median(e["program_ms"] for e in events),
                    "grouped_stage_event_samples_ms":[e["grouped_stage_ms"] for e in events],
                    "public_resident_completed_samples_ms":resident,
                    "public_resident_completed_median_ms":median(resident),
                    "public_host_completed_samples_ms":host,
                    "public_host_completed_median_ms":median(host),
                    "resident_over_host_median_ratio":median(resident)/median(host),
                    "program_contract_digest":digest,"correctness":"independent_fp64_stage_oracle_and_bitwise_host_resident",
                    "package_reuse":"one_program_and_owner_across_all_frames"})
    for output,snapshot in retained:np.testing.assert_array_equal(output,snapshot)
    return {"dtype":dtype,"depth_per_operand":depth,"cold_compile_and_public_call_ms":cold,
            "program_contract_digest":digest,"native_plan":json.loads(program.native_plan_json),
            "image_digests":[p.image.image_digest for p in (
                *(program.producer_chain or (program.edge.producer,)),*program.rhs_chain,program.edge.consumer)],
            "scratch":scratch,"rows":rows}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if rt._nvidia_device_name()!="sm_120":raise RuntimeError("exact sm_120 required")
    programs=[record(dtype,depth) for dtype in ("fp16","bf16") for depth in (1,2)]
    packet={"architecture":"sm_120","device":identity(),"shape_bounds":BOUNDS,
        "route":"abstract frontend Graph->native bounded projection->Schedule->Tile->Target->LLVM/PTX->ordered CUDA ABI",
        "timing_scope":"Public completed wall time includes ordering and result download; native program/grouped member events are separate domains",
        "programs":programs}
    sources=("python/tessera/compiler/bounded_nvidia_lhs.py","python/tessera/compiler/jit.py",
        "python/tessera/compiler/resident_nvidia_tensor.py","python/tessera/compiler/prepared_nvidia_lhs.py",
        "python/tessera/compiler/nvidia_tensor_dag.py","tests/unit/test_bounded_resident_tensor_frontend.py",
        "tests/device/nvidia/test_bounded_resident_tensor_frontend.py",
        "benchmarks/nvidia/record_bounded_resident_tensor_frontend.py",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/matmul_prepared.cpp")
    packet["sources"]={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    packet["tools"]={key:hashlib.sha256(Path(os.environ[key]).read_bytes()).hexdigest()
                     for key in ("TESSERA_OPT","TESSERA_NVIDIA_OPT","TESSERA_NVIDIA_PTX_LAUNCH_LIB")}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2)+"\n")
    print(f"Recorded {sum(len(p['rows']) for p in packet['programs'])} active frames on {packet['device']}")

if __name__=="__main__":main()
