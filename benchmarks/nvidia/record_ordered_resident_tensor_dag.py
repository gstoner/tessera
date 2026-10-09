"""Exact sm_120 ordered borrowed-root DAG numerical and timing packet."""
from contextlib import ExitStack, closing
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
from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
from tessera.compiler import nvidia_tensor_lhs as lhs
from tests.device.nvidia.test_native_tensor_dag import public_dag, public_dag_deep, oracle
from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity():
    driver=ct.CDLL("libcuda.so.1")
    device=ct.c_int();name=ct.create_string_buffer(256);uuid=(ct.c_ubyte*16)()
    driver.cuCtxGetDevice.argtypes=[ct.POINTER(ct.c_int)]
    driver.cuDeviceGetName.argtypes=[ct.c_void_p,ct.c_int,ct.c_int]
    driver.cuDeviceGetUuid_v2.argtypes=[ct.c_void_p,ct.c_int]
    if (driver.cuCtxGetDevice(ct.byref(device)) or driver.cuDeviceGetName(name,256,device.value)
            or driver.cuDeviceGetUuid_v2(ct.byref(uuid),device.value)):
        raise RuntimeError("active CUDA device identity failed")
    return {"name":name.value.decode(),"uuid_hex":bytes(uuid).hex(),"ordinal":device.value}


def check(output,a,b,depth):
    expected=oracle(a,b,depth)
    np.testing.assert_allclose(output,expected,rtol=.015,atol=.015)
    return float(np.max(np.abs(output.astype(np.float64)-expected)))


def record(dtype,depth):
    storage=np.float16 if dtype=="fp16" else ml_dtypes.bfloat16
    rng=np.random.default_rng(19240+depth)
    def host_frame(m,n,k):
        return (rng.normal(0,.2,(m,k)).astype(storage),
                rng.normal(0,.2,(k,n)).astype(storage))
    bounds={"M":257,"N":128,"K":1024}
    function=ts.jit(target="nvidia_sm120",shape_bounds=bounds)(
        (public_dag if depth==1 else public_dag_deep)._fn)
    a,b=host_frame(17,19,35)
    start=time.perf_counter()
    program=function.compile_native_lhs_matmul(a,b)
    compile_ms=(time.perf_counter()-start)*1000
    program=lhs.from_manifest(program.manifest())
    profiles=[]
    retained=[]
    with ExitStack() as stack:
        left=stack.enter_context(NvidiaDeviceSession())
        right=stack.enter_context(NvidiaDeviceSession())
        launch=stack.enter_context(NvidiaDeviceSession())
        prepared=stack.enter_context(closing(PreparedLhsCall(program)))
        host_owner=stack.enter_context(closing(PreparedLhsCall(program)))
        a_capacity=left.upload(np.zeros((bounds["M"],bounds["K"]),storage))
        b_capacity=right.upload(np.zeros((bounds["K"],bounds["N"]),storage))
        if left.synchronize() or right.synchronize():raise RuntimeError("initial root uploads failed")
        def forbidden(*args,**kwargs):raise RuntimeError("resident timing invoked compiler/eager frontend")
        with patch.object(subprocess,"run",forbidden),patch.object(function,"_fn",forbidden):
            for m,n,k in ((17,19,35),(129,65,513)):
                av=a_capacity.view(0,(m,k),storage)
                bv=b_capacity.view(0,(k,n),storage)
                roots=[Borrowed(av),Borrowed(bv)]
                output=launch.empty((m,n),prepared.output_dtype)
                # Prime native capacities before collecting timing windows.
                prepared.invoke_resident(roots,output,stream=launch.stream)
                windows=[];wall=[];errors=[]
                for _ in range(7):
                    a,b=host_frame(m,n,k)
                    for session,buffer,host in ((left,av,a),(right,bv,b)):
                        status=session.lib.tessera_nvidia_device_upload(
                            ct.c_void_p(buffer.ptr),ct.c_void_p(host.ctypes.data),
                            host.nbytes,ct.c_void_p(session.stream))
                        if status:raise RuntimeError("producer upload failed")
                    prepared.invoke_resident(roots,output,stream=launch.stream)
                    errors.append(check(output.numpy(),a,b,depth))
                    window=prepared.profile_resident(roots,output,stream=launch.stream,repeats=128)
                    if window["program_ms"]<=0 or len(window["grouped_stage_ms"])!=2*depth+1:
                        raise RuntimeError("resident event profile geometry differs")
                    windows.append(window)
                    errors.append(check(output.numpy(),a,b,depth))
                    host_output,_=host_owner([a,b])
                    np.testing.assert_array_equal(output.numpy(),host_output)
                    start=time.perf_counter()
                    result=program.execute_resident(*roots)
                    stack.callback(result.close)
                    wall.append((time.perf_counter()-start)*1000)
                    try:
                        errors.append(check(result.output.numpy(),a,b,depth))
                        np.testing.assert_array_equal(result.output.numpy(),host_output)
                        if result.consumer_receipt["native_call_binding"]!="prepared_cpp_ordered_resident_tensor_dag":
                            raise RuntimeError("resident ownership receipt differs")
                        held=result.output.numpy()
                        retained.append((result,held))
                    except Exception:
                        result.close()
                        raise
                profiles.append({
                    "shape_mnk":[m,n,k],"dtype":dtype,"depth_per_operand":depth,
                    "correctness":"independent f64 oracle before/after every native/profile/package window",
                    "native_host_vs_resident":"bitwise equal at every window",
                    "max_abs_error":max(errors),"program_event_samples_ms":[w["program_ms"] for w in windows],
                    "program_event_median_ms":median(w["program_ms"] for w in windows),
                    "grouped_stage_event_samples_ms":[w["grouped_stage_ms"] for w in windows],
                    "grouped_stage_event_medians_ms":[median(w["grouped_stage_ms"][index] for w in windows)
                        for index in range(2*depth+1)],
                    "resident_package_wall_samples_ms":wall,"resident_package_wall_median_ms":median(wall),
                    "repetitions_per_native_window":128,"allocation_stats":prepared.scratch_stats(),
                })
        # Explicitly retire external root/stream owners while returned outputs
        # remain alive in independent result sessions.
        left.close();right.close()
        for result,held in retained:np.testing.assert_array_equal(result.output.numpy(),held)
        for result,_ in retained:result.close()
    members=[*(program.producer_chain or (program.edge.producer,)),
             *program.rhs_chain,program.edge.consumer]
    return {"bounds":bounds,"compile_ms":compile_ms,
            "member_policies":[{"entry":member.descriptor.entry_symbol,
                "kind":member.descriptor.provenance.get("kind","matmul"),
                "schedule":member.descriptor.provenance.get("schedule"),
                "physical_route":member.descriptor.provenance.get("physical_route"),
                "geometry":member.descriptor.geometry.policy,
                "tile_ir_sha256":hashlib.sha256(member.tile_ir.encode()).hexdigest(),
                "target_ir_sha256":hashlib.sha256(member.target_ir.encode()).hexdigest()}
                for member in members],
            "native_plan_sha256":hashlib.sha256(program.native_plan_json.encode()).hexdigest(),
            "member_image_digests":[receipt["image_digest"] for receipt in prepared.component_receipts],
            "member_descriptor_digests":[receipt["launch_descriptor_digest"] for receipt in prepared.component_receipts],
            "profiles":profiles}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    options=parser.parse_args()
    if rt._nvidia_device_name()!="sm_120":raise RuntimeError("exact sm_120 required")
    jobs=subprocess.run(["pgrep","-af","[p]ytest|[g]raphify update"],capture_output=True,text=True)
    if jobs.returncode==0 and jobs.stdout.strip():raise RuntimeError("test/graph job is active")
    profiles=[record(dtype,depth) for dtype in ("fp16","bf16") for depth in (1,2)]
    sources=[
        "python/tessera/compiler/nvidia_tensor_dag.py",
        "python/tessera/compiler/nvidia_tensor_lhs.py",
        "python/tessera/compiler/native_sm120_tensor_program.py",
        "python/tessera/compiler/prepared_nvidia_lhs.py",
        "python/tessera/compiler/resident_nvidia_tensor.py",
        "python/tessera/compiler/emit/nvidia_cuda.py",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/matmul_prepared.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.h",
        "tests/device/nvidia/test_ordered_resident_tensor_dag.py"]
    packet={
        "schema":"tessera.sm120.ordered_resident_tensor_dag.v1",
        "architecture":"sm_120","device":identity(),
        "source_sha256":{path:digest(path) for path in sources},
        "tools":{name:digest(os.environ[name]) for name in (
            "TESSERA_OPT","TESSERA_NVIDIA_OPT","TESSERA_NVIDIA_PTX_LAUNCH_LIB","TESSERA_NVIDIA_GEMM_LIB")},
        "recorder_sha256":digest(__file__),
        "route":"Python frontend->Graph MLIR->native SSA export->Schedule/Tile/views/fragments->NVIDIA Target->LLVM NVPTX->PTX->checked C++ resident ownership",
        "incoming_ordering":"native per-producer CUDA event/wait; caller roots/streams remain live and immutable until synchronous completion",
        "native_timing_domain":"CUDA program events after incoming waits; separate grouped repeated-stage events; 128 launches/window; stages are not additive",
        "wall_timing_domain":"warm resident package call: metadata/seal checks, owner/session/output allocation, native execution/completion; existing device roots; no input upload or output download in the measured wall interval",
        "claim":"numerical/lifetime and resident-route characterization; no speedup, selector promotion, direct-CuPy/PyTorch or public-JIT-CUDA-root claim",
        "records":profiles}
    options.output.parent.mkdir(parents=True,exist_ok=True)
    options.output.write_text(json.dumps(packet,indent=2)+"\n")


if __name__=="__main__":
    main()
