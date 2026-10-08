"""Ordinary NVIDIA LHS JIT proof with separate checked timing windows."""
from pathlib import Path
import hashlib
import json
import os
import statistics
import subprocess
import time
import numpy as np
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.nvidia_tensor_rhs import NvidiaNormRhsProgram
from tests.device.nvidia.test_lhs_tensor_jit import (
    rms_lhs,layer_lhs,softmax_lhs,rms_lhs_fused,layer_lhs_fused,softmax_lhs_fused,
    _storage,_oracle,
)


def record():
    gpu=subprocess.check_output([
        "/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,driver_version,compute_cap",
        "--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or gpu.split(",")[-1].strip()!="12.0":
        raise RuntimeError("exact SM120 device required")
    rhs_order=os.environ.get("TESSERA_LHS_RHS_ORDER","F")
    if rhs_order not in {"C","F"}:raise ValueError("RHS order must be C or F")
    rows=[]
    from tessera.compiler.prepared_nvidia_lhs import clear_portable_lhs_owners
    clear_portable_lhs_owners()
    for dtype in ("fp16","bf16"):
        for kind,plain,fused_function in (
            ("rmsnorm",rms_lhs,rms_lhs_fused),
            ("layernorm",layer_lhs,layer_lhs_fused),
            ("softmax",softmax_lhs,softmax_lhs_fused)):
            for shape in ((17,35,19),(128,1024,64)):
                for fused in (False,True):
                    m,k,n=shape
                    storage=_storage(dtype)
                    rng=np.random.default_rng(120517)
                    padded=np.zeros((m,k+7),storage)
                    padded[:,:k]=(rng.normal(size=(m,k))*.2).astype(storage)
                    source=padded[:,:k]
                    rhs=np.array(rng.normal(size=(k,n))*.2,dtype=storage,order=rhs_order)
                    bias=(rng.normal(size=n)*.2).astype(np.float32)
                    residual=(rng.normal(size=(m,n))*.2).astype(np.float32)
                    function=fused_function if fused else plain
                    args=(source,rhs,bias,residual) if fused else (source,rhs)
                    expected=_oracle(source,rhs,kind,bias if fused else None,residual if fused else None)
                    start=time.perf_counter_ns();actual=function(*args)
                    cold=(time.perf_counter_ns()-start)/1e6
                    np.testing.assert_allclose(actual,expected,rtol=.015,atol=.015)
                    if function.execution_kind!="native_gpu":
                        raise RuntimeError("lost native JIT route")
                    program=function._nvidia_lhs_last_program
                    if program.native_plan_json is None:
                        raise RuntimeError("public JIT lost compiler-owned actual Graph partition")
                    packages=function.native_lhs_packages()
                    images=[p.image.image_digest for p in packages]
                    warm=[]
                    for _ in range(3):
                        start=time.perf_counter_ns();actual=function(*args)
                        warm.append((time.perf_counter_ns()-start)/1e6)
                        np.testing.assert_allclose(actual,expected,rtol=.015,atol=.015)
                        if [p.image.image_digest for p in function.native_lhs_packages()]!=images:
                            raise RuntimeError("warm package identity changed")
                    artifact=rt.RuntimeArtifact.from_json(function.runtime_artifact().to_json())
                    start=time.perf_counter_ns();receipt=rt.launch(artifact,args)
                    replay=(time.perf_counter_ns()-start)/1e6
                    if not receipt.get("ok"):raise RuntimeError(receipt)
                    np.testing.assert_array_equal(actual,receipt["output"])
                    replay_warm=[]
                    for _ in range(5):
                        start=time.perf_counter_ns();receipt=rt.launch(artifact,args)
                        replay_warm.append((time.perf_counter_ns()-start)/1e6)
                        if not receipt.get("ok"):raise RuntimeError(receipt)
                        np.testing.assert_array_equal(actual,receipt["output"])
                    replay_binding=receipt["component_receipts"][0].get(
                        "native_call_binding","portable_checked_descriptor")
                    edge=program.edge
                    with NvidiaDeviceSession() as session:
                        ds=session.upload(np.ascontiguousarray(source))
                        db=session.upload(rhs,layout=packages[1].descriptor.provenance["b_layout"])
                        intermediate=session.empty((m,k),storage)
                        output=session.empty((m,n),np.float16 if fused else np.float32)
                        pa={edge.producer_input_name:ds,edge.intermediate_name:intermediate}
                        scalars={"Rows":m,"Columns":k,"K":k}
                        pa.update({s.name:scalars[s.name] for s in packages[0].descriptor.scalars})
                        ca={edge.consumer_input_name:intermediate,edge.consumer_rhs_name:db,
                            edge.output_name:output,"M":m,"N":n,"K":k}
                        if fused:
                            for name,value in edge._epilogue_inputs(bias,residual,m,n).items():
                                binding=edge._binding(packages[1],name,"input")
                                ca[name]=session.upload(value,layout=binding.layout)
                        for p,values in zip(packages,(pa,ca),strict=True):
                            receipt=rt.launch(NvidiaNormRhsProgram.runtime_artifact(p),values,stream=session.stream)
                            if not receipt.get("ok"):raise RuntimeError(receipt)
                        session.synchronize()
                        np.testing.assert_allclose(session.download(output),expected,rtol=.015,atol=.015)
                        samples={}
                        for role,p,values in zip(("producer","consumer"),packages,(pa,ca),strict=True):
                            samples[role]=[rt._nvidia_native_descriptor_resident_device_latency(
                                p.image,p.descriptor,values,stream=session.stream,warmup=20,reps=100)
                                for _ in range(3)]
                        session.synchronize()
                        np.testing.assert_allclose(session.download(output),expected,rtol=.015,atol=.015)
                    rows.append(dict(dtype=dtype,kind=kind,fused=fused,shape_mkn=list(shape),
                        native_partition_schema=json.loads(program.native_plan_json)["schema"],
                        native_partition_digest=hashlib.sha256(program.native_plan_json.encode()).hexdigest(),
                        rhs_layout=packages[1].descriptor.provenance["b_layout"],
                        consumer_abi=packages[1].descriptor.abi_id,
                        public_binding=function._nvidia_lhs_last_receipts[0].get("native_call_binding","portable_checked_descriptor"),
                        cold_wall_ms=cold,warm_wall_samples_ms=warm,warm_wall_median_ms=statistics.median(warm),
                        portable_replay_wall_ms=replay,portable_warm_wall_samples_ms=replay_warm,
                        portable_warm_wall_median_ms=statistics.median(replay_warm),
                        portable_binding=replay_binding,resident_event_dispatch_samples_ms=samples,
                        resident_event_dispatch_median_ms={role:statistics.median(v) for role,v in samples.items()},
                        max_abs_error=float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)))),
                        image_digests=images,compiler_path=artifact.metadata["compiler_path"],
                        contract_digest=artifact.metadata["native_program"]["contract_digest"]))
                    print("verified",dtype,kind,shape,"fused",fused,flush=True)
    sources=("python/tessera/compiler/nvidia_tensor_lhs.py","python/tessera/compiler/nvidia_native.py","python/tessera/compiler/jit.py",
        "python/tessera/compiler/prepared_nvidia_lhs.py","python/tessera/compiler/prepared_nvidia_matmul.py",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/matmul_prepared.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.h",
        "python/tessera/compiler/scheduled_matmul.py","python/tessera/runtime.py",
        "python/tessera/compiler/execution_matrix.py","tests/device/nvidia/test_lhs_tensor_jit.py",
        "benchmarks/nvidia/benchmark_lhs_jit_dispatch.py",
        "python/tessera/compiler/native_sm120_tensor_program.py",
        "src/transforms/lib/NativeSM120TensorProgram.h","src/transforms/lib/NativeScaledMatmulProgram.h")
    return dict(schema="tessera.nvidia.lhs_jit_dispatch.v1",gpu=gpu,rows=rows,
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        runtime_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        timing_scope="Cold includes tracing/compiler; warm/replay include host staging, allocation, dispatch, readback and cleanup. Resident CUDA event windows include device dispatch, not isolated kernel-only measurements.",
        scope="Static primal fp16/BF16 LHS RMSNorm/LayerNorm/last-axis softmax -> native matmul, with optional bias/ReLU/residual and fp16 output. No general AD/dynamic/FP8/MXFP8/MXFP4 closure.")


if __name__=="__main__":
    packet=record()
    Path(os.environ["TESSERA_LHS_PACKET"]).write_text(json.dumps(packet,indent=2,sort_keys=True)+"\n")
