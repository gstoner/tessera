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
from tessera.compiler.nvidia_tensor_lhs import package_traced_lhs,runtime_artifact
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
    rows=[]
    for dtype in ("fp16","bf16"):
        for kind,plain,fused_function in (
            ("rmsnorm",rms_lhs,rms_lhs_fused),
            ("layernorm",layer_lhs,layer_lhs_fused)):
            for shape in ((17,255,19),(17,256,19),(17,257,19),(128,1024,64),(128,4096,64)):
                if dtype=="bf16" and shape[1]==4096:
                    continue  # Separate retained numerical failure; no accepted timing claim.
                for fused in (False,True):
                    m,k,n=shape
                    storage=_storage(dtype)
                    rng=np.random.default_rng(120517)
                    padded=np.zeros((m,k+7),storage)
                    padded[:,:k]=(rng.normal(size=(m,k))*.2).astype(storage)
                    source=padded[:,:k]
                    rhs=(rng.normal(size=(k,n))*.2).astype(storage)
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
                    baseline=package_traced_lhs(function._traced_autodiff_module(args,{}),producer_schedule="serial")
                    if baseline.edge.consumer.image.image_digest!=packages[1].image.image_digest:
                        raise RuntimeError("consumer image changed between paired arms")
                    serial_artifact=runtime_artifact(baseline)
                    serial_receipt=rt.launch(serial_artifact,args)
                    if not serial_receipt.get("ok"):raise RuntimeError(serial_receipt)
                    np.testing.assert_allclose(serial_receipt["output"],expected,rtol=.015,atol=.015)
                    paired_wall={"selected":[],"serial":[]}
                    for trial in range(5):
                        order=("selected","serial") if trial%2==0 else ("serial","selected")
                        for arm in order:
                            start=time.perf_counter_ns()
                            receipt=rt.launch(artifact if arm=="selected" else serial_artifact,args)
                            paired_wall[arm].append((time.perf_counter_ns()-start)/1e6)
                            if not receipt.get("ok"):raise RuntimeError(receipt)
                            np.testing.assert_allclose(receipt["output"],expected,rtol=.015,atol=.015)
                    edge=program.edge
                    with NvidiaDeviceSession() as session:
                        ds=session.upload(np.ascontiguousarray(source))
                        db=session.upload(np.array(rhs,copy=True,order="F"),layout="col_major")
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
                        samples={"selected":{"producer":[],"consumer":[]},"serial":{"producer":[],"consumer":[]}}
                        paired_packages={"selected":packages,"serial":(baseline.edge.producer,baseline.edge.consumer)}
                        for trial in range(5):
                            order=("selected","serial") if trial%2==0 else ("serial","selected")
                            for arm in order:
                                for role,p,values in zip(("producer","consumer"),paired_packages[arm],(pa,ca),strict=True):
                                    samples[arm][role].append(rt._nvidia_native_descriptor_resident_device_latency(
                                        p.image,p.descriptor,values,stream=session.stream,warmup=20,reps=100))
                        session.synchronize()
                        np.testing.assert_allclose(session.download(output),expected,rtol=.015,atol=.015)
                    rows.append(dict(dtype=dtype,kind=kind,fused=fused,shape_mkn=list(shape),
                        cold_wall_ms=cold,warm_wall_samples_ms=warm,warm_wall_median_ms=statistics.median(warm),
                        portable_replay_wall_ms=replay,resident_event_dispatch_samples_ms=samples,
                        resident_event_dispatch_median_ms={arm:{role:statistics.median(v) for role,v in arms.items()} for arm,arms in samples.items()},
                        paired_wall_samples_ms=paired_wall,
                        paired_wall_median_ms={arm:statistics.median(v) for arm,v in paired_wall.items()},
                        producer_schedule=packages[0].descriptor.provenance["schedule"],
                        serial_producer_image=baseline.edge.producer.image.image_digest,
                        static_barriers={arm:pair[0].target_ir.count("nvvm.barrier") for arm,pair in paired_packages.items()},
                        producer_resources={arm:dict(pair[0].image.resource_record.metrics) for arm,pair in paired_packages.items()},
                        max_abs_error=float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)))),
                        image_digests=images,compiler_path=artifact.metadata["compiler_path"],
                        contract_digest=artifact.metadata["native_program"]["contract_digest"]))
                    Path(os.environ["TESSERA_LHS_PACKET"]+".partial").write_text(json.dumps(rows,indent=2,sort_keys=True)+"\n")
                    print("verified",dtype,kind,shape,"fused",fused,flush=True)
    sources=("python/tessera/compiler/nvidia_tensor_lhs.py","python/tessera/compiler/jit.py",
        "python/tessera/compiler/scheduled_matmul.py","python/tessera/runtime.py",
        "python/tessera/compiler/execution_matrix.py","tests/device/nvidia/test_lhs_tensor_jit.py",
        "benchmarks/nvidia/benchmark_native_norm_selection.py",
        "src/compiler/programming_model/lib/PMPasses.cpp","python/tessera/compiler/scheduled_kernel.py")
    return dict(schema="tessera.nvidia.native_norm_selection.v1",gpu=gpu,rows=rows,
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        nvidia_opt_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_OPT"]).read_bytes()).hexdigest(),
        bridge_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        timing_scope="Cold includes tracing/compiler; warm/replay include host staging, allocation, dispatch, readback and cleanup. Resident CUDA event windows include device dispatch, not isolated kernel-only measurements.",
        excluded_envelopes=[{"dtype":"bf16","K":4096,"reason":"Both schedules exceed the original composed float64-oracle tolerance; bf16-long-attribution.json retains the diagnosis."}],
        scope="Native selected versus forced serial, identical consumer image. Static primal fp16/BF16 LHS RMSNorm/LayerNorm -> native matmul, with optional bias/ReLU/residual and fp16 output. No general AD/dynamic/FP8/MXFP8/MXFP4 closure.")


if __name__=="__main__":
    packet=record()
    Path(os.environ["TESSERA_LHS_PACKET"]).write_text(json.dumps(packet,indent=2,sort_keys=True)+"\n")
