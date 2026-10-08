"""Ordinary JIT native RHS execution, correctness before separate timing windows."""
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
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.unit.test_nvidia_rhs_jit import rhs_product,rhs_reordered


def record(kind="rmsnorm"):
    if not nvidia_cuda_host_ready():
        raise RuntimeError("owning CUDA host required")
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or gpu.split(",")[-1].strip()!="12.0":
        raise RuntimeError("requires exact SM120 GPU")
    if kind=="layernorm":
        from tests.device.nvidia.test_layernorm_rhs_jit import layer_rhs,layer_rhs_reordered
        functions=(layer_rhs,layer_rhs_reordered)
    elif kind=="rmsnorm":
        functions=(rhs_product,rhs_reordered)
    else:
        raise ValueError("unsupported normalization kind")
    rows=[]
    for dtype in ("fp16","bf16"):
        storage=np.float16
        if dtype=="bf16":
            import ml_dtypes
            storage=ml_dtypes.bfloat16
        for m,k,n in ((16,16,8),(17,35,19),(64,256,64),(128,1024,64),(256,1024,128)):
            function=functions[0] if m%2 else functions[1]
            rng=np.random.default_rng(120414)
            a=(rng.normal(size=(m,k))*.2).astype(storage)
            # Exercise explicit host compaction by the resident program.
            padded=np.full((k,n+5),13,storage)
            padded[:,:n]=(rng.normal(size=(k,n))*.2).astype(storage)
            x=padded[:,:n]
            xf=x.astype(np.float64)
            centered=xf-np.mean(xf,axis=1,keepdims=True) if kind=="layernorm" else xf
            normalized=(centered/np.sqrt(np.mean(centered*centered,axis=1,keepdims=True)+1e-5)).astype(storage)
            expected=a.astype(np.float64)@normalized.astype(np.float64)
            start=time.perf_counter_ns()
            actual=function(lhs=a,source=x)
            cold_ms=(time.perf_counter_ns()-start)/1e6
            np.testing.assert_allclose(actual,expected,rtol=.015,atol=.015)
            if function.execution_kind!="native_gpu" or len(function.native_rhs_packages())!=2:
                raise RuntimeError("ordinary JIT did not execute checked native program")
            program=function._nvidia_rhs_last_program
            packages=function.native_rhs_packages()
            images=[p.image.image_digest for p in packages]
            warm=[]
            for _ in range(3):
                start=time.perf_counter_ns()
                actual=function(lhs=a,source=x)
                warm.append((time.perf_counter_ns()-start)/1e6)
                np.testing.assert_allclose(actual,expected,rtol=.015,atol=.015)
                if [p.image.image_digest for p in function.native_rhs_packages()]!=images:
                    raise RuntimeError("warm call changed package identity")
                if not all(r.get("ok") and r.get("execution_kind")=="native_gpu" for r in function._nvidia_rhs_last_receipts):
                    raise RuntimeError("warm call lost native execution")
            artifact=rt.RuntimeArtifact.from_json(function.runtime_artifact().to_json())
            replay=[]
            for _ in range(3):
                start=time.perf_counter_ns()
                receipt=rt.launch(artifact,{"lhs":a,"source":x})
                replay.append((time.perf_counter_ns()-start)/1e6)
                if not receipt.get("ok") or receipt.get("execution_kind")!="native_gpu":
                    raise RuntimeError(receipt)
                np.testing.assert_allclose(receipt["output"],expected,rtol=.015,atol=.015)
                if len(receipt["component_receipts"])!=2:
                    raise RuntimeError("portable program lost component execution")
            edge=program.edge
            with NvidiaDeviceSession() as session:
                dx=session.upload(np.ascontiguousarray(x))
                da=session.upload(a)
                intermediate=session.empty((k,n),storage)
                output=session.empty((m,n),np.float32)
                pa,ca=edge.arguments(dx,da,intermediate,output)
                for package,args in zip(packages,(pa,ca),strict=True):
                    receipt=rt.launch(edge.runtime_artifact(package),args,stream=session.stream)
                    if not receipt.get("ok") or receipt.get("execution_kind")!="native_gpu":
                        raise RuntimeError(receipt)
                session.synchronize()
                actual_edge=session.download(intermediate)
                np.testing.assert_allclose(actual_edge.astype(np.float32),normalized.astype(np.float32),rtol=.012,atol=.012)
                consumer_oracle=a.astype(np.float64)@actual_edge.astype(np.float64)
                np.testing.assert_allclose(session.download(output),consumer_oracle,rtol=4e-5,atol=4e-5)
                producer_error=float(np.max(np.abs(actual_edge.astype(np.float32)-normalized.astype(np.float32))))
                consumer_error=float(np.max(np.abs(session.download(output)-consumer_oracle)))
                samples={}
                for role,package,args in zip(("producer","consumer"),packages,(pa,ca),strict=True):
                    samples[role]=[rt._nvidia_native_descriptor_resident_device_latency(
                        package.image,package.descriptor,args,stream=session.stream,warmup=20,reps=100) for _ in range(3)]
                session.synchronize()
                np.testing.assert_allclose(session.download(output),expected,rtol=.015,atol=.015)
            rows.append(dict(dtype=dtype,shape_mkn=[m,k,n],frontend_argument_names=list(program.argument_names),
                max_abs_pipeline_error=float(np.max(np.abs(actual-expected))),
                max_abs_producer_error=producer_error,max_abs_consumer_error=consumer_error,
                portable_contract_digest=artifact.metadata["native_program"]["contract_digest"],
                portable_replay_samples_ms=replay,portable_replay_median_ms=statistics.median(replay),
                native_graph_sha256=hashlib.sha256(program.graph_ir.encode()).hexdigest(),
                cold_call_wall_ms=cold_ms,warm_call_samples_ms=warm,warm_call_median_ms=statistics.median(warm),
                resident_dispatch_window_samples_ms=samples,image_digests=images,
                compiler_path=function.runtime_artifact().metadata["compiler_path"],
                producer_provenance=dict(packages[0].descriptor.provenance),
                consumer_provenance=dict(packages[1].descriptor.provenance)))
    sources=("python/tessera/compiler/jit.py","python/tessera/compiler/nvidia_tensor_rhs.py",
        "python/tessera/runtime.py","python/tessera/compiler/execution_matrix.py",
        "tests/device/nvidia/test_rhs_jit_dispatch.py","tests/device/nvidia/test_layernorm_rhs_jit.py",
        "benchmarks/nvidia/benchmark_rhs_jit_dispatch.py")
    return dict(schema="tessera.nvidia.rhs_jit_dispatch.v1",gpu=gpu,normalization_kind=kind,rows=rows,
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        scope="ordinary primal static half normalization RHS -> fp32 matmul calls; cached native packages; host compaction; no composed AD/general graph closure",
        timing_scope="cold wall includes trace/compile/execution; warm wall includes staging/module dispatch/download; resident producer/consumer event dispatch windows are separate, not isolated kernel-only timing",
        decision_gate="FP8/MXFP8/MXFP4 correctness and performance required before tuning/default strategy decisions")


if __name__=="__main__":
    output=Path(os.environ["TESSERA_RHS_DISPATCH_PACKET"])
    packet=record(os.environ.get("TESSERA_RHS_NORM_KIND","rmsnorm"))
    output.write_text(json.dumps(packet,indent=2,sort_keys=True)+"\n")
    print("verified",len(packet["rows"]),"ordinary native JIT RHS cases")
