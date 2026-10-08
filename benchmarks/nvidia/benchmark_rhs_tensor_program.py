"""Numerical and ownership proof for native RMSNorm RHS -> typed matmul B."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
import numpy as np
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.unit.test_nvidia_rhs_tensor_program import program
from tests._support.nvidia import nvidia_cuda_host_ready


def record(frontend=False):
    if not nvidia_cuda_host_ready():
        raise RuntimeError("requires owning CUDA host")
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or gpu.split(",")[-1].strip()!="12.0":
        raise RuntimeError("requires exact SM120 GPU")
    rows=[]
    for dtype in ("fp16","bf16"):
        storage=np.float16
        if dtype=="bf16":
            import ml_dtypes
            storage=ml_dtypes.bfloat16
        for shape in ((16,16,8),(17,35,19),(64,256,64),(128,1024,64)):
            m,k,n=shape
            traced=None
            if frontend:
                from tests.unit.test_nvidia_rhs_jit import rhs_product,rhs_reordered
                function=rhs_product if m%2 else rhs_reordered
                traced=function.compile_native_rhs_matmul(
                    **{"lhs":np.zeros((m,k),storage),"source":np.ones((k,n),storage)})
                edge_program=traced.edge
            else:
                edge_program=program(shape,dtype)
            rng=np.random.default_rng(120412)
            x=(rng.normal(size=(k,n))*.2).astype(storage)
            a=(rng.normal(size=(m,k))*.2).astype(storage)
            xf=x.astype(np.float64)
            normalized=(xf/np.sqrt(np.mean(xf*xf,axis=1,keepdims=True)+1e-5)).astype(storage)
            oracle=a.astype(np.float64)@normalized.astype(np.float64)
            start=time.perf_counter_ns()
            runner=(lambda: traced.execute_resident(lhs=a,source=x)) if frontend else (
                lambda: edge_program.execute_resident(x,a))
            with runner() as result:
                wall_ms=(time.perf_counter_ns()-start)/1e6
                actual_edge=result.device_session.download(result.intermediate)
                actual=result.device_session.download(result.output)
                np.testing.assert_allclose(actual_edge.astype(np.float32),normalized.astype(np.float32),rtol=.012,atol=.012)
                # Isolate consumer correctness from producer rounding.
                consumer_oracle=a.astype(np.float64)@actual_edge.astype(np.float64)
                np.testing.assert_allclose(actual,consumer_oracle,rtol=4e-5,atol=4e-5)
                np.testing.assert_allclose(actual,oracle,rtol=.015,atol=.015)
                consumer_error=float(np.max(np.abs(actual-consumer_oracle)))
                pipeline_error=float(np.max(np.abs(actual-oracle)))
            # The result owns all allocations; retained handles refuse downloads
            # after closing their session instead of exposing stale pointers.
            try:
                result.device_session.download(result.intermediate)
            except (RuntimeError,ValueError):
                pass
            else:
                raise RuntimeError("closed RHS allocation remained readable")
            with NvidiaDeviceSession() as session:
                dx=session.upload(x)
                da=session.upload(a)
                db=session.empty((k,n),storage)
                output=session.empty((m,n),np.float32)
                pa,ca=edge_program.arguments(dx,da,db,output)
                for package,args in ((edge_program.producer,pa),(edge_program.consumer,ca)):
                    receipt=rt.launch(edge_program.runtime_artifact(package),args,stream=session.stream)
                    if not receipt.get("ok") or receipt.get("execution_kind")!="native_gpu":
                        raise RuntimeError(receipt)
                session.synchronize()
                np.testing.assert_allclose(session.download(output),oracle,rtol=.015,atol=.015)
                timings={}
                for name,package,args in (("producer",edge_program.producer,pa),("consumer",edge_program.consumer,ca)):
                    timings[name]=[rt._nvidia_native_descriptor_resident_device_latency(
                        package.image,package.descriptor,args,stream=session.stream,warmup=20,reps=100) for _ in range(3)]
                session.synchronize()
                np.testing.assert_allclose(session.download(output),oracle,rtol=.015,atol=.015)
            rows.append(dict(dtype=dtype,shape_mkn=list(shape),
                frontend_argument_names=list(traced.argument_names) if traced else None,
                frontend_graph_sha256=hashlib.sha256(traced.graph_ir.encode()).hexdigest() if traced else None,consumer_max_abs_error=consumer_error,
                pipeline_max_abs_error=pipeline_error,checked_capture_and_execution_wall_ms=wall_ms,
                resident_dispatch_window_samples_ms=timings,producer_provenance=dict(edge_program.producer.descriptor.provenance),
                consumer_provenance=dict(edge_program.consumer.descriptor.provenance)))
    paths=["python/tessera/compiler/nvidia_tensor_rhs.py","benchmarks/nvidia/benchmark_rhs_tensor_program.py","tests/unit/test_nvidia_rhs_tensor_program.py","python/tessera/compiler/jit.py","tests/unit/test_nvidia_rhs_jit.py"]
    return dict(schema="tessera.nvidia.rmsnorm_rhs_edge.v1",frontend_trace=frontend,gpu=gpu,rows=rows,
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        scope="static RMSNorm KxN row-major RHS -> native Schedule typed B fragments, same-stream resident lifetime; no dynamic/fused/low-precision widening",
        decision_gate="FP8, MXFP8, MXFP4 correctness and performance required before strategy/default selection")


if __name__=="__main__":
    Path(os.environ["TESSERA_RHS_EDGE_PACKET"]).write_text(json.dumps(record(),indent=2,sort_keys=True)+"\n")
    print("verified 8 RMSNorm RHS producer/consumer cases")
