"""Numerical and timing evidence for compiler-owned actual SM120 tensor members."""
from pathlib import Path
import hashlib
import json
import os
import statistics
import subprocess
import time
import numpy as np
from tessera import runtime as rt
from tessera.compiler.nvidia_tensor_lhs import _semantic_graph
from tessera.compiler.native_sm120_tensor_program import export_native_sm120_tensor_graph
from tessera.compiler.nvidia_native import package_scheduled_tensor_matmul
from tessera.compiler.nvidia_tensor_rhs import NvidiaNormRhsProgram
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession

def record():
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or gpu.split(",")[-1].strip()!="12.0":
        raise RuntimeError("exact SM120 device required")
    rows=[]
    for dtype in ("fp16","bf16"):
        storage=np.float16
        if dtype=="bf16":
            import ml_dtypes
            storage=ml_dtypes.bfloat16
        for operation in ("tessera.rmsnorm","tessera.layer_norm","tessera.softmax"):
            for order in ("row_major","col_major"):
                for m,k,n in ((17,32,11),(128,1024,64)):
                    sem={"producer":operation,"producer_attrs":{"axis":-1} if operation=="tessera.softmax" else {"eps":1e-5},
                        "consumer":"tessera.matmul","consumer_attrs":{"output_dtype":"fp32","rhs_storage_order":order},
                        "roles":{"source":1,"rhs":0}}
                    graph=_semantic_graph(m,k,n,dtype,sem).to_mlir(target="nvidia_sm120",canonical=True)
                    start=time.perf_counter_ns()
                    native=export_native_sm120_tensor_graph(graph)
                    edge=package_scheduled_tensor_matmul(*native.scheduled_members(),
                        pipeline_name="tessera-nvidia-pipeline-sm120")
                    package_wall=(time.perf_counter_ns()-start)/1e6
                    rng=np.random.default_rng(120081)
                    x=rng.normal(0,.25,(m,k)).astype(storage)
                    rhs=np.array(rng.normal(0,.25,(k,n)),dtype=storage,order="F" if order=="col_major" else "C")
                    xf=x.astype(np.float64)
                    if operation=="tessera.softmax":
                        ex=np.exp(xf-xf.max(axis=-1,keepdims=True))
                        expected_edge=(ex/ex.sum(axis=-1,keepdims=True)).astype(storage)
                    else:
                        z=xf-xf.mean(axis=-1,keepdims=True) if operation=="tessera.layer_norm" else xf
                        expected_edge=(z/np.sqrt(np.mean(z*z,axis=-1,keepdims=True)+1e-5)).astype(storage)
                    expected=expected_edge.astype(np.float64)@rhs.astype(np.float64)
                    tolerance=dict(rtol=.02 if dtype=="bf16" else .003,atol=.01 if dtype=="bf16" else .002)
                    wall=[]
                    for _ in range(3):
                        start=time.perf_counter_ns()
                        with edge.execute_resident(x,rhs) as result:
                            actual=result.output.numpy()
                            np.testing.assert_allclose(result.intermediate.numpy().astype(np.float64),
                                expected_edge.astype(np.float64),rtol=.008 if dtype=="bf16" else .002,atol=.002)
                            np.testing.assert_allclose(actual,expected,**tolerance)
                            if any(r["execution_kind"]!="native_gpu" for r in (result.producer_receipt,result.consumer_receipt)):
                                raise RuntimeError("native execution lost")
                        wall.append((time.perf_counter_ns()-start)/1e6)
                    with NvidiaDeviceSession() as session:
                        ds=session.upload(x)
                        db=session.upload(rhs,layout=order)
                        intermediate=session.empty((m,k),storage)
                        output=session.empty((m,n),np.float32)
                        pa={edge.producer_input_name:ds,edge.intermediate_name:intermediate}
                        scalar_values={"Rows":m,"Columns":k,"K":k}
                        pa.update({s.name:scalar_values[s.name] for s in edge.producer.descriptor.scalars})
                        ca={edge.consumer_input_name:intermediate,edge.consumer_rhs_name:db,
                            edge.output_name:output,"M":m,"N":n,"K":k}
                        for package,values in ((edge.producer,pa),(edge.consumer,ca)):
                            receipt=rt.launch(NvidiaNormRhsProgram.runtime_artifact(package),values,stream=session.stream)
                            if not receipt.get("ok"):raise RuntimeError(receipt)
                        session.synchronize()
                        np.testing.assert_allclose(session.download(output),expected,**tolerance)
                        samples={role:[rt._nvidia_native_descriptor_resident_device_latency(
                            p.image,p.descriptor,values,stream=session.stream,warmup=20,reps=100)
                            for _ in range(3)] for role,p,values in
                            (("producer",edge.producer,pa),("consumer",edge.consumer,ca))}
                        session.synchronize()
                        np.testing.assert_allclose(session.download(output),expected,**tolerance)
                    rows.append(dict(dtype=dtype,producer=operation,rhs_layout=order,shape_mkn=[m,k,n],
                        role_indices=native.manifest["role_indices"],
                        compiler_program_sha256=hashlib.sha256(native.plan_json.encode()).hexdigest(),
                        image_digests=[p.image.image_digest for p in (edge.producer,edge.consumer)],
                        package_wall_ms=package_wall,end_to_end_samples_ms=wall,
                        end_to_end_median_ms=statistics.median(wall),
                        resident_event_dispatch_samples_ms=samples,
                        resident_event_dispatch_median_ms={role:statistics.median(v) for role,v in samples.items()},
                        max_abs_error=float(np.max(np.abs(actual.astype(np.float64)-expected))),
                        correctness="independent_float64_oracle_with_producer_storage_rounding"))
                    print("verified",dtype,operation,order,(m,k,n),flush=True)
    sources=("src/transforms/lib/NativeSM120TensorProgram.h","src/transforms/lib/NativeScaledMatmulProgram.h",
        "python/tessera/compiler/native_sm120_tensor_program.py","python/tessera/compiler/nvidia_native.py",
        "tests/unit/test_native_sm120_tensor_partition.py","benchmarks/nvidia/benchmark_native_sm120_tensor_partition.py")
    return dict(schema="tessera.nvidia.native_tensor_partition_benchmark.v1",gpu=gpu,rows=rows,
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        runtime_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        timing_scope="Resident CUDA event windows include device dispatch. End-to-end includes allocation, upload, native execution, readback and cleanup. No kernel-only or speedup claim.",
        scope="Static actual Graph member outlining and scheduled resident execution; public JIT and portable witness integration remain open.")

if __name__=="__main__":
    Path(os.environ["TESSERA_NATIVE_TENSOR_PACKET"]).write_text(json.dumps(record(),indent=2,sort_keys=True)+"\n")
