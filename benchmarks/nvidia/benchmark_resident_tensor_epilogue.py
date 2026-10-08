"""Correctness-gated resident RMSNorm -> fused matmul measurements on SM120."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"python")]
SOURCE_PATHS=(
    "src/compiler/programming_model/lib/PMPasses.cpp",
    "src/compiler/ir/TileOps.cpp",
    "src/compiler/ir/include/Tessera/Dialect/Tile/TileOps.td",
    "src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
    "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp",
    "python/tessera/compiler/pass_metadata.py",
    "python/tessera/compiler/diagnostic_codes.py",
    "python/tessera/compiler/nvidia_native.py",
    "python/tessera/compiler/scheduled_matmul.py",
    "python/tessera/runtime.py",
    "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp",
    "tests/device/nvidia/test_scheduled_matmul_consumers.py",
    "tests/unit/test_nvidia_tensor_program.py",
    "tests/unit/test_scheduled_matmul_consumers.py",
    "benchmarks/nvidia/benchmark_resident_tensor_epilogue.py",
)

def record(samples:int,reps:int,warmup:int)->dict:
    from tessera import runtime as rt
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    from tests.unit.test_nvidia_tensor_program import _program
    from tests._support.nvidia import nvidia_cuda_host_ready
    if not nvidia_cuda_host_ready():
        raise RuntimeError("requires owning SM120 host and matching toolchain")
    device=subprocess.run(["nvidia-smi","--query-gpu=name,uuid,driver_version,compute_cap",
                           "--format=csv,noheader"],check=True,capture_output=True,text=True).stdout.strip()
    if device.split(",")[-1].strip()!="12.0":
        raise RuntimeError(f"requires SM120; got {device}")
    rows=[]
    def artifact(package):
        return rt.RuntimeArtifact(metadata={"target":"nvidia_sm120"},native_image=package.image,
                                  launch_descriptor=package.descriptor,tile_ir=package.tile_ir,
                                  target_ir=package.target_ir)
    for shape in ((16,16,8),(17,67,23),(256,256,512)):
        for dtype in ("fp16","bf16"):
            if dtype=="bf16":
                import ml_dtypes
                storage=ml_dtypes.bfloat16
            else:storage=np.float16
            for output_dtype in ("fp16","fp32"):
                for dynamic in (False,True):
                    start=time.perf_counter()
                    program=_program(dtype=dtype,output_dtype=output_dtype,dynamic_m=dynamic,
                                     bias=True,residual=True,activation="relu",shape_mkn=shape)
                    build_ms=(time.perf_counter()-start)*1e3
                    if ("tile.fragment_pack" not in program.consumer.tile_ir
                            or "tile.fragment_unpack" not in program.consumer.tile_ir
                            or "tile.matmul_kernel" in program.consumer.tile_ir):
                        raise RuntimeError("consumer did not use the explicit typed epilogue producer")
                    m=max(1,program.m//2-1) if dynamic else program.m
                    k,n=program.k,program.n
                    rng=np.random.default_rng(120405)
                    x=(rng.normal(size=(m,k))*.2).astype(storage)
                    b=np.asfortranarray((rng.normal(size=(k,n))*.2).astype(storage))
                    bias=(rng.normal(size=(n,))*.2).astype(np.float32)
                    residual=(rng.normal(size=(m,n))*.2).astype(np.float32)
                    with program.execute_resident(x,b,bias=bias,residual=residual) as result:
                        edge=result.device_session.download(result.intermediate)[:m,:k]
                        actual=result.device_session.download(result.output)
                        xf=x.astype(np.float32)
                        reference=(xf/np.sqrt(np.mean(xf*xf,axis=1,keepdims=True)+1e-5)).astype(storage)
                        np.testing.assert_allclose(edge.astype(np.float32),reference.astype(np.float32),
                                                   rtol=.012,atol=.012)
                        expected=np.maximum(edge.astype(np.float32)@b.astype(np.float32)+bias,0)+residual
                        if output_dtype=="fp16":expected=expected.astype(np.float16)
                        np.testing.assert_allclose(actual,expected,rtol=.002,atol=.002)
                        error=float(np.max(np.abs(actual.astype(np.float32)-expected.astype(np.float32))))
                    with NvidiaDeviceSession() as session:
                        dx=session.upload(x)
                        db=session.upload(b,layout="strided" if dynamic else "col_major")
                        extras={name:session.upload(value,layout=program._binding(
                            program.consumer,name,"input").layout)
                            for name,value in program._epilogue_inputs(bias,residual,m,n).items()}
                        edge=session.empty((program.m,k),storage)
                        output=session.empty((m,n),np.float16 if output_dtype=="fp16" else np.float32,
                                             layout="strided" if dynamic else "row_major")
                        pa={program.producer_input_name:dx,program.intermediate_name:edge,"Rows":m,"Columns":k}
                        ca={program.consumer_input_name:edge.view(0,(m,k),edge.dtype,layout="strided") if dynamic else edge,
                            program.consumer_rhs_name:db,program.output_name:output,"M":m,"N":n,"K":k,**extras}
                        if dynamic:ca.update(LDA=k,LDB=k,LDD=n)
                        for package,args in ((program.producer,pa),(program.consumer,ca)):
                            receipt=rt.launch(artifact(package),args,stream=session.stream)
                            if not receipt.get("ok") or receipt.get("execution_kind")!="native_gpu":
                                raise RuntimeError(receipt)
                        session.synchronize()
                        producer=[rt._nvidia_native_descriptor_resident_device_latency(
                            program.producer.image,program.producer.descriptor,pa,stream=session.stream,
                            reps=reps,warmup=warmup) for _ in range(samples)]
                        consumer=[rt._nvidia_native_descriptor_resident_device_latency(
                            program.consumer.image,program.consumer.descriptor,ca,stream=session.stream,
                            reps=reps,warmup=warmup) for _ in range(samples)]
                        np.testing.assert_allclose(session.download(output),expected,rtol=.002,atol=.002)
                    e2e=[]
                    for _ in range(samples):
                        start=time.perf_counter()
                        for _ in range(10):
                            with program.execute_resident(x,b,bias=bias,residual=residual):pass
                        e2e.append((time.perf_counter()-start)*1e3/10)
                    def timing(values):
                        return {"samples_ms":values,"median_ms":statistics.median(values),
                                "cv_percent":100*statistics.pstdev(values)/statistics.mean(values)}
                    rows.append({"dtype":dtype,"output_dtype":output_dtype,"active_shape_mkn":[m,k,n],
                                 "bounds_mkn":[program.m,k,n],"dynamic_m":dynamic,"epilogue_order":"matmul_bias_relu_residual_store",
                                 "consumer_route":"Schedule->typed fragments->epilogue store->NVIDIA Target->PTX",
                                 "correctness":"passed_before_timing_and_after_device_replay","max_abs_error":error,
                                 "package_build_ms":build_ms,"producer":timing(producer),"consumer":timing(consumer),
                                 "end_to_end":timing(e2e),"producer_entry":program.producer.descriptor.entry_symbol,
                                 "consumer_entry":program.consumer.descriptor.entry_symbol,
                                 "consumer_abi":program.consumer.descriptor.abi_id,
                                 "consumer_resources":rt._nvidia_native_descriptor_resources(
                                     program.consumer.image,program.consumer.descriptor,block_size=32),
                                 "producer_provenance":dict(program.producer.descriptor.provenance),
                                 "consumer_provenance":dict(program.consumer.descriptor.provenance),
                                 "image_digests":[program.producer.image.image_digest,program.consumer.image.image_digest]})
    return {"schema":"tessera.nvidia.resident_tensor_epilogue.v2","work_item":"W1.1","target":"nvidia_sm120",
            "device":device,"rows":rows,"source_revision":subprocess.run(["git","rev-parse","HEAD"],
            check=True,capture_output=True,text=True).stdout.strip(),
            "source_worktree_dirty":bool(subprocess.run(["git","status","--porcelain"],check=True,
                 capture_output=True,text=True).stdout.strip()),
            "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_PATHS},
            "compiler_sha256":hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
            "target_compiler_sha256":hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_OPT"]).read_bytes()).hexdigest(),
            "native_launch_library_sha256":hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
            "method":{"samples":samples,"device_reps":reps,"warmup":warmup,"e2e_reps":10,
                      "end_to_end_domain":"allocate/upload/producer/consumer/synchronize/cleanup; excludes compilation and download",
                      "device_domain":"independent CUDA-event producer and consumer measurements; not fused total",
                      "selector_changed":False}}
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=7)
    parser.add_argument("--reps",type=int,default=1000)
    parser.add_argument("--warmup",type=int,default=20)
    args=parser.parse_args()
    args.output.write_text(json.dumps(record(args.samples,args.reps,args.warmup),indent=2,sort_keys=True)+"\n")
    print(f"wrote {args.output}")
if __name__=="__main__":main()
