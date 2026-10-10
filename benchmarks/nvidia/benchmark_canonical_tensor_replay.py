"""Exact SM120 canonical tensor tiling -> native Schedule/Tile package proof."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"python"),str(ROOT/"tests/unit")]
from tessera import runtime as rt
from tessera.compiler import scheduled_matmul,nvidia_native
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.unit.test_scheduled_matmul_consumers import _module
from tests._support.nvidia import nvidia_cuda_host_ready

SOURCES=("src/transforms/lib/TileIRLoweringPass.cpp",
 "src/transforms/lib/TilingPass.cpp","src/transforms/lib/Passes.cpp",
 "src/transforms/include/Tessera/Transforms/Passes.h","src/compiler/programming_model/lib/PMPasses.cpp",
 "src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
 "python/tessera/compiler/scheduled_matmul.py","python/tessera/compiler/nvidia_native.py",
 "python/tessera/runtime.py",
 "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp",
 "tests/unit/test_sm120_legacy_scheduled_producer.py",
 "tests/unit/test_scheduled_matmul_consumers.py",
 "benchmarks/nvidia/benchmark_canonical_tensor_replay.py")

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def timed(call):
    start=time.perf_counter_ns();value=call()
    return value,(time.perf_counter_ns()-start)/1e6

def record(samples,reps,*,permute_arguments=False):
    if not nvidia_cuda_host_ready():
        raise RuntimeError("owning SM120 GPU and matching native toolchain required")
    device=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(device.splitlines())!=1 or device.split(",")[-1].strip()!="12.0":
        raise RuntimeError("requires one selected SM120 GPU: "+device)
    tool=scheduled_matmul.find_tessera_opt()
    rows=[]
    for shape in ((16,32,16),(17,35,19),(64,256,64)):
      for dtype in ("fp16","bf16"):
       for fused in (False,True):
        module=_module(target="nvidia_sm120",dtype=dtype,shape=shape,
            bias=fused,residual=fused,activation="relu" if fused else "none")
        order=(3,1,0,2) if fused else (1,0)
        if permute_arguments:
            fn=module.functions[0]
            fn.args=[fn.args[i] for i in order]
        else:
            order=tuple(range(4 if fused else 2))
        canonical,graph_ms=timed(lambda:scheduled_matmul.lower_scheduled_matmul(module,target="nvidia_sm120"))
        generic=canonical.graph_ir.replace('"nvidia_sm120"','"generic_tensor_replay"')
        tensor_ir,tiling_ms=timed(lambda:scheduled_matmul.run_tessera_opt(tool,generic,
            "--tessera-tiling=tile-m=16 tile-n=16 tile-k=16"))
        tensor_ir=tensor_ir.replace('"generic_tensor_replay"','"nvidia_sm120"')
        if "tessera.canonical_k_step" not in tensor_ir:raise RuntimeError("no actual tensor producer")
        tile,recovery_ms=timed(lambda:scheduled_matmul.run_tessera_opt(tool,tensor_ir,
            "--tessera-tile-ir-lowering=sm=120"))
        if tile!=canonical.tile_ir:raise RuntimeError("recovered Tile differs from canonical native replay")
        if "tensor.extract_slice" in tile or "tile.async_copy" in tile:
            raise RuntimeError("unbufferized tensor producer survived")
        artifact=replace(canonical,tile_ir=tile);artifact.validate()
        package,package_ms=timed(lambda:nvidia_native.package_scheduled_matmul(artifact,
            pipeline_name="tessera-nvidia-pipeline-sm120"))
        full,full_pipeline_ms=timed(lambda:scheduled_matmul.run_tessera_opt(
            tool,tensor_ir,"--tessera-nvidia-pipeline-sm120"))
        def function_body(source):
            return source[source.index("  llvm.func"):source.rindex("\n}")].strip()
        full_normal=scheduled_matmul.run_tessera_opt(tool,full,"--canonicalize")
        tile_normal=scheduled_matmul.run_tessera_opt(tool,tile,"--canonicalize")
        if function_body(full_normal)!=function_body(tile_normal):
            raise RuntimeError("registered pipeline changed the native executable function/ABI")
        compiled,full_native_ms=timed(lambda:nvidia_native._compile_tile_ir(
            full,package.descriptor.entry_symbol))
        if compiled[1].encode()!=package.image.payload:
            raise RuntimeError("registered pipeline PTX differs from checked Schedule package")
        runtime=rt.RuntimeArtifact(metadata={"target":package.image.target},
            native_image=package.image,launch_descriptor=package.descriptor,
            tile_ir=package.tile_ir,target_ir=package.target_ir)
        m,k,n=shape;rng=np.random.default_rng(120_402)
        storage=np.float16
        if dtype=="bf16":
            import ml_dtypes
            storage=ml_dtypes.bfloat16
        a=(rng.normal(size=(m,k))*.2).astype(storage)
        b=np.asfortranarray((rng.normal(size=(k,n))*.2).astype(storage))
        output=np.full((m,n),np.nan,np.float32)
        host={artifact.a_name:a,artifact.b_name:b,artifact.output_name:output,
              "M":m,"N":n,"K":k}
        expected=a.astype(np.float32)@b.astype(np.float32)
        if fused:
            bias=(rng.normal(size=n)*.1).astype(np.float32)
            residual=(rng.normal(size=(m,n))*.05).astype(np.float32)
            host[artifact.bias_name]=bias;host[artifact.residual_name]=residual
            expected=np.maximum(expected+bias,0)+residual
        receipt=rt.launch(runtime,host)
        if not receipt.get("ok") or receipt.get("execution_kind")!="native_gpu":
            raise RuntimeError(receipt)
        np.testing.assert_allclose(output,expected,rtol=4e-5,atol=4e-5)
        with NvidiaDeviceSession() as session:
            resident={"M":m,"N":n,"K":k}
            for binding in package.descriptor.buffers:
                value=host[binding.name]
                resident[binding.name]=(session.empty(value.shape,value.dtype,layout=binding.layout)
                    if binding.direction=="output" else session.upload(value,layout=binding.layout))
            receipt=rt.launch(runtime,resident,stream=session.stream)
            if not receipt.get("ok") or receipt.get("execution_kind")!="native_gpu":
                raise RuntimeError(receipt)
            session.synchronize()
            actual=session.download(resident[artifact.output_name])
            np.testing.assert_allclose(actual,expected,rtol=4e-5,atol=4e-5)
            device_samples=[rt._nvidia_native_descriptor_resident_device_latency(
                package.image,package.descriptor,resident,stream=session.stream,
                reps=reps,warmup=20) for _ in range(samples)]
            session.synchronize()
            np.testing.assert_allclose(session.download(resident[artifact.output_name]),expected,
                rtol=4e-5,atol=4e-5)
        wall=[]
        for _ in range(samples):
            receipt,elapsed=timed(lambda:rt.launch(runtime,host))
            if not receipt.get("ok"):raise RuntimeError(receipt)
            np.testing.assert_allclose(output,expected,rtol=4e-5,atol=4e-5)
            wall.append(elapsed)
        rows.append({"shape_mkn":list(shape),"dtype":dtype,
            "frontend_argument_order":list(order),"fused_bias_relu_residual":fused,
            "canonical_graph_to_tile_wall_ms":graph_ms,"generic_tensor_tiling_wall_ms":tiling_ms,
            "verified_recovery_to_tile_wall_ms":recovery_ms,"native_package_wall_ms":package_ms,
            "registered_full_pipeline_wall_ms":full_pipeline_ms,
            "registered_full_native_compile_wall_ms":full_native_ms,
            "registered_full_ptx_sha256":hashlib.sha256(compiled[1].encode()).hexdigest(),
            "registered_full_image_parity":"byte_identical_to_executed_checked_schedule_package",
            "resident_device_event_samples_ms":device_samples,
            "resident_device_event_median_ms":statistics.median(device_samples),
            "host_package_launch_wall_samples_ms":wall,
            "host_package_launch_wall_median_ms":statistics.median(wall),
            "max_abs_error":float(np.max(np.abs(actual-expected))),
            "canonical_tensor_ir_sha256":hashlib.sha256(tensor_ir.encode()).hexdigest(),
            "tile_ir_sha256":hashlib.sha256(tile.encode()).hexdigest(),
            "target_ir_sha256":hashlib.sha256(package.target_ir.encode()).hexdigest(),
            "image_sha256":hashlib.sha256(package.image.payload).hexdigest(),
            "entry_symbol":package.descriptor.entry_symbol,"abi_id":package.descriptor.abi_id,
            "correctness":"independent_numpy_oracle_before_and_after_resident_timing",
            "tile_parity":"byte_identical_to_direct_graph_schedule_tile"})
    return {"schema":"tessera.nvidia.canonical_tensor_replay.v1",
        "architecture":"sm_120","device":device,"kernel_release":platform.release(),
        "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "dirty_worktree":bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()),
        "source_sha256":{p:digest(ROOT/p) for p in SOURCES},"compiler_sha256":digest(tool),
        "timing_scope":"CUDA-event resident C++ launch windows include any driver dispatch gaps; host package wall includes allocation/transfers/module lifecycle; compilation stages separate; no isolated instruction time or speedup claim",
        "samples":samples,"reps":reps,"rows":rows}

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--permute-arguments",action="store_true")
    parser.add_argument("--samples",type=int,default=5);parser.add_argument("--reps",type=int,default=200)
    args=parser.parse_args()
    if args.samples<3 or args.reps<20:raise ValueError("at least three samples and 20 repetitions")
    packet=record(args.samples,args.reps,permute_arguments=args.permute_arguments)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2,sort_keys=True)+"\n")
    print("verified",len(packet["rows"]),"SM120 canonical tensor replay rows")
