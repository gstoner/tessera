"""Correctness-gated native serial/flat ReLU ABBA measurement on exact SM120."""
from pathlib import Path
import argparse
import ctypes as ct
import hashlib
import json
import os
import statistics
import subprocess
import time
import numpy as np
import ml_dtypes
from tests.unit.test_sm120_coalesced_activation import activation_package
from tests.unit.test_sm120_scheduled_activation import _frontend_program
from tessera import runtime as rt
from tessera.compiler.nvidia_tensor_lhs import package_traced_lhs
from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=args.harness.resolve().parent
    assert rt._nvidia_device_name()=="sm_120"
    helper=ct.CDLL(str(args.harness.resolve()))
    fn=helper.benchmark
    fn.argtypes=[ct.c_char_p,ct.c_char_p,ct.c_void_p,ct.c_void_p,ct.c_int64,ct.c_int64,ct.POINTER(ct.c_float),ct.POINTER(ct.c_float),ct.c_int,ct.c_int]
    fn.restype=ct.c_int
    rows=[]
    for storage,dtype in (("f16",np.float16),("bf16",ml_dtypes.bfloat16)):
     for shape in ((7,32),(17,64),(129,256),(256,256),(257,257),(65537,1)):
      rng=np.random.default_rng(51015+sum(shape))
      source=rng.uniform(-1,1,shape).astype(dtype)
      expected=np.maximum(0,source).view(np.uint16)
      packages={mode:activation_package(storage,shape,mode) for mode in ("serial","flat_128")}
      arms=[]
      for mode in ("serial","flat_128","flat_128","serial"):
       artifact,package=packages[mode]
       output=np.empty_like(source)
       rt._submit_nvidia_sm120_native(package.image,package.descriptor,
         {artifact.input_name:source,artifact.output_name:output},{"Rows":shape[0],"Columns":shape[1]},None)
       np.testing.assert_array_equal(output.view(np.uint16),expected)
       ordinary=(ct.c_float*9)();captured=(ct.c_float*9)()
       rc=fn(package.image.payload,package.descriptor.entry_symbol.encode(),
          source.ctypes.data,output.ctypes.data,*shape,ordinary,captured,9,128)
       if rc:raise RuntimeError(f"native matched timing rc={rc}")
       np.testing.assert_array_equal(output.view(np.uint16),expected)
       arms.append({"mode":mode,"native_enqueue_event_samples_ms":list(ordinary),"graph_batch_event_samples_ms":list(captured)})
      result={"storage":storage,"shape":list(shape),"arms":arms,"images":{mode:{"entry":p.descriptor.entry_symbol,"image":p.image.image_digest,"schedule":p.descriptor.provenance["schedule_digest"],"ptx_sha256":hashlib.sha256(p.image.payload).hexdigest()} for mode,(_,p) in packages.items()}}
      for mode in packages:
       result[mode+"_graph_median_us"]=1000*statistics.median(v for a in arms if a["mode"]==mode for v in a["graph_batch_event_samples_ms"])
      rows.append(result)
      print(storage,shape,"serial_us",result["serial_graph_median_us"],"flat_us",result["flat_128_graph_median_us"],flush=True)
    composed=[]
    for storage,dtype in (("f16",np.float16),("bf16",ml_dtypes.bfloat16)):
     for shape in ((7,32,9),(17,64,19),(129,256,33)):
      _,values,public=_frontend_program(dtype,shape)
      graph,_=public._trace_frontend_capture(values,{})
      programs={mode:package_traced_lhs(graph,activation_schedule=mode) for mode in ("serial","flat_128")}
      assert programs["serial"].producer_chain[0].image.payload==programs["flat_128"].producer_chain[0].image.payload
      assert programs["serial"].edge.consumer.image.payload==programs["flat_128"].edge.consumer.image.payload
      a,b=values;wide=a.astype(np.float32)
      normalized=(wide/np.sqrt(np.mean(wide*wide,axis=-1,keepdims=True)+np.float32(1e-5))).astype(dtype)
      expected=np.maximum(0,normalized).astype(np.float32) @ b.astype(np.float32)
      arms=[]
      for mode in ("serial","flat_128","flat_128","serial"):
       owner=PreparedLhsCall(programs[mode])
       try:
        output,receipt=owner(values)
        np.testing.assert_allclose(output,expected,atol=.02 if storage=="bf16" else .003,rtol=.02 if storage=="bf16" else .003)
        for _ in range(25):owner(values)
        profiles=[owner.profile(repeats=128) for _ in range(9)]
        for p in profiles:np.testing.assert_allclose(p["output"],expected,atol=.02 if storage=="bf16" else .003,rtol=.02 if storage=="bf16" else .003)
        wall=[]
        for _ in range(9):
         start=time.perf_counter()
         for __ in range(16):owner(values)
         wall.append((time.perf_counter()-start)*1000/16)
        arms.append({"mode":mode,"program_event_samples_ms":[p["program_ms"] for p in profiles],"stage_event_samples_ms":[p["grouped_stage_ms"] for p in profiles],"completed_prepared_host_samples_ms":wall,"max_abs_error":float(np.max(np.abs(output-expected)))})
       finally:owner.close()
      result={"storage":storage,"shape_mkn":list(shape),"normalization_and_consumer_ptx_identical":True,"arms":arms}
      composed.append(result)
      print("composed",storage,shape,[(a["mode"],statistics.median(a["program_event_samples_ms"]),statistics.median(a["completed_prepared_host_samples_ms"])) for a in arms],flush=True)
    paths=['src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp', 'src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/matmul_prepared.cpp', 'src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp', 'src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp', 'src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.h', 'src/compiler/ir/TileOps.cpp', 'src/compiler/programming_model/ir/ScheduleDialect.cpp', 'src/compiler/programming_model/lib/PMPasses.cpp', 'src/transforms/lib/NativeSM120TensorProgram.h', 'python/tessera/compiler/native_sm120_tensor_program.py', 'python/tessera/compiler/nvidia_native.py', 'python/tessera/compiler/nvidia_tensor_lhs.py', 'python/tessera/compiler/nvidia_tensor_dag.py', 'python/tessera/compiler/prepared_nvidia_lhs.py', 'python/tessera/compiler/backend_manifest.py', 'python/tessera/compiler/execution_matrix.py', 'python/tessera/runtime.py', 'tests/unit/test_sm120_coalesced_activation.py', 'tests/unit/test_sm120_scheduled_activation.py', 'benchmarks/nvidia/benchmark_relu_schedule.py', 'benchmarks/nvidia/relu_capture_probe.cpp']
    packet={"revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"source_sha256":{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},"patch_sha256":hashlib.sha256(subprocess.check_output(["git","diff","--binary"])).hexdigest(),"device":subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,driver_version","--format=csv,noheader"],text=True).strip(),"binary_sha256":{k:hashlib.sha256(Path(os.environ[k]).read_bytes()).hexdigest() for k in ("TESSERA_OPT","TESSERA_NVIDIA_OPT","TESSERA_NVIDIA_PTX_LAUNCH_LIB")},"harness_sha256":{p:hashlib.sha256(path.read_bytes()).hexdigest() for p,path in (("relu_capture_probe.cpp",Path("benchmarks/nvidia/relu_capture_probe.cpp")),("native_harness",args.harness))},"method":{"order":"serial,flat,flat,serial","producer_event":"resident same-allocation CUDA-graph event batch means, 128 kernels, nine samples per arm; excludes capture and transfers, includes graph dispatch","ordinary_events":"native C++ enqueue loop; may include CPU enqueue gaps","composed_events":"native prepared three-stage program and isolated member event windows; may include CPU enqueue gaps","host_wall":"completed prepared host calls including validation/copies/synchronization, nine batches of sixteen","public_default_changed":False,"arithmetic_lowering":"compiler-owned Graph->Schedule->Tile->NVIDIA LLVM/NVVM->PTX, no handwritten kernel or PTX rewrite","promotion":"none; explicit optional schedule only","limits":"no FP8/MXFP8/MXFP4 strategy or sibling proof, no composed captured-event metric"},"standalone":rows,"composed":composed,"script_source":Path(__file__).read_text()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2)+"\n")


if __name__ == "__main__":
    main()
