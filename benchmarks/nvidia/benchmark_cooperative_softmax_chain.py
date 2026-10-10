"""Matched native producer-chain timing for explicit softmax Schedule policy."""
from pathlib import Path
import time
from statistics import median
import json, hashlib, subprocess, argparse
import numpy as np
import ml_dtypes
from tessera.compiler import nvidia_tensor_lhs as lhs
from tessera.compiler import prepared_nvidia_lhs as prepared
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tessera import runtime as rt
from benchmarks.nvidia.benchmark_native_producer_chain import chain, three_chain

def main():
    parser=argparse.ArgumentParser(description="Matched serial/cooperative prepared native producer-chain host timing")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    device=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,compute_cap","--format=csv,noheader"],text=True).strip()
    if "12.0" not in device or rt._nvidia_device_name()!="sm_120":
        raise RuntimeError("requires exact SM120 device")

    live=subprocess.check_output(["pgrep","-af","[p]ytest|[g]raphify update"],text=True) if subprocess.run(["pgrep","-f","[p]ytest|[g]raphify update"],capture_output=True).returncode==0 else ""
    if live: raise RuntimeError("validation or graph extraction is active: "+live)
    rows=[]
    for dtype,storage in (("fp16",np.float16),("bf16",ml_dtypes.bfloat16)):
     for count,function in ((2,chain),(3,three_chain)):
      m,k,n=128,4096,64
      rng=np.random.default_rng(5070128);x=rng.normal(0,.2,(m,k)).astype(storage)
      rhs=np.array(rng.normal(0,.2,(k,n)),dtype=storage,order="F")
      graph=lhs.project_rhs_storage(function._traced_autodiff_module((x,rhs),{}),[x,rhs])
      value=x.astype(np.float64)
      if count==3:
       centered=value-value.mean(axis=-1,keepdims=True)
       value=(centered/np.sqrt(np.mean(centered*centered,axis=-1,keepdims=True)+1e-5)).astype(storage).astype(np.float64)
      value=(value/np.sqrt(np.mean(value*value,axis=-1,keepdims=True)+1e-5)).astype(storage).astype(np.float64)
      exp=np.exp(value-value.max(axis=-1,keepdims=True));soft=(exp/exp.sum(axis=-1,keepdims=True)).astype(storage)
      expected=soft.astype(np.float64)@rhs.astype(np.float64)
      programs={mode:lhs.package_traced_lhs(graph,softmax_schedule=mode) for mode in ("serial","cooperative_128")}
      owners={mode:prepared.PreparedLhsCall(p) for mode,p in programs.items()}
      samples={mode:[] for mode in programs}
      try:
       for mode,owner in owners.items():
        actual,receipt=owner([x,rhs]);np.testing.assert_allclose(actual,expected,rtol=.015,atol=.002)
       for window in range(5):
        for mode in (("serial","cooperative_128") if window%2==0 else ("cooperative_128","serial")):
         start=time.perf_counter()
         for _ in range(11):actual,receipt=owners[mode]([x,rhs])
         samples[mode].append((time.perf_counter()-start)*1000/11)
         np.testing.assert_allclose(actual,expected,rtol=.015,atol=.002)
       row=dict(dtype=dtype,shape_mkn=[m,k,n],producer_count=count,prepared_host_wall_samples_ms=samples,host_speedup=median(samples["serial"])/median(samples["cooperative_128"]),consumer_image=programs["serial"].edge.consumer.image.image_digest,correctness="before_and_after_timing")
       assert programs["serial"].edge.consumer.image.image_digest==programs["cooperative_128"].edge.consumer.image.image_digest
       rows.append(row);print(json.dumps(row),flush=True)
      finally:
       for owner in owners.values():owner.close()
    tool=find_tessera_opt()
    runtime=Path(rt._load_nvidia_ptx_launch()._name).resolve()
    sources=("python/tessera/compiler/nvidia_tensor_lhs.py",
             "python/tessera/compiler/prepared_nvidia_lhs.py",
             "python/tessera/compiler/nvidia_native.py",
             "python/tessera/compiler/native_sm120_tensor_program.py",
             "src/compiler/programming_model/lib/PMPasses.cpp",
             "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/matmul_prepared.cpp",
             "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp")
    packet=dict(schema="tessera.sm120.cooperative_softmax_chain_ab.v1",device=device,rows=rows,
        compiler_sha256=hashlib.sha256(Path(tool).read_bytes()).hexdigest(),
        runtime_path=str(runtime),runtime_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest(),
        sources={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources},
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        timing="alternating five windows of eleven prepared whole-chain host calls; copies and completion included; no device-kernel claim",
        automatic_selection=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2)+"\n")


if __name__ == "__main__":
    main()
