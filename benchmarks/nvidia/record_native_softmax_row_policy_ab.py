"""Matched compiler-owned SM120 softmax row-policy DAG A/B evidence."""
from contextlib import ExitStack,closing
from copy import deepcopy
import ctypes as ct
import json
import time
import os
import hashlib
import argparse
import subprocess
from pathlib import Path
from statistics import median
import numpy as np
import ml_dtypes
from tessera import runtime as rt
from tessera.compiler import nvidia_tensor_lhs as lhs
from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_native_tensor_dag import public_dag_deep,oracle
from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed
from benchmarks.nvidia.record_ordered_resident_tensor_dag import identity
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--control-opt",type=Path,required=True)
    options=parser.parse_args()
    candidate=Path(os.environ["TESSERA_OPT"]).resolve()
    control=options.control_opt.resolve()
    assert rt._nvidia_device_name()=="sm_120"
    jobs=subprocess.run(["pgrep","-af","[p]ytest|[g]raphify update"],capture_output=True,text=True)
    if jobs.returncode==0 and jobs.stdout.strip():raise RuntimeError("validation/graph extraction active")
    records=[]
    for dtype,storage in (("fp16",np.float16),("bf16",ml_dtypes.bfloat16)):
     for bounded,(m,n,k) in [(False,shape) for shape in ((17,19,35),(17,65,255),(17,65,256),(129,65,513),(17,257,1024))]+[(True,shape) for shape in ((17,19,35),(129,65,513))]:
      rng=np.random.default_rng(90251)
      a=rng.normal(0,.2,(m,k)).astype(storage);b=rng.normal(0,.2,(k,n)).astype(storage)
      graph=public_dag_deep._traced_autodiff_module((a,b),{})
      programs={}
      for mode in ("serial","row_policy"):
       os.environ["TESSERA_OPT"]=str(control if mode=="serial" else candidate)
       programs[mode]=lhs.from_manifest(lhs.package_traced_lhs(deepcopy(graph),**({"shape_bounds":{"M":257,"N":128,"K":1024}} if bounded else {})).manifest())
      samples={mode:[] for mode in programs};walls={mode:[] for mode in programs};errors={mode:[] for mode in programs}
      with ExitStack() as stack:
       l=stack.enter_context(NvidiaDeviceSession());r=stack.enter_context(NvidiaDeviceSession());s=stack.enter_context(NvidiaDeviceSession())
       av=l.upload(a);bv=r.upload(b)
       assert not l.synchronize() and not r.synchronize()
       roots=[Borrowed(av),Borrowed(bv)]
       owners={mode:stack.enter_context(closing(PreparedLhsCall(p))) for mode,p in programs.items()}
       outputs={mode:s.empty((m,n),np.float32) for mode in programs}
       for mode,o in owners.items():o.invoke_resident(roots,outputs[mode],stream=s.stream)
       expected=oracle(a,b,2)
       for window in range(7):
        for mode in (("serial","row_policy") if window%2==0 else ("row_policy","serial")):
         start=time.perf_counter();owners[mode].invoke_resident(roots,outputs[mode],stream=s.stream)
         walls[mode].append((time.perf_counter()-start)*1000)
         actual=outputs[mode].numpy()
         np.testing.assert_allclose(actual,expected,rtol=.015,atol=.015)
         errors[mode].append(float(np.max(np.abs(actual.astype(np.float64)-expected))))
         samples[mode].append(owners[mode].profile_resident(roots,outputs[mode],stream=s.stream,repeats=128))
         np.testing.assert_allclose(outputs[mode].numpy(),expected,rtol=.015,atol=.015)
       med={mode:median(x["program_ms"] for x in rows) for mode,rows in samples.items()}
       row=dict(dtype=dtype,bounded=bounded,shape_mnk=[m,n,k],program_ms=med,speedup=med["serial"]/med["row_policy"],
         invoke_wall_ms={mode:median(rows) for mode,rows in walls.items()},samples=samples,
         stages_ms={mode:[median(x["grouped_stage_ms"][i] for x in rows) for i in range(5)] for mode,rows in samples.items()},
         max_abs_error=errors,member_tile_ir_sha256={mode:[hashlib.sha256(x.tile_ir.encode()).hexdigest() for x in [*p.producer_chain,*p.rhs_chain,p.edge.consumer]] for mode,p in programs.items()},consumer_physical_target_equal=programs["serial"].edge.consumer.target_ir==programs["row_policy"].edge.consumer.target_ir,
         schedules={mode:[x.descriptor.provenance.get("schedule") for x in [*p.producer_chain,*p.rhs_chain]] for mode,p in programs.items()})
       assert row["consumer_physical_target_equal"]
       records.append(row);print(json.dumps({key:row[key] for key in ("dtype","bounded","shape_mnk","program_ms","speedup","stages_ms")}),flush=True)

    leaf_records=[]
    from benchmarks.nvidia.benchmark_public_softmax_alias import ordinary
    from tessera.compiler.scheduled_kernel import lower_scheduled_kernel
    from tessera.compiler import nvidia_native
    for shape in ((3,35),(3,255),(3,256),(129,513),(17,1024),(3,4097)):
     source=np.random.default_rng(90251).uniform(-20,20,shape).astype(np.float32)
     graph=ordinary._traced_autodiff_module((source,),{})
     value=source.astype(np.float64);exponent=np.exp(value-value.max(axis=-1,keepdims=True))
     expected=exponent/exponent.sum(axis=-1,keepdims=True)
     packages={};outputs={};samples={mode:[] for mode in ("serial","row_policy")}
     for mode,tool in (("serial",control),("row_policy",candidate)):
      os.environ["TESSERA_OPT"]=str(tool)
      scheduled=lower_scheduled_kernel(graph,target="nvidia_sm120")
      package=nvidia_native.package_scheduled_kernel(scheduled,pipeline_name="tessera-nvidia-pipeline-sm120")
      artifact=rt.RuntimeArtifact(metadata={"target":"nvidia_sm120"},native_image=package.image,
       launch_descriptor=package.descriptor,tile_ir=package.tile_ir,target_ir=package.target_ir)
      bindings=sorted(package.descriptor.buffers,key=lambda x:x.ordinal)
      output=np.empty_like(source)
      args={bindings[0].name:source,bindings[1].name:output,"Rows":shape[0],"K":shape[1]}
      packages[mode]=(package,artifact,args);outputs[mode]=output
     for window in range(7):
      for mode in (("serial","row_policy") if window%2==0 else ("row_policy","serial")):
       package,artifact,args=packages[mode]
       result=rt.launch(artifact,args)
       if not result.get("ok"):raise RuntimeError(result)
       np.testing.assert_allclose(outputs[mode].astype(np.float64),expected,rtol=3e-5,atol=2e-6)
       samples[mode].append(rt._nvidia_native_descriptor_device_latency(
        package.image,package.descriptor,args,reps=128,warmup=4))
       result=rt.launch(artifact,args)
       if not result.get("ok"):raise RuntimeError(result)
       np.testing.assert_allclose(outputs[mode].astype(np.float64),expected,rtol=3e-5,atol=2e-6)
     med={mode:median(rows) for mode,rows in samples.items()}
     row=dict(dtype="fp32",shape=list(shape),device_samples_ms=samples,device_medians_ms=med,
      speedup=med["serial"]/med["row_policy"],
      schedules={mode:package[0].descriptor.provenance["schedule"] for mode,package in packages.items()},
      max_abs_error={mode:float(np.max(np.abs(output.astype(np.float64)-expected))) for mode,output in outputs.items()})
     leaf_records.append(row);print(json.dumps(row),flush=True)

    sources=("src/compiler/programming_model/lib/PMPasses.cpp","python/tessera/compiler/scheduled_kernel.py",
    "python/tessera/compiler/nvidia_tensor_dag.py","python/tessera/compiler/prepared_nvidia_lhs.py",
    "tests/device/nvidia/test_cooperative_softmax.py","tests/unit/test_native_softmax_row_policy.py")
    packet=dict(schema="tessera.sm120.native_softmax_row_policy_ab.v1",device=identity(),records=records,leaf_records=leaf_records,
    compiler_sha256={mode:hashlib.sha256(path.read_bytes()).hexdigest() for mode,path in (("serial",control),("row_policy",candidate))},
    control_commit="8d1c9c800c796261e1fa9309dd14f8e7e80fc65e",
    source_sha256={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sources},
    runtime_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
    recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    route="ordinary frontend Graph->native Schedule/Tile->NVIDIA Target->LLVM NVPTX->PTX->C++ ordered resident DAG",
    timing="alternating seven 128-launch CUDA windows after incoming waits; grouped stage windows are separate and nonadditive; prepared invoke wall includes metadata and synchronous completion, excludes compilation/upload/download",
    claim="native device-program comparison for these profiles only; no FP8/MXFP8/MXFP4, sibling or full compiler closure claim")
    options.output.parent.mkdir(parents=True,exist_ok=True)
    options.output.write_text(json.dumps(packet,indent=2)+"\n")


if __name__ == "__main__":
    main()
