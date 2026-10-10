"""Ordinary frontend attention -> native Schedule/Tile -> checked SM120 launch."""
import argparse
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from benchmarks.nvidia.benchmark_jvp_argument_order import function
from tests.device.nvidia.test_public_attention_forward import forward_oracle


def record(shape,order,causal,samples,reps):
    b,hq,hkv,sq,sk,d,dv=shape
    rng=np.random.default_rng(5070)
    arrays={n:rng.normal(size=s).astype(np.float32)*.2 for n,s in zip(
        ("q","k","v"),((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv)),strict=True)}
    expected=forward_oracle(*(arrays[n] for n in ("q","k","v")),causal)
    fn=ts.jit(target="nvidia_sm120")(function(order,causal))
    start=time.perf_counter_ns()
    output=fn(**arrays)
    cold=(time.perf_counter_ns()-start)/1e6
    if fn.execution_kind!="native_gpu":raise RuntimeError("ordinary attention did not execute natively")
    np.testing.assert_allclose(output,expected,rtol=3e-5,atol=3e-5)
    artifact=fn.runtime_artifact()
    descriptor=artifact.launch_descriptor
    module,_=fn._trace_frontend_capture(tuple(arrays[n] for n in order),{})
    host={arg.name:arrays[name] for arg,name in zip(module.functions[0].args,order,strict=True)}
    output_name=next(x.name for x in descriptor.buffers if x.direction=="output")
    host[output_name]=np.empty(output.shape,np.float32)
    scalars=dict(zip(("B","Hq","Hkv","Sq","Sk","D","Dv"),shape,strict=True))
    warm=[]
    for _ in range(samples):
        start=time.perf_counter_ns();value=fn(*(arrays[n] for n in order))
        warm.append((time.perf_counter_ns()-start)/1e6)
        np.testing.assert_allclose(value,expected,rtol=3e-5,atol=3e-5)
    with NvidiaDeviceSession() as session:
        resident=dict(scalars)
        for binding in descriptor.buffers:
            value=host[binding.name]
            resident[binding.name]=(session.empty(value.shape,value.dtype,layout=binding.layout)
                if binding.direction=="output" else session.upload(value,layout=binding.layout))
        receipt=rt.launch(artifact,resident,stream=session.stream)
        if not receipt["ok"]:raise RuntimeError(receipt)
        session.synchronize()
        np.testing.assert_allclose(session.download(resident[output_name]),expected,rtol=3e-5,atol=3e-5)
        events=[rt._nvidia_native_descriptor_resident_device_latency(
            artifact.native_image,descriptor,resident,stream=session.stream,reps=reps,warmup=20)
            for _ in range(samples)]
        session.synchronize()
        actual=session.download(resident[output_name])
        np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=3e-5)
    return {"shape_bhqhkvsqskddv":list(shape),"argument_order":list(order),"causal":causal,
        "cold_compile_launch_wall_ms":cold,"warm_public_wall_samples_ms":warm,
        "warm_public_wall_median_ms":median(warm),"resident_event_samples_ms":events,
        "resident_event_median_ms":median(events),"event_reps":reps,"event_warmup":20,
        "max_abs_error":float(np.max(np.abs(actual-expected))),
        "correctness":"independent_fp64_before_and_after_timing",
        "image_digest":artifact.native_image.image_digest,"abi_id":descriptor.abi_id,
        "entry_symbol":descriptor.entry_symbol,"provenance":descriptor.provenance}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=3)
    parser.add_argument("--reps",type=int,default=100)
    args=parser.parse_args()
    if args.samples<3 or args.reps<1:raise ValueError("at least three windows and positive repetitions required")
    device=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(device.splitlines())!=1 or device.split(",")[-1].strip()!="12.0":
        raise RuntimeError("one selected exact SM120 device required")
    root=Path(__file__).resolve().parents[2]
    sources=("python/tessera/compiler/jit.py","python/tessera/compiler/scheduled_attention.py",
        "python/tessera/compiler/native_attention_contract.py","python/tessera/compiler/nvidia_native.py",
        "src/compiler/programming_model/lib/PMPasses.cpp","benchmarks/nvidia/benchmark_ordinary_attention.py")
    tool_paths={name:Path(os.environ[variable]) for name,variable in
                (("tessera-opt","TESSERA_OPT"),("tessera-nvidia-opt","TESSERA_NVIDIA_OPT"),
                 ("native-ptx-runtime","TESSERA_NVIDIA_PTX_LAUNCH_LIB"))}
    tool_hashes={name:hashlib.sha256(path.read_bytes()).hexdigest()
                 for name,path in tool_paths.items()}
    packet={"architecture":"sm_120","device":device,"selector_promotion":False,
        "timing_scope":"ordinary warm walls include checked host allocation/copies; native resident C++ CUDA event windows include dispatch gaps; no isolated instruction or saved-LSE speedup claim",
        "source_sha256":{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
        "tool_sha256":tool_hashes,"rows":[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for shape in ((1,2,1,3,5,4,3),(2,4,2,5,3,4,6),(1,4,2,17,19,16,12)):
        for causal in (False,True):
            for order in permutations(("q","k","v")):
                row=record(shape,order,causal,args.samples,args.reps)
                packet["rows"].append(row)
                args.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+"\n")
                print("verified",shape,causal,order,flush=True)


if __name__=="__main__":main()
