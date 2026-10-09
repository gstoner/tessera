"""Correctness-gated SM120 resident saved-LSE JVP timing packet."""
from contextlib import ExitStack
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import tessera as ts

from benchmarks.nvidia.benchmark_jvp_argument_order import function
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_attention_jvp_runtime import prepared,clear_prepared
from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed
from tests.device.nvidia.test_resident_attention_jvp import oracle

ROOT=Path(__file__).resolve().parents[2]


def run(order,wrt,sk,causal,repetitions):
    rng=np.random.default_rng(202610091)
    shapes={"q":(1,2,3,4),"k":(1,1,sk,4),"v":(1,1,sk,3)}
    xs={name:rng.normal(0,.2,shape).astype(np.float32) for name,shape in shapes.items()}
    ds={name:rng.normal(0,.1,shape).astype(np.float32) if name in wrt
        else np.zeros(shape,np.float32) for name,shape in shapes.items()}
    expected=oracle(xs,causal)
    h=1e-4
    tangent=(oracle({n:x.astype(np.float64)+h*ds[n] for n,x in xs.items()},causal)-
             oracle({n:x.astype(np.float64)-h*ds[n] for n,x in xs.items()},causal))/(2*h)
    fn=ts.jit(target="nvidia_sm120",autodiff="forward",wrt=wrt)(function(order,causal))
    samples={"resident":[],"host":[]};events={"resident":[],"host":[]}
    errors=[]
    with ExitStack() as stack:
        sessions={name:stack.enter_context(NvidiaDeviceSession()) for name in xs}
        roots={name:Borrowed(sessions[name].upload(value)) for name,value in xs.items()}
        seeds={name:Borrowed(sessions[name].upload(ds[name])) for name in wrt}
        for session in sessions.values():assert session.synchronize()==0
        def call(mode):
            values=roots if mode=="resident" else xs
            directions=seeds if mode=="resident" else ds
            return fn.native_jvp(*(values[name] for name in order),
                                 tangents=tuple(directions[name] for name in wrt))
        for mode in samples:
            outputs=call(mode)
            for actual,reference in zip(outputs,(expected,tangent),strict=True):
                np.testing.assert_allclose(actual,reference,rtol=3e-5,atol=3e-5)
                errors.append(float(np.max(np.abs(actual-reference))))
        package=next(iter(fn._native_jvp_packages.values()))
        owner=prepared(package.contract["steps"][0]["child_metadata"])
        handle=owner.handle
        # Alternate order to avoid systematically assigning drift to either arm.
        for index in range(repetitions):
            for mode in (("resident","host") if index%2 else ("host","resident")):
                start=time.perf_counter();outputs=call(mode)
                samples[mode].append((time.perf_counter()-start)*1e3)
                events[mode].append(owner.last_device_ms)
                assert owner.handle==handle
                for actual,reference in zip(outputs,(expected,tangent),strict=True):
                    np.testing.assert_allclose(actual,reference,rtol=3e-5,atol=3e-5)
        medians={mode:statistics.median(times) for mode,times in samples.items()}
        row={"order":order,"wrt":wrt,"sk":sk,"causal":causal,
             "correctness":"passed_before_timing","max_abs_error":max(errors),
             "public_completed_call_samples_ms":samples,"public_completed_call_medians_ms":medians,
             "native_forward_jvp_event_samples_ms":events,
             "native_forward_jvp_event_medians_ms":{
                 mode:[statistics.median(pair[i] for pair in pairs) for i in range(2)]
                 for mode,pairs in events.items()},
             "resident_over_host":medians["resident"]/medians["host"],
             "native_package_digest":package.artifact_hash}
        # The final alternating call can be host; retain the actual resident proof.
        call("resident")
        row["resident_frontend_certificate"]=fn.last_jvp_execution["frontend_certificate"]
        row["compiler_receipt"]=dict(fn.last_jvp_execution)
        clear_prepared()
        return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repetitions",type=int,default=9)
    args=parser.parse_args()
    if args.repetitions<3:raise ValueError("requires at least three alternating rounds")
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,driver_version","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or "RTX 5070" not in gpu or gpu.split(",")[2].strip()!="12.0":
        raise RuntimeError("requires owning RTX5070/SM120")
    rows=[run(order,wrt,sk,causal,args.repetitions) for order,wrt,sk,causal in (
        (("q","k","v"),("q",),5,False),(("v","q","k"),("v","k","q"),5,False),
        (("k","v","q"),("v",),129,True),(("v","k","q"),("q","k","v"),129,True))]
    files=("python/tessera/compiler/frontend_authority.py","python/tessera/compiler/jit.py",
           "python/tessera/compiler/native_attention_jvp_runtime.py",
           "python/tessera/compiler/resident_nvidia_tensor.py",
           "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp",
           "benchmarks/nvidia/record_resident_attention_jvp.py")
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    packet={"architecture":"sm120","device":gpu,"rows":rows,
            "fingerprints":{name:digest(ROOT/name) for name in files},
            "compiler_sha256":digest(os.environ["TESSERA_OPT"]),
            "native_provider_sha256":digest(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]),
            "timing_scope":"forward and JVP device events exclude producer waits/snapshots; public completed calls include snapshots and host downloads",
            "resident_storage":"compact rank-four fp32 roots; native private Q/K/V/tangent snapshots",
            "frontend_numerical_execution":0}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"rows":len(rows),
          "wall_ratios":[row["resident_over_host"] for row in rows]},indent=2))


if __name__=="__main__":main()
