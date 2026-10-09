"""Source-pinned resident/host public saved-LSE reverse comparison."""
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
from benchmarks.nvidia.benchmark_public_attention_vjp import oracle,biased,biased_causal
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.prepared_attention_vjp import prepared,clear_prepared
from tests.device.nvidia.test_ordered_resident_tensor_dag import Borrowed

ROOT=Path(__file__).resolve().parents[2]


def run(order,wrt,sk,causal,bias_shape,repetitions):
    b,hq,hkv=(2,4,2) if bias_shape else (1,2,1)
    shapes={"q":(b,hq,3,4),"k":(b,hkv,sk,4),"v":(b,hkv,sk,3)}
    if bias_shape:shapes["bias"]=bias_shape
    rng=np.random.default_rng(202610093)
    xs={name:rng.normal(0,.2,shape).astype(np.float32) for name,shape in shapes.items()}
    cot=rng.normal(0,.1,(b,hq,3,3)).astype(np.float32)
    expected=oracle(xs,cot,causal)
    body=(biased_causal if causal else biased) if bias_shape else function(order,causal)
    fn=ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=wrt)(body)
    samples={"resident":[],"host":[]};events={"resident":[],"host":[]};errors=[]
    with ExitStack() as stack:
        sessions={name:stack.enter_context(NvidiaDeviceSession()) for name in shapes}
        seed_session=stack.enter_context(NvidiaDeviceSession())
        roots={name:Borrowed(sessions[name].upload(value)) for name,value in xs.items()}
        seed=Borrowed(seed_session.upload(cot))
        for session in (*sessions.values(),seed_session):assert session.synchronize()==0
        def call(mode):
            values=roots if mode=="resident" else xs
            return fn.native_backward(*(values[name] for name in order),
                out_cotangents=seed if mode=="resident" else cot)
        def verify(outputs):
            for name,gradient in zip(wrt,outputs,strict=True):
                np.testing.assert_allclose(gradient,expected[name],rtol=3e-5,atol=3e-5)
                errors.append(float(np.max(np.abs(gradient-expected[name]))))
        for mode in samples:verify(call(mode))
        artifact=fn.native_backward_runtime_artifact()
        owner=prepared(artifact.metadata);handle=owner.handle
        for index in range(repetitions):
            for mode in (("resident","host") if index%2 else ("host","resident")):
                start=time.perf_counter();outputs=call(mode)
                samples[mode].append((time.perf_counter()-start)*1e3)
                events[mode].append(owner.last_device_ms)
                assert owner.handle==handle
                verify(outputs)
        call("resident")
        receipt=dict(fn.last_backward_execution)
        assert receipt["execution_certificate"]["evidence_scope"]=="exact_device"
        assert receipt["execution_certificate"]["source_reexecution"]=="prohibited"
        medians={mode:statistics.median(times) for mode,times in samples.items()}
        row={"order":order,"wrt":wrt,"sk":sk,"causal":causal,"bias_shape":bias_shape,
             "correctness":"independent_fp64_gradients_before_timing","max_abs_error":max(errors),
             "public_completed_call_samples_ms":samples,"public_completed_call_medians_ms":medians,
             "native_forward_backward_event_samples_ms":events,
             "native_forward_backward_event_medians_ms":{
                 mode:[statistics.median(pair[i] for pair in pairs) for i in range(2)]
                 for mode,pairs in events.items()},
             "resident_over_host":medians["resident"]/medians["host"],
             "compiler_receipt":receipt,"resident_frontend_certificate":dict(fn.last_frontend_differential.contract)}
        clear_prepared()
        return row


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repetitions",type=int,default=9);args=parser.parse_args()
    if args.repetitions<3:raise ValueError("requires at least three alternating rounds")
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,driver_version","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or "RTX 5070" not in gpu or gpu.split(",")[2].strip()!="12.0":
        raise RuntimeError("requires owning RTX5070/SM120")
    rows=[run(order,wrt,sk,causal,bias,args.repetitions) for order,wrt,sk,causal,bias in (
        (("q","k","v"),("q",),5,False,None),(("v","q","k"),("v","k","q"),5,False,None),
        (("k","v","q"),("v",),129,True,None),(("v","k","q"),("q","k","v"),129,True,None),
        (("bias","v","q","k"),("bias","v","k","q"),5,False,(1,4,1,1)),
        (("bias","v","q","k"),("bias","k","q","v"),5,True,(1,4,1,1)))]
    files=("python/tessera/compiler/jit.py","python/tessera/compiler/native_vjp_plugins.py",
           "python/tessera/compiler/native_attention_vjp_runtime.py",
           "python/tessera/compiler/prepared_attention_vjp.py",
           "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp",
           "benchmarks/nvidia/record_resident_attention_vjp.py")
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    packet={"architecture":"sm120","device":gpu,"rows":rows,
            "fingerprints":{name:digest(ROOT/name) for name in files},
            "compiler_sha256":digest(os.environ["TESSERA_OPT"]),
            "native_provider_sha256":digest(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]),
            "timing_scope":"forward/backward device events exclude producer waits/snapshots; public completed calls include snapshots and host gradient downloads",
            "resident_storage":"compact rank-four fp32 roots/cotangent; native private saved-LSE generation"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"rows":len(rows),
          "wall_ratios":[row["resident_over_host"] for row in rows]},indent=2))


if __name__=="__main__":main()
