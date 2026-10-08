"""Ordinary saved-LSE attention: native route, independent oracle and split timing."""
import argparse
import hashlib
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
from tests.device.nvidia.test_public_attention_forward import saved_function, saved_alias_function, saved_forward_oracle

ROOT=Path(__file__).resolve().parents[2]


def record(shape,order,causal,bias,samples,reps,tuple_aliases=False):
    b,hq,hkv,sq,sk,d,dv=shape
    rng=np.random.default_rng(120620)
    values={name:rng.normal(size=s).astype(np.float32)*.2 for name,s in zip(
        ("q","k","v"),((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv)),strict=True)}
    if bias: values["bias"]=rng.normal(size=(1,hq,1,sk)).astype(np.float32)*.1
    expected=saved_forward_oracle(values["q"],values["k"],values["v"],causal,values.get("bias"))
    factory=saved_alias_function if tuple_aliases else saved_function
    call=ts.jit(target="nvidia_sm120")(factory(order,causal,bias))
    start=time.perf_counter_ns();result=call(**values);cold=(time.perf_counter_ns()-start)/1e6
    if call.execution_kind!="native_gpu": raise RuntimeError("saved-LSE call did not execute natively")
    def check(actual):
        if not isinstance(actual,tuple) or len(actual)!=2: raise RuntimeError("saved result contract was lost")
        for value,reference in zip(actual,expected,strict=True):
            np.testing.assert_allclose(value,reference,rtol=3e-5,atol=3e-5)
    check(result)
    warm=[]
    for _ in range(samples):
        start=time.perf_counter_ns();result=call(**values);warm.append((time.perf_counter_ns()-start)/1e6)
        check(result)
    artifact=rt.RuntimeArtifact.from_json(call.runtime_artifact().to_json())
    descriptor=artifact.launch_descriptor
    module,_=call._trace_frontend_capture(tuple(values[name] for name in order),{})
    host={arg.name:values[name] for arg,name in zip(module.functions[0].args,order,strict=True)}
    outputs=[binding for binding in descriptor.buffers if binding.direction=="output"]
    for binding,array in zip(outputs,result,strict=True): host[binding.name]=np.empty_like(array)
    scalars=dict(zip(("B","Hq","Hkv","Sq","Sk","D","Dv"),shape,strict=True))
    if descriptor.provenance["bias_shape"]:
        scalars.update(zip(("BiasB","BiasH","BiasQ","BiasK"),descriptor.provenance["bias_shape"],strict=True))
    receipt=rt.launch(artifact,{"buffers":host,"scalars":scalars})
    if not receipt["ok"] or receipt["execution_kind"]!="native_gpu": raise RuntimeError(receipt)
    check(tuple(host[binding.name] for binding in outputs))
    with NvidiaDeviceSession() as session:
        resident=dict(scalars)
        for binding in descriptor.buffers:
            array=host[binding.name]
            resident[binding.name]=(session.empty(array.shape,array.dtype,layout=binding.layout)
                if binding.direction=="output" else session.upload(array,layout=binding.layout))
        receipt=rt.launch(artifact,resident,stream=session.stream)
        if not receipt["ok"]: raise RuntimeError(receipt)
        session.synchronize()
        check(tuple(session.download(resident[binding.name]) for binding in outputs))
        events=[rt._nvidia_native_descriptor_resident_device_latency(
            artifact.native_image,descriptor,resident,stream=session.stream,reps=reps,warmup=20)
            for _ in range(samples)]
        session.synchronize()
        actual=tuple(session.download(resident[binding.name]) for binding in outputs)
        check(actual)
    return {"shape_bhqhkvsqskddv":list(shape),"argument_order":list(order),"causal":causal,"bias":bias,
        "cold_compile_launch_wall_ms":cold,"warm_public_samples_ms":warm,"warm_public_median_ms":median(warm),
        "resident_event_samples_ms":events,"resident_event_median_ms":median(events),"event_reps":reps,
        "max_abs_error":[float(np.max(np.abs(x-y))) for x,y in zip(actual,expected,strict=True)],
        "input_sha256":{name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in values.items()},
        "correctness":"independent_fp64_O_and_natural_log_LSE_before_each_wall_sample_and_after_resident_timing",
        "abi_id":descriptor.abi_id,"entry_symbol":descriptor.entry_symbol,
        "image_digest":artifact.native_image.image_digest,"provenance":descriptor.provenance}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=5)
    parser.add_argument("--reps",type=int,default=100)
    parser.add_argument("--tuple-aliases",action="store_true",help="record bound/copied/destructured saved-LSE tuples")
    args=parser.parse_args()
    if args.samples<3 or args.reps<1: parser.error("at least three windows and positive repetitions required")
    device=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(device.splitlines())!=1 or device.split(",")[-1].strip()!="12.0":
        raise RuntimeError("one selected exact SM120 device required")
    sources=("python/tessera/__init__.py","python/tessera/ops.pyi","python/tessera/apple_gpu_ops_interception.py",
        "python/tessera/compiler/jit.py","python/tessera/compiler/trace.py","python/tessera/compiler/graph_ir.py",
        "python/tessera/compiler/driver.py","python/tessera/compiler/scheduled_checkpoint.py",
        "python/tessera/compiler/nvidia_native.py","python/tessera/autodiff/vjp.py","python/tessera/autodiff/jvp.py",
        "python/tessera/runtime.py","src/compiler/programming_model/lib/PMPasses.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
        "tests/device/nvidia/test_public_attention_forward.py","benchmarks/nvidia/record_public_saved_lse_attention.py")
    tools={name:os.environ[name] for name in ("TESSERA_OPT","TESSERA_NVIDIA_OPT","TESSERA_NVIDIA_PTX_LAUNCH_LIB")}
    packet={"schema":"tessera.public_saved_lse_attention.v1","work_item":"NVIDIA-LSE-1",
        "sync_key":"NVIDIA-PUBLIC-SAVED-LSE-2026-10-06","architecture":"sm_120","device":device,
        "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        "tool_sha256":{name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in tools.items()},
        "route":"ordinary Python frontend -> original Graph -> native Schedule -> Tile -> NVIDIA Target -> PTX -> checked ABI",
        "timing_scope":{"warm_public":"checked binding/allocation/transfers/synchronized native launch wall",
            "resident_event":"resident repeated CUDA event window including dispatch gaps; upload/readback excluded"},
        "selector_promotion":False,"frontend_spelling":"tuple_aliases" if args.tuple_aliases else "direct_call","rows":[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for shape in ((1,2,1,3,5,4,3),(2,4,2,5,3,4,6),(1,4,2,17,19,16,12)):
        for causal in (False,True):
            for order,bias in ((("q","k","v"),False),(("v","k","q"),False),(("bias","v","q","k"),True)):
                packet["rows"].append(record(shape,order,causal,bias,args.samples,args.reps,args.tuple_aliases))
                args.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+"\n")
                print("verified",shape,causal,order,flush=True)


if __name__=="__main__": main()
