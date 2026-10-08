"""Owning SM120 public row-softmax package characterization."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import subprocess
import time
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.scheduled_matmul import find_tessera_opt

@ts.jit(target="nvidia_sm120")
def safe(x):
    return ts.ops.softmax_safe(x, axis=-1)

@ts.jit(target="nvidia_sm120")
def ordinary(x):
    return ts.ops.softmax(x, axis=-1)

def profile(shape, dtype):
    import ml_dtypes
    storage={"fp32":np.float32,"fp16":np.float16,"bf16":ml_dtypes.bfloat16}[dtype]
    x=np.random.default_rng(5070120).uniform(-20,20,shape).astype(storage)
    x.reshape(-1,shape[-1])[0]=storage(1000)
    xf=x.astype(np.float64)
    ex=np.exp(xf-xf.max(axis=-1,keepdims=True))
    oracle=ex/ex.sum(axis=-1,keepdims=True)
    rows=[]
    for name,function in (("softmax",ordinary),("softmax_safe",safe)):
        actual=function(x)
        if function.execution_kind!="native_gpu":
            raise RuntimeError("ordinary frontend did not execute a native package")
        np.testing.assert_allclose(actual.astype(np.float64),oracle,rtol=.01,atol=2e-4)
        artifact=function.runtime_artifact()
        image,descriptor=artifact.native_image,artifact.launch_descriptor
        bindings=sorted(descriptor.buffers,key=lambda b:b.ordinal)
        session=NvidiaDeviceSession()
        try:
            resident={bindings[0].name:session.upload(x),
                      bindings[1].name:session.empty(shape,storage),
                      "Rows":math.prod(shape[:-1]),"K":shape[-1]}
            samples=[rt._nvidia_native_descriptor_resident_device_latency(
                image,descriptor,resident,stream=session.stream,reps=100,warmup=10)
                for _ in range(3)]
            np.testing.assert_allclose(session.download(resident[bindings[1].name]).astype(np.float64),
                                       oracle,rtol=.01,atol=2e-4)
        finally:
            session.close()
        wall=[]
        previous=subprocess.run
        def forbidden(*args,**kwargs):
            raise AssertionError("timed warm native call spawned a compiler process")
        subprocess.run=forbidden
        try:
            for _ in range(3):
                start=time.perf_counter()
                for _ in range(11):
                    actual=function(x)
                wall.append((time.perf_counter()-start)*1000/11)
        finally:
            subprocess.run=previous
        np.testing.assert_allclose(actual.astype(np.float64),oracle,rtol=.01,atol=2e-4)
        rows.append({"operation":name,"shape":list(shape),"dtype":dtype,
                     "image_digest":image.image_digest,"descriptor_digest":descriptor.descriptor_digest,
                     "provenance":dict(descriptor.provenance),"abi_id":descriptor.abi_id,
                     "correctness":"passed_before_and_after_timing",
                     "max_abs_error":float(np.max(np.abs(actual.astype(np.float64)-oracle))),
                     "resident_device_event_samples_ms":samples,"resident_device_event_median_ms":median(samples),
                     "public_host_wall_samples_ms":wall,"public_host_wall_median_ms":median(wall)})
    if rows[0]["image_digest"]!=rows[1]["image_digest"]:
        raise RuntimeError("safe alias did not produce the same canonical native image")
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    device=subprocess.check_output(
        ["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,compute_cap","--format=csv,noheader"],
        text=True).strip()
    if "12.0" not in device:
        raise RuntimeError("requires owning sm_120 hardware")
    tool=find_tessera_opt()
    if tool is None:
        raise RuntimeError("requires production tessera-opt")
    runtime=rt._load_nvidia_ptx_launch()
    if runtime is None:
        raise RuntimeError("requires native CUDA runtime")
    runtime_path=Path(runtime._name).resolve()
    root=Path(__file__).resolve().parents[2]
    sources=("python/tessera/compiler/jit.py",
             "python/tessera/compiler/scheduled_kernel.py",
             "python/tessera/compiler/backend_manifest.py",
             "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp",
             "tests/device/nvidia/test_scheduled_semantic_kernels.py")
    packet={"schema":"tessera.sm120.public_softmax_alias.v1","device":device,
            "runtime_path":str(runtime_path),
            "runtime_sha256":hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
            "source_sha256":{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources},
            "compiler_sha256":hashlib.sha256(Path(tool).read_bytes()).hexdigest(),
            "recorder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "timing_scope":"resident CUDA-event kernel loop and separate warm public host-wall calls",
            "promotion":False,"rows":[]}
    for shape in ((3,1),(3,17),(2,3,257)):
        for dtype in ("fp32","fp16","bf16"):
            packet["rows"].extend(profile(shape,dtype))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"rows":len(packet["rows"]),"max_abs_error":max(r["max_abs_error"] for r in packet["rows"])}))

if __name__=="__main__":
    main()
