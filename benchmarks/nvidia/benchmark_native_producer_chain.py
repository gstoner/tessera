"""Correctness-gated native RMSNorm/softmax/matmul chain characterization."""
from pathlib import Path
from contextlib import closing
import argparse
import hashlib
import json
from statistics import median
import subprocess
import time
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.prepared_nvidia_lhs import PreparedLhsCall

@ts.jit(target="nvidia_sm120")
def chain(source, rhs):
    return ts.ops.matmul(ts.ops.softmax(ts.ops.rmsnorm(source, eps=1e-5), axis=-1),
                         rhs, output_dtype="fp32")

@ts.jit(target="nvidia_sm120")
def three_chain(source, rhs):
    value=ts.ops.layer_norm(source, eps=1e-5)
    value=ts.ops.rmsnorm(value, eps=1e-5)
    return ts.ops.matmul(ts.ops.softmax(value, axis=-1), rhs, output_dtype="fp32")

def profile(shape, dtype, producer_count=2, rhs_order="C"):
    m,k,n=shape
    storage=np.float16
    if dtype=="bf16":
        import ml_dtypes
        storage=ml_dtypes.bfloat16
    rng=np.random.default_rng(120518)
    x=rng.normal(0,.2,(m,k)).astype(storage)
    rhs=np.array(rng.normal(0,.2,(k,n)),dtype=storage,order=rhs_order)
    xf=x.astype(np.float64)
    layer=None
    if producer_count==3:
        centered=xf-xf.mean(axis=-1,keepdims=True)
        layer=(centered/np.sqrt(np.mean(centered*centered,axis=-1,keepdims=True)+1e-5)).astype(storage)
        xf=layer.astype(np.float64)
    norm=(xf/np.sqrt(np.mean(xf*xf,axis=-1,keepdims=True)+1e-5)).astype(storage)
    nf=norm.astype(np.float64)
    exp=np.exp(nf-nf.max(axis=-1,keepdims=True))
    soft=(exp/exp.sum(axis=-1,keepdims=True)).astype(storage)
    expected=soft.astype(np.float64)@rhs.astype(np.float64)
    function=three_chain if producer_count==3 else chain
    actual=function(x,rhs)
    if function.execution_kind!="native_gpu":
        raise RuntimeError("public chain did not execute natively")
    np.testing.assert_allclose(actual,expected,rtol=.015,atol=.002)
    program=function._nvidia_lhs_last_program
    packages=function.native_lhs_packages()
    wall=[]
    with closing(PreparedLhsCall(program)) as owner:
        owner([x,rhs])
        for _ in range(3):
            start=time.perf_counter()
            for _ in range(10):
                result,receipt=owner([x,rhs])
            wall.append((time.perf_counter()-start)*1000/10)
        np.testing.assert_allclose(result,expected,rtol=.015,atol=.002)
    session=NvidiaDeviceSession()
    stages=[]
    try:
        for index,(package,host,oracle) in enumerate(zip(packages,
                (x,layer,norm,soft) if producer_count==3 else (x,norm,soft),
                (layer,norm,soft,expected) if producer_count==3 else (norm,soft,expected),strict=True)):
            source=session.upload(host)
            output=session.empty(oracle.shape,host.dtype if index<producer_count else np.float32)
            if index<producer_count:
                args={"source":source,"edge":output,"Rows":m,
                      "K" if package.descriptor.provenance["kind"]=="softmax" else "Columns":k}
            else:
                args={"edge":source,"rhs":session.upload(rhs,layout=package.descriptor.provenance["b_layout"]),"out":output,"M":m,"N":n,"K":k}
            samples=[rt._nvidia_native_descriptor_resident_device_latency(
                package.image,package.descriptor,args,stream=session.stream,warmup=3,reps=32)
                for _ in range(3)]
            np.testing.assert_allclose(session.download(output),oracle,rtol=.015,atol=.002)
            stages.append({"kind":package.descriptor.provenance.get("kind","matmul"),
                           "device_event_samples_ms":samples,"median_ms":median(samples)})
    finally:
        session.close()
    return {"shape_mkn":list(shape),"storage":dtype,"producer_count":producer_count,"correctness":"passed_before_and_after_timing",
            "max_abs_error":float(np.max(np.abs(actual.astype(np.float64)-expected))),
            "images":[p.image.image_digest for p in packages],
            "descriptor_digests":[p.descriptor.descriptor_digest for p in packages],
            "consumer_physical_route":packages[-1].descriptor.provenance["physical_route"],
            "rhs_storage_order":packages[-1].descriptor.provenance["b_layout"],
            "plan_digest":hashlib.sha256(program.native_plan_json.encode()).hexdigest(),
            "stages":stages,"prepared_host_wall_samples_ms":wall,"prepared_host_wall_median_ms":median(wall)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--long-macro",action="store_true",
                        help="add K4096 column-major two/three-producer macro profiles")
    args=parser.parse_args()
    active=subprocess.run(["pgrep","-af","[p]ytest|[g]raphify update"],
                          capture_output=True,text=True)
    if active.returncode==0 and active.stdout.strip():
        raise RuntimeError("test or graph extraction is active; matched timing must wait")
    if rt._nvidia_device_name()!="sm_120":
        raise RuntimeError("exact SM120 device required")
    device=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,driver_version","--format=csv,noheader"],text=True).strip()
    root=Path(__file__).resolve().parents[2]
    compiler=Path(__import__("tessera.compiler.scheduled_matmul",fromlist=["find_tessera_opt"]).find_tessera_opt())
    runtime=Path(rt._load_nvidia_ptx_launch()._name).resolve()
    profiles=[profile(shape,dtype,count) for shape in ((17,35,19),(64,256,64)) for dtype in ("fp16","bf16") for count in (2,3)]
    if args.long_macro:
        profiles.extend(profile((128,4096,64),dtype,count,"F")
                        for dtype in ("fp16","bf16") for count in (2,3))
    packet={"schema":"tessera.sm120.native_producer_chain_benchmark.v1",
            "device":device,"target":"nvidia_sm120",
            "compiler_sha256":hashlib.sha256(compiler.read_bytes()).hexdigest(),
            "runtime_path":str(runtime),"runtime_sha256":hashlib.sha256(runtime.read_bytes()).hexdigest(),
            "source_sha256":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),root/"python/tessera/compiler/nvidia_tensor_lhs.py",root/"python/tessera/compiler/resident_nvidia_tensor.py",root/"python/tessera/compiler/prepared_nvidia_lhs.py",root/"src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/matmul_prepared.cpp",root/"src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp"]},
            "method":"independent float64 oracle with each producer rounded to storage; separate resident stage CUDA events; prepared complete-program host wall includes copies and synchronization",
            "profiles":profiles,
            "promotion":"none; characterization only"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2)+"\n")
    print(json.dumps({"profiles":len(packet["profiles"]),"max_error":max(p["max_abs_error"] for p in packet["profiles"]) }))
if __name__=="__main__":
    main()
