"""Native-bound frontend input and cotangent role permutation proof on SM120."""
from __future__ import annotations
import argparse, ctypes, hashlib, json, os, statistics, subprocess, time
from pathlib import Path
import numpy as np
import tessera as ts
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests._support.nvidia import nvidia_cuda_host_ready
from benchmarks.nvidia.benchmark_checkpoint_bias_gradient import reference
from benchmarks.nvidia.benchmark_jit_attention_vjp import download
from benchmarks.nvidia.benchmark_jit_attention_bias_vjp import function as canonical

ORDERS = {
    "canonical": ("q","k","v","bias"),
    "k_bias_v_q": ("k","bias","v","q"),
    "bias_v_q_k": ("bias","v","q","k"),
    "v_q_k": ("v","q","k"),
}


def function(order,wrt,causal):
    if order=="canonical":
        return canonical(wrt,causal)
    if order=="k_bias_v_q":
        if causal:
            @ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=wrt)
            def attention(k,bias,v,q):
                return ts.ops.flash_attn(q,k,v,attn_bias=bias,causal=True)
        else:
            @ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=wrt)
            def attention(k,bias,v,q):
                return ts.ops.flash_attn(q,k,v,attn_bias=bias,causal=False)
    elif order=="bias_v_q_k":
        if causal:
            @ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=wrt)
            def attention(bias,v,q,k):
                return ts.ops.flash_attn(q,k,v,attn_bias=bias,causal=True)
        else:
            @ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=wrt)
            def attention(bias,v,q,k):
                return ts.ops.flash_attn(q,k,v,attn_bias=bias,causal=False)
    elif order=="v_q_k":
        if causal:
            @ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=wrt)
            def attention(v,q,k):
                return ts.ops.flash_attn(q,k,v,causal=True)
        else:
            @ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=wrt)
            def attention(v,q,k):
                return ts.ops.flash_attn(q,k,v,causal=False)
    else:
        raise ValueError("unknown frontend order")
    return attention


def run_case(shape,order,wrt,causal,samples=3):
    b,hq,hkv,sq,sk,d,dv=shape;rng=np.random.default_rng(120_406)
    q,k,v,bias,seed=[(rng.normal(size=s)*.3).astype(np.float32) for s in
        ((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),(b,hq,sq,sk),(b,hq,sq,dv))]
    labels=("q","k","v","bias")
    roles=(q,k,v,bias)
    values=[roles[labels.index(name)] for name in ORDERS[order]]
    expected,lse,grad=reference(q,k,v,bias if len(values)==4 else np.zeros_like(bias),seed,causal)
    start=time.perf_counter_ns()
    program=function(order,wrt,causal).compile_native_attention_vjp(
        *values,compiler=Path(os.environ["TESSERA_OPT"]))
    compile_ms=(time.perf_counter_ns()-start)/1e6
    expected_mapping=tuple(ORDERS[order].index(role) for role in labels[:len(values)])
    expected_active=tuple(labels.index(name) for name in wrt)
    assert program.input_indices==expected_mapping
    assert program.active==expected_active
    captures=[];backwards=[];errors=[]
    with NvidiaDeviceSession() as session:
        resident=[session.upload(value) for value in values]
        cotangent=session.upload(seed)
        start=time.perf_counter_ns();frame=program.capture(*resident)
        captures.append((time.perf_counter_ns()-start)/1e6)
        try:
            np.testing.assert_allclose(download(session,frame.primal),expected,atol=4e-5,rtol=4e-5)
            for value in resident:
                changed=np.full(value.shape,19,np.float32)
                rc=session.lib.tessera_nvidia_device_upload(ctypes.c_void_p(value.ptr),
                    ctypes.c_void_p(changed.ctypes.data),changed.nbytes,ctypes.c_void_p(session.stream))
                if rc:raise RuntimeError("caller mutation failed")
            session.synchronize()
            for _ in range(samples):
                start=time.perf_counter_ns();results=frame.backward(cotangent)
                backwards.append((time.perf_counter_ns()-start)/1e6)
                for result,index in zip(results,expected_active,strict=True):
                    host=download(session,result)
                    np.testing.assert_allclose(host,grad[index],atol=4e-5,rtol=4e-5)
                    errors.append(float(np.max(np.abs(host-grad[index]))))
        finally:
            frame.close()
        fresh=[session.upload(value) for value in values]
        for _ in range(samples-1):
            start=time.perf_counter_ns();frame=program.capture(*fresh)
            captures.append((time.perf_counter_ns()-start)/1e6)
            try:np.testing.assert_allclose(download(session,frame.primal),expected,atol=4e-5,rtol=4e-5)
            finally:frame.close()
    return dict(shape=list(shape),frontend_order=list(ORDERS[order]),wrt=list(wrt),causal=causal,
        native_frontend_argument_indices=list(program.input_indices),
        native_gradient_result_indices=list(program.active),compile_ms=compile_ms,
        max_abs_gradient_error=max(errors),
        capture_wall_samples_ms=captures,capture_wall_median_ms=statistics.median(captures),
        backward_wall_samples_ms=backwards,backward_wall_median_ms=statistics.median(backwards),
        forward_schedule=program.pair.forward.descriptor.provenance["schedule_digest"],
        backward_schedule=program.pair.backward.descriptor.provenance["schedule_digest"],
        backward_abi=program.pair.backward.descriptor.abi_id,
        forward_image_sha256=hashlib.sha256(program.pair.forward.image.payload).hexdigest(),
        backward_image_sha256=hashlib.sha256(program.pair.backward.image.payload).hexdigest(),
        route="public frontend -> native paired AD argument role map -> replay-bound Schedule/Tile -> NVIDIA Target/PTX -> checked resident capture")


def record(samples):
    if not nvidia_cuda_host_ready():raise RuntimeError("owning SM120 host required")
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or gpu.split(",")[-1].strip()!="12.0":
        raise RuntimeError("requires one exact SM120 GPU")
    rows=[]
    for shape in ((1,2,1,3,5,4,3),(2,4,2,16,11,8,6)):
        for causal in (False,True):
            for order in ("canonical","k_bias_v_q","bias_v_q_k"):
                for wrt in (("bias","q"),("v","bias","k","q")):
                    rows.append(run_case(shape,order,wrt,causal,samples))
            rows.append(run_case(shape,"v_q_k",("v","q"),causal,samples))
        print("verified shape",shape,flush=True)
    root=Path(__file__).resolve().parents[2]
    sources=("src/transforms/lib/AutodiffPairedPass.cpp",
        "src/compiler/programming_model/lib/NativeCheckpoint.h",
        "python/tessera/compiler/scheduled_checkpoint.py","python/tessera/compiler/nvidia_native.py",
        "python/tessera/compiler/native_attention_program.py","python/tessera/compiler/resident_attention.py",
        "benchmarks/nvidia/benchmark_attention_argument_order.py",
        "benchmarks/nvidia/benchmark_checkpoint_bias_gradient.py")
    return dict(schema="tessera.nvidia.attention_argument_order.v1",architecture="sm_120",gpu=gpu,
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
        rows=rows,samples=samples,
        timing_scope="capture/backward wall includes allocation, module loading/copies and synchronization; no device-time or speedup claim; separate device windows remain in the checkpoint packet")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=3);args=parser.parse_args()
    if args.samples<3:raise ValueError("requires >=3 samples")
    result=record(args.samples);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("verified",len(result["rows"]),"frontend argument-order rows")
