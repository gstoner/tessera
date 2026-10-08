"""Public biased attention JIT reverse AD with private resident checkpoint proof."""
from __future__ import annotations
import argparse, ctypes, hashlib, json, os, statistics, subprocess, time
from pathlib import Path
import numpy as np
import tessera as ts
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests._support.nvidia import nvidia_cuda_host_ready
from benchmarks.nvidia.benchmark_checkpoint_bias_gradient import reference
from benchmarks.nvidia.benchmark_jit_attention_vjp import download


def function(wrt, causal):
    if causal:
        @ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=wrt)
        def attention(q,k,v,bias):
            return ts.ops.flash_attn(q,k,v,attn_bias=bias,causal=True)
    else:
        @ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=wrt)
        def attention(q,k,v,bias):
            return ts.ops.flash_attn(q,k,v,attn_bias=bias,causal=False)
    return attention


def run_case(shape, causal, wrt, samples=3):
    b,hq,hkv,sq,sk,d,dv=shape
    rng=np.random.default_rng(120_405)
    values=[(rng.normal(size=s)*.3).astype(np.float32) for s in
        ((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),(b,hq,sq,sk),(b,hq,sq,dv))]
    expected,lse,grad=reference(*values,causal)
    start=time.perf_counter_ns()
    program=function(wrt,causal).compile_native_attention_vjp(
        *values[:4],compiler=Path(os.environ["TESSERA_OPT"]))
    compile_ms=(time.perf_counter_ns()-start)/1e6
    capture_times=[];backward_times=[];errors=[]
    with NvidiaDeviceSession() as session:
        resident=[session.upload(x) for x in values]
        start=time.perf_counter_ns();frame=program.capture(*resident[:4])
        capture_times.append((time.perf_counter_ns()-start)/1e6)
        try:
            np.testing.assert_allclose(download(session,frame.primal),expected,atol=4e-5,rtol=4e-5)
            for value in resident[:4]:
                changed=np.full(value.shape,19,np.float32)
                status=session.lib.tessera_nvidia_device_upload(
                    ctypes.c_void_p(value.ptr),ctypes.c_void_p(changed.ctypes.data),
                    changed.nbytes,ctypes.c_void_p(session.stream))
                if status:raise RuntimeError("CUDA caller mutation failed")
            session.synchronize()
            for _ in range(samples):
                start=time.perf_counter_ns();actual=frame.backward(resident[4])
                backward_times.append((time.perf_counter_ns()-start)/1e6)
                for result,i in zip(actual,program.active,strict=True):
                    host=download(session,result)
                    np.testing.assert_allclose(host,grad[i],atol=4e-5,rtol=4e-5)
                    errors.append(float(np.max(np.abs(host-grad[i]))))
            # A second cotangent must reuse the same captured generation.
            changed_seed=(rng.normal(size=values[4].shape)*.4).astype(np.float32)
            seed2=session.upload(changed_seed)
            _,_,grad2=reference(*values[:4],changed_seed,causal)
            second=frame.backward(seed2)
            for result,i in zip(second,program.active,strict=True):
                host=download(session,result)
                np.testing.assert_allclose(host,grad2[i],atol=4e-5,rtol=4e-5)
                errors.append(float(np.max(np.abs(host-grad2[i]))))
        finally:
            frame.close()
        try:
            frame.backward(resident[4])
        except ValueError:
            pass
        else:
            raise AssertionError("closed frame accepted backward")
        fresh=[session.upload(x) for x in values[:4]]
        for _ in range(samples-1):
            start=time.perf_counter_ns();again=program.capture(*fresh)
            capture_times.append((time.perf_counter_ns()-start)/1e6)
            try:
                np.testing.assert_allclose(download(session,again.primal),expected,atol=4e-5,rtol=4e-5)
            finally:
                again.close()
    return dict(shape=list(shape),causal=causal,wrt=list(wrt),active=list(program.active),
        compile_ms=compile_ms,capture_wall_samples_ms=capture_times,
        capture_wall_median_ms=statistics.median(capture_times),
        backward_wall_samples_ms=backward_times,backward_wall_median_ms=statistics.median(backward_times),
        max_abs_gradient_error=max(errors),forward_abi=program.pair.forward.descriptor.abi_id,
        backward_abi=program.pair.backward.descriptor.abi_id,
        forward_image_sha256=hashlib.sha256(program.pair.forward.image.payload).hexdigest(),
        backward_image_sha256=hashlib.sha256(program.pair.backward.image.payload).hexdigest(),
        checkpoint_identity=program.pair.contract_digest,
        ownership="private Q/K/V/bias/O/LSE generation survives caller mutation, repeated and changed cotangents; closed frame rejects backward",
        route="public JIT trace -> paired native AD -> checkpoint Graph/Schedule/Tile -> NVIDIA Target/NVVM/PTX -> checked resident tape ABI")


def record(samples):
    if not nvidia_cuda_host_ready():raise RuntimeError("owning SM120 host required")
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or gpu.split(",")[-1].strip()!="12.0":
        raise RuntimeError("requires one exact SM120 GPU")
    rows=[run_case(shape,causal,wrt,samples) for shape in
        ((1,2,1,3,5,4,3),(1,2,1,5,3,4,3),(2,4,2,16,19,8,6))
        for causal in (False,True)
        for wrt in (("bias",),("bias","k"),("v","bias","q","k"))]
    root=Path(__file__).resolve().parents[2]
    sources=("src/compiler/ir/AttentionADContract.h","src/compiler/ir/AdjointInterface.cpp",
        "src/transforms/lib/AutodiffPairedPass.cpp","python/tessera/compiler/graph_ir.py",
        "python/tessera/compiler/scheduled_checkpoint.py","python/tessera/compiler/nvidia_native.py",
        "python/tessera/compiler/native_attention_program.py","python/tessera/compiler/resident_attention.py",
        "benchmarks/nvidia/benchmark_jit_attention_bias_vjp.py",
        "benchmarks/nvidia/benchmark_checkpoint_bias_gradient.py")
    return dict(schema="tessera.nvidia.jit_attention_bias_vjp.v1",architecture="sm_120",gpu=gpu,
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
        samples=samples,rows=rows,
        timing_scope="capture wall includes private copies, module loading, forward and synchronization; backward wall includes gradient allocation and synchronization; selected wrt computes all four native gradients; device windows remain in the explicit checkpoint packet; no speedup claim")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=3);args=parser.parse_args()
    if args.samples<3:raise ValueError("requires >=3 samples")
    result=record(args.samples);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("verified",len(result["rows"]),"JIT bias-gradient rows")
