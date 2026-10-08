"""Traced bias JVP, independent fp64 derivative and resident SM120 timings."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import tessera as ts
from benchmarks.record_device_ring_protocol import Device
from benchmarks.record_jit_attention_program import timed_native_product
from tessera.compiler.native_attention_program import NativeAttentionJVPProgram

def attention(bias,v,q,k):
    return ts.ops.flash_attn(q,k,v,attn_bias=bias,causal=False)

def causal_attention(bias,v,q,k):
    return ts.ops.flash_attn(q,k,v,attn_bias=bias,causal=True)

def reference(values,directions,causal):
    q,k,v=(values[n].astype(np.float64) for n in ("q","k","v"))
    dq,dk,dv=(directions[n].astype(np.float64) for n in ("q","k","v"))
    group=q.shape[1]//k.shape[1]
    k,dk,v,dv=(np.repeat(x,group,axis=1) for x in (k,dk,v,dv))
    scale=1/np.sqrt(q.shape[-1])
    score=scale*(q@k.swapaxes(-1,-2))+values["bias"].astype(np.float64)
    direction=scale*(dq@k.swapaxes(-1,-2)+q@dk.swapaxes(-1,-2))+directions["bias"]
    if causal:
        sq,sk=q.shape[-2],k.shape[-2]
        legal=np.arange(sk)[None,:]<=np.arange(sq)[:,None]+max(sk-sq,0)
        score=np.where(legal,score,-np.inf)
    p=np.exp(score-score.max(axis=-1,keepdims=True));p/=p.sum(axis=-1,keepdims=True)
    dp=p*(direction-(p*direction).sum(axis=-1,keepdims=True))
    return p@v,dp@v+p@dv

def run(device,sk,causal,bias_shape,wrt,artifacts):
    rng=np.random.default_rng(917)
    shapes={"q":(2,4,3,4),"k":(2,2,sk,4),"v":(2,2,sk,3),"bias":bias_shape}
    values={n:rng.normal(size=s).astype(np.float32)*.2 for n,s in shapes.items()}
    directions={n:rng.normal(size=s).astype(np.float32)*.1 if n in wrt else np.zeros(s,np.float32)
                for n,s in shapes.items()}
    primal,expected=reference(values,directions,causal)
    step=1e-4
    plus={n:values[n].astype(np.float64)+step*directions[n] for n in values}
    minus={n:values[n].astype(np.float64)-step*directions[n] for n in values}
    fd=(reference(plus,directions,causal)[0]-reference(minus,directions,causal)[0])/(2*step)
    np.testing.assert_allclose(fd,expected,atol=2e-9,rtol=2e-7)
    fn=ts.jit(target="nvidia_sm120",autodiff="forward",wrt=wrt)(
        causal_attention if causal else attention)
    begin=time.perf_counter()
    program=fn.compile_native_attention_jvp(**values,compiler=Path(os.environ["TESSERA_OPT"]),
        llvm_bin=Path("/usr/lib/llvm-23/bin"))
    compile_ms=(time.perf_counter()-begin)*1e3
    encoded=program.to_json()
    program=NativeAttentionJVPProgram.from_json(encoded,expected_digest=program.program_digest)
    assert program.to_json()==encoded
    pointers=[]
    def upload(x):
        x=np.ascontiguousarray(x,np.float32)
        ptr=ct.c_void_p();device.check(device.alloc(ct.byref(ptr),x.nbytes));pointers.append(ptr)
        device.check(device.htod(ptr,x.ctypes.data,x.nbytes))
        return SimpleNamespace(__cuda_array_interface__={
            "version":3,"shape":x.shape,"typestr":x.dtype.str,"data":(ptr.value,False)})
    def download(x):
        spec=x.__cuda_array_interface__;out=np.empty(spec["shape"],np.float32)
        device.check(device.dtoh(out.ctypes.data,ct.c_void_p(spec["data"][0]),out.nbytes))
        return out
    try:
        resident={n:upload(x) for n,x in values.items()}
        tangents={n:upload(directions[n]) for n in wrt}
        begin=time.perf_counter()
        frame=program.capture(**resident)
        capture_ms=(time.perf_counter()-begin)*1e3
        with frame:
            np.testing.assert_allclose(download(frame.primal),primal,atol=3e-5,rtol=3e-5)
            result=frame.jvp(*(tangents[n] for n in wrt))
            actual=download(result)
            np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=3e-5)
            # Captured source/bias generations must survive caller mutation.
            clear=device.driver.cuMemsetD32_v2 if hasattr(device,"driver") else ct.CDLL("libcuda.so.1").cuMemsetD32_v2
            clear.argtypes=[ct.c_uint64,ct.c_uint,ct.c_size_t];clear.restype=ct.c_int
            for n,value in resident.items():
                device.check(clear(value.__cuda_array_interface__["data"][0],0,values[n].size))
            device.check(device.sync())
            def forbidden(*a,**kw):
                raise AssertionError("compiler subprocess during resident replay")
            walls=[]
            with patch("subprocess.run",forbidden),patch("subprocess.Popen",forbidden),patch("subprocess.check_output",forbidden):
                for multiplier in (2.,1.,-1.):
                    inputs=[upload(directions[n]*multiplier) for n in wrt]
                    begin=time.perf_counter()
                    repeat=frame.jvp(*inputs)
                    walls.append((time.perf_counter()-begin)*1e3)
                    np.testing.assert_allclose(download(repeat),multiplier*expected,atol=3e-5,rtol=3e-5)
                    np.testing.assert_array_equal(download(result),actual)
            native=frame._frame
            slots={**frame._zeros,**dict(zip(program.active,(tangents[n] for n in wrt),strict=True))}
            raw=[x.pointer.value for x in native._saved]
            raw.extend(slots[i].__cuda_array_interface__["data"][0] for i in range(3))
            raw.extend((native._bias.pointer.value,slots[3].__cuda_array_interface__["data"][0],
                        result.__cuda_array_interface__["data"][0],128))
            timing=timed_native_product(device,program.tangent,raw,24)
            np.testing.assert_allclose(download(result),expected,atol=3e-5,rtol=3e-5)
        try:frame.jvp(*(tangents[n] for n in wrt))
        except ValueError:pass
        else:raise AssertionError("closed frame executed")
        label=f"k{sk}_c{int(causal)}_"+"x".join(map(str,bias_shape))+"_"+"_".join(wrt)
        (artifacts/(label+".program.json")).write_text(encoded)
        (artifacts/(label+".tile.mlir")).write_text(program.tangent.arena_ir)
        np.savez(artifacts/(label+".npz"),**{f"primal_{n}":x for n,x in values.items()},
                 **{f"direction_{n}":x for n,x in directions.items()},expected=expected,primal=primal)
        return dict(case=label,bias_shape=bias_shape,wrt=wrt,active=program.active,
            max_abs_error=float(np.max(np.abs(actual-expected))),program_digest=program.program_digest,
            compile_ms=compile_ms,capture_wall_ms=capture_ms,jvp_wall_samples_ms=walls,
            jvp_wall_median_ms=statistics.median(walls),
            private_generation_repeated_directions_and_close="passed",
            correctness="independent_fp64_analytic_JVP_and_central_difference",**timing)
    finally:
        for ptr in pointers:device.check(device.free(ptr))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--limit",type=int);args=ap.parse_args()
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,driver_version","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or "RTX 5070" not in gpu or gpu.split(",")[2].strip()!="12.0":
        raise RuntimeError("owning RTX5070 / SM120 required")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    artifacts=args.output.parent/"artifacts";artifacts.mkdir(exist_ok=True)
    cases=[(sk,causal,bias,wrt) for sk in (5,129) for causal in (False,True)
           for bias in ((1,4,1,1),(1,4,1,sk),(2,4,3,sk))
           for wrt in (("bias",),("v",),("bias","v","k","q"))]
    if args.limit is not None:cases=cases[:args.limit]
    rows=[];device=Device("nvidia")
    for case in cases:
        rows.append(run(device,*case,artifacts));print("verified",rows[-1]["case"],flush=True)
    sources=("python/tessera/compiler/native_attention_program.py","python/tessera/compiler/native_attention_jvp.py",
        "python/tessera/compiler/native_attention_jvp_artifact.py","python/tessera/compiler/resident_attention.py",
        "src/compiler/programming_model/lib/NativeAttentionJvp.h","src/transforms/lib/AutodiffForwardPass.cpp",
        "src/compiler/ir/TangentInterface.cpp","benchmarks/nvidia/benchmark_bias_attention_jvp.py")
    args.output.write_text(json.dumps(dict(device=gpu,architecture="sm_120",rows=rows,
        compiler_digest=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        fingerprints={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in sources},
        route="public compile/capture -> native AD Graph -> Schedule -> Tile -> LLVM/NVVM -> CUDA",
        timing_scope="compile and allocating capture/JVP wall; preloaded tangent CUDA-event dispatch window; no performance promotion"),
        indent=2)+"\n")
if __name__=="__main__":main()
