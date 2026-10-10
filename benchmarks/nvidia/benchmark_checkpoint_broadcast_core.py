"""Native broadcast-checkpoint arithmetic gate; checked package ABI is not yet integrated."""
from __future__ import annotations
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import numpy as np
from tessera import runtime as rt
from tessera.compiler import nvidia_native as native
from tessera.compiler.scheduled_checkpoint import lower_scheduled_checkpoint
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from benchmarks.nvidia.benchmark_checkpoint_bias_gradient import reference


def run_case(physical, causal):
    dims = (2,4,2,5,7,4,3)
    b,hq,hkv,sq,sk,d,dv = dims
    rng = np.random.default_rng(120_409)
    q,k,v,bias,seed = [(rng.normal(size=shape)*.3).astype(np.float32) for shape in
        ((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv),physical,(b,hq,sq,dv))]
    output,lse,grads = reference(q,k,v,bias,seed,causal)
    axes = tuple(i for i,(p,n) in enumerate(zip(physical,(b,hq,sq,sk),strict=True)) if p==1 and n!=1)
    expected = (*grads[:3],grads[3].sum(axis=axes,keepdims=True))
    f=lower_scheduled_checkpoint(("q","k","v","bias","output","lse"),dims,float(1/np.sqrt(d)),causal,
        bias=True,bias_shape=physical)
    g=lower_scheduled_checkpoint(("do","q","k","v","output","bias","lse","dq","dk","dv","dbias"),
        dims,float(1/np.sqrt(d)),causal,backward=True,bias=True,bias_gradient=True,bias_shape=physical)
    lib = rt._load_nvidia_ptx_launch()
    assert lib is not None
    compiled=[]
    for artifact in (f,g):
        target,ptx,*_=native._compile_tile_ir(artifact.tile_ir,artifact.entry)
        assert rt._register_nvidia_ptx(lib,artifact.entry,ptx)==0
        compiled.append(dict(entry=artifact.entry,schedule_hash=artifact.schedule_digest,
            graph_sha256=hashlib.sha256(artifact.graph_ir.encode()).hexdigest(),
            schedule_sha256=hashlib.sha256(artifact.schedule_ir.encode()).hexdigest(),
            tile_sha256=hashlib.sha256(artifact.tile_ir.encode()).hexdigest(),
            target_sha256=hashlib.sha256(target.encode()).hexdigest(),
            ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest()))
    with NvidiaDeviceSession() as session:
        # Guard words follow the actual physical result, not the dense score count.
        bias_count=int(np.prod(physical))
        host=[q,k,v,bias,np.full(output.shape,np.nan,np.float32),
              np.full(lse.shape,np.nan,np.float32)]
        resident=[session.upload(x) for x in host]
        gradient=[session.upload(np.full(x.shape,np.nan,np.float32)) for x in expected[:3]]
        guard=session.upload(np.full((bias_count+32,),np.nan,np.float32))
        backward=[session.upload(seed),*resident[:3],resident[4],resident[3],resident[5],*gradient,guard]
        dim_array=(ctypes.c_int64*7)(*dims)

        def pointers(values):
            return (ctypes.c_void_p*len(values))(*[
                int(x.__cuda_array_interface__["data"][0]) for x in values])

        def launch(artifact,values):
            rc=lib.tessera_nvidia_ptx_invoke_resident(artifact.entry.encode(),pointers(values),
                len(values),dim_array,7,ctypes.c_void_p(session.stream))
            assert rc==0,rc
            session.synchronize()

        launch(f,resident)
        np.testing.assert_allclose(session.download(resident[4]),output,atol=4e-5,rtol=4e-5)
        np.testing.assert_allclose(session.download(resident[5]),lse,atol=4e-5,rtol=4e-5)
        launch(g,backward)

        def check():
            actual=[session.download(x) for x in gradient]
            tail=session.download(guard)
            assert np.isnan(tail[bias_count:]).all(), "native reduction wrote outside physical bias capacity"
            actual.append(tail[:bias_count].reshape(physical))
            for x,y in zip(actual,expected,strict=True):
                np.testing.assert_allclose(x,y,atol=4e-5,rtol=4e-5)
            return [float(np.max(np.abs(x-y))) for x,y in zip(actual,expected,strict=True)]

        errors=check()
        timings=[]
        for _ in range(5):
            latency=ctypes.c_float()
            rc=lib.tessera_nvidia_ptx_benchmark_resident(g.entry.encode(),pointers(backward),
                len(backward),dim_array,7,ctypes.c_void_p(session.stream),20,100,ctypes.byref(latency))
            assert rc==0,rc
            timings.append(latency.value)
        session.synchronize()
        check()
    return dict(logical_dims=list(dims),physical_bias_shape=list(physical),causal=causal,
        reduction_axes=list(axes),max_abs_gradient_errors=errors,guard_words_untouched=32,
        device_dispatch_window_samples_ms=timings,device_dispatch_window_median_ms=statistics.median(timings),
        compiled=compiled,scope="raw resident native arithmetic proof; no checked broadcast package/tape ABI")


def main():
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or gpu.split(",")[-1].strip()!="12.0":
        raise RuntimeError("exact SM120 device required")
    shapes=[(1,4,5,7),(2,1,5,7),(2,4,1,7),(2,4,5,1),(1,1,1,7),(1,1,1,1)]
    root=Path(__file__).resolve().parents[2]
    sources=["src/compiler/ir/AttentionADContract.h",
        "src/compiler/tile_opt_fa4/lib/Dialect/Attn/AttnOps.cpp",
        "src/compiler/programming_model/lib/NativeCheckpoint.h",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
        "python/tessera/compiler/scheduled_checkpoint.py",
        "benchmarks/nvidia/benchmark_checkpoint_broadcast_core.py"]
    packet=dict(schema="tessera.nvidia.broadcast_checkpoint_core.v1",gpu=gpu,
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in sources},
        rows=[run_case(shape,causal) for shape in shapes for causal in (False,True)],
        production_package_support=False)
    output=Path(os.environ["TESSERA_BROADCAST_CORE_PACKET"])
    output.write_text(json.dumps(packet,indent=2)+"\n")
    print(f"wrote {output}")


if __name__=="__main__":
    main()
