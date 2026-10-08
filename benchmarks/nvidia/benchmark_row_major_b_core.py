"""Native Tile physical RHS gather probe; Schedule/package integration remains open."""
from __future__ import annotations
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import numpy as np
from tessera import runtime as rt
from tessera.compiler import nvidia_native as native
from tessera.compiler.scheduled_matmul import lower_scheduled_matmul
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.unit.test_scheduled_matmul_consumers import _module


def tile_probe(shape, dtype, pitch, *, bounded=True, transpose=True):
    m,k,n=shape
    base=lower_scheduled_matmul(_module(target="nvidia_sm120",shape=shape,dtype=dtype),target="nvidia_sm120")
    entry=f"nvidia_sm120_scheduled_matmul_row_b_core_{dtype}_{m}_{k}_{n}_{pitch}_{int(bounded)}"
    text=base.tile_ir.replace("@"+base.function_name+"(", "@"+entry+"(",1)
    old=f'<[{k}, {n}], [1, {k}],'
    assert text.count(old)==1
    text=text.replace(old,f'<[{k}, {n}], [{pitch}, 1],',1)
    old=f'order = "col_major", leading_dim = {k}>'
    assert text.count(old)==1
    text=text.replace(old,f'order = "row_major", leading_dim = {pitch}>',1)
    if transpose:
        assert text.count('role = "b"}')==1
        text=text.replace('role = "b"}','role = "b", transpose}',1)
    if not bounded:
        lines=text.splitlines()
        for i,line in enumerate(lines):
            if 'tile.view %arg1,' in line:
                if ', %arg5, %arg4 {' not in line:
                    assert '(!llvm.ptr, i64, i64, i64)' in line
                line=line.replace(', %arg5, %arg4 {',' {',1)
                line=line.replace('(!llvm.ptr, i64, i64, i64, i64, i64)','(!llvm.ptr, i64, i64, i64)',1)
                lines[i]=line
        text="\n".join(lines)+"\n"
    # This is a raw physical experiment, not a replay-certified Schedule arm.
    text=re.sub(r', tessera.schedule_hash = "[0-9a-f]+"','',text)
    return entry,text


def run_case(shape,dtype,pad,bounded):
    m,k,n=shape; pitch=n+pad
    if not bounded and (m%16 or k%16 or n%8):
        raise ValueError("unbounded probe requires complete fragments")
    entry,tile=tile_probe(shape,dtype,pitch,bounded=bounded)
    target,ptx,*_=native._compile_tile_ir(tile,entry)
    lib=rt._load_nvidia_ptx_launch()
    assert lib is not None and rt._register_nvidia_ptx(lib,entry,ptx)==0
    storage=np.float16
    if dtype=="bf16":
        import ml_dtypes
        storage=ml_dtypes.bfloat16
    rng=np.random.default_rng(120_410)
    a=(rng.normal(size=(m,k))*.2).astype(storage)
    b=np.full((k,pitch),13,storage)
    b[:,:n]=(rng.normal(size=(k,n))*.2).astype(storage)
    expected=a.astype(np.float32)@b[:,:n].astype(np.float32)
    with NvidiaDeviceSession() as session:
        av,bv,out=[session.upload(v) for v in (a,b,np.full((m,n),np.nan,np.float32))]
        pointers=(ctypes.c_void_p*3)(av.ptr,bv.ptr,out.ptr)
        dims=(ctypes.c_int64*3)(m,n,k)
        rc=lib.tessera_nvidia_ptx_invoke_resident(entry.encode(),pointers,3,dims,3,ctypes.c_void_p(session.stream))
        assert rc==0,rc
        session.synchronize()
        actual=session.download(out)
        np.testing.assert_allclose(actual,expected,rtol=4e-5,atol=4e-5)
        samples=[]
        for _ in range(3):
            latency=ctypes.c_float()
            rc=lib.tessera_nvidia_ptx_benchmark_resident(entry.encode(),pointers,3,dims,3,
                ctypes.c_void_p(session.stream),20,100,ctypes.byref(latency))
            assert rc==0,rc
            samples.append(latency.value)
        session.synchronize()
        np.testing.assert_allclose(session.download(out),expected,rtol=4e-5,atol=4e-5)
    return dict(shape_mkn=list(shape),dtype=dtype,pitch=pitch,bounded=bounded,
        max_abs_error=float(np.max(np.abs(actual-expected))),
        device_dispatch_window_samples_ms=samples,device_dispatch_window_median_ms=statistics.median(samples),
        tile_sha256=hashlib.sha256(tile.encode()).hexdigest(),
        target_sha256=hashlib.sha256(target.encode()).hexdigest(),
        ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest())


def main():
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,driver_version,compute_cap",
        "--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or gpu.split(",")[-1].strip()!="12.0":
        raise RuntimeError("requires one exact SM120 GPU")
    rows=[run_case(shape,dtype,pad,bounded) for dtype in ("fp16","bf16")
        for shape,bounded in (((16,16,8),False),((16,64,16),False),((17,35,19),True),((33,67,13),True))
        for pad in (0,5)]
    packet=dict(schema="tessera.nvidia.row_major_b_core.v1",gpu=gpu,rows=rows,
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ("src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
             "benchmarks/nvidia/benchmark_row_major_b_core.py")},
        production_package_support=False,
        scope="raw Tile typed-fragment gather; native Schedule/profile/checked package and tensor RHS edge remain open")
    dest=Path(os.environ["TESSERA_ROW_MAJOR_B_PACKET"])
    dest.write_text(json.dumps(packet,indent=2,sort_keys=True)+"\n")
    print("verified",len(rows),"row-major RHS physical gather cases")


if __name__=="__main__":
    main()
