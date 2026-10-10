#!/usr/bin/env python3
"""Correctness-gated ordinary NVFP4 JIT characterization on SM120."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.nvfp4_tensor import NVFP4Tensor
from tests.device.nvidia.test_nvfp4_tensor_jit import rank_two, shared, independent
from tests.device.nvidia.test_e2e_spine_native import _pack_nvfp4, _decode_e2m1, _decode_ue4m3

ROOT=Path(__file__).resolve().parents[2]

def record(mode, rows, n, k, samples, use_vmap=False, typed=False):
    batch=3
    ashape=(rows,k) if mode=="rank_two" else (batch,rows,k)
    bshape=(batch,k,n) if mode=="independent" else (k,n)
    sk=k//16+(k%16!=0)
    rng=np.random.default_rng(120613+k)
    ac=rng.integers(0,16,ashape,dtype=np.uint8)
    bc=rng.integers(0,16,bshape,dtype=np.uint8)
    choices=np.array([0x30,0x33,0x35,0x38,0x3a,0x40],np.uint8)
    sa=np.ascontiguousarray(choices[rng.integers(0,len(choices),(*ashape[:-1],sk))])
    sbshape=(batch,sk,n) if mode=="independent" else (sk,n)
    sb=np.ascontiguousarray(choices[rng.integers(0,len(choices),sbshape)])
    a=NVFP4Tensor(_pack_nvfp4(ac,len(ashape)-1),ashape,len(ashape)-1)
    b=NVFP4Tensor(_pack_nvfp4(bc,len(bshape)-2),bshape,len(bshape)-2)
    decoded_a=_decode_e2m1(ac)*np.repeat(_decode_ue4m3(sa),16,axis=len(ashape)-1)[..., :k]
    rhs_scale=np.repeat(_decode_ue4m3(sb),16,axis=len(bshape)-2)
    rhs_scale=rhs_scale[:, :k, :] if mode=="independent" else rhs_scale[:k,:]
    oracle=decoded_a.astype(np.float64)@(_decode_e2m1(bc)*rhs_scale).astype(np.float64)
    call=ts.jit({"rank_two":rank_two,"shared":shared,"independent":independent}[mode],target="nvidia_sm120")
    if use_vmap:
        from tessera.autodiff import vmap
        from tests.unit.test_native_nvfp4_vmap import typed_product
        call = vmap(ts.jit(typed_product if typed else rank_two, target="nvidia_sm120"),
                    in_axes=(0, None, 0, None) if mode == "shared" else 0)
    start=time.perf_counter()
    result=call(a,b,sa,sb)
    cold_ms=(time.perf_counter()-start)*1e3
    np.testing.assert_allclose(result,oracle,rtol=0,atol=2e-3)
    error=float(np.max(np.abs(result.astype(np.float64)-oracle)))
    artifact=call._cached_artifact
    descriptor=artifact.launch_descriptor
    image=artifact.native_image
    assert descriptor is not None and image is not None
    assert call._native_descriptor_last_receipt["execution_kind"]=="native_gpu"
    wall=[]
    for _ in range(samples):
        start=time.perf_counter()
        result=call(a,b,sa,sb)
        wall.append((time.perf_counter()-start)*1e3)
        np.testing.assert_allclose(result,oracle,rtol=0,atol=2e-3)
    arrays=(a.storage,b.storage,sa,sb)
    inputs=sorted((item for item in descriptor.buffers if item.direction=="input"),key=lambda item:item.ordinal)
    output=next(item for item in descriptor.buffers if item.direction=="output")
    buffers=dict(zip((item.name for item in inputs),arrays,strict=True))
    buffers[output.name]=np.empty_like(result)
    dims=list(descriptor.provenance["shape"])
    if mode=="independent":
        dims.extend((rows,batch))
    scalars=dict(zip((item.name for item in sorted(descriptor.scalars,key=lambda item:item.ordinal)),dims,strict=True))
    arguments={"buffers":buffers,"scalars":scalars}
    events=[rt._nvidia_native_descriptor_device_latency(image,descriptor,arguments,reps=100,warmup=20) for _ in range(samples)]
    receipt=rt.launch(artifact,arguments)
    assert receipt["ok"] and receipt["execution_kind"]=="native_gpu"
    np.testing.assert_allclose(buffers[output.name],oracle,rtol=0,atol=2e-3)
    return {"mode":mode,"frontend_transform":"vmap" if use_vmap else "explicit_batching","symbolic_annotations":typed,"logical_a_shape":ashape,"logical_b_shape":bshape,
        "physical_a_shape":a.storage.shape,"physical_b_shape":b.storage.shape,
        "correctness":"passed_before_timing_after_each_public_sample_and_final_portable_launch",
        "maximum_absolute_error":error,"cold_public_ms":cold_ms,
        "warm_public_ms":wall,"warm_public_median_ms":statistics.median(wall),
        "resident_device_event_ms":events,"resident_device_event_median_ms":statistics.median(events),
        "image_sha256":hashlib.sha256(image.payload).hexdigest(),
        "entry_symbol":descriptor.entry_symbol,"abi_id":descriptor.abi_id,
        "schedule_digest":descriptor.provenance["schedule_digest"],
        "tile_ir_digest":descriptor.provenance["tile_ir_digest"]}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=7)
    parser.add_argument("--vmap", action="store_true", help="record public native batch transformation")
    parser.add_argument("--typed", action="store_true", help="exercise symbolic scalar annotations with native vmap")
    args=parser.parse_args()
    if args.typed and not args.vmap:
        parser.error("--typed requires --vmap")
    if args.samples<3:
        parser.error("at least three samples required")
    device=subprocess.check_output(["nvidia-smi","--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(device.splitlines())!=1 or not device.endswith("12.0"):
        raise RuntimeError("recorder requires one visible owning SM120 device")
    source_paths=("python/tessera/compiler/graph_ir.py","python/tessera/compiler/constraints.py","tests/unit/test_native_nvfp4_vmap.py","python/tessera/compiler/native_vmap.py","python/tessera/autodiff/transforms.py","python/tessera/compiler/nvfp4_tensor.py","python/tessera/compiler/trace.py",
        "python/tessera/compiler/jit.py","python/tessera/compiler/nvidia_native.py",
         "python/tessera/compiler/backend_manifest.py","src/compiler/programming_model/lib/PMPasses.cpp",
        "benchmarks/nvidia/record_nvfp4_tensor_jit.py","tests/device/nvidia/test_nvfp4_tensor_jit.py",
        "tests/device/nvidia/test_e2e_spine_native.py")
    source={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_paths}
    import os
    tool_paths=(os.environ["TESSERA_OPT"],os.environ["TESSERA_NVIDIA_OPT"],os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"])
    packet={"schema":"tessera.nvfp4.logical_host_jit.v1","work_item":"W1.1",
        "sync_key":"NVIDIA-NVFP4-NATIVE-VMAP-2026-10-06" if args.vmap else "NVIDIA-NVFP4-LOGICAL-JIT-2026-10-06","architecture":"sm_120a","device":device,
        "source_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "source_worktree_dirty":bool(subprocess.check_output(["git","status","--porcelain","-uno"],cwd=ROOT,text=True).strip()),
        "source_sha256":source,"tool_sha256":{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in tool_paths},
        "route":"Python frontend -> typed Graph MLIR -> Schedule -> Tile -> NVIDIA Target -> PTX -> checked ABI",
        "selector_changed":False,
        "timing_domains":{"cold_public":"trace/compile/package/checked host-array launch wall",
            "warm_public":"cached Graph/package validation, host output allocation, copies and synchronized launch wall",
            "device_event":"resident 100-launch native CUDA-event window; host upload/readback excluded; dispatch gaps included"},
        "rows":[record(mode,m,n,k,args.samples,args.vmap,args.typed) for mode in (("shared","independent") if args.vmap else ("rank_two","shared","independent")) for m,n,k in ((7,5,31),(17,19,129))]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2)+"\n")

if __name__=="__main__":
    main()
