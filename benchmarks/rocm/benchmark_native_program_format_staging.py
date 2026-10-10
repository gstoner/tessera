"""Diagnostic native staging A/B for existing FP8/MXFP8/folded-MXFP4 images.

This marshals a one-member test plan from an existing package. It does not
claim a compiler-projected primal/AD program for these three formats.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
import subprocess
from pathlib import Path
from statistics import median
import time
import ml_dtypes
import numpy as np
from tessera import runtime as rt
from tessera.compiler.native_scaled_program import _Buffer, _Step
from tessera.compiler.rocm_fp8_blockscale import BlockScaleShape, compile_blockscale
from tessera.compiler.rocm_mxfp8_blockscale import compile_mxfp8
from tessera.compiler.rocm_mxfp4_folded_frontend import compile_folded_scaled_matmul
from tessera.compiler.rocm_mxfp4_folded import FoldedPrefillSchedule, folded_prefill_grid
from benchmarks.rocm.benchmark_gfx1201_three_formats import quantize_fp8, quantize_folded, verify

def check(code):
    if code:raise RuntimeError("native diagnostic program status "+str(code))

class DiagnosticOwner:
    def __init__(self,lib,package,inputs,output,grid,block,shape):
        self.lib=lib;self.handle=C.c_uint64();self.inputs=inputs;self.output=output
        arrays=inputs+[output]
        contract=(_Buffer*len(arrays))(*[
            _Buffer(a.nbytes,a.size,-1 if i<4 else 0,0 if i<4 else 1,0 if i<4 else 2,0)
            for i,a in enumerate(arrays)])
        self.blob=C.create_string_buffer(package.image.payload)
        step=_Step(C.cast(self.blob,C.c_void_p),len(package.image.payload),
            package.descriptor.entry_symbol.encode(),4,3,(C.c_uint32*6)(0,1,2,3),4,
            (C.c_uint32*6)(*grid,*block),(C.c_int64*8)(*shape))
        status=lib.tessera_rocm_program_prepare(b"gfx1201",4,5,contract,1,C.byref(step),
                self.pointers(inputs),self.sizes(inputs),C.byref(self.handle))
        if status:
            if self.handle.value:check(lib.tessera_rocm_program_close(self.handle))
            check(status)
    @staticmethod
    def pointers(values):return (C.c_void_p*4)(*(a.ctypes.data for a in values))
    @staticmethod
    def sizes(values):return (C.c_uint64*4)(*(a.nbytes for a in values))
    def update(self,values):
        check(self.lib.tessera_rocm_program_update(self.handle,self.pointers(values),self.sizes(values)))
    def invoke(self,repeats=1,timed=False):
        generation=C.c_uint64();elapsed=C.c_float()
        check(self.lib.tessera_rocm_program_invoke(self.handle,repeats,C.byref(generation),C.byref(elapsed) if timed else None))
        return generation.value,elapsed.value
    def read(self,generation):
        result=np.empty_like(self.output)
        check(self.lib.tessera_rocm_program_read(self.handle,4,generation,C.c_void_p(result.ctypes.data),result.nbytes))
        return result
    def close(self):
        if self.handle.value:
            check(self.lib.tessera_rocm_program_close(self.handle));self.handle.value=0

def library():
    lib=rt._load_rocm_native_movement_runtime()
    if lib is None:raise RuntimeError("native owner library unavailable")
    lib.tessera_rocm_program_prepare.argtypes=[C.c_char_p,C.c_uint32,C.c_uint32,C.POINTER(_Buffer),C.c_uint32,C.POINTER(_Step),C.POINTER(C.c_void_p),C.POINTER(C.c_uint64),C.POINTER(C.c_uint64)]
    lib.tessera_rocm_program_update.argtypes=[C.c_uint64,C.POINTER(C.c_void_p),C.POINTER(C.c_uint64)]
    lib.tessera_rocm_program_invoke.argtypes=[C.c_uint64,C.c_uint32,C.POINTER(C.c_uint64),C.POINTER(C.c_float)]
    lib.tessera_rocm_program_read.argtypes=[C.c_uint64,C.c_uint32,C.c_uint64,C.c_void_p,C.c_uint64]
    lib.tessera_rocm_program_close.argtypes=[C.c_uint64]
    lib.tessera_rocm_program_cache_clear.argtypes=[]
    return lib

def run(shape,name,lib):
    m,n,k=shape
    rng=np.random.default_rng(712+m+n+k)
    a=rng.normal(size=(m,k)).astype(np.float32)
    b=rng.normal(size=(k,n)).astype(np.float32)
    b*=np.repeat(np.exp2(rng.integers(-3,4,(k//32,n))).astype(np.float32),32,axis=0)
    output=np.empty((m,n),ml_dtypes.bfloat16)
    start=time.perf_counter()
    if name=="mxfp4_folded":
        qa,sa,folded,da,exact_b,db=quantize_folded(a,b)
        package=compile_folded_scaled_matmul(qa,sa,folded,tessera_opt=Path(os.environ["TESSERA_OPT"]),allow_approximate=True).package
        buffers=dict(a=qa,b_folded=folded.weight_bytes,a_scale=sa,row_reference=folded.row_reference,output=output)
        p=package.descriptor.provenance
        schedule=FoldedPrefillSchedule(raster_group_m=p["raster_group_m"],workgroup_mode=p["workgroup_mode"],
            staging_prefetch=p["staging_prefetch"],epilogue=p["epilogue_schedule"],row_guard=p["row_guard"])
        grid=folded_prefill_grid(m,n,schedule);block=(256,1,1)
    else:
        mx=name=="mxfp8"
        group_k,group_n=(32,1) if mx else (128,128)
        qa,qb,(sa,sb),da,db=quantize_fp8(a,b,group_k,group_n,e8m0=mx)
        spec=BlockScaleShape(m,n,k,group_k,group_n,"nk","bf16")
        package=compile_mxfp8(spec) if mx else compile_blockscale(spec)
        buffers=dict(a=qa,b=np.ascontiguousarray(qb.T),a_scale=sa,b_scale=sb,o=output)
        bm,bn=package.descriptor.provenance["macro_tile"]
        grid=((n+bn-1)//bn,(m+bm-1)//bm,1);block=tuple(package.descriptor.provenance["workgroup"])
    compile_ms=(time.perf_counter()-start)*1000
    artifact=rt.RuntimeArtifact(metadata={"target":package.image.target},native_image=package.image,
        launch_descriptor=package.descriptor,tile_ir=package.tile_ir,target_ir=package.target_ir)
    receipt=rt.launch(artifact,dict(buffers=buffers,scalars=dict(M=m,N=n,K=k)))
    if not receipt["ok"] or receipt["execution_kind"]!="native_gpu":raise RuntimeError(receipt)
    ideal=da@db;abs_product=np.abs(da)@np.abs(db)
    verify(output,ideal,abs_product,k)
    baseline=output.copy()
    bindings=sorted(package.descriptor.buffers,key=lambda binding:binding.ordinal)
    inputs=[np.ascontiguousarray(buffers[binding.name]) for binding in bindings[:-1]]
    if len(inputs)!=4:raise ValueError("diagnostic requires the existing four-input image ABI")
    arms={}
    for mode in ("0","1"):
        os.environ["TESSERA_ROCM_PROGRAM_PINNED"]=mode
        check(lib.tessera_rocm_program_cache_clear())
        owner=DiagnosticOwner(lib,package,inputs,output,grid,block,shape)
        try:
            generation,_=owner.invoke()
            np.testing.assert_array_equal(owner.read(generation).view(np.uint16),baseline.view(np.uint16))
            events=[];updated=[]
            for _ in range(11):
                generation,elapsed=owner.invoke(100,True);events.append(elapsed)
            changed=list(inputs)
            if name=="mxfp8":
                if np.any(inputs[2]<=1):raise ValueError("half-scale diagnostic requires finite E8M0 exponents above one")
                changed[2]=np.ascontiguousarray(inputs[2]-np.uint8(1))
            else:
                changed[2]=np.ascontiguousarray(inputs[2]*.5)
            owner.update(changed);generation,_=owner.invoke()
            changed_output=owner.read(generation)
            verify(changed_output,ideal*.5,abs_product*.5,k)
            owner.update(inputs)
            for _ in range(11):
                start=time.perf_counter();owner.update(inputs);generation,_=owner.invoke();owner.read(generation)
                updated.append((time.perf_counter()-start)*1000)
        finally:owner.close()
        warm=[]
        for _ in range(7):
            start=time.perf_counter()
            for _ in range(10):
                owner=DiagnosticOwner(lib,package,inputs,output,grid,block,shape)
                try:generation,_=owner.invoke();owner.read(generation)
                finally:owner.close()
            warm.append((time.perf_counter()-start)*100)
        arms[mode]={"native_event_ms":events,"native_event_median_ms":median(events),
            "update_invoke_read_ms":updated,"update_invoke_read_median_ms":median(updated),
            "prepare_invoke_read_close_ms":warm,"prepare_invoke_read_close_median_ms":median(warm)}
    check(lib.tessera_rocm_program_cache_clear())
    return {"shape_mnk":list(shape),"format":name,"compile_ms":compile_ms,"abi":package.descriptor.abi_id,
        "entry":package.descriptor.entry_symbol,"grid":list(grid),"block":list(block),
        "image_sha256":hashlib.sha256(package.image.payload).hexdigest(),
        "correctness":"independent_forward_bound_and_bitwise_existing_launcher_parity",
        "changed_scale_correctness":"independent_forward_bound","arms":arms,
        "boundary":"diagnostic_one_member_plan_from_existing_compiler_package"}

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if rt._rocm_live_arch()!="gfx1201":raise RuntimeError("owning gfx1201 required")
    os.environ["TESSERA_ROCM_PROGRAM_CACHE"]="1"
    lib=library()
    hip=rt._load_hip_for_launch()
    ordinal=C.c_int();device=C.create_string_buffer(256)
    check(hip.hipGetDevice(C.byref(ordinal)))
    check(hip.hipDeviceGetName(device,len(device),ordinal.value))
    device_info=subprocess.run(["rocminfo"],text=True,capture_output=True,check=True).stdout
    rows=[run(shape,name,lib) for shape in [(200,256,256),(256,256,1536)]
          for name in ["fp8","mxfp8","mxfp4_folded"]]
    packet={"architecture":"gfx1201","device":device.value.decode(),"device_ordinal":ordinal.value,"rocminfo":device_info,"rows":rows,"runtime_sha256":hashlib.sha256(Path(lib._name).read_bytes()).hexdigest(),
        "timing_note":"native one-member events include enqueue gaps; host intervals include copy/readback/ABI marshaling; diagnostic plan is not public JIT/AD projection"}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(packet,indent=2)+"\n")
    print(json.dumps([{"shape":r["shape_mnk"],"format":r["format"],"off":r["arms"]["0"]["prepare_invoke_read_close_median_ms"],
        "on":r["arms"]["1"]["prepare_invoke_read_close_median_ms"]} for r in rows],indent=2))
if __name__=="__main__":main()
