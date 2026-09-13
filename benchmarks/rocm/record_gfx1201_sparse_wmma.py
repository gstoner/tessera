"""Registered Target-IR SWMMAC probe, not a public sparse Schedule/Tile package."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import ml_dtypes
import numpy as np
from tessera.compiler.rocm_sparse_packing import pack_sparse_wmma_inputs, sparse_wmma_target_ir, sparse_wmma_schedule_ir
from tessera.compiler.native_gpu_storage import _decode_image


def record(route="schedule_mlir"):
    if route not in {"schedule_mlir", "target_mlir", "native_llvm"}:
        raise ValueError("unknown sparse compiler route")
    from tessera import runtime as rt
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("requires gfx1201 owning device")
    hip=rt._load_hip_for_launch(); P=ct.c_void_p
    def check(rc):
        if rc: raise RuntimeError(f"HIP status {rc}")
    llvm=Path(os.environ["TESSERA_LLVM_BIN"])
    # Profile HIP execution in this process, not compiler/linker subprocesses.
    # rocprof injection into ld.lld crashes on the owning WSL toolchain.
    compiler_env={k:v for k,v in os.environ.items() if not k.startswith("ROCP") and k != "LD_PRELOAD"}
    rows=[]
    for dtype,elem in ((np.float16,"half"),(ml_dtypes.bfloat16,"i16")):
        storage="f16" if elem=="half" else "bf16"
        suffix="f16" if elem=="half" else "i16"
        intrinsic=f"llvm.amdgcn.swmmac.f32.16x16x32.{storage}.v8f32.v8{suffix}.v16{suffix}.i32"
        source=f'''target triple = "amdgcn-amd-amdhsa"
declare i32 @llvm.amdgcn.workitem.id.x()
declare <8 x float> @{intrinsic}(<8 x {elem}>, <16 x {elem}>, <8 x float>, i32)
define amdgpu_kernel void @probe(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %idx, ptr addrspace(1) %out) #0 {{
 %tid = call i32 @llvm.amdgcn.workitem.id.x()
 %ap = getelementptr <8 x {elem}>, ptr addrspace(1) %a, i32 %tid
 %bp = getelementptr <16 x {elem}>, ptr addrspace(1) %b, i32 %tid
 %ip = getelementptr i32, ptr addrspace(1) %idx, i32 %tid
 %op = getelementptr <8 x float>, ptr addrspace(1) %out, i32 %tid
 %av = load <8 x {elem}>, ptr addrspace(1) %ap, align 16
 %bv = load <16 x {elem}>, ptr addrspace(1) %bp, align 16
 %iv = load i32, ptr addrspace(1) %ip, align 4
 %r = call <8 x float> @{intrinsic}(<8 x {elem}> %av, <16 x {elem}> %bv, <8 x float> zeroinitializer, i32 %iv)
 store <8 x float> %r, ptr addrspace(1) %op, align 16
 ret void
}}
attributes #0 = {{ "amdgpu-flat-work-group-size"="32,32" "target-features"="+wavefrontsize32" }}
'''
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'probe.ll').write_text(source)
            if route == "native_llvm":
                subprocess.run([str(llvm/'clang'),'-target','amdgcn-amd-amdhsa','-mcpu=gfx1201','-nogpulib','-x','ir','-c',str(root/'probe.ll'),'-o',str(root/'probe.o')],check=True,env=compiler_env)
                subprocess.run([str(llvm/'ld.lld'),'-shared',str(root/'probe.o'),'-o',str(root/'probe.hsaco')],check=True,env=compiler_env)
                image=(root/'probe.hsaco').read_bytes()
            else:
                producer = sparse_wmma_schedule_ir if route == "schedule_mlir" else sparse_wmma_target_ir
                source=producer("float16" if storage=="f16" else "bfloat16")
                passes = ["--tessera-schedule-to-tile", "--lower-tile-to-rocm"] if route == "schedule_mlir" else []
                lowered=subprocess.check_output([os.environ["TESSERA_OPT"],*passes,"--lower-tessera-target-to-rocdl"],
                    input=source,text=True,env=compiler_env)
                if "tessera_rocm.swmmac" in lowered:
                    raise RuntimeError("sparse Target op did not lower")
                pipeline="builtin.module(gpu.module(convert-vector-to-llvm,convert-scf-to-cf,convert-gpu-to-rocdl,reconcile-unrealized-casts),rocdl-attach-target{chip=gfx1201},gpu-module-to-binary{toolkit="+os.environ["ROCM_PATH"]+"})"
                binary=subprocess.check_output([str(llvm/"mlir-opt"),"--pass-pipeline="+pipeline],
                    input=lowered,text=True,env=compiler_env)
                literals=re.findall(r'"((?:\\.|[^"\\])*)"',binary)
                image=_decode_image(literals[-1])
                (root/"probe.hsaco").write_bytes(image)
            assembly=subprocess.check_output([str(llvm/'llvm-objdump'),'--disassemble',str(root/'probe.hsaco')],text=True,env=compiler_env)
        mnemonic=f'v_swmmac_f32_16x16x32_{storage}'
        if not re.search(r'\b'+mnemonic+r'\b',assembly): raise RuntimeError('missing sparse instruction')
        module,fn=P(),P(); blob=ct.create_string_buffer(image)
        check(hip.hipModuleLoadData(ct.byref(module),blob))
        check(hip.hipModuleGetFunction(ct.byref(fn),module,b'probe'))
        try:
            # Exercise all six index pairs, varying by row/group, and zeros.
            for seed in range(3):
                rng=np.random.default_rng(seed)
                a=np.zeros((16,32),dtype=dtype); b=(rng.integers(-4,5,size=(32,16))/4).astype(dtype)
                pairs=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
                for row in range(16):
                    for group in range(8):
                        pair=pairs[(row+group+seed)%6]
                        a[row,4*group+np.array(pair)]=(rng.integers(-4,5,size=2)/4).astype(dtype)
                packed=pack_sparse_wmma_inputs(a,b)
                raw=[ct.create_string_buffer(x) for x in (packed.a,packed.b,packed.indices)]
                out=np.zeros((32,8),np.float32); ptrs=[]
                try:
                    for size in (len(packed.a),len(packed.b),len(packed.indices),out.nbytes):
                        ptr=P();check(hip.hipMalloc(ct.byref(ptr),size));ptrs.append(ptr)
                    for ptr,host,size in zip(ptrs,raw,(len(packed.a),len(packed.b),len(packed.indices))):
                        check(hip.hipMemcpy(ptr,host,size,1))
                    values=list(ptrs)
                    if route != "native_llvm":
                        values=[]
                        for ptr,size in zip(ptrs,(256,512,32,256),strict=True):
                            values.extend((P(ptr.value),P(ptr.value),ct.c_int64(0),ct.c_int64(size),ct.c_int64(1)))
                    args=(P*len(values))(*[ct.cast(ct.byref(p),P) for p in values])
                    check(hip.hipModuleLaunchKernel(fn,1,1,1,32,1,1,0,None,args,None))
                    check(hip.hipDeviceSynchronize())
                    check(hip.hipMemcpy(out.ctypes.data_as(P),ptrs[3],out.nbytes,2))
                    actual=np.empty((16,16),np.float32)
                    for row in range(16):
                        for col in range(16): actual[row,col]=out[(row//8)*16+col,row%8]
                    expected=a.astype(np.float32)@b.astype(np.float32)
                    np.testing.assert_array_equal(actual,expected)
                    rows.append(dict(dtype=storage,seed=seed,max_abs_error=float(np.max(np.abs(actual-expected))),instruction=mnemonic,image_sha256=hashlib.sha256(image).hexdigest(),source_sha256=hashlib.sha256(source.encode()).hexdigest()))
                finally:
                    # Probe runs in its own process; require completion before cleanup.
                    check(hip.hipDeviceSynchronize())
                    for ptr in reversed(ptrs): check(hip.hipFree(ptr))
        finally:
            check(hip.hipDeviceSynchronize());check(hip.hipModuleUnload(module))
    return dict(target='gfx1201',route=route+'_sparse_probe',production_admission=False,performance_eligible=False,rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--route',choices=('schedule_mlir','target_mlir','native_llvm'),default='schedule_mlir')
    args=parser.parse_args();args.output.write_text(json.dumps(record(args.route),indent=2)+'\n')
