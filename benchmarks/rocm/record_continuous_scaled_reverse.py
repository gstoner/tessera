"""Record compiler-owned gfx1201 continuous reverse residual execution."""
import argparse
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time

import numpy as np
import tessera as ts
from tessera import runtime
from tessera.compiler.native_scaled_program import PreparedScaledProgram
from tests.unit.test_continuous_scaled_ssa_chain import case,chained
from tests.device.rocm.test_continuous_scaled_ssa_chain import reverse_oracle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    options=parser.parse_args()
    if runtime._rocm_live_arch()!="gfx1201":
        raise RuntimeError("exact gfx1201 required")
    hip=c.CDLL("/opt/rocm/lib/libamdhip64.so")
    name=c.create_string_buffer(256)
    uuid=(c.c_ubyte*16)()
    if hip.hipInit(0) or hip.hipDeviceGetName(name,256,0) or hip.hipDeviceGetUuid(c.byref(uuid),0):
        raise RuntimeError("device identity query failed")
    _,_,values=case()
    seed=np.array([[.25,-.5,.125],[-.25,.0625,.5]],dtype=np.float32)
    roles=("a","b","sa","sb","c","sc","sd")
    owner=ts.jit(target="rocm_gfx1201",autodiff="reverse",wrt=roles)(chained)
    start=time.perf_counter()
    actual=owner.native_backward(*values,out_cotangents=seed)
    cold_ms=(time.perf_counter()-start)*1000
    expected=reverse_oracle(values,seed)
    def check(results,wanted):
        errors=[]
        for got,want in zip(results,wanted,strict=True):
            np.testing.assert_allclose(got,want,rtol=4e-5,atol=3e-6)
            errors.append(float(np.max(np.abs(got-want))))
        return max(errors)
    error=check(actual,expected)
    package=owner._native_backward_artifact
    native,member_samples,public=[],[],[]
    with PreparedScaledProgram(package,(*values,seed),runtime_library=os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]) as prepared:
        for index in range(7):
            factor=np.float32(1 if index%2==0 else -.875)
            selected=tuple(np.ascontiguousarray(value*factor) for value in values)
            cotangent=np.ascontiguousarray(seed*(1 if index%2==0 else -.5))
            wanted=reverse_oracle(selected,cotangent)
            prepared.update((*selected,cotangent))
            generation,elapsed=prepared.invoke(repeats=128,timed=True)
            error=max(error,check(prepared.read(generation),wanted))
            native.append(elapsed)
            generation,parts=prepared.profile_members(repeats=128)
            error=max(error,check(prepared.read(generation),wanted))
            member_samples.append(parts)
            start=time.perf_counter()
            result=owner.native_backward(*selected,out_cotangents=cotangent)
            public.append((time.perf_counter()-start)*1000)
            error=max(error,check(result,wanted))
    program=json.loads(package.program_json)
    paths=["python/tessera/compiler/native_scaled_program.py",
           "python/tessera/compiler/rocm_typed_scaled_native.py",
           "src/compiler/ir/LinearTransposeInterface.cpp",
           "src/transforms/lib/NativeScaledMatmulProgram.h",
           "src/compiler/programming_model/lib/NativeScaleTranspose.h",
           "tests/device/rocm/test_continuous_scaled_ssa_chain.py",
           "benchmarks/rocm/record_continuous_scaled_reverse.py"]
    packet={"architecture":"gfx1201","device":name.value.decode(),"uuid_hex":bytes(uuid).hex(),
        "route":"Python frontend->Graph->native AD/residual SSA->Schedule->Tile->ROCm Target->LLVM->HSACO->checked HIP ABI",
        "input_shapes":[list(value.shape) for value in values],"requested_roles":roles,
        "max_abs_error":error,"correctness":"independent scalar adjoint oracle before and after every timing window",
        "cold_compile_and_call_ms":cold_ms,"native_program_samples_ms":native,
        "native_program_median_ms":median(native),"public_compile_warm_samples_ms":public,
        "public_compile_warm_median_ms":median(public),"member_samples_ms":member_samples,
        "member_medians_ms":[median(row[i] for row in member_samples) for i in range(len(program["steps"]))],
        "member_timing_scope":"Grouped repeated SSA members including HIP graph device dispatch; not additive interleaved program times.",
        "public_scope":"Compilation warm; preparation/allocation not excluded. No speedup claim.",
        "dependency_policy":program["dependency_policy"],"steps":program["steps"],"buffers":program["buffers"],
        "image_sha256":[hashlib.sha256(image).hexdigest() for image in package.images],
        "sources":{path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths},
        "tools":{key:hashlib.sha256(Path(os.environ[key]).read_bytes()).hexdigest()
                 for key in ("TESSERA_OPT","TESSERA_ROCM_OPT")}}
    options.output.parent.mkdir(parents=True,exist_ok=True)
    options.output.write_text(json.dumps(packet,indent=2)+"\n")
    print(json.dumps({key:packet[key] for key in ("device","max_abs_error","native_program_median_ms","member_medians_ms","public_compile_warm_median_ms")}))
if __name__=="__main__":
    main()
