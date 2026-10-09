"""Record gfx1201 continuous SSA producer/consumer execution."""
import argparse
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time

import numpy as np
from tessera import runtime
from tessera.compiler.native_scaled_program import PreparedScaledProgram
from tests.unit.test_continuous_scaled_ssa_chain import case
from tests.device.rocm.test_continuous_scaled_ssa_chain import oracle


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
    owner,_,values=case()
    directions=tuple(np.full_like(value,.03125) for value in values)
    expected=oracle(values,directions)[0]
    start=time.perf_counter()
    actual=owner(*values)
    cold_ms=(time.perf_counter()-start)*1000
    np.testing.assert_allclose(actual,expected,rtol=4e-5,atol=3e-6)
    package=owner._native_composed_scaled_last_program
    native,member_samples,public=[],[],[]
    error=float(np.max(np.abs(actual-expected)))
    with PreparedScaledProgram(package,values,runtime_library=os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]) as prepared:
        for index in range(7):
            selected=tuple(np.ascontiguousarray(value*np.float32(1 if index%2==0 else 1.125)) for value in values)
            wanted=oracle(selected,directions)[0]
            prepared.update(selected)
            generation,elapsed=prepared.invoke(repeats=128,timed=True)
            result=prepared.read(generation)[0]
            np.testing.assert_allclose(result,wanted,rtol=4e-5,atol=3e-6)
            error=max(error,float(np.max(np.abs(result-wanted))))
            native.append(elapsed)
            generation,parts=prepared.profile_members(repeats=128)
            np.testing.assert_allclose(prepared.read(generation)[0],wanted,rtol=4e-5,atol=3e-6)
            member_samples.append(parts)
            start=time.perf_counter()
            result=owner(*selected)
            public.append((time.perf_counter()-start)*1000)
            np.testing.assert_allclose(result,wanted,rtol=4e-5,atol=3e-6)
    program=json.loads(package.program_json)
    paths=["python/tessera/compiler/rocm_typed_scaled_native.py",
           "src/transforms/lib/NativeScaledMatmulProgram.h",
           "src/transforms/lib/NativeFloatingScaledProduct.h",
           "tests/device/rocm/test_continuous_scaled_ssa_chain.py",
           "benchmarks/rocm/record_continuous_scaled_ssa_chain.py"]
    packet={"architecture":"gfx1201","device":name.value.decode(),"uuid_hex":bytes(uuid).hex(),
        "route":"Python frontend->Graph->native program->Schedule->Tile->ROCm Target->LLVM->HSACO->checked HIP ABI",
        "input_shapes":[list(value.shape) for value in values],"max_abs_error":error,
        "correctness":"independent scalar oracle checked before and after all timing windows",
        "cold_compile_and_call_ms":cold_ms,"native_program_samples_ms":native,
        "native_program_median_ms":median(native),"public_compile_warm_samples_ms":public,
        "public_compile_warm_median_ms":median(public),"member_samples_ms":member_samples,
        "member_medians_ms":[median(row[i] for row in member_samples) for i in range(len(program["steps"]))],
        "member_timing_scope":"Grouped repeated member HIP graph event windows, including device dispatch; not additive interleaved program timings.",
        "public_scope":"Compilation warm; frame allocation/cache preparation may occur. No universal performance claim.",
        "steps":program["steps"],"buffers":program["buffers"],
        "image_sha256":[hashlib.sha256(image).hexdigest() for image in package.images],
        "sources":{path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths},
        "tools":{key:hashlib.sha256(Path(os.environ[key]).read_bytes()).hexdigest()
                 for key in ("TESSERA_OPT","TESSERA_ROCM_OPT")}}
    options.output.parent.mkdir(parents=True,exist_ok=True)
    options.output.write_text(json.dumps(packet,indent=2)+"\n")
    print(json.dumps({key:packet[key] for key in ("device","max_abs_error","native_program_median_ms","member_medians_ms","public_compile_warm_median_ms")}))
if __name__=="__main__":
    main()
