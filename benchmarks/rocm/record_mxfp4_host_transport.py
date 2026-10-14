"""Exact-device public transport attribution; no kernel or policy promotion."""
import argparse
import ctypes as c
import hashlib
import json
import os
import time
import subprocess
from pathlib import Path
from statistics import median
import numpy as np
from tessera import runtime
from tessera.autodiff import jvp
from tests.unit.test_native_mxfp4_residual_gate import typed_case
from tests.device.rocm.test_mxfp4_residual_native_execution import frame, coefficient


def main():
    parser = argparse.ArgumentParser(description="Diagnostic ABBA pinned/pageable public MXFP4 calls")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    options = parser.parse_args()
    if not 1 <= options.iterations <= 10000:
        parser.error("iterations must be between 1 and 10000")
    assert runtime._rocm_live_arch()=="gfx1201"
    hip = c.CDLL("/opt/rocm/lib/libamdhip64.so")
    device, name, uuid = c.c_int(), c.create_string_buffer(256), (c.c_ubyte*16)()
    if (hip.hipInit(0) or hip.hipGetDevice(c.byref(device)) or
            hip.hipDeviceGetName(name, 256, device.value) or hip.hipDeviceGetUuid(c.byref(uuid), device.value)):
        raise RuntimeError("active device identity query failed")

    rows=[]
    for mode in ("0","1","1","0"):
     os.environ["TESSERA_ROCM_PROGRAM_PINNED"]=mode
     for n in (80,160):
      owner,_=typed_case((256,128,n))
      values,cot=frame((256,128,n))
      directions=(None,None,None,None,cot)
      expected=coefficient(values)
      times={"primal":[],"jvp":[],"reverse":[]}
      original_run=subprocess.run
      try:
       owner(*values); jvp(owner,values,directions); owner.native_backward(*values,out_cotangents=cot)
       def forbidden(*args,**kwargs):
        raise AssertionError("warm host transport arm invoked compiler")
       subprocess.run=forbidden
       for _ in range(options.iterations):
        start=time.perf_counter()
        p=owner(*values)
        times["primal"].append((time.perf_counter()-start)*1000)
        np.testing.assert_array_equal(p,expected*values[4])
        start=time.perf_counter()
        p,t=jvp(owner,values,directions)
        times["jvp"].append((time.perf_counter()-start)*1000)
        np.testing.assert_array_equal(p,expected*values[4])
        np.testing.assert_array_equal(t,expected*cot)
        start=time.perf_counter()
        grad=owner.native_backward(*values,out_cotangents=cot)
        times["reverse"].append((time.perf_counter()-start)*1000)
        np.testing.assert_array_equal(grad[0],expected*cot)
       row={"requested_pinned":mode,"shape_mkn":[256,128,n],"samples_ms":times,
            "median_ms":{key:median(value) for key,value in times.items()},
            "correctness":"exact_after_each_completed_call","warm_compiler":"forbidden",
            "input_bytes":sum(v.nbytes for v in values),
            "jvp_input_bytes":sum(v.nbytes for v in values)+cot.nbytes,
            "jvp_output_bytes":2*cot.nbytes}
       rows.append(row)
       print(row["requested_pinned"],row["shape_mkn"],row["median_ms"],flush=True)
      finally:
       subprocess.run=original_run
       owner.close_native_storage()
    packet={"schema":"tessera.mxfp4_host_transport.abba.v1",
            "architecture":runtime._rocm_live_arch(),"device":name.value.decode(),
            "device_uuid_hex":bytes(uuid).hex(),"device_index":device.value,"timing_scope":"completed public calls; host copies included, no kernel-only claim",
            "source_sha256":{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [
               "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_program_runtime.cpp",
               "python/tessera/compiler/native_scaled_program.py",
               "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/ProgramHostTransportPolicy.h",
               "python/tessera/compiler/rocm_typed_scaled_native.py",
               "tests/unit/test_native_mxfp4_residual_gate.py",
               "tests/device/rocm/test_mxfp4_residual_native_execution.py",
               str(Path(__file__).relative_to(Path.cwd()))]},
            "provider_sha256":hashlib.sha256(Path(os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]).read_bytes()).hexdigest(),
            "rows":rows}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(packet,indent=2)+"\n")


if __name__ == "__main__":
    main()
