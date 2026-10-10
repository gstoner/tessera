"""Exact gfx1201 bounded NVFP4 producer/product characterization."""
import argparse
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time
from unittest.mock import patch

import ml_dtypes
import numpy as np
from tessera import runtime as rt
from tessera.compiler.rocm_mxfp4 import folded_weights
from tessera.compiler.rocm_mxfp4_folded import prepare_folded_weights
from tests.device.rocm.test_nvfp4_resident_jit import make_function
from tests.unit.test_rocm_nvfp4_resident import inputs_and_oracle


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_output(actual,expected):
    np.testing.assert_allclose(actual.astype("f4"),expected,rtol=.008,atol=.015625)
    return float(np.max(np.abs(actual.astype("f4")-expected)))


def check_conversion(session,converted):
    diagnostic=session.conversion_diagnostics()
    for name,wanted in zip(("packed","exponents","stats"),converted,strict=True):
        if name=="stats":np.testing.assert_allclose(diagnostic[name],wanted,rtol=1e-13,atol=1e-30)
        else:np.testing.assert_array_equal(diagnostic[name],wanted)


def check_producer(session,converted,stored):
    check_conversion(session,converted)
    diagnostic=session.storage_diagnostics()
    np.testing.assert_array_equal(diagnostic["fragment"],stored[0])
    np.testing.assert_array_equal(diagnostic["plane"],stored[1])


def profile(bound,n,k):
    frames=[inputs_and_oracle(rows,n,k) for rows in (17,bound,1,200)]
    function=make_function(n,k)
    start=time.perf_counter()
    program=function.compile_native_nvfp4_program(*frames[0][0],m_bound=bound)
    compile_ms=(time.perf_counter()-start)*1000
    rows=[]
    with program.native.native_session(*frames[0][0],reuse=True) as session:
        capacity=session.frame_stats()
        handle=session.handle
        for arguments,_,converted,stored,expected in frames:
            session.update_inputs(*arguments)
            session.run_combined()
            error=check_output(session.read_output(),expected)
            check_producer(session,converted,stored)
            stage_samples={}
            graph_samples={}
            for stage in ("converter","storage","ingest","consumer","combined"):
                samples=[]
                captured=[]
                windows=[]
                # Alternate ordering in matched rounds to reduce clock/order bias.
                for round_index in range(7):
                    modes=("direct","graph") if round_index%2==0 else ("graph","direct")
                    for mode in modes:
                        session.run_combined()
                        error=max(error,check_output(session.read_output(),expected))
                        check_producer(session,converted,stored)
                        if mode=="direct":
                            samples.extend(session.measure(stage,samples=1,repeats=128))
                        else:
                            window=session.measure_graph(stage,samples=1,repeats=128)[0]
                            nodes=128*{"converter":1,"storage":1,"ingest":2,"consumer":1,"combined":3}[stage]
                            if (window["graph_nodes"]!=nodes or window["repeats"]!=128
                                    or window["host_graph_submissions"]!=1):
                                raise RuntimeError("captured window differs from declared stage/repetitions")
                            captured.append(window["per_iteration_ms"])
                            windows.append(window)
                        if stage=="converter":check_conversion(session,converted)
                        else:check_producer(session,converted,stored)
                        if stage in ("consumer","combined"):
                            error=max(error,check_output(session.read_output(),expected))
                stage_samples[stage]={"samples_ms":samples,"median_ms":median(samples)}
                graph_samples[stage]={"samples_ms":captured,"median_ms":median(captured),
                                     "windows":windows,
                                     "direct_over_graph_median":median(samples)/median(captured)}
            stats=session.frame_stats()
            assert session.handle==handle
            assert stats["allocation_bytes"]==capacity["allocation_bytes"]
            assert stats["allocation_count"]==capacity["allocation_count"]
            weight=folded_weights(prepare_folded_weights(*converted[:2],allow_approximate=True)).astype("f8")
            public=[]
            for index in range(7):
                values=[x.copy() for x in arguments]
                values[3]=np.roll(values[3],index,axis=1).copy()
                values[4]*=np.float32(2.0**(index%3-1))
                decoded=values[3].view(ml_dtypes.float8_e4m3fn).astype("f8")*values[4][:,None]
                wanted=(decoded@weight.T).astype(ml_dtypes.bfloat16).astype("f4")
                public.append((values,wanted))
            latency=[]
            with patch.object(subprocess,"run",side_effect=RuntimeError("compiler during warm replay")):
                for values,wanted in public:
                    start=time.perf_counter()
                    actual=function(*values)
                    latency.append((time.perf_counter()-start)*1000)
                    error=max(error,check_output(actual,wanted))
            rows.append({"active_mnk":[arguments[3].shape[0],n,k],
                         "correctness":"checked_around_every_native_window_and_changed_public_call",
                         "max_abs_error":error,"native_frame":stats,
                         "native_event_stages":stage_samples,
                         "native_graph_event_stages":graph_samples,
                         "public_warm_samples_ms":latency,"public_warm_median_ms":median(latency)})
    return {"bounds_mnk":[bound,n,k],"compile_ms":compile_ms,
            "plan_sha256":hashlib.sha256(program.native.native_plan_json.encode()).hexdigest(),
            "component_image_digests":program.native.receipt["component_image_digests"],
            "component_descriptors":program.native.receipt["component_descriptors"],"rows":rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    options=parser.parse_args()
    if rt._rocm_live_arch()!="gfx1201":raise RuntimeError("exact gfx1201 device required")
    active=subprocess.run(["pgrep","-af","[p]ytest|[g]raphify update"],capture_output=True,text=True)
    if active.returncode==0 and active.stdout.strip():raise RuntimeError("test/graph jobs active")
    hip=c.CDLL("/opt/rocm/lib/libamdhip64.so")
    name=c.create_string_buffer(256);uuid=(c.c_ubyte*16)();ordinal=c.c_int()
    if (hip.hipInit(0) or hip.hipGetDevice(c.byref(ordinal)) or
            hip.hipDeviceGetName(name,256,ordinal.value) or hip.hipDeviceGetUuid(c.byref(uuid),ordinal.value)):
        raise RuntimeError("active HIP device identity query failed")
    sources=[
        "src/transforms/lib/NativeNVFP4Program.h",
        "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_nvfp4_runtime.cpp",
        "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_image_cache.cpp",
        "python/tessera/compiler/jit.py","python/tessera/compiler/native_nvfp4_program.py",
        "python/tessera/compiler/native_resident_nvfp4.py","python/tessera/compiler/rocm_nvfp4_program.py",
        "python/tessera/compiler/rocm_nvfp4_resident.py",
        "tests/device/rocm/test_nvfp4_bounded_rows.py","tests/unit/test_native_nvfp4_bounded_rows.py"]
    packet={"schema":"tessera.gfx1201.bounded_nvfp4_rows.v1","target":"rocm_gfx1201",
            "device":{"name":name.value.decode(),"uuid_hex":bytes(uuid).hex(),"ordinal":ordinal.value},
            "compiler_sha256":digest(os.environ["TESSERA_OPT"]),
            "target_tool_sha256":digest(os.environ["TESSERA_ROCM_OPT"]),
            "runtime_sha256":digest(os.environ["TESSERA_ROCM_NATIVE_MOVEMENT_LIB"]),
            "image_runtime_sha256":digest(os.environ["TESSERA_ROCM_NATIVE_IMAGE_LIB"]),
            "source_sha256":{name:digest(name) for name in sources},"recorder_sha256":digest(__file__),
            "numeric_policy":"explicit NVFP4 requantization and folded-row approximate product; no original-BF16/model quality claim",
            "native_domain":"HIP events; seven interleaved direct/captured rounds per stage; 128 repetitions/window; direct includes repeated host dispatch gaps; captured has one graph submission; converter and storage have checked independent producer receipts; ingest, consumer and three-stage program timed independently",
            "public_domain":"warm ordinary JIT call: host checks/packing, all-input upload, native program, synchronization and readback; changed values; compiler forbidden",
            "claim":"bounded-row correctness and matched native dispatch attribution; identical images/operands; graph/direct ratio is submission-policy characterization, not a kernel algorithm speedup or selector promotion",
            "profiles":[profile(*shape) for shape in ((257,32,64),(513,80,256),(256,64,1024))]}
    options.output.parent.mkdir(parents=True,exist_ok=True)
    options.output.write_text(json.dumps(packet,indent=2)+"\n")


if __name__=="__main__":
    main()
