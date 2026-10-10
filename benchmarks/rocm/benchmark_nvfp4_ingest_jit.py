"""Checked canonical JIT ingest; separate compile/wall/resident timing scopes."""
from pathlib import Path
from statistics import median
import ctypes as C
import hashlib
import json
import os
import time
import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler import rocm_nvfp4_ingest as oracle
from tessera.compiler.rocm_native import ROCMNativePackage
from tessera.compiler.rocm_nvfp4_ingest_native import NVFP4IngestPackage,execute_nvfp4_ingest

def make_converter(offsets):
    @ts.jit(target="rocm_gfx1201")
    def convert(codes,scales,globals_):
        return ts.ops.nvfp4_requantize(codes,scales,globals_,
            row_offsets=offsets,numeric_policy=oracle.nvfp4_requantization_policy())
    return convert

def record():
    architecture=rt._rocm_live_arch()
    if architecture!="gfx1201":
        raise RuntimeError(f"requires exact gfx1201, found {architecture}")
    hip=rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):
        raise RuntimeError("HIP runtime unavailable")
    ordinal,name=C.c_int(),C.create_string_buffer(256)
    hip.hipGetDevice.argtypes=[C.POINTER(C.c_int)]
    hip.hipDeviceGetName.argtypes=[C.c_char_p,C.c_int,C.c_int]
    if hip.hipGetDevice(C.byref(ordinal)) or hip.hipDeviceGetName(name,len(name),ordinal.value):
        raise RuntimeError("HIP device identity probe failed")
    gpu=name.value.decode()
    if not gpu or str(gpu).lower()=="unknown":
        raise RuntimeError("device identity required")
    rows=[]
    for n,k,offsets in ((7,64,[0,3,7]),(67,256,[0,31,67]),(513,1024,[0,255,513])):
        rng=np.random.default_rng(120164)
        args=(rng.integers(0,256,(n,k//2),dtype=np.uint8),
            rng.choice(np.array([0,.125,.5,1,2,6]),(n,k//16)).astype(ml_dtypes.float8_e4m3fn),
            np.array([.5,2.],np.float64))
        expected=oracle.reference_nvfp4_requantize(*args,row_offsets=offsets,
            numeric_policy=oracle.nvfp4_requantization_policy())
        function=make_converter(offsets)
        def check(result):
            np.testing.assert_array_equal(result[0],expected[0])
            np.testing.assert_array_equal(result[1],expected[1])
            np.testing.assert_allclose(result[2],expected[2],rtol=1e-13,atol=1e-30)
        start=time.perf_counter_ns()
        actual=function(*args)
        cold=(time.perf_counter_ns()-start)/1e6
        check(actual)
        artifact=function.runtime_artifact()
        if function.execution_kind!="native_gpu" or not artifact.metadata["canonical_executable"]:
            raise RuntimeError("canonical native JIT required")
        warm=[]
        for _ in range(5):
            start=time.perf_counter_ns()
            result=function(codes=args[0],scales=args[1],globals_=args[2])
            warm.append((time.perf_counter_ns()-start)/1e6)
            check(result)
            if function.runtime_artifact().native_image.image_digest!=artifact.native_image.image_digest:
                raise RuntimeError("warm package identity changed")
        package=NVFP4IngestPackage(artifact.graph_ir,artifact.schedule_ir,
            ROCMNativePackage(artifact.tile_ir,artifact.target_ir,"",
                artifact.native_image,artifact.launch_descriptor))
        events=[]
        check(execute_nvfp4_ingest(package,*args,event_samples=events))
        rows.append({"shape_nk":[n,k],"row_offsets":offsets,
            "correctness":"bitwise_codes_exponents_and_f64_statistics_before_timing",
            "cold_compile_and_call_wall_ms":cold,"warm_jit_wall_samples_ms":warm,
            "warm_jit_wall_median_ms":median(warm),"resident_event_samples_ms":events,
            "resident_event_median_ms":median(events),
            "image_digest":artifact.native_image.image_digest,
            "abi_id":artifact.launch_descriptor.abi_id,
            "provenance":dict(artifact.launch_descriptor.provenance)})
    sources=("python/tessera/compiler/jit.py","python/tessera/compiler/trace.py",
        "python/tessera/compiler/graph_ir.py","python/tessera/compiler/op_catalog.py",
        "python/tessera/compiler/backend_manifest.py","python/tessera/compiler/capabilities.py",
        "python/tessera/compiler/driver.py","python/tessera/compiler/rocm_nvfp4_ingest_native.py",
        "python/tessera/compiler/rocm_nvfp4_ingest.py",
        "python/tessera/runtime.py","benchmarks/rocm/benchmark_nvfp4_ingest_jit.py")
    return {"schema":"tessera.rocm.nvfp4_ingest_jit.v1","architecture":architecture,
        "device":gpu,"rows":rows,
        "compiler_sha256":hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        "source_sha256":{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        "route":"Python tracer -> Graph MLIR -> Schedule MLIR -> Tile MLIR -> ROCm Target -> LLVM -> HSACO -> checked runtime",
        "timing_scope":"cold wall includes trace/compile/execution; warm wall includes guards, allocation, upload, module launch, completion and readback; resident events exclude module/allocation/transfer",
        "scope":"isolated static checkpoint conversion; private device outputs; host readback; no resident consumer edge or model-quality/default promotion",
        "format_gates":"FP8, MXFP8 and MXFP4 remain independent correctness/quality/performance gates"}

if __name__=="__main__":
    packet=record()
    Path(os.environ["TESSERA_INGEST_JIT_PACKET"]).write_text(json.dumps(packet,indent=2,sort_keys=True)+"\n")
    print("verified",len(packet["rows"]),"canonical JIT ingest rows")
