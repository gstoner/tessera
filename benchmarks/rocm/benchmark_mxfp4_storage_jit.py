"""Checked canonical JIT lossless storage; separate compile/wall/resident timing scopes."""
from pathlib import Path
from statistics import median
import ctypes as C
import hashlib
import json
import os
import time
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4_storage as oracle
from tessera.compiler.rocm_native import ROCMNativePackage
from tessera.compiler.rocm_mxfp4_storage_native import MXFP4StoragePackage,execute_mxfp4_storage

def make_storage():
    @ts.jit(target="rocm_gfx1201")
    def storage(codes,exponents):
        return ts.ops.mxfp4_folded_storage(codes,exponents,
            storage_contract=oracle.MXFP4_STORAGE_CONTRACT)
    return storage

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
    for n,k in ((16,64),(32,256),(64,1024)):
        rng=np.random.default_rng(120164)
        args=(rng.integers(0,256,(n,k//2),dtype=np.uint8),
            rng.integers(0,256,(k//32,n),dtype=np.uint8))
        args[1][:,0]=0
        args[1][:,1]=127
        expected=oracle.reference_mxfp4_folded_storage(*args,
            storage_contract=oracle.MXFP4_STORAGE_CONTRACT)
        function=make_storage()
        def check(result):
            for actual,wanted in zip(result,expected):
                np.testing.assert_array_equal(actual,wanted)
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
            result=function(codes=args[0],exponents=args[1])
            warm.append((time.perf_counter_ns()-start)/1e6)
            check(result)
            if function.runtime_artifact().native_image.image_digest!=artifact.native_image.image_digest:
                raise RuntimeError("warm package identity changed")
        package=MXFP4StoragePackage(artifact.graph_ir,artifact.schedule_ir,
            ROCMNativePackage(artifact.tile_ir,artifact.target_ir,"",
                artifact.native_image,artifact.launch_descriptor))
        events=[]
        check(execute_mxfp4_storage(package,*args,event_samples=events))
        rows.append({"shape_nk":[n,k],
            "correctness":"bitwise_fragment_bytes_and_extended_scale_plane_before_timing",
            "cold_compile_and_call_wall_ms":cold,"warm_jit_wall_samples_ms":warm,
            "warm_jit_wall_median_ms":median(warm),"resident_event_samples_ms":events,
            "resident_event_median_ms":median(events),
            "image_digest":artifact.native_image.image_digest,
            "abi_id":artifact.launch_descriptor.abi_id,
            "provenance":dict(artifact.launch_descriptor.provenance)})
    sources=("python/tessera/compiler/jit.py","python/tessera/compiler/trace.py",
        "python/tessera/compiler/graph_ir.py","python/tessera/compiler/op_catalog.py",
        "python/tessera/compiler/backend_manifest.py","python/tessera/compiler/capabilities.py",
        "python/tessera/compiler/driver.py","python/tessera/compiler/rocm_mxfp4_storage_native.py",
        "python/tessera/compiler/rocm_mxfp4_storage.py",
        "python/tessera/runtime.py","benchmarks/rocm/benchmark_mxfp4_storage_jit.py")
    return {"schema":"tessera.rocm.mxfp4_storage_jit.v1","architecture":architecture,
        "device":gpu,"rows":rows,
        "compiler_sha256":hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        "source_sha256":{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        "route":"Python tracer -> Graph MLIR -> Schedule MLIR -> Tile MLIR -> ROCm Target -> LLVM -> HSACO -> checked runtime",
        "timing_scope":"cold wall includes trace/compile/execution; warm wall includes guards, allocation, upload, module launch, completion and readback; resident events exclude module/allocation/transfer but include host dispatch gaps, not isolated kernel time",
        "scope":"isolated static lossless storage bridge; private device outputs; host readback; no resident converter/consumer edge or model-quality/default promotion",
        "format_gates":"FP8, MXFP8 and MXFP4 remain independent correctness/quality/performance gates"}

if __name__=="__main__":
    packet=record()
    Path(os.environ["TESSERA_STORAGE_JIT_PACKET"]).write_text(json.dumps(packet,indent=2,sort_keys=True)+"\n")
    print("verified",len(packet["rows"]),"canonical JIT lossless storage rows")
