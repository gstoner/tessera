"""Ordinary JIT and portable ROCm native math, with separate resident event timing."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
from statistics import median
import time

import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.scheduled_matmul import find_tessera_opt
from benchmarks.rocm.benchmark_native_math_schedule import MathCase, expected
from tests.device.rocm.test_native_math_package_jit import FUNCTIONS


def record(hip, target, name, shape, samples, iterations, storage):
    kind=name.removesuffix("_reverse")
    rng=np.random.default_rng(606)
    dtype=np.float32 if storage=="f32" else np.float16
    if storage=="bf16":
        import ml_dtypes
        dtype=ml_dtypes.bfloat16
    values=[rng.uniform(.125,2,shape).astype(dtype)]
    if kind in {"add","div"}:
        values.append(rng.uniform(.5,2,shape).astype(dtype))
    def ordered():
        promoted=[value.astype(np.float32) for value in values]
        return list(reversed(promoted)) if name.endswith("_reverse") else promoted
    functions=FUNCTIONS
    if storage!="f32":
        from tests.device.rocm.test_native_math_widening import FUNCTIONS as widening_functions
        functions=widening_functions
    fn=ts.jit(target=target)(functions[name])
    actual=fn(*values)
    np.testing.assert_allclose(actual,expected(kind,ordered()),rtol=2e-5,atol=2e-5)
    assert fn.execution_kind=="native_gpu"
    artifact=fn.runtime_artifact()
    image=artifact.native_image
    descriptor=artifact.launch_descriptor
    assert image is not None and descriptor is not None
    info=descriptor.provenance["native_math"]
    buffers=dict(zip(info["bindings"][:-1],values))
    output=next(b.name for b in descriptor.buffers if b.direction=="output")
    buffers[output]=np.empty(shape,np.float32)
    scalars={"Rows":info["rows"],"Columns":info["columns"]} if kind.startswith("cum") else {"N":info["elements"]}
    restored=rt.RuntimeArtifact.from_json(artifact.to_json())
    ordered_inputs=[buffers[b.name] for b in sorted(descriptor.buffers,key=lambda b:b.ordinal) if b.direction=="input"]
    case=MathCase(hip,image.payload,descriptor.entry_symbol,ordered_inputs,kind.startswith("cum"),output_dtype=np.float32)
    try:
        np.testing.assert_allclose(case.download(),expected(kind,ordered()),rtol=2e-5,atol=2e-5)
        events=case.measure(trials=samples,iterations=iterations,warmup=5)
    finally:
        case.close()
    jit_walls=[]
    portable_walls=[]
    for sample in range(samples):
        # Same addresses, changed values: retained packages must not retain input content.
        factor=np.float32(.99 if sample%2 else 1.01)
        for value in values:
            value[:]=value.astype(np.float32)*factor
        start=time.perf_counter_ns()
        actual=fn(*values)
        jit_walls.append((time.perf_counter_ns()-start)/1e6)
        np.testing.assert_allclose(actual,expected(kind,ordered()),rtol=2e-5,atol=2e-5)
        start=time.perf_counter_ns()
        receipt=rt.launch(restored,{"buffers":buffers,"scalars":scalars})
        portable_walls.append((time.perf_counter_ns()-start)/1e6)
        assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
        np.testing.assert_allclose(buffers[output],expected(kind,ordered()),rtol=2e-5,atol=2e-5)
        assert fn.runtime_artifact().native_image.image_digest==image.image_digest
    return {
        "kind":kind,"storage":storage,"output_storage":"f32","shape":list(shape),"reversed_binary_roles":name.endswith("_reverse"),
        "correctness":"independent NumPy oracle before device timing and after each changed-input JIT/portable call",
        "max_abs_error":float(np.max(np.abs(actual-expected(kind,ordered())))),
        "route":"Python frontend->native Graph/Schedule/Tile->ROCm Target->ROCDL/LLVM->HSACO->checked ABI",
        "compiler_path":artifact.metadata["compiler_path"],"entry_symbol":descriptor.entry_symbol,
        "image_sha256":image.image_digest,"abi_id":descriptor.abi_id,
        "ir_sha256":{key:hashlib.sha256(getattr(artifact,key).encode()).hexdigest()
                     for key in ("graph_ir","schedule_ir","tile_ir","target_ir")},
        "device_event_samples_ms":events,"device_event_median_ms":median(events),
        "warm_jit_end_to_end_samples_ms":jit_walls,"warm_jit_end_to_end_median_ms":median(jit_walls),
        "portable_end_to_end_samples_ms":portable_walls,"portable_end_to_end_median_ms":median(portable_walls),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture",choices=("gfx1151","gfx1201"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=7)
    parser.add_argument("--storage",choices=("f32","all"),default="all")
    parser.add_argument("--iterations",type=int,default=100)
    args=parser.parse_args()
    if args.samples<3 or args.iterations<1:
        raise ValueError("invalid timing counts")
    live=rt._rocm_live_arch()
    if live!=args.architecture:
        raise RuntimeError(f"owning architecture mismatch: {live}")
    hip=rt._load_hip_for_launch()
    if hip is None:
        raise RuntimeError("HIP unavailable")
    MathCase.check(hip.hipInit(0),"init")
    ordinal=C.c_int()
    name=C.create_string_buffer(256)
    uuid=C.create_string_buffer(16)
    MathCase.check(hip.hipGetDevice(C.byref(ordinal)),"ordinal")
    MathCase.check(hip.hipDeviceGetName(name,len(name),ordinal.value),"name")
    hip.hipDeviceGetUuid.argtypes=[C.c_void_p,C.c_int]
    hip.hipDeviceGetUuid.restype=C.c_int
    MathCase.check(hip.hipDeviceGetUuid(uuid,ordinal.value),"UUID")
    storages=("f32",) if args.storage=="f32" else ("f32","f16","bf16")
    rows=[record(hip,"rocm_"+live,op,shape,args.samples,args.iterations,storage)
          for storage in storages for shape in ((3,17),(2,3,257),(256,1024)) for op in FUNCTIONS]
    for kind in ("sqrt","exp","add","div","cumsum","cummax"):
        for storage in storages:
            assert len({row["image_sha256"] for row in rows if row["kind"]==kind and row["storage"]==storage})==1
        assert len({row["image_sha256"] for row in rows if row["kind"]==kind})==len(storages)
    root=Path(__file__).resolve().parents[2]
    sources=("python/tessera/compiler/rocm_math_native.py","python/tessera/compiler/rocm_native.py",
             "python/tessera/compiler/jit.py","python/tessera/compiler/driver.py","python/tessera/runtime.py",
             "src/compiler/programming_model/lib/NativeROCMMath.h",
             "src/compiler/programming_model/lib/PMPasses.cpp",
             "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/ROCMKernelIdentity.cpp",
             "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp",
             "benchmarks/rocm/benchmark_native_math_package.py",
             "tests/device/rocm/test_native_math_package_jit.py",
             "tests/device/rocm/test_native_math_widening.py",
             "src/compiler/ir/TileOps.cpp",
             "src/compiler/codegen/Tessera_ROCM_Backend/include/TesseraROCM/IR/TesseraROCMOps.td",
             "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateROCMUnaryKernel.cpp",
             "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateROCMBinaryKernel.cpp",
             "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateROCMScanKernel.cpp")
    packet={"schema":"tessera.rocm.native-math-package.v1","architecture":live,"device":name.value.decode(),
            "device_ordinal":ordinal.value,"opaque_hip_uuid":uuid.raw.hex(),
            "compiler_binary_sha256":hashlib.sha256(find_tessera_opt().read_bytes()).hexdigest(),
            "source_sha256":{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},
            "input_storages":list(storages),"output_storage":"f32","rows":rows,"shape_independent_images":"verified per operation across all shapes and binary roles",
            "timing_domains":{"device":"HIP events around resident launches; upload/readback/allocation excluded",
                              "warm_jit":"ordinary JIT host-array call including validation/upload/launch/readback; compiler excluded",
                              "portable":"checked deserialized package host-array launch; deserialization and compiler excluded"},
            "scope":"static compact f32 math with explicit exact f16/bf16 Graph widening casts; general composition and dynamic shapes remain open"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"architecture":live,"device":packet["device"],"rows":len(rows),
                      "correctness":"passed","image_reuse":"passed"}))


if __name__=="__main__":
    main()
