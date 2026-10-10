"""Checked ROCm movement: Python, native cold allocations, native staging reuse."""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import numpy as np
from tessera import runtime as rt
from tessera.compiler import rocm_native
from benchmarks.rocm.benchmark_rocm_e2e_movement import (
    _paged_module, _moe_module, _ResidentDescriptor, _event_ms,
)


def native_stats(lib, symbol, names):
    values = [ct.c_uint64() for _ in names]
    if getattr(lib, symbol)(*(ct.byref(value) for value in values)):
        raise RuntimeError("native statistics failed")
    return dict(zip(names, (value.value for value in values), strict=True))


def device_identity(hip, expected):
    active = ct.c_int()
    if hip.hipGetDevice(ct.byref(active)):
        raise RuntimeError("HIP active device query failed")
    if active.value != 0:
        raise RuntimeError("recorder requires the single owning device at ordinal zero")
    name, pci = ct.create_string_buffer(256), ct.create_string_buffer(256)
    hip.hipDeviceGetName.argtypes = [ct.c_void_p, ct.c_int, ct.c_int]
    hip.hipDeviceGetPCIBusId.argtypes = [ct.c_void_p, ct.c_int, ct.c_int]
    if hip.hipDeviceGetName(name, 256, active) or hip.hipDeviceGetPCIBusId(pci, 256, active):
        raise RuntimeError("HIP name/PCI query failed")
    live = rt._rocm_live_arch()
    if live != expected:
        raise RuntimeError(f"live architecture {live!r} disagrees with {expected!r}")
    return dict(name=name.value.decode(), pci_bus_id=pci.value.decode(),
                architecture=live, ordinal=active.value)


def run_case(hip, movement, image_cache, arch, family, shape, repeats, directory):
    rng = np.random.default_rng(120_509)
    begin = time.perf_counter_ns()
    if family == "paged_kv":
        p,page,h,d,start,tokens = shape
        module = _paged_module(*shape)
        package = rocm_native.package_paged_kv_read(module,
            pipeline_name="tessera-lower-to-rocm", architecture=arch)
        x = rng.normal(size=(p,page,h,d)).astype(np.float32)
        indices = rng.permutation(p).astype(np.int32)
        output = np.zeros((tokens,h,d),np.float32)
        expected = x[indices].reshape(p*page,h,d)[start:start+tokens]
        dimensions = (p,p,page,h,d,start,tokens)
        kwargs = dict(zip(("P","LP","PageSize","H","D","Start","Tokens"), dimensions, strict=True))
        kwargs.update(pages=x,page_table=indices,slice=output)
    else:
        t,s,h = shape
        module = _moe_module(*shape)
        package = rocm_native.package_moe_dispatch(module,
            pipeline_name="tessera-lower-to-rocm", architecture=arch)
        x = rng.normal(size=(t,h)).astype(np.float32)
        indices = rng.integers(0,t,size=s,dtype=np.int32)
        output = np.zeros((s,h),np.float32)
        expected = x[indices]
        dimensions = shape
        kwargs = dict(T=t,S=s,H=h,x=x,token=indices,o=output)
    compile_ms = (time.perf_counter_ns()-begin)/1e6
    artifact = rt.RuntimeArtifact(graph_ir=module.to_mlir(),
        tile_ir=package.tile_ir,target_ir=package.target_ir,
        metadata={"target":"rocm_"+arch,"compiler_path":"rocm_"+arch+"_native_descriptor"},
        native_image=package.image,launch_descriptor=package.descriptor)
    image_digest = package.image.image_digest
    def launch():
        result = rt.launch(artifact, kwargs)
        if not result.get("ok") or result.get("execution_kind") != "native_gpu":
            raise RuntimeError(f"checked launch failed: {result}")
    os.environ["TESSERA_ROCM_NATIVE_MOVEMENT"]="1"
    rt._clear_rocm_native_image_cache()
    resident = _ResidentDescriptor(hip,package,(x,indices),np.zeros_like(output),
                                   dimensions,output.size)
    try:
        resident.launch()
        np.testing.assert_array_equal(resident.read(),expected)
        events = [_event_ms(hip,resident,64) for _ in range(5)]
        resources = dict(block_threads=256,launch_blocks=resident.grid)
        for label,code in (("registers",4),("local_bytes",3),("shared_bytes",1)):
            value=ct.c_int()
            hip.hipFuncGetAttribute.argtypes=[ct.POINTER(ct.c_int),ct.c_int,ct.c_void_p]
            if hip.hipFuncGetAttribute(ct.byref(value),code,resident.function):
                raise RuntimeError("HIP resource query failed")
            resources[label]=value.value
        samples={name:[] for name in ("python","native_unpooled","native_pooled")}
        counters={name:[] for name in samples}
        cache_counters={name:[] for name in samples}
        for trial in range(9):
            names=tuple(samples)
            names=names[trial%3:]+names[:trial%3]
            for name in names:
                os.environ["TESSERA_ROCM_NATIVE_MOVEMENT"]="0" if name=="python" else "1"
                os.environ["TESSERA_ROCM_MOVEMENT_STAGING_REUSE"]="1" if name=="native_pooled" else "0"
                # Rewarm outside the measurement after changing allocation policy.
                for _ in range(2):
                    launch()
                np.testing.assert_array_equal(output,expected)
                before=native_stats(movement,"tessera_rocm_movement_stats",
                                    ("allocations","frees","reuses","launches"))
                image_before=native_stats(image_cache,"tessera_rocm_image_stats",
                                          ("loads","hits","function_lookups","unloads"))
                start=time.perf_counter_ns()
                for _ in range(repeats):
                    launch()
                elapsed=(time.perf_counter_ns()-start)/1e6/repeats
                after=native_stats(movement,"tessera_rocm_movement_stats",
                                   ("allocations","frees","reuses","launches"))
                image_after=native_stats(image_cache,"tessera_rocm_image_stats",
                                         ("loads","hits","function_lookups","unloads"))
                np.testing.assert_array_equal(output,expected)
                delta={key:after[key]-before[key] for key in before}
                if name=="native_pooled" and (delta["allocations"] or delta["frees"] or
                    delta["reuses"] != 3*repeats or delta["launches"] != repeats):
                    raise RuntimeError(f"warm native staging reuse not proved: {delta}")
                image_delta={key:image_after[key]-image_before[key] for key in image_before}
                if image_delta["loads"] or image_delta["function_lookups"]:
                    raise RuntimeError(f"unexpected per-call image reload: {image_delta}")
                samples[name].append(elapsed)
                counters[name].append(delta)
                cache_counters[name].append(image_delta)
        assert package.image.image_digest==image_digest
        medians={name:statistics.median(values) for name,values in samples.items()}
        label=family+"_"+"_".join(map(str,shape))
        (directory/(label+".graph.mlir")).write_text(module.to_mlir())
        (directory/(label+".tile.mlir")).write_text(package.tile_ir)
        (directory/(label+".target.mlir")).write_text(package.target_ir)
        (directory/(label+".hsaco")).write_bytes(package.image.payload)
        return dict(family=family,shape=list(shape),correctness="bit_exact_before_and_after_timing",
            image_digest=image_digest,payload_sha256=hashlib.sha256(package.image.payload).hexdigest(),
            schedule_digest=package.descriptor.provenance["schedule_digest"],abi_id=package.descriptor.abi_id,
            compile_package_wall_ms=compile_ms,resident_dispatch_samples_ms=events,
            resident_dispatch_median_ms=statistics.median(events),resources=resources,
            host_wall_samples_ms=samples,host_wall_medians_ms=medians,
            python_over_native_pooled=medians["python"]/medians["native_pooled"],
            native_unpooled_over_pooled=medians["native_unpooled"]/medians["native_pooled"],
            staging_counter_deltas=counters,image_cache_counter_deltas=cache_counters)
    finally:
        os.environ["TESSERA_ROCM_NATIVE_MOVEMENT"]="1"
        rt._clear_rocm_native_image_cache()
        resident.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture",choices=("gfx1151","gfx1201"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=10)
    args=parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    hip=rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0) or hip.hipDeviceSynchronize():
        raise RuntimeError("HIP device initialization failed")
    identity=device_identity(hip,args.architecture)
    os.environ["TESSERA_ROCM_NATIVE_MOVEMENT"]="1"
    movement=rt._load_rocm_native_movement_runtime()
    image_cache=rt._load_rocm_native_image_runtime()
    if movement is None or image_cache is None:
        raise RuntimeError("native movement and image lease libraries are required")
    directory=args.output.parent/"artifacts"
    directory.mkdir(parents=True,exist_ok=True)
    rows=[]
    cases=[("paged_kv",shape) for shape in
        ((4,4,3,8,1,5),(32,16,4,64,3,31),(64,32,8,128,7,249))]
    if args.architecture=="gfx1151":
        cases += [("moe_dispatch",shape) for shape in ((7,9,13),(64,128,256),(512,768,1024))]
    for family,shape in cases:
        rows.append(run_case(hip,movement,image_cache,args.architecture,family,shape,
                             args.repeats,directory))
        print("verified",family,shape,flush=True)
    sources=("python/tessera/runtime.py",
        "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_movement_runtime.cpp",
        "src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/native_image_cache.cpp",
        "python/tessera/compiler/rocm_native.py",
        "benchmarks/rocm/benchmark_native_movement.py")
    result=dict(device=identity,architecture=args.architecture,rows=rows,repeats=args.repeats,
        timing_scope="nine rotating paired trials of checked host launch wall; native image cache enabled for all arms, warm native staging verified by counters; separate preloaded resident HIP-event dispatch windows include host launch gaps; no kernel speedup claim",
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        native_movement_library_sha256=hashlib.sha256(Path(movement._name).read_bytes()).hexdigest(),
        native_image_library_sha256=hashlib.sha256(Path(image_cache._name).read_bytes()).hexdigest(),
        source_sha256={file:hashlib.sha256(Path(file).read_bytes()).hexdigest() for file in sources})
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
