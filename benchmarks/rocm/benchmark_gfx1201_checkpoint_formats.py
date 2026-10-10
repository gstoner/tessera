#!/usr/bin/env python3
"""Pinned gate/up weight comparison across compiler-owned FP8/MXFP8/MXFP4 arms."""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import time
import numpy as np
from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx,rocm_nvfp4_ingest as ingest
from benchmarks.rocm import benchmark_rocm_nvfp4_checkpoint as checkpoint
from benchmarks.rocm import benchmark_gfx1201_three_formats as formats
from benchmarks.rocm.benchmark_gfx1201_mxfp8_package import Resident
from benchmarks.rocm.record_gfx1201_mxfp4_folded_load_schedule import DeviceClock

ROOT=Path(__file__).resolve().parents[2]


def measure(args):
    if args.windows<3 or not args.m or any(m<=64 for m in args.m):
        raise ValueError("at least three timing windows and M>64 required")
    os.environ["TESSERA_OPT"]=str(args.compiler.resolve())
    if rt._rocm_live_arch()!="gfx1201":
        raise RuntimeError("checkpoint formats require exact live gfx1201")
    hip=rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):
        raise RuntimeError("HIP unavailable")
    ordinal,name=C.c_int(),C.create_string_buffer(256)
    Resident.check(hip.hipGetDevice(C.byref(ordinal)))
    Resident.check(hip.hipDeviceGetName(name,len(name),ordinal.value))
    tensors=("model.layers.0.mlp.gate_proj.weight","model.layers.0.mlp.up_proj.weight")
    loaded=[checkpoint._load_projection(tensor) for tensor in tensors]
    sources=[item["source"] for item in loaded]
    projections=[item["projection"] for item in loaded]
    base=np.concatenate([np.asarray(item["bf16"],np.float32) for item in loaded])
    # Keep source identities and independent globals; free BF16/raw duplication.
    del loaded
    started=time.perf_counter()
    ingested=ingest.ingest_nvfp4_projections(projections)
    ingest_ms=(time.perf_counter()-started)*1e3
    native_ingest=None
    if getattr(args,"native_ingest",False):
        from tessera.compiler.rocm_nvfp4_ingest_native import (
            build_nvfp4_ingest_graph,package_nvfp4_ingest_graph,execute_nvfp4_ingest)
        source_codes=np.ascontiguousarray(np.concatenate([p.packed_codes for p in projections]))
        source_scales=np.ascontiguousarray(np.concatenate([p.e4m3_scales for p in projections]))
        source_globals=np.array([p.global_scale for p in projections],np.float64)
        started=time.perf_counter()
        conversion=package_nvfp4_ingest_graph(build_nvfp4_ingest_graph(
            *ingested.shape,ingested.row_offsets,
            numeric_policy=ingest.nvfp4_requantization_policy()))
        native_compile_ms=(time.perf_counter()-started)*1e3
        events=[]
        converted=execute_nvfp4_ingest(conversion,source_codes,source_scales,
            source_globals,event_samples=events)
        def verify_conversion(output):
            np.testing.assert_array_equal(output[0],ingested.packed_codes)
            np.testing.assert_array_equal(output[1],ingested.scale_exponents)
            assert np.isfinite(output[2]).all()
        verify_conversion(converted)
        wall=[]
        for _ in range(3):
            started=time.perf_counter()
            converted=execute_nvfp4_ingest(conversion,source_codes,source_scales,source_globals)
            wall.append((time.perf_counter()-started)*1e3)
            verify_conversion(converted)
        # Independently sum checkpoint-domain signal/error in bounded row batches.
        signal=error=0.
        for i,(a,b) in enumerate(zip(ingested.row_offsets,ingested.row_offsets[1:])):
            for start in range(a,b,32):
                stop=min(b,start+32)
                source=ingest._E2M1[mx.unpack_e2m1_codes(source_codes[start:stop])].astype(np.float64)
                source*=np.repeat(source_scales[start:stop].astype(np.float64)*source_globals[i],16,axis=1)
                decoded=mx.exact_weights(mx.unpack_e2m1_codes(converted[0][start:stop]),
                    converted[1][:,start:stop]).astype(np.float64)
                signal+=float(np.square(source).sum())
                error+=float(np.square(source-decoded).sum())
        np.testing.assert_allclose(converted[2][...,0].sum(),signal,rtol=1e-12)
        np.testing.assert_allclose(converted[2][...,1].sum(),error,rtol=1e-12)
        native_ingest=dict(
            route=conversion.native.descriptor.provenance["route"],
            abi_id=conversion.native.descriptor.abi_id,compile_ms=native_compile_ms,
            schedule_digest=conversion.native.descriptor.provenance["schedule_digest"],
            image_sha256=conversion.native.image.payload_digest,
            device_event_samples_ms=events,device_event_median_ms=statistics.median(events),
            checked_wall_samples_ms=wall,checked_wall_median_ms=statistics.median(wall),
            source_signal_f64=signal,requantization_error_f64=error,
            correctness="bitwise packed codes/exponents and independent bounded f64 loss statistics",
            consumer_edge="checked host readback; consumer uploads converted native results",
            timing_scope="conversion resident events and checked conversion wall measured separately from consumer",
            selector_promotion=False)
        # The following packed consumers receive native-produced values.
        ingested=ingest.MXFP4IngestedWeights(converted[0],converted[1],
            ingested.projection_names,ingested.row_offsets,ingested.metadata)
        del source_codes,source_scales,source_globals,converted,conversion
    del projections
    started=time.perf_counter()
    direct_codes,direct_scales,direct_values=checkpoint._direct_bf16_to_mxfp4(base)
    direct_ms=(time.perf_counter()-started)*1e3
    direct_quality=checkpoint._relative_rms(base,direct_values)
    del direct_values
    ingested_values=mx.exact_weights(mx.unpack_e2m1_codes(ingested.packed_codes),ingested.scale_exponents)
    ingested_quality=checkpoint._relative_rms(base,ingested_values)
    del ingested_values
    packed=dict(mxfp4_nvfp4_ingested=(ingested.packed_codes,ingested.scale_exponents),
                mxfp4_direct_joint_sse=(mx.pack_e2m1_codes(direct_codes),direct_scales))
    del direct_codes
    packet=dict(schema="tessera.gfx1201.checkpoint_format_gate.v1",
        architecture="gfx1201",device=name.value.decode(),device_ordinal=ordinal.value,host=socket.gethostname(),
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        source_checkpoints=sources,projection_names=list(ingested.projection_names),
        projection_row_offsets=list(ingested.row_offsets),
        source_activation="seeded synthetic f32; not captured model activations",
        host_ingest_ms=ingest_ms,host_direct_mxfp4_ms=direct_ms,native_ingest=native_ingest,
        ingested_weight_error=ingested_quality,direct_joint_weight_error=direct_quality,
        ingest_numeric_policy=ingested.numeric_policy(),
        selector_promotion=False,public_dtype_promotion=False,
        scale_scope="standard E8M0 MXFP8; folded MXFP4 named legacy zero-block and approximate E4M3 expansion",
        timing_scope="resident graph device execution/dispatch; checked host staging/transfer/sync separately",
        rows=[])
    files=(__file__,formats.__file__,checkpoint.__file__,
        ROOT/"python/tessera/compiler/rocm_nvfp4_ingest.py",
        ROOT/"python/tessera/compiler/rocm_nvfp4_ingest_native.py",
        ROOT/"python/tessera/compiler/graph_ir.py",
        ROOT/"python/tessera/compiler/capabilities.py",
        ROOT/"src/compiler/ir/include/Tessera/IR/NVFP4IngestContract.h",
        ROOT/"src/compiler/programming_model/lib/NativeNVFP4Ingest.h",
        ROOT/"src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/NativeNVFP4Ingest.h",
        ROOT/"python/tessera/compiler/rocm_mxfp8_blockscale.py",
        ROOT/"python/tessera/compiler/rocm_fp8_blockscale.py",
        ROOT/"python/tessera/compiler/rocm_mxfp4.py",
        ROOT/"python/tessera/compiler/rocm_mxfp4_packed_folded.py",
        ROOT/"python/tessera/runtime.py",
        ROOT/"src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/ROCMFoldedW4A8Contract.h",
        ROOT/"python/tessera/compiler/rocm_mxfp4_folded_frontend.py",
        ROOT/"src/compiler/programming_model/lib/PMPasses.cpp",
        ROOT/"src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateWMMAGemmKernel.cpp",
        ROOT/"src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp")
    packet["source_sha256"]={str(Path(p).relative_to(ROOT)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    clock=DeviceClock(hip,args.compiler,args.llvm_bin)
    try:
        for m in args.m:
            a=np.random.default_rng(args.seed+m).normal(size=(m,base.shape[1])).astype(np.float32)
            row=formats.run_operands(a,base.T,args,hip,clock,packed_weights=packed)
            row["activation_sha256"]=hashlib.sha256(a.tobytes()).hexdigest()
            row["source_weight_sha256"]=hashlib.sha256(base.tobytes()).hexdigest()
            packet["rows"].append(row)
            args.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+"\n")
            print(json.dumps(dict(m=m,arms={name:dict(
                weight_relative_rms=arm["quantized_weight_error"]["relative_rms"],
                output_relative_rms=arm["native_output_error"]["relative_rms"],
                device_us=arm["device_execution_dispatch_median_ms"]*1e3,
                wall_ms=arm["end_to_end_median_ms"]) for name,arm in row["arms"].items()})),flush=True)
    finally:
        clock.close()
    return packet


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler",type=Path,required=True)
    parser.add_argument("--llvm-bin",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--m",type=int,nargs="+",default=[128,256])
    parser.add_argument("--include-native-packed",action="store_true")
    parser.add_argument("--native-ingest",action="store_true")
    parser.add_argument("--windows",type=int,default=3)
    parser.add_argument("--seed",type=int,default=271)
    measure(parser.parse_args())
