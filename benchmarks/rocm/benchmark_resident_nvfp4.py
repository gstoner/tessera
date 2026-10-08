#!/usr/bin/env python3
"""Exact gfx1201 resident ingest chain; source quality and timing stay separate."""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time

import ml_dtypes
import numpy as np

from tessera import runtime as rt
from tessera.compiler.rocm_nvfp4_ingest import (
    nvfp4_requantization_policy,reference_nvfp4_requantize,
)
from tessera.compiler.rocm_mxfp4_storage import (
    MXFP4_STORAGE_CONTRACT,reference_mxfp4_folded_storage,
)
from tessera.compiler.rocm_mxfp4_folded import prepare_folded_weights
from tessera.compiler.rocm_nvfp4_resident import NVFP4ResidentProgram,package_resident_nvfp4_matmul
from benchmarks.rocm import benchmark_gfx1201_three_formats as formats
from benchmarks.rocm import benchmark_rocm_nvfp4_checkpoint as checkpoint

ROOT=Path(__file__).resolve().parents[2]


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _source_quality(output,weights_nk,decoded_activation,source_base,source_activation):
    if weights_nk.shape!=source_base.shape or source_activation.shape!=decoded_activation.shape:
        raise ValueError("quality comparison requires matched source N/K weights and M/K activations")
    original=source_activation.astype(np.float64) @ source_base.astype(np.float64).T
    dequantized=decoded_activation @ source_base.astype(np.float64).T
    return {
        "folded_weight_error":formats.errors(weights_nk,source_base),
        "native_output_vs_source":formats.errors(output.astype(np.float64),original),
        "native_output_vs_dequantized_activation_bf16_weights":formats.errors(output.astype(np.float64),dequantized),
        "source_activation_sha256":sha(source_activation),"source_weight_sha256":sha(source_base),
        "activation_source":"same seeded synthetic f32 input as format controls; not model activations",
    }


def record_case(codes,scales,globals_,offsets,a,a_scale,*,windows,source_base=None,source_activation=None,jit=False,runtime_mn=True):
    n,half_k=codes.shape
    k,m=half_k*2,a.shape[0]
    start=time.perf_counter_ns()
    converted=reference_nvfp4_requantize(codes,scales,globals_,
        row_offsets=offsets,numeric_policy=nvfp4_requantization_policy())
    stored=reference_mxfp4_folded_storage(*converted[:2],storage_contract=MXFP4_STORAGE_CONTRACT)
    folded=prepare_folded_weights(*converted[:2],allow_approximate=True)
    # Decode the oracle to f64 independently of the native integer table.
    da=a.view(ml_dtypes.float8_e4m3fn).astype(np.float64)*a_scale.astype(np.float64)[:,None]
    db=(folded.weight_bytes.view(ml_dtypes.float8_e4m3fn).astype(np.float64)*
        np.where(folded.row_reference==0,0.,np.exp2(folded.row_reference.astype(np.int16)-127))[:,None])
    ideal=da @ db.T
    abs_product=np.abs(da) @ np.abs(db).T
    oracle_ms=(time.perf_counter_ns()-start)/1e6
    start=time.perf_counter_ns()
    program=package_resident_nvfp4_matmul(m,n,k,offsets,
        numeric_policy=nvfp4_requantization_policy(),approximate_policy="explicit_allow",runtime_mn=runtime_mn)
    compile_ms=(time.perf_counter_ns()-start)/1e6

    with program.session(codes,scales,globals_,a,a_scale) as session:
        session.run_combined()
        output=session.read_output()
        correctness=formats.verify(output,ideal,abs_product,k)
        diagnostics=session.diagnostics()
        for name,wanted in zip(("packed","exponents","stats"),converted):
            if name=="stats":
                np.testing.assert_allclose(diagnostics[name],wanted,rtol=1e-12,atol=1e-30)
            else:
                np.testing.assert_array_equal(diagnostics[name],wanted)
        np.testing.assert_array_equal(diagnostics["fragment"],stored[0])
        np.testing.assert_array_equal(diagnostics["plane"],stored[1])
        source_signal=float(diagnostics["stats"][...,0].sum())
        conversion_sse=float(diagnostics["stats"][...,1].sum())
        events,graphs={},{}
        stages=("converter","storage","consumer","combined")
        repeats_by_stage={}
        for stage in stages:
            events[stage]=session.measure(stage,samples=windows,repeats=3)
            probe=session.measure_graph(stage,samples=1,repeats=1)[0]
            repeats_by_stage[stage]=max(1,min(512,int(3/max(probe["window_ms"],1e-6))))
            session.measure_graph(stage,samples=1,repeats=repeats_by_stage[stage])
            graphs[stage]=[]
        for trial in range(windows):
            order=list(stages[trial%len(stages):]+stages[:trial%len(stages)])
            if trial%2:
                order.reverse()
            for stage in order:
                graphs[stage].extend(session.measure_graph(stage,samples=1,
                    repeats=repeats_by_stage[stage]))
        formats.verify(session.read_output(),ideal,abs_product,k)
        # Weights remain resident; these walls include new activation upload and output readback.
        reuse_wall=[]
        for _ in range(windows):
            start=time.perf_counter_ns()
            session.update_activations(a,a_scale)
            session.launch_matmul()
            result=session.read_output()
            reuse_wall.append((time.perf_counter_ns()-start)/1e6)
            formats.verify(result,ideal,abs_product,k)
        assert len(session._buffers)==11 and len(set(p.value for p in session._buffers.values()))==11
    # Full checked walls include modules, all uploads, allocations, three stages and result readback.
    combined_wall=[]
    for _ in range(windows):
        start=time.perf_counter_ns()
        with program.session(codes,scales,globals_,a,a_scale) as active:
            active.run_combined()
            result=active.read_output()
        combined_wall.append((time.perf_counter_ns()-start)/1e6)
        formats.verify(result,ideal,abs_product,k)
    encoded=program.to_json()
    restored_samples=[]
    replay_wall=[]
    for _ in range(windows):
        start=time.perf_counter_ns()
        replay=NVFP4ResidentProgram.from_json(encoded)
        restored_samples.append((time.perf_counter_ns()-start)/1e6)
        assert replay.receipt==program.receipt
        start=time.perf_counter_ns()
        replay=NVFP4ResidentProgram.from_json(encoded)
        with replay.session(codes,scales,globals_,a,a_scale) as active:
            active.run_combined()
            result=active.read_output()
        replay_wall.append((time.perf_counter_ns()-start)/1e6)
        formats.verify(result,ideal,abs_product,k)
    row={
        "shape_mnk":[m,n,k],"row_offsets":offsets,
        "input_sha256":dict(zip(("codes","scales","globals","a","a_scale"),
            map(sha,(codes,scales,globals_,a,a_scale)))),
        "compiler_receipt":program.receipt,"host_oracle_preparation_ms":oracle_ms,
        "package_compile_ms":compile_ms,"correctness":correctness,
        "conversion_correctness":"bitwise codes/exponents and f64 SSE; lossless bitwise storage bridge",
        "conversion_signal_f64":source_signal,"conversion_sse_f64":conversion_sse,
        "resident_event_samples_ms":events,
        "resident_event_median_ms":{stage:median(values) for stage,values in events.items()},
        "graph_windows":graphs,"graph_window_order":"rotate and reverse stages per trial",
        "graph_per_iteration_median_ms":{stage:median(v["per_iteration_ms"] for v in values)
            for stage,values in graphs.items()},
        "checked_combined_wall_samples_ms":combined_wall,"checked_combined_wall_median_ms":median(combined_wall),
        "resident_weight_reuse_wall_samples_ms":reuse_wall,"resident_weight_reuse_wall_median_ms":median(reuse_wall),
        "converted_weights_readback":"diagnostic/oracle validation only; never reuploaded into the consumer",
        "input_host_snapshot":"private immutable uploads; activation updates retain staging until synchronization",
        "portable_program":{
            "schema":program.to_dict()["schema"],"program_digest":program.to_dict()["program_digest"],
            "serialized_bytes":len(encoded.encode()),
            "restore_validate_samples_ms":restored_samples,
            "restore_validate_median_ms":median(restored_samples),
            "restore_and_checked_combined_wall_samples_ms":replay_wall,
            "restore_and_checked_combined_wall_median_ms":median(replay_wall),
            "compiler_required_for_replay":False,
            "timing_scope":"JSON decode and validation; full wall also owns modules, allocations, uploads, three kernels, readback and cleanup"},
        "allocation_count":11,
    }
    if jit:
        from benchmarks.rocm.benchmark_jit_nvfp4_program import record_arrays
        row["ordinary_jit"]=record_arrays((codes,scales,globals_,a,a_scale),ideal,
            reordered=False,windows=windows,converted=converted,stored=stored,row_offsets=offsets,
            verify=lambda output:formats.verify(output,ideal,abs_product,k))
    if source_base is not None:
        if source_activation is None:
            raise ValueError("checkpoint quality requires original matched activations")
        row["source_quality"]=_source_quality(output,db,da,source_base,source_activation)
    return row


def record(args):
    if args.windows<3:
        raise ValueError("at least three independent windows required")
    os.environ["TESSERA_OPT"]=str(args.compiler.resolve())
    if rt._rocm_live_arch()!="gfx1201":
        raise RuntimeError("requires exact live gfx1201")
    hip=rt._load_hip_for_launch()
    if hip is None or hip.hipInit(0):
        raise RuntimeError("HIP runtime unavailable")
    hip.hipDeviceGetName.argtypes=[C.c_char_p,C.c_int,C.c_int]
    hip.hipGetDevice.argtypes=[C.POINTER(C.c_int)]
    ordinal,name=C.c_int(),C.create_string_buffer(256)
    if hip.hipGetDevice(C.byref(ordinal)) or hip.hipDeviceGetName(name,len(name),ordinal.value):
        raise RuntimeError("HIP device identity probe failed")
    if not name.value or name.value.lower()==b"unknown":
        raise RuntimeError("device name required")
    packet={
        "schema":"tessera.rocm.nvfp4_resident_chain.v1","architecture":"gfx1201",
        "device":name.value.decode(),"device_ordinal":ordinal.value,
        "compiler_sha256":hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        "selector_promotion":False,
        "timing_scope":"event loops include host submission gaps; graphs use one submission per window and include GPU graph dispatch; checked walls include uploads/readback",
        "format_gates":"FP8/MXFP8/MXFP4 remain independent correctness/quality/performance gates",
        "scope":"static explicit three-package chain; ordinary composed JIT/general AD/dynamic producer graphs remain open",
        "rows":[],
    }
    sources=("python/tessera/compiler/rocm_nvfp4_program.py",
        "python/tessera/compiler/graph_ir.py","python/tessera/compiler/op_catalog.py",
        "python/tessera/compiler/jit.py","python/tessera/runtime.py",
        "benchmarks/rocm/benchmark_jit_nvfp4_program.py",
        "src/compiler/programming_model/lib/PMPasses.cpp",
        "python/tessera/compiler/rocm_nvfp4_resident.py",
        "python/tessera/compiler/rocm_nvfp4_ingest_native.py",
        "python/tessera/compiler/rocm_mxfp4_storage_native.py",
        "python/tessera/compiler/rocm_mxfp4_packed_folded.py",
        "python/tessera/compiler/rocm_mxfp4_storage.py",
        "benchmarks/rocm/benchmark_resident_nvfp4.py",
        "benchmarks/rocm/benchmark_gfx1201_three_formats.py")
    packet["source_sha256"]={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in sources}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.checkpoint:
        loaded=[]
        for tensor in ("model.layers.0.mlp.gate_proj.weight","model.layers.0.mlp.up_proj.weight"):
            print("loading pinned tensor",tensor,flush=True)
            loaded.append(checkpoint._load_projection(tensor))
            print("loaded",tensor,loaded[-1]["projection"].packed_codes.shape,flush=True)
        packet["source_checkpoints"]=[item["source"] for item in loaded]
        base=np.concatenate([np.asarray(item["bf16"],np.float32) for item in loaded])
        projections=[item["projection"] for item in loaded]
        codes=np.concatenate([p.packed_codes for p in projections])
        scales=np.concatenate([p.e4m3_scales for p in projections])
        globals_=np.array([p.global_scale for p in projections],np.float64)
        offsets=[0,projections[0].packed_codes.shape[0],codes.shape[0]]
        del loaded,projections
        for m in args.m:
            values=np.random.default_rng(271+m).normal(size=(m,base.shape[1])).astype(np.float32)
            a_scale=np.maximum(np.max(np.abs(values),axis=1)/448,np.finfo(np.float32).tiny).astype(np.float32)
            a=np.ascontiguousarray((values/a_scale[:,None]).astype(ml_dtypes.float8_e4m3fn).view(np.uint8))
            print("resident checkpoint proof",m,codes.shape[0],a.shape[1],flush=True)
            row=record_case(codes,scales,globals_,offsets,a,a_scale,windows=args.windows,source_base=base,source_activation=values,jit=args.jit,runtime_mn=not args.static_consumer)
            packet["rows"].append(row)
            args.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+"\n")
            print("verified checkpoint",row["shape_mnk"],row["graph_per_iteration_median_ms"],flush=True)
        if args.format_controls:
            from benchmarks.rocm.record_gfx1201_mxfp4_folded_load_schedule import DeviceClock
            # Same pinned sources and seeded activations; never mix historical timings.
            clock=DeviceClock(hip,args.compiler,args.llvm_bin)
            try:
                packet["format_controls"]=[]
                for m in args.m:
                    values=np.random.default_rng(271+m).normal(size=(m,base.shape[1])).astype(np.float32)
                    control_args=argparse.Namespace(compiler=args.compiler,windows=args.windows,
                        output=args.output,include_native_packed=True)
                    controls=formats.run_operands(values,base.T,control_args,hip,clock)
                    packet["format_controls"].append(controls)
                    args.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+"\n")
                    print("verified FP8/MXFP8/MXFP4 controls",m,flush=True)
            finally:
                clock.close()
    else:
        for m,n,k in ((128,32,256),(257,80,1024),(256,64,64)):
            rng=np.random.default_rng(120105+m+n+k)
            codes=rng.integers(0,256,(n,k//2),np.uint8)
            scales=rng.choice(np.array([0,.03125,.125,1,2,6]),(n,k//16)).astype(ml_dtypes.float8_e4m3fn)
            scales[0,:]=0
            globals_=np.array([.5,2.],np.float64)
            a=rng.integers(-4,5,(m,k)).astype(ml_dtypes.float8_e4m3fn).view(np.uint8).copy()
            a_scale=np.exp2(rng.integers(-1,2,m)).astype(np.float32)
            row=record_case(codes,scales,globals_,[0,n//2,n],a,a_scale,windows=args.windows,jit=args.jit,runtime_mn=not args.static_consumer)
            packet["rows"].append(row)
            args.output.write_text(json.dumps(packet,indent=2,allow_nan=False)+"\n")
            print("verified synthetic",row["shape_mnk"],row["graph_per_iteration_median_ms"],flush=True)
    return packet


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--windows",type=int,default=3)
    parser.add_argument("--checkpoint",action="store_true")
    parser.add_argument("--jit",action="store_true")
    parser.add_argument("--static-consumer",action="store_true",help="retain static image for matched control")
    parser.add_argument("--m",type=int,nargs="+",default=[256])
    parser.add_argument("--format-controls",action="store_true")
    parser.add_argument("--llvm-bin",type=Path)
    args=parser.parse_args()
    if any(m<=64 for m in args.m) or (args.format_controls and (not args.checkpoint or args.llvm_bin is None)):
        parser.error("M>64; format controls require checkpoint and LLVM bin")
    record(args)
