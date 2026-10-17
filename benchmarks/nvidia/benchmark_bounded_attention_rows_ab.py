"""Matched inline versus native row residual, bounded resident programs."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import ctypes as ct
import hashlib
import json
import lzma
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import time
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.resident_attention import ResidentAttentionTape
from benchmarks.nvidia.benchmark_jit_attention_vjp import download
from benchmarks.nvidia.benchmark_typed_resident_attention_capture import GraphWindows
from tests.device.nvidia.test_public_typed_attention_vjp import function
from tests.device.nvidia.test_saved_lse_half_backward_kernel import compare,reference


def inputs(dtype,sq,sk):
    rng=np.random.default_rng(20261018+sq*13+sk)
    shapes=((1,4,sq,32),(1,2,sk,32),(1,2,sk,16),(1,4,sq,16))
    values=tuple(rng.normal(0,.2,shape).astype(dtype) for shape in shapes)
    return (*values,rng.normal(0,.1,(1,4,sq)).astype(np.float32))


def stage_clock(tape,dispatch,repeats,windows):
    # Reuse capture/events for the exact owned functions and pointers. This
    # diagnostic omits allocations and copies, which remain in host timings.
    proxy=SimpleNamespace(sync=tape.sync,check=tape.check,_stream=tape._stream,
        _execute=lambda _backward,_buffers:dispatch())
    return GraphWindows(proxy).measure(True,(),repeats=repeats,windows=windows)


def row_launch(tape,producer):
    arguments=[ct.c_void_p(value.pointer.value) for value in producer]+[ct.c_int64(x) for x in tape.dims]
    pointers=(ct.c_void_p*len(arguments))(*(ct.cast(ct.pointer(x),ct.c_void_p) for x in arguments))
    count=math.prod(tape.shapes[4])
    tape.check(tape._launch(tape._functions[2],(count+31)//32,1,1,32,1,1,0,tape._stream,pointers,None))


def run(output,samples=12,repeats=32,windows=3):
    if min(samples,repeats,windows)<3:raise ValueError("at least three samples, repeats and windows")
    if rt._nvidia_device_name()!="sm_120":raise RuntimeError("owning RTX5070 required")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    device=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(device.splitlines())!=1 or device.split(",")[-1].strip()!="12.0":raise RuntimeError("one exact SM120 GPU required")
    rows=[];pins=[]
    for storage,dtype in (("float16",np.float16),("bfloat16",ml_dtypes.bfloat16)):
        trace=inputs(dtype,1,4)
        programs={}
        for label,use_prepass in (("inline",False),("prepass",True)):
            fn=ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=("q","k","v"))(function(True,False,True))
            program=fn.compile_native_attention_vjp(*trace[:3],compiler=os.environ["TESSERA_OPT"],
                sequence_bounds=(33,129),compact_gradients=True,row_correction=use_prepass)
            programs[label]=program
            text=program.to_json().encode();name=f"{storage}-{label}.program.json.xz"
            (output/name).write_bytes(lzma.compress(text))
            pins.append(dict(storage=storage,arm=label,program_digest=program.program_digest,
                archive=name,decompressed_sha256=hashlib.sha256(text).hexdigest(),
                image_hashes=[x.image.image_digest for x in getattr(program,"members",(program.pair.forward,program.pair.backward))]))
        assert programs["inline"].pair.forward.image.payload==programs["prepass"].pair.forward.image.payload,"forward images differ"
        assert programs["inline"].pair.contract_digest==programs["prepass"].pair.contract_digest
        for sq,sk in ((3,5),(17,129),(33,129)):
            values=inputs(dtype,sq,sk)
            primal,expected=reference(values[0],values[1],values[2],values[3],None,values[4])
            with ExitStack() as stack:
                session=stack.enter_context(NvidiaDeviceSession())
                roots=tuple(session.upload(x) for x in values[:3])
                seeds=tuple(session.upload(x) for x in values[3:])
                frames={label:stack.enter_context(program.capture(*roots,asynchronous=True)) for label,program in programs.items()}
                held={}
                for label,frame in frames.items():
                    frame.wait_on(session.stream)
                    for actual,wanted in zip(frame.primal,primal,strict=True):compare(download(session,actual),wanted)
                    held[label]=frame.backward(seeds);frame.wait_on(session.stream)
                    for actual,wanted in zip(held[label],expected[:3],strict=True):compare(download(session,actual),wanted)
                for a,b in zip(held["inline"],held["prepass"],strict=True):np.testing.assert_array_equal(download(session,a),download(session,b))
                host: dict[str,list[float]]={label:[] for label in frames}
                for i in range(samples):
                    order=("inline","prepass") if i%2==0 else ("prepass","inline")
                    for label in order:
                        start=time.perf_counter_ns();actual=frames[label].backward(seeds);frames[label].synchronize()
                        elapsed=(time.perf_counter_ns()-start)/1e6
                        host[label].append(elapsed)
                        frames[label].wait_on(session.stream)
                        for value,wanted in zip(actual,expected[:3],strict=True):compare(download(session,value),wanted)
                stages={}
                for label,frame in frames.items():
                    tape=frame._frame
                    external=[SimpleNamespace(pointer=ct.c_void_p(value.ptr)) for value in seeds]
                    gradients=[value._buffer for value in held[label]]
                    forward=lambda t=tape:ResidentAttentionTape._execute(t,False,t._saved)
                    if label=="inline":
                        buffers=[external[0],*tape._saved[:4],tape._saved[4],external[1],*gradients]
                        dispatch=lambda t=tape,b=buffers:ResidentAttentionTape._execute(t,True,b)
                        stages[label]=dict(forward_ms=stage_clock(tape,forward,repeats,windows),
                            backward_ms=stage_clock(tape,dispatch,repeats,windows))
                    else:
                        scratch=tape._allocate(tape.shapes[4])
                        producer=[external[0],*tape._saved[:3],tape._saved[4],external[1],scratch]
                        consumer=[external[0],*tape._saved[:3],tape._saved[4],scratch,*gradients]
                        produce=lambda t=tape,b=producer:row_launch(t,b)
                        consume=lambda t=tape,b=consumer:ResidentAttentionTape._execute(t,True,b)
                        def pair():produce();consume()
                        produce()
                        stages[label]=dict(forward_ms=stage_clock(tape,forward,repeats,windows),
                            row_producer_ms=stage_clock(tape,produce,repeats,windows),
                            backward_consumer_ms=stage_clock(tape,consume,repeats,windows),
                            combined_reverse_ms=stage_clock(tape,pair,repeats,windows))
                    frame.wait_on(session.stream)
                    for value,wanted in zip(held[label],expected[:3],strict=True):compare(download(session,value),wanted)
                for a,b in zip(held["inline"],held["prepass"],strict=True):np.testing.assert_array_equal(download(session,a),download(session,b))
                medians={label:{key:median(vals) for key,vals in samples.items()} for label,samples in stages.items()}
                host_medians={label:median(vals) for label,vals in host.items()}
                row=dict(storage=storage,shape=[1,4,2,sq,sk,32,16],host_reverse_ms=host,
                    host_reverse_medians_ms=host_medians,dispatch_stages_ms=stages,dispatch_medians_ms=medians,
                    host_ratio_inline_over_prepass=host_medians["inline"]/host_medians["prepass"],
                    dispatch_ratio_inline_over_prepass=medians["inline"]["backward_ms"]/medians["prepass"]["combined_reverse_ms"],
                    correctness="independent_fp64_before_after; arms bit-identical before_after")
                rows.append(row);print(json.dumps(row),flush=True)
    packet=dict(device=device,rows=rows,programs=pins,samples=samples,repeats=repeats,windows=windows,
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
            "python/tessera/compiler/native_attention_rows.py",
            "python/tessera/compiler/resident_attention_rows.py",
            "python/tessera/compiler/resident_attention.py",
            "python/tessera/compiler/native_attention_program.py")},
        compiler_hashes={name:hashlib.sha256(Path(os.environ[name]).read_bytes()).hexdigest() for name in ("TESSERA_OPT","TESSERA_NVIDIA_OPT")},
        timing_scope=dict(host="resident completed backward with seed snapshots and fresh gradient allocations; forward, external upload/readback and oracle excluded",
            dispatch="CUDA graphs repeat exact owned native functions on preallocated buffers; separate producer and consumer; excludes allocations/copies; graph dispatch overhead included"),
        selector_promotion=False)
    (output/"timings.json").write_text(json.dumps(packet,indent=2)+"\n")
    return packet


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=12);args=parser.parse_args()
    run(args.output,args.samples)
