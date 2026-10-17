"""Pure AD-witness digest reuse A/B; identical native images and validation."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import json
import lzma
import os
from pathlib import Path
from statistics import median
import subprocess
import time

import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
import tessera.compiler.native_attention_rows as contracts
from benchmarks.nvidia.benchmark_jit_attention_vjp import download
from tests.device.nvidia.test_public_typed_attention_vjp import function
from tests.device.nvidia.test_saved_lse_half_backward_kernel import compare,reference

LABELS=("control","witness_memo")


def capture(label,program,roots):
    limit=contracts._WITNESS_CACHE_BYTE_LIMIT
    contracts._WITNESS_CACHE_BYTE_LIMIT=0 if label=="control" else 256*1024
    try:return program.capture(*roots,asynchronous=True)
    finally:contracts._WITNESS_CACHE_BYTE_LIMIT=limit


def inputs(dtype,sq,sk):
    rng=np.random.default_rng(202610171+sq*13+sk)
    shapes=((1,4,sq,32),(1,2,sk,32),(1,2,sk,16),(1,4,sq,16))
    values=tuple(rng.normal(0,.2,shape).astype(dtype) for shape in shapes)
    return (*values,rng.normal(0,.1,(1,4,sq)).astype(np.float32))


def check_outputs(session,outputs,expected):
    host=tuple(download(session,x) for x in outputs)
    for actual,wanted in zip(host,expected[:3],strict=True):compare(actual,wanted)
    return host


def run(output,samples,one_shot_samples):
    if rt._nvidia_device_name()!="sm_120":raise RuntimeError("owning SM120 required")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
    rows=[];programs=[]
    for storage,dtype in (("float16",np.float16),("bfloat16",ml_dtypes.bfloat16)):
        trace=inputs(dtype,1,4)
        fn=ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=("q","k","v"))(function(True,False,True))
        program=fn.compile_native_attention_vjp(*trace[:3],compiler=os.environ["TESSERA_OPT"],
            sequence_bounds=(33,129),compact_gradients=True,row_correction=True)
        text=program.to_json().encode();archive=f"{storage}.program.json.xz"
        (output/archive).write_bytes(lzma.compress(text))
        programs.append(dict(storage=storage,archive=archive,program_digest=program.program_digest,
            decompressed_sha256=hashlib.sha256(text).hexdigest()))
        for sq,sk in ((3,5),(17,129),(33,129)):
            values=inputs(dtype,sq,sk)
            primal,expected=reference(values[0],values[1],values[2],values[3],None,values[4])
            warm: dict[str,list[float]]={label:[] for label in LABELS}
            capture_host: dict[str,list[float]]={label:[] for label in LABELS}
            one_shot: dict[str,list[float]]={label:[] for label in LABELS}
            counts={}
            with ExitStack() as stack:
                session=stack.enter_context(NvidiaDeviceSession())
                roots=tuple(session.upload(x) for x in values[:3]);seeds=tuple(session.upload(x) for x in values[3:])
                frames={label:stack.enter_context(capture(label,program,roots)) for label in LABELS}
                prior={}
                for label,frame in frames.items():
                    frame.wait_on(session.stream)
                    for value,wanted in zip(frame.primal,primal,strict=True):compare(download(session,value),wanted)
                    for _ in range(3):
                        result=frame.backward(seeds);frame.synchronize()
                        prior[label]=check_outputs(session,result,expected)
                    # FFI allocation attribution is outside all timed windows.
                    tape=frame._frame;calls={"alloc":0,"free":0}
                    alloc,free=tape.alloc_async,tape._free_async
                    def count_alloc(*args):calls["alloc"]+=1;return alloc(*args)
                    def count_free(*args):calls["free"]+=1;return free(*args)
                    tape.alloc_async=count_alloc;tape._free_async=count_free
                    try:result=frame.backward(seeds);frame.synchronize()
                    finally:tape.alloc_async=alloc;tape._free_async=free
                    check_outputs(session,result,expected);counts[label]=dict(calls)
                for a,b in zip(prior["control"],prior["witness_memo"],strict=True):np.testing.assert_array_equal(a,b)
                for index in range(samples):
                    order=("control","witness_memo") if index%2==0 else ("witness_memo","control")
                    for label in order:
                        start=time.perf_counter_ns();result=frames[label].backward(seeds);frames[label].synchronize()
                        warm[label].append((time.perf_counter_ns()-start)/1e6)
                        prior[label]=check_outputs(session,result,expected)
                for a,b in zip(prior["control"],prior["witness_memo"],strict=True):np.testing.assert_array_equal(a,b)
                # Include the shifted allocation cost, rather than hiding it.
                for index in range(one_shot_samples):
                    order=("control","witness_memo") if index%2==0 else ("witness_memo","control")
                    for label in order:
                        start=time.perf_counter_ns();frame=capture(label,program,roots);frame.synchronize()
                        capture_elapsed=(time.perf_counter_ns()-start)/1e6
                        try:result=frame.backward(seeds);frame.synchronize();elapsed=(time.perf_counter_ns()-start)/1e6
                        except BaseException:frame.close();raise
                        capture_host[label].append(capture_elapsed);one_shot[label].append(elapsed)
                        check_outputs(session,result,expected);frame.close()
            row=dict(storage=storage,shape=[1,4,2,sq,sk,32,16],program_digest=program.program_digest,
                image_sha256=[hashlib.sha256(member.image.payload).hexdigest() for member in program.members],
                allocation_calls_per_reverse=counts,warm_reverse_ms=warm,capture_ms=capture_host,
                capture_plus_first_reverse_ms=one_shot,
                warm_reverse_medians_ms={label:median(x) for label,x in warm.items()},
                capture_medians_ms={label:median(x) for label,x in capture_host.items()},
                one_shot_medians_ms={label:median(x) for label,x in one_shot.items()},
                warm_ratio_control_over_witness_memo=median(warm["control"])/median(warm["witness_memo"]),
                one_shot_ratio_control_over_witness_memo=median(one_shot["control"])/median(one_shot["witness_memo"]))
            rows.append(row);print(json.dumps({key:row[key] for key in ("storage","shape","allocation_calls_per_reverse","warm_reverse_medians_ms","one_shot_medians_ms","warm_ratio_control_over_witness_memo","one_shot_ratio_control_over_witness_memo")}),flush=True)
    packet=dict(rows=rows,programs=programs,device=gpu,samples=samples,one_shot_samples=one_shot_samples,
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tool_sha256={name:hashlib.sha256(Path(os.environ[name]).read_bytes()).hexdigest() for name in ("TESSERA_OPT","TESSERA_NVIDIA_OPT","TESSERA_NVIDIA_PTX_LAUNCH_LIB")},
        control_source_sha256=hashlib.sha256(Path(contracts.__file__).read_bytes()).hexdigest(),
        control_mode="zero witness-cache byte budget; all validation still executes",
        candidate_mode="default bounded pure-digest cache; identical parser and native images",
        source_revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        correctness="same program and native images; independent FP64 before/after every measured call; bit-identical control/candidate outputs",
        scopes=dict(warm="completed backward on retained capture; uploads/oracle/readback excluded",one_shot="completed capture plus first backward; external upload, readback and close excluded",allocation_counts="separate untimed diagnostic call"),
        promotion=False,cache_info=contracts._WITNESS_DIGEST_CACHE.cache_info()._asdict())
    (output/"timings.json").write_text(json.dumps(packet,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=30);parser.add_argument("--one-shot-samples",type=int,default=10)
    args=parser.parse_args();run(args.output,args.samples,args.one_shot_samples)
