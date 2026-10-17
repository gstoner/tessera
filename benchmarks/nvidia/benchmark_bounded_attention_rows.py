"""Correctness-gated public bounded row-owner stage and completed-call timings."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time
import lzma

import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.native_attention_program import NativeAttentionVJPProgram
from tessera.compiler.prepared_attention_vjp import clear_prepared,prepared
from tests.device.nvidia.test_saved_lse_half_backward_kernel import compare,reference


def attention(q,k,v):
    return ts.ops.flash_attn(q,k,v,causal=True,lse_checkpoint="saved")


def operands(storage,sq,sk):
    rng=np.random.default_rng(20261017+sq*13+sk)
    shapes=((1,4,sq,32),(1,2,sk,32),(1,2,sk,16),(1,4,sq,16))
    values=tuple(rng.normal(0,.2,shape).astype(storage) for shape in shapes)
    return (*values,rng.normal(0,.1,(1,4,sq)).astype(np.float32))


def record(output: Path,samples: int):
    if samples<3:raise ValueError("at least three samples required")
    if rt._nvidia_device_name()!="sm_120":raise RuntimeError("owning SM120 device required")
    output.mkdir(parents=True,exist_ok=False)
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or gpu.split(",")[-1].strip()!="12.0":raise RuntimeError("one exact SM120 GPU required")
    rows=[];programs=[]
    try:
        for name,storage in (("float16",np.float16),("bfloat16",ml_dtypes.bfloat16)):
            fn=ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=("q","k","v"))(attention)
            trace=operands(storage,1,4)
            program=fn.compile_native_attention_vjp(*trace[:3],compiler=os.environ["TESSERA_OPT"],
                sequence_bounds=(33,129),compact_gradients=True,row_correction=True)
            program=NativeAttentionVJPProgram.from_json(program.to_json(),expected_digest=program.program_digest)
            archive=f"{name}-program.json.xz"
            raw=program.to_json().encode();(output/archive).write_bytes(lzma.compress(raw))
            programs.append(dict(storage=name,program_digest=program.program_digest,archive=archive,
                decompressed_sha256=hashlib.sha256(raw).hexdigest(),
                images=[member.image.image_digest for member in program.members]))
            metadata=dict(target="nvidia_sm120",compiler_path="nvidia_sm120_attention_vjp_compiled",
                execution_kind="native_gpu",execution_mode="cuda_runtime",executable=True,
                arg_names=["primal_0","primal_1","primal_2","cotangent","lse_cotangent"],
                program_json=program.to_json(),program_digest=program.program_digest)
            artifact=rt.RuntimeArtifact(graph_ir=fn.last_traced_graph_ir if hasattr(fn,"last_traced_graph_ir") else "",
                target_ir=program.pair.backward.target_ir,metadata=metadata)
            cases=[]
            for sq,sk in ((3,5),(17,129),(33,129)):
                values=operands(storage,sq,sk)
                _,expected=reference(*values[:4],None,values[4])
                case=dict(storage=name,shape=[1,4,2,sq,sk,32,16],public_ms=[],stages_ms=[])
                for _ in range(3):
                    receipt=rt.launch(artifact,values)
                    if not receipt.get("ok"):raise RuntimeError(receipt)
                    for actual,wanted in zip(receipt["output"],expected[:3],strict=True):compare(actual,wanted)
                cases.append((case,values,expected))
            # Interleave concrete frames while retaining the same bounded images.
            for _ in range(samples):
                for case,values,expected in cases:
                    start=time.perf_counter_ns();receipt=rt.launch(artifact,values)
                    elapsed=(time.perf_counter_ns()-start)/1e6
                    if not receipt.get("ok") or receipt.get("execution_kind")!="native_gpu":raise RuntimeError(receipt)
                    owner=prepared(metadata)
                    if tuple(case["shape"])!=owner.dims:owner=prepared(metadata,runtime_dims=tuple(case["shape"]))
                    times=owner.last_device_ms
                    if times is None or len(times)!=4:raise RuntimeError("missing native row stage timing")
                    case["public_ms"].append(elapsed);case["stages_ms"].append(list(times))
                    for actual,wanted in zip(receipt["output"],expected[:3],strict=True):compare(actual,wanted)
            for case,_,_ in cases:
                case.update(public_median_ms=median(case["public_ms"]),
                    stage_medians_ms=[median(row[i] for row in case["stages_ms"]) for i in range(4)],
                    correctness="independent_fp64_checked_before_and_after_every_sample")
                rows.append(case)
    finally:clear_prepared()
    root=Path(__file__).resolve().parents[2]
    files=("python/tessera/compiler/native_attention_rows.py","python/tessera/compiler/resident_attention_rows.py",
           "python/tessera/compiler/prepared_attention_vjp.py","src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp")
    packet=dict(device=gpu,target="nvidia_sm120",samples=samples,programs=programs,rows=rows,
        measured_source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},
        tools={name:hashlib.sha256(Path(os.environ[name]).read_bytes()).hexdigest() for name in
               ("TESSERA_OPT","TESSERA_NVIDIA_OPT","TESSERA_NVIDIA_PTX_LAUNCH_LIB")},
        scope=dict(public="completed RuntimeArtifact launch including validation, snapshots, native stages and host output copies; compilation/oracle excluded",
            stages=["forward event window","row producer event window","backward event window","complete native forward-to-backward event interval"],
            limitations="synthetic compact QKV; no A/B speedup, isolated instruction latency, model quality or default selection claim"))
    (output/"timings.json").write_text(json.dumps(packet,indent=2)+"\n")
    return packet


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=10);args=parser.parse_args()
    result=record(args.output,args.samples)
    print("verified",len(result["rows"]),"bounded native row timing cases")
