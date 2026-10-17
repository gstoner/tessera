"""Correctness-gated prepared typed JVP characterization; no default promotion."""
import argparse
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
from tessera.compiler.native_attention_jvp_runtime import prepared
from tests.device.nvidia.test_public_typed_attention_jvp import reference
from tests.device.nvidia.test_public_typed_attention_vjp import function
from tests.device.nvidia.test_saved_lse_half_backward_kernel import compare

def record(output: Path, repetitions: int = 9):
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be positive")
    device = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"], text=True).strip()
    if len(device.splitlines()) != 1 or not device.endswith(", 12.0"):
        raise RuntimeError("exact single SM120 owning device required")
    rows=[]
    for sq,sk in [(3,5),(17,129)]:
        for half in ["float16","bfloat16"]:
            half_dtype=np.dtype(ml_dtypes.bfloat16 if half=="bfloat16" else half)
            rng=np.random.default_rng(20261017)
            common={n:rng.normal(0,.3,shape).astype(half_dtype) for n,shape in {
                "q":(1,2,sq,16),"k":(1,1,sk,16),"v":(1,1,sk,16)}.items()}
            directions={n:rng.normal(0,.1,x.shape).astype(half_dtype) for n,x in common.items()}
            for wide in [False,True]:
                values={n:x.astype(np.float32) if wide else x for n,x in common.items()}
                seeds=tuple(x.astype(np.float32) if wide else x for x in directions.values())
                fn=ts.jit(target="nvidia_sm120",autodiff="forward",wrt=("q","k","v"))(function(True,False,True))
                try:
                    expected=reference(*values.values(),*seeds)
                    actual=fn.native_jvp(**values,tangents=seeds)
                    for got,wanted in zip(actual,expected,strict=True):
                        for a,b in zip(got,wanted,strict=True):compare(a,b)
                    package=next(iter(fn._native_jvp_packages.values()))
                    metadata=package.contract["steps"][0]["child_metadata"]
                    owner=prepared(metadata)
                    artifact_path=output.parent/"artifacts"/(metadata["program_digest"]+".json.xz")
                    artifact_path.parent.mkdir(parents=True,exist_ok=True)
                    artifact_bytes=lzma.compress(metadata["program_json"].encode())
                    artifact_path.write_bytes(artifact_bytes)
                    for _ in range(3):fn.native_jvp(**values,tangents=seeds)
                    e2e=[];device=[]
                    for _ in range(repetitions):
                        start=time.perf_counter()
                        last=fn.native_jvp(**values,tangents=seeds)
                        e2e.append(1000*(time.perf_counter()-start))
                        device.append(list(owner.last_device_ms))
                    for got,wanted in zip(last,expected,strict=True):
                        for a,b in zip(got,wanted,strict=True):compare(a,b)
                    row=dict(shape=[1,2,1,sq,sk,16,16],source_quantization=half,
                        storage="float32" if wide else half,saved_lse=True,
                        correctness="independent_fp64_before_timing",
                        program_digest=metadata["program_digest"],
                        program_artifact=str(artifact_path.relative_to(output.parent)),
                        program_artifact_sha256=hashlib.sha256(artifact_bytes).hexdigest(),
                        forward_image_sha256=hashlib.sha256(owner.program.pair.forward.image.payload).hexdigest(),
                        tangent_image_sha256=hashlib.sha256(owner.program.tangent.image).hexdigest(),
                        host_sizer_sha256=hashlib.sha256(owner.program.tangent.host_library).hexdigest(),
                        forward_entry=owner.program.pair.forward.descriptor.entry_symbol,
                        tangent_entry=owner.program.tangent.entry,
                        completed_public_host_samples_ms=e2e,completed_public_host_median_ms=median(e2e),
                        native_forward_jvp_event_samples_ms=device,
                        native_forward_event_median_ms=median(x[0] for x in device),
                        native_jvp_event_median_ms=median(x[1] for x in device))
                    rows.append(row);print(json.dumps(row),flush=True)
                finally:fn.close_native_storage()
    root=Path.cwd()
    paths=[
        "benchmarks/nvidia/benchmark_typed_attention_jvp.py",
        "python/tessera/compiler/native_attention_jvp_runtime.py",
        "python/tessera/compiler/native_attention_jvp.py",
        "python/tessera/compiler/native_attention_jvp_artifact.py",
        "python/tessera/compiler/native_attention_program.py",
        "python/tessera/compiler/native_gpu_tensor.py",
        "python/tessera/compiler/native_jvp_plugins.py",
        "python/tessera/compiler/jit.py",
        "python/tessera/compiler/resident_attention.py",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp",
        "src/compiler/programming_model/lib/NativeAttentionJvp.h",
        "src/compiler/ir/TangentInterface.cpp",
        "src/compiler/ir/include/Tessera/IR/AttentionTangentZero.h",
        "src/compiler/tile_opt_fa4/lib/Dialect/Attn/AttnOps.cpp",
        "src/transforms/lib/AutodiffForwardPass.cpp",
    ]
    result=dict(device=subprocess.check_output(["nvidia-smi","--query-gpu=name,compute_cap","--format=csv,noheader"],text=True).strip(),
        source_revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        source_dirty=bool(subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],text=True)),
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        runtime_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        scope="static saved-LSE, causal, GQA2, no bias; quantized-common-value f32 controls; compile, initial preparation and oracle excluded; public calls include validation/upload/readback; CUDA event dispatch windows reported separately, not hardware-counter isolated kernel proof; no default or speedup promotion",
        rows=rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repetitions",type=int,default=9)
    args=parser.parse_args()
    record(args.output,args.repetitions)


if __name__=="__main__":
    main()
