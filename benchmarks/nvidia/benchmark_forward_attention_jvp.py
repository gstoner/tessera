"""Matched compiler work attribution for saved-LSE JVP forward products."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
from unittest.mock import patch
from benchmarks.nvidia.benchmark_public_attention_jvp import run
from tessera.compiler import native_attention_program as native

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,driver_version","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or "RTX 5070" not in gpu or gpu.split(",")[2].strip()!="12.0":
        raise RuntimeError("requires owning RTX 5070 / SM120")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    artifacts=args.output.parent/"artifacts";artifacts.mkdir(exist_ok=True)
    original=native.compile_attention_program
    original_run=subprocess.run
    rows=[]
    for order,wrt,sk,causal in (
        (("q","k","v"),("q",),5,False),
        (("k","v","q"),("k","q"),5,False),
        (("v","q","k"),("v",),129,True),
        (("v","k","q"),("q","k","v"),129,True)):
        samples={False:[],True:[]}; counts={False:[],True:[]}; sizes={}; current=None
        def compile(source,active,**kwargs):
            nonlocal current
            for round_index in range(3):
                for retain in ((True,False) if round_index%2==0 else (False,True)):
                    calls=[]
                    def traced(*a,**k):
                        calls.append(a[0] if a else k.get("args"))
                        return original_run(*a,**k)
                    start=time.perf_counter()
                    with patch("subprocess.run",traced):
                        product=original(source,active,**kwargs,retain_reverse=retain)
                    samples[retain].append((time.perf_counter()-start)*1e3)
                    counts[retain].append(len(calls))
                    sizes[retain]=len(product.to_json().encode())
                    if retain:
                        control=product
                    else:
                        assert product.pair.forward.image.payload==control.pair.forward.image.payload
                        assert product.tangent.image==control.tangent.image
                        assert not hasattr(product.pair,"backward")
                        current=product
            return current
        with patch.object(native,"compile_attention_program",compile):
            public=run(order,wrt,sk,causal,artifacts)
        rows.append(dict(public=public,
            compile_samples_ms={"forward_only":samples[False],"paired":samples[True]},
            compiler_subprocess_counts={"forward_only":counts[False],"paired":counts[True]},
            program_bytes={"forward_only":sizes[False],"paired":sizes[True]},
            native_images_unchanged=True))
        print("verified",order,wrt,sk,causal,flush=True)
    paths=("python/tessera/compiler/native_attention_program.py",
        "python/tessera/compiler/native_attention_jvp_artifact.py",
        "python/tessera/compiler/nvidia_native.py","python/tessera/compiler/resident_attention.py",
        "benchmarks/nvidia/benchmark_forward_attention_jvp.py")
    packet=dict(device=gpu,rows=rows,timing_scope="alternating matched compilation wall; public execution oracle gate",
        fingerprints={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        median_forward_over_paired=statistics.median(statistics.median(r["compile_samples_ms"]["forward_only"])/
            statistics.median(r["compile_samples_ms"]["paired"]) for r in rows))
    args.output.write_text(json.dumps(packet,indent=2)+"\n")

if __name__=="__main__":
    main()
