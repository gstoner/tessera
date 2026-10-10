"""Matched common-runtime A/B over the same pinned native attention images."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from unittest.mock import patch
import numpy as np
from tessera.compiler.native_attention_jvp_runtime import prepared,clear_prepared
from tessera.compiler import native_attention_jvp_runtime as adapter
from tessera.compiler.native_jvp import NativeJVPArtifact
from tessera.runtime import RuntimeArtifact,launch,backend_capabilities

def run(file,repetitions):
    raw=json.loads(file.read_text())
    parent=NativeJVPArtifact(raw["native_jvp"]);parent.validate()
    child=parent.contract["steps"][0]["child_metadata"]
    owner=prepared(child);program=owner.program
    policy=program.pair.forward.descriptor.provenance
    b,hq,hkv,sq,sk,d,dv=policy["shape"]
    rng=np.random.default_rng(915)
    xs=[rng.normal(size=s).astype(np.float32)*.2 for s in
        ((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv))]
    ts=[rng.normal(size=x.shape).astype(np.float32)*.1
        if i in program.active else np.zeros_like(x) for i,x in enumerate(xs)]
    def oracle(values):
        q,k,v=(x.astype(np.float64) for x in values)
        score=float(policy["scale"])*(q@np.swapaxes(np.repeat(k,hq//hkv,axis=1),-1,-2))
        if policy["causal"]:
            score=np.where(np.arange(sk)[None,:]<=np.arange(sq)[:,None]+max(sk-sq,0),score,-np.inf)
        p=np.exp(score-score.max(axis=-1,keepdims=True));p/=p.sum(axis=-1,keepdims=True)
        return p@np.repeat(v,hq//hkv,axis=1)
    h=1e-4
    primal=oracle(xs)
    expected=(oracle([x.astype(np.float64)+h*t for x,t in zip(xs,ts,strict=True)])-
              oracle([x.astype(np.float64)-h*t for x,t in zip(xs,ts,strict=True)]))/(2*h)
    frontend=[None]*3
    for i,index in enumerate(program.input_indices):frontend[index]=xs[i]
    values=tuple(frontend)+tuple(ts[i] for i in program.active)
    artifact=RuntimeArtifact(metadata=parent.runtime_metadata())
    def call():
        result=launch(artifact,values)
        if not result.get("ok") or result.get("execution_mode")!="cuda_runtime":
            raise RuntimeError(f"native product failed: {result}")
        return result["output"]
    outputs={}
    for name in ("prepared","unprepared"):
        if name=="unprepared":
            with patch.object(adapter,"execute",adapter.execute_unprepared):outputs[name]=call()
        else:outputs[name]=call()
        np.testing.assert_allclose(outputs[name][0],primal,atol=3e-5,rtol=3e-5)
        np.testing.assert_allclose(outputs[name][1],expected,atol=3e-5,rtol=3e-5)
    def forbidden(*a,**k):raise AssertionError("compiler subprocess during matched replay")
    samples={name:[] for name in outputs};events=[];retained=outputs["prepared"][1].copy()
    with patch("subprocess.run",forbidden),patch("subprocess.Popen",forbidden),patch("subprocess.check_output",forbidden):
        for i in range(repetitions):
            for name in (("prepared","unprepared") if i%2==0 else ("unprepared","prepared")):
                start=time.perf_counter()
                if name=="unprepared":
                    with patch.object(adapter,"execute",adapter.execute_unprepared):result=call()
                else:result=call()
                samples[name].append((time.perf_counter()-start)*1e3)
                if name=="prepared":events.append(owner.last_device_ms)
                np.testing.assert_allclose(result[0],primal,atol=3e-5,rtol=3e-5)
                np.testing.assert_allclose(result[1],expected,atol=3e-5,rtol=3e-5)
                np.testing.assert_array_equal(outputs["prepared"][1],retained)
    medians={name:statistics.median(values) for name,values in samples.items()}
    return dict(case=file.stem,artifact_hash=parent.artifact_hash,
        correctness="passed_before_timing",max_abs_error=float(np.max(np.abs(outputs["prepared"][1]-expected))),
        matched_common_runtime_samples_ms=samples,medians_ms=medians,
        prepared_over_unprepared=medians["prepared"]/medians["unprepared"],
        native_forward_tangent_event_samples_ms=events,
        native_forward_event_median_ms=statistics.median(x[0] for x in events),
        native_tangent_event_median_ms=statistics.median(x[1] for x in events),
        compiler_subprocesses="forbidden")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--artifacts",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--repetitions",type=int,default=7)
    args=ap.parse_args()
    if args.repetitions<3:raise ValueError("requires at least three alternating rounds")
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,driver_version","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or "RTX 5070" not in gpu or gpu.split(",")[2].strip()!="12.0":
        raise RuntimeError("requires owning RTX5070 / SM120")
    backend_capabilities("nvidia_sm120")
    rows=[]
    try:
        for file in sorted(args.artifacts.glob("*.json")):
            rows.append(run(file,args.repetitions));print("verified",file.stem,rows[-1]["prepared_over_unprepared"],flush=True)
    finally:clear_prepared()
    if len(rows)!=72:raise ValueError("expected complete 72-envelope matched packet")
    root=Path(__file__).resolve().parents[2]
    files=("python/tessera/compiler/native_attention_jvp_runtime.py",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.h",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/CMakeLists.txt",
        "benchmarks/nvidia/benchmark_prepared_attention_jvp.py")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(device=gpu,architecture="sm120",rows=rows,
        fingerprints={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
        runtime_library_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
        timing_scope="alternating synchronous common-runtime wall samples; separate native CUDA forward/JVP event windows",
        median_prepared_over_unprepared=statistics.median(x["prepared_over_unprepared"] for x in rows)),indent=2)+"\n")
    print("wrote",args.output,flush=True)
if __name__=="__main__":main()
