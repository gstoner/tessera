"""Public native_jvp and matched prepared/native-owned versus captured replay."""
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
import tessera as ts
from benchmarks.nvidia.benchmark_bias_attention_jvp import attention,causal_attention,reference
from tessera.compiler import native_attention_jvp_runtime as adapter
from tessera.compiler.native_jvp import NativeJVPArtifact
from tessera.runtime import RuntimeArtifact,launch

def run(row,inputs,artifacts,repetitions):
    values={n:inputs["primal_"+n] for n in ("bias","v","q","k")}
    directions={n:inputs["direction_"+n] for n in values}
    expected=inputs["expected"];primal=inputs["primal"]
    causal="_c1_" in row["case"];wrt=tuple(row["wrt"])
    independently_recomputed=reference(values,directions,causal)
    np.testing.assert_allclose(primal,independently_recomputed[0],atol=2e-9,rtol=2e-7)
    np.testing.assert_allclose(expected,independently_recomputed[1],atol=2e-9,rtol=2e-7)
    fn=ts.jit(target="nvidia_sm120",autodiff="forward",wrt=wrt)(
        causal_attention if causal else attention)
    begin=time.perf_counter()
    output=fn.native_jvp(**values,tangents=tuple(directions[n] for n in wrt))
    cold_ms=(time.perf_counter()-begin)*1e3
    for actual,oracle in zip(output,(primal,expected),strict=True):
        np.testing.assert_allclose(actual,oracle,atol=3e-5,rtol=3e-5)
    receipt=dict(fn.last_jvp_execution)
    assert receipt["family"]=="attention_checkpoint" and receipt["execution_kind"]=="native_gpu"
    assert receipt["compiler_path"]=="nvidia_sm120_jvp_compiled"
    package=list(fn._native_jvp_packages.values())[0]
    metadata=package.runtime_metadata()
    parent=NativeJVPArtifact(metadata["native_jvp"]);parent.validate()
    child=parent.contract["steps"][0]["child_metadata"];owner=adapter.prepared(child)
    args=tuple(values[n] for n in ("bias","v","q","k"))+tuple(directions[n] for n in wrt)
    artifact=RuntimeArtifact(metadata=metadata)
    def call():
        result=launch(artifact,args)
        if not result.get("ok") or result.get("execution_mode")!="cuda_runtime":
            raise RuntimeError(result)
        return result["output"]
    samples={"prepared":[],"unprepared":[]};events=[]
    retained=tuple(x.copy() for x in output)
    def forbidden(*a,**k):raise AssertionError("compiler subprocess in warm native_jvp")
    with patch("subprocess.run",forbidden),patch("subprocess.Popen",forbidden),patch("subprocess.check_output",forbidden):
        for multiplier in (2.,1.,-1.):
            p,t=fn.native_jvp(**values,tangents=tuple(multiplier*directions[n] for n in wrt))
            np.testing.assert_allclose(p,primal,atol=3e-5,rtol=3e-5)
            np.testing.assert_allclose(t,multiplier*expected,atol=3e-5,rtol=3e-5)
        for iteration in range(repetitions):
            for name in (("prepared","unprepared") if iteration%2==0 else ("unprepared","prepared")):
                begin=time.perf_counter()
                if name=="unprepared":
                    with patch.object(adapter,"execute",adapter.execute_unprepared):actual=call()
                else:actual=call()
                samples[name].append((time.perf_counter()-begin)*1e3)
                if name=="prepared":events.append(owner.last_device_ms)
                for result,oracle in zip(actual,(primal,expected),strict=True):
                    np.testing.assert_allclose(result,oracle,atol=3e-5,rtol=3e-5)
                for result,prior in zip(output,retained,strict=True):np.testing.assert_array_equal(result,prior)
    medians={n:statistics.median(x) for n,x in samples.items()}
    (artifacts/(row["case"]+".json")).write_text(json.dumps(metadata,sort_keys=True))
    np.savez(artifacts/(row["case"]+".npz"),**{f"arg_{i}":v for i,v in enumerate(args)},
             expected_primal=primal,expected_tangent=expected)
    return dict(case=row["case"],wrt=wrt,bias_shape=row["bias_shape"],
        artifact_hash=parent.artifact_hash,program_digest=owner.program.program_digest,
        max_abs_error=float(np.max(np.abs(output[1]-expected))),cold_compile_launch_ms=cold_ms,
        matched_wall_samples_ms=samples,matched_wall_medians_ms=medians,
        prepared_over_unprepared=medians["prepared"]/medians["unprepared"],
        native_forward_tangent_event_samples_ms=events,
        native_forward_event_median_ms=statistics.median(x[0] for x in events),
        native_tangent_event_median_ms=statistics.median(x[1] for x in events),
        compiler_subprocesses_on_warm_call="forbidden",receipt=receipt)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--limit",type=int)
    ap.add_argument("--repetitions",type=int,default=5);args=ap.parse_args()
    if args.repetitions<3:raise ValueError("requires at least three alternating rounds")
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,driver_version","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or "RTX 5070" not in gpu or gpu.split(",")[2].strip()!="12.0":
        raise RuntimeError("owning RTX5070 / SM120 required")
    packet=json.loads(args.input.read_text())
    rows=packet["rows"] if args.limit is None else packet["rows"][:args.limit]
    artifacts=args.output.parent/"public-artifacts";artifacts.mkdir(parents=True,exist_ok=True)
    output=[]
    try:
        for row in rows:
            with np.load(args.input.parent/"artifacts"/(row["case"]+".npz")) as inputs:
                output.append(run(row,inputs,artifacts,args.repetitions))
                print("verified",row["case"],output[-1]["prepared_over_unprepared"],flush=True)
    finally:adapter.clear_prepared()
    files=("python/tessera/compiler/native_attention_jvp_runtime.py",
        "python/tessera/compiler/native_attention_program.py","python/tessera/compiler/native_attention_jvp_artifact.py",
        "python/tessera/compiler/native_jvp_plugins.py",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.h",
        "benchmarks/nvidia/benchmark_public_bias_attention_jvp.py")
    args.output.write_text(json.dumps(dict(device=gpu,architecture="sm_120",rows=output,
        fingerprints={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        runtime_library_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
        median_prepared_over_unprepared=statistics.median(r["prepared_over_unprepared"] for r in output),
        timing_scope="alternating common-runtime synchronous host calls including transfers; separate native CUDA forward/tangent windows; no GPU algorithm speedup claim"),indent=2)+"\n")
if __name__=="__main__":main()
