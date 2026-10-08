"""Matched native reverse ownership comparison over identical pinned packages."""
from __future__ import annotations
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
from tessera.compiler import native_attention_vjp_runtime as adapter
from tessera.compiler.prepared_attention_vjp import prepared, clear_prepared
from tessera.runtime import RuntimeArtifact, launch, backend_capabilities
from benchmarks.nvidia.benchmark_public_attention_vjp import oracle

def run(file, repetitions):
    artifact = RuntimeArtifact.from_json(file.read_text())
    metadata = artifact.metadata
    owner = prepared(metadata)
    with np.load(file.with_suffix(".npz")) as data:
        values = tuple(data[n] for n in metadata["arg_names"])
    physical = {name: values[owner.program.input_indices[i]]
                for i, name in enumerate(("q", "k", "v", "bias")[:len(values)-1])}
    expected_by_name = oracle(physical, values[-1], owner.program.pair.forward.descriptor.provenance["causal"])
    expected = tuple(expected_by_name[("q", "k", "v", "bias")[i]] for i in owner.program.active)
    def call():
        result = launch(artifact, values)
        if not result.get("ok") or result.get("execution_kind") != "native_gpu":
            raise RuntimeError(f"native reverse failed: {result}")
        return tuple(result["output"])
    outputs = {}
    for name in ("prepared", "unprepared"):
        if name == "unprepared":
            with patch.object(adapter, "execute", adapter.execute_unprepared):
                outputs[name] = call()
        else:
            outputs[name] = call()
        for x, reference in zip(outputs[name], expected, strict=True):
            np.testing.assert_allclose(x, reference, atol=3e-5, rtol=3e-5)
    retained = tuple(x.copy() for x in outputs["prepared"])
    samples = {n: [] for n in outputs}
    events = []
    def forbidden(*a, **k):
        raise AssertionError("compiler/process access during matched warm reverse")
    with patch("subprocess.run", forbidden), patch("subprocess.Popen", forbidden), patch("subprocess.check_output", forbidden):
        for trial in range(repetitions):
            for name in (("prepared", "unprepared") if trial % 2 == 0 else ("unprepared", "prepared")):
                start = time.perf_counter()
                if name == "unprepared":
                    with patch.object(adapter, "execute", adapter.execute_unprepared):
                        result = call()
                else:
                    result = call()
                samples[name].append((time.perf_counter()-start)*1e3)
                if name == "prepared":
                    events.append(owner.last_device_ms)
                for x, reference in zip(result, expected, strict=True):
                    np.testing.assert_allclose(x, reference, atol=3e-5, rtol=3e-5)
                for x, prior in zip(outputs["prepared"], retained, strict=True):
                    np.testing.assert_array_equal(x, prior)
    medians = {name: statistics.median(samples[name]) for name in samples}
    return dict(case=file.stem, artifact_hash=artifact.artifact_hash, program_digest=metadata["program_digest"],
                correctness="passed_before_timing", max_abs_error=max(float(np.max(np.abs(x-y))) for x,y in zip(outputs["prepared"],expected,strict=True)),
                matched_common_runtime_samples_ms=samples, medians_ms=medians,
                prepared_over_unprepared=medians["prepared"]/medians["unprepared"],
                native_forward_backward_event_samples_ms=events,
                native_forward_event_median_ms=statistics.median(x[0] for x in events),
                native_backward_event_median_ms=statistics.median(x[1] for x in events),
                compiler_subprocesses="forbidden", requested_physical_roles=owner.program.active)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--artifacts",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repetitions",type=int,default=9)
    args=parser.parse_args()
    if args.repetitions<3:
        raise ValueError("requires at least three alternating trials")
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,driver_version","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or "RTX 5070" not in gpu or gpu.split(",")[2].strip()!="12.0":
        raise RuntimeError("requires owning RTX 5070 / SM120")
    backend_capabilities("nvidia_sm120")
    rows=[]
    try:
        for file in sorted(args.artifacts.glob("*.json")):
            rows.append(run(file,args.repetitions))
            print("verified",file.stem,rows[-1]["prepared_over_unprepared"],flush=True)
    finally:
        clear_prepared()
    if len(rows)!=88:
        raise ValueError("expected complete 88-case matched reverse packet")
    files=("python/tessera/compiler/native_attention_vjp_runtime.py",
           "python/tessera/compiler/prepared_attention_vjp.py",
           "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp",
           "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.h",
           "benchmarks/nvidia/benchmark_prepared_attention_vjp.py")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(device=gpu,architecture="sm120",rows=rows,
        fingerprints={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
        runtime_library_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
        timing_scope="alternating synchronous common-runtime host wall calls, including upload/download; independent CUDA forward/backward event windows",
        median_prepared_over_unprepared=statistics.median(r["prepared_over_unprepared"] for r in rows)),indent=2)+"\n")
if __name__=="__main__":main()
