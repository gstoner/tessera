"""Source-bound bounded SM120 producer-chain public/stage characterization."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from statistics import median

import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_bounded_producer_chain import inputs, oracle, two, three


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def intermediates(source, producers):
    storage = source.dtype
    value = source.astype(np.float64)
    results = []
    if producers == 3:
        centered = value-value.mean(axis=-1,keepdims=True)
        stored = (centered/np.sqrt(np.mean(centered*centered,axis=-1,keepdims=True)+1e-5)).astype(storage)
        results.append(stored)
        value = stored.astype(np.float64)
    stored = (value/np.sqrt(np.mean(value*value,axis=-1,keepdims=True)+1e-5)).astype(storage)
    results.append(stored)
    value = stored.astype(np.float64)
    exponential = np.exp(value-value.max(axis=-1,keepdims=True))
    results.append((exponential/exponential.sum(axis=-1,keepdims=True)).astype(storage))
    return results


def profile(dtype, producers):
    bounds = {"M":32,"N":24,"K":64}
    function = ts.jit(target="nvidia_sm120",shape_bounds=bounds)(two if producers==2 else three)
    first_values = inputs({"M":17,"N":11,"K":32},dtype)
    start = time.perf_counter()
    result = function(*first_values)
    first_call_ms = (time.perf_counter()-start)*1000
    np.testing.assert_allclose(result,oracle(first_values,producers),rtol=.015,atol=.002)
    packages = function.native_lhs_packages()
    program = function._nvidia_lhs_last_program
    owner = next(iter(function._nvidia_lhs_prepared_calls.values()))
    stats = owner.scratch_stats()
    rows = []
    try:
        for frame in ({"M":17,"N":11,"K":32}, bounds, {"M":1,"N":1,"K":1}):
            values = inputs(frame,dtype)
            wanted = oracle(values,producers)
            actual = function(*values)
            error = float(np.max(np.abs(actual.astype(np.float64)-wanted)))
            np.testing.assert_allclose(actual,wanted,rtol=.015,atol=.002)
            public = []
            original_run = subprocess.run
            def forbidden(*args,**kwargs):
                raise AssertionError("warm bounded-chain benchmark invoked a compiler subprocess")
            subprocess.run = forbidden
            try:
                for index in range(7):
                    changed = inputs(frame,dtype,seed=1270+index)
                    start = time.perf_counter()
                    actual = function(*changed)
                    public.append((time.perf_counter()-start)*1000)
                    expected = oracle(changed,producers)
                    np.testing.assert_allclose(actual,expected,rtol=.015,atol=.002)
                    error = max(error,float(np.max(np.abs(actual.astype(np.float64)-expected))))
                    assert function._nvidia_lhs_last_program is program
                    assert owner.scratch_stats() == stats
            finally:
                subprocess.run = original_run
            produced = intermediates(values[0],producers)
            stages = []
            session = NvidiaDeviceSession()
            try:
                for index,package in enumerate(packages):
                    host = values[0] if index==0 else produced[index-1]
                    expected = produced[index] if index<producers else wanted
                    layouts = {binding.name:binding.layout for binding in package.descriptor.buffers}
                    source = session.upload(host,layout=layouts["source" if index<producers else "edge"])
                    output = session.empty(expected.shape,host.dtype if index<producers else np.float32,
                                           layout=layouts["edge" if index<producers else "out"])
                    m,n,k = (frame[axis] for axis in ("M","N","K"))
                    if index<producers:
                        args = {"source":source,"edge":output,"Rows":m,
                                "K" if package.descriptor.provenance["kind"]=="softmax" else "Columns":k}
                    else:
                        rhs_host = np.array(values[1],order="F" if package.descriptor.provenance["b_layout"]=="col_major" else "C")
                        args = {"edge":source,"rhs":session.upload(rhs_host,layout=layouts["rhs"]),
                                "out":output,"M":m,"N":n,"K":k,"LDA":k,
                                "LDB":k if package.descriptor.provenance["b_layout"]=="col_major" else n,"LDD":n}
                    rt._nvidia_native_descriptor_resident_device_latency(
                        package.image,package.descriptor,args,stream=session.stream,warmup=1,reps=1)
                    np.testing.assert_allclose(session.download(output),expected,rtol=.015,atol=.002)
                    samples = []
                    for _ in range(7):
                        samples.append(rt._nvidia_native_descriptor_resident_device_latency(
                            package.image,package.descriptor,args,stream=session.stream,warmup=3,reps=128))
                        np.testing.assert_allclose(session.download(output),expected,rtol=.015,atol=.002)
                    stages.append({"kind":package.descriptor.provenance.get("kind","matmul"),
                                   "device_event_samples_ms":samples,"median_ms":median(samples)})
            finally:
                session.close()
            rows.append({"active_mnk":[frame[axis] for axis in ("M","N","K")],
                         "correctness":"checked_before_and_after_every_public_round_and_each_stage",
                         "max_abs_error":error,"public_warm_samples_ms":public,
                         "public_warm_median_ms":median(public),"stages":stages,
                         "staging_bytes":stats[0],"staging_allocations":stats[1]})
    finally:
        function.close_native_storage()
    return {"storage":dtype,"producer_count":producers,"shape_bounds_mnk":[32,24,64],
            "first_public_call_ms":first_call_ms,"plan_sha256":hashlib.sha256(program.native_plan_json.encode()).hexdigest(),
            "native_schema":json.loads(program.native_plan_json)["schema"],
            "image_digests":[p.image.image_digest for p in packages],
            "descriptor_digests":[p.descriptor.descriptor_digest for p in packages],"rows":rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    options = parser.parse_args()
    if rt._nvidia_device_name()!="sm_120":
        raise RuntimeError("owning SM120 device required")
    active = subprocess.run(["pgrep","-af","[p]ytest|[g]raphify update"],capture_output=True,text=True)
    if active.returncode==0 and active.stdout.strip():
        raise RuntimeError("tests or graph extraction still active")
    device = subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,driver_version",
                                      "--format=csv,noheader"],text=True).strip()
    profiles = [profile(dtype,count) for dtype in ("fp16","bf16") for count in (2,3)]
    sources = [
        "src/transforms/lib/NativeSM120TensorProgram.h",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_gemm.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/aot/tessera_nvidia_mma_f16_sm120_v1.cu",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/matmul_prepared.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/attention_jvp_prepared.cpp",
        "python/tessera/compiler/native_sm120_tensor_program.py",
        "python/tessera/compiler/nvidia_tensor_lhs.py",
        "python/tessera/compiler/prepared_nvidia_lhs.py",
        "python/tessera/compiler/bounded_nvidia_lhs.py",
        "tests/unit/test_native_bounded_sm120_chain.py",
        "tests/device/nvidia/test_bounded_producer_chain.py",
    ]
    packet = {"schema":"tessera.sm120.bounded_chain_benchmark.v1","target":"nvidia_sm120","device":device,
              "compiler_sha256":digest(os.environ["TESSERA_OPT"]),
              "target_tool_sha256":digest(os.environ["TESSERA_NVIDIA_OPT"]),
              "runtime_sha256":digest(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]),
              "resident_runtime_sha256":digest(os.environ["TESSERA_NVIDIA_GEMM_LIB"]),
              "compiler_version":subprocess.check_output([os.environ["TESSERA_OPT"],"--version"],text=True),
              "recorder_sha256":digest(__file__),"source_sha256":{path:digest(path) for path in sources},
              "stage_domain":"separate resident CUDA-event launch windows; compilation/upload/readback excluded; stages are not summed into a whole-program time",
              "public_domain":"warm ordinary public call including host checks, packing, upload, producer/consumer execution, synchronization and readback",
              "first_call_domain":"first wrapper call includes tracing, native compilation and preparation; process caches may already be warm",
              "claim":"numerical/capacity characterization only; no speedup or selector promotion","profiles":profiles}
    options.output.parent.mkdir(parents=True,exist_ok=True)
    options.output.write_text(json.dumps(packet,indent=2)+"\n")


if __name__=="__main__":
    main()
