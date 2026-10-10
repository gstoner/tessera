"""Paired native SM120 norm schedules; correctness gates precede timing."""
from pathlib import Path
import hashlib
import json
import os
import statistics
import subprocess
import time
import numpy as np
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.nvidia_tensor_rhs import NvidiaNormRhsProgram
from tests.device.nvidia.test_cooperative_norm import package,launch,oracle
from tests.device.nvidia.test_lhs_tensor_jit import _storage


def record():
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or gpu.split(",")[-1].strip()!="12.0":
        raise RuntimeError("owning SM120 required")
    current=os.environ["TESSERA_NVIDIA_OPT"]
    baseline=str(Path(".build-sm120-w1-1/norm-accuracy-ab-20261005/tessera-nvidia-opt-baseline").resolve())
    rows=[]
    for dtype in ("fp16","bf16","fp32"):
        storage=np.float32 if dtype=="fp32" else _storage(dtype)
        for kind in ("rmsnorm","layernorm"):
            for shape in ((1,32),(17,35),(129,257),(128,1024),(256,4096),(2,4097)):
                source=(np.random.default_rng(512).normal(size=shape)*.2).astype(storage)
                expected=oracle(source,kind)
                candidates={}
                errors={}
                for mode in ("serial","cooperative_128"):
                    for arm,tool in (("baseline",baseline),("candidate",current)):
                        os.environ["TESSERA_NVIDIA_OPT"]=tool
                        strategy=arm+"_"+mode
                        artifact,native=package(kind,source,mode)
                        actual=launch(native,source)
                        np.testing.assert_allclose(actual.astype(np.float32),expected.astype(np.float32),rtol=.015,atol=.015)
                        errors[strategy]=float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64))))
                        candidates[strategy]=(artifact,native)
                os.environ["TESSERA_NVIDIA_OPT"]=current
                samples={strategy:[] for strategy in candidates}
                wall={strategy:[] for strategy in candidates}
                with NvidiaDeviceSession() as session:
                    ds=session.upload(source)
                    outputs={strategy:session.empty(shape,storage) for strategy in candidates}
                    arguments={strategy:{
                        native.descriptor.buffers[0].name:ds,native.descriptor.buffers[1].name:outputs[strategy],
                        "Rows":shape[0],"Columns":shape[1]} for strategy,(_,native) in candidates.items()}
                    for strategy,(_,native) in candidates.items():
                        receipt=rt.launch(NvidiaNormRhsProgram.runtime_artifact(native),arguments[strategy],stream=session.stream)
                        if not receipt.get("ok"):raise RuntimeError(receipt)
                    session.synchronize()
                    for strategy in candidates:
                        np.testing.assert_allclose(session.download(outputs[strategy]).astype(np.float32),
                            expected.astype(np.float32),rtol=.015,atol=.015)
                    for trial in range(5):
                        order=tuple(candidates) if trial%2==0 else tuple(reversed(candidates))
                        for strategy in order:
                            native=candidates[strategy][1]
                            samples[strategy].append(rt._nvidia_native_descriptor_resident_device_latency(
                                native.image,native.descriptor,arguments[strategy],
                                stream=session.stream,warmup=20,reps=100))
                            start=time.perf_counter_ns()
                            launch(native,source)
                            wall[strategy].append((time.perf_counter_ns()-start)/1e6)
                    session.synchronize()
                    for strategy in candidates:
                        np.testing.assert_allclose(session.download(outputs[strategy]).astype(np.float32),
                            expected.astype(np.float32),rtol=.015,atol=.015)
                medians={strategy:statistics.median(v) for strategy,v in samples.items()}
                row=dict(dtype=dtype,kind=kind,shape=list(shape),max_abs_error=errors,
                    resident_event_dispatch_samples_ms=samples,resident_event_dispatch_median_ms=medians,
                    checked_host_wall_samples_ms=wall,checked_host_wall_median_ms={s:statistics.median(v) for s,v in wall.items()},
                    event_speedup_baseline_over_candidate={mode:medians["baseline_"+mode]/medians["candidate_"+mode] for mode in ("serial","cooperative_128")},
                    image_digests={s:p.image.image_digest for s,(_,p) in candidates.items()},
                    schedule_digests={s:a.schedule_digest for s,(a,_) in candidates.items()},
                    resources={s:dict(p.image.resource_record.metrics) for s,(_,p) in candidates.items()},
                    static_barriers={s:p.target_ir.count("nvvm.barrier") for s,(_,p) in candidates.items()})
                rows.append(row)
                print(dtype,kind,shape,row["event_speedup_baseline_over_candidate"],flush=True)
    sources=("src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp",
        "src/compiler/programming_model/lib/PMPasses.cpp","src/compiler/programming_model/ir/ScheduleDialect.cpp",
        "src/compiler/programming_model/ir/schedule/ScheduleMeshPipelineOps.td","src/compiler/ir/TileOps.cpp",
        "python/tessera/compiler/scheduled_kernel.py","python/tessera/compiler/nvidia_native.py",
        "tests/device/nvidia/test_cooperative_norm.py","benchmarks/nvidia/benchmark_norm_accuracy_performance.py")
    return dict(schema="tessera.nvidia.norm_accuracy_performance.v1",gpu=gpu,rows=rows,
        baseline_compiler_sha256=hashlib.sha256(Path(baseline).read_bytes()).hexdigest(),
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        nvidia_opt_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_OPT"]).read_bytes()).hexdigest(),
        bridge_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
        source_sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},
        timing_scope="Alternating schedule order over five trials. CUDA event windows include resident dispatch; checked host wall includes module loading, allocation, upload, launch, readback and cleanup.",
        scope="Frozen rsqrt/simple-sum baseline versus sqrt/division and compensated serial FP32 sum, paired by schedule. Norm-only numerical gate; composed failures are retained in accuracy.json. No FP8/MXFP8/MXFP4 or sibling architecture inference.")


if __name__=="__main__":
    Path(os.environ["TESSERA_COOP_NORM_PACKET"]).write_text(json.dumps(record(),indent=2,sort_keys=True)+"\n")
