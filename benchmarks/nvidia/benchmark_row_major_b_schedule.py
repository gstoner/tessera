"""Native Schedule-owned RHS layout packages with matched-input numerical proof."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import numpy as np
from tessera import runtime as rt
from tessera.compiler import nvidia_native as native
from tessera.compiler.scheduled_matmul import lower_scheduled_matmul
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.unit.test_scheduled_matmul_consumers import _module
from tests._support.nvidia import nvidia_cuda_host_ready


def run_case(shape,dtype,order):
    m,k,n=shape
    module=_module(target="nvidia_sm120",shape=shape,dtype=dtype)
    if order=="row_major":
        module.functions[0].body[0].kwargs["rhs_storage_order"]=order
    artifact=lower_scheduled_matmul(module,target="nvidia_sm120")
    package=native.package_scheduled_matmul(artifact,pipeline_name="tessera-nvidia-pipeline-sm120")
    storage=np.float16
    if dtype=="bf16":
        import ml_dtypes
        storage=ml_dtypes.bfloat16
    rng=np.random.default_rng(120_411)
    a=(rng.normal(size=(m,k))*.2).astype(storage)
    values=(rng.normal(size=(k,n))*.2).astype(storage)
    # The static ABI requires compact physical storage. Check that a padded
    # host view is rejected, then explicitly compact the logical values.
    padded=np.full((k,n+5),13,storage)
    padded[:,:n]=values
    b=np.ascontiguousarray(padded[:,:n]) if order=="row_major" else np.asfortranarray(values)
    output=np.full((m,n),np.nan,np.float32)
    expected=a.astype(np.float32)@values.astype(np.float32)
    runtime=rt.RuntimeArtifact(metadata={"target":"nvidia_sm120"},native_image=package.image,
        launch_descriptor=package.descriptor,tile_ir=package.tile_ir,target_ir=package.target_ir)
    scalars={"M":m,"N":n,"K":k}
    args={**scalars,artifact.a_name:a,artifact.b_name:b,artifact.output_name:output}
    if order=="row_major":
        refused=rt.launch(runtime,{**args,artifact.b_name:padded[:,:n]})
        if refused.get("ok") or refused.get("diagnostic_code")!="E_LAUNCH_BINDING_MISMATCH":
            raise RuntimeError(f"padded static RHS must be rejected: {refused}")
    result=rt.launch(runtime,args)
    if not result.get("ok") or result.get("execution_kind")!="native_gpu":
        raise RuntimeError(result)
    np.testing.assert_allclose(output,expected,rtol=4e-5,atol=4e-5)
    host=[]
    for _ in range(3):
        start=time.perf_counter_ns()
        result=rt.launch(runtime,args)
        if not result.get("ok"):raise RuntimeError(result)
        host.append((time.perf_counter_ns()-start)/1e6)
    np.testing.assert_allclose(output,expected,rtol=4e-5,atol=4e-5)
    with NvidiaDeviceSession() as session:
        buffers={**scalars,artifact.a_name:session.upload(a),
                 artifact.b_name:session.upload(values,layout=order),
                 artifact.output_name:session.upload(np.full((m,n),np.nan,np.float32))}
        # Numerical resident proof precedes all event windows.
        receipt=rt.launch(runtime,buffers,stream=session.stream)
        if not receipt.get("ok"):raise RuntimeError(receipt)
        session.synchronize()
        np.testing.assert_allclose(session.download(buffers[artifact.output_name]),expected,rtol=4e-5,atol=4e-5)
        device=[rt._nvidia_native_descriptor_resident_device_latency(
            package.image,package.descriptor,buffers,stream=session.stream,warmup=20,reps=100)
            for _ in range(3)]
        session.synchronize()
        np.testing.assert_allclose(session.download(buffers[artifact.output_name]),expected,rtol=4e-5,atol=4e-5)
    return dict(shape_mkn=list(shape),dtype=dtype,order=order,
        max_abs_error=float(np.max(np.abs(output-expected))),abi=package.descriptor.abi_id,
        schedule_digest=artifact.schedule_digest,graph_sha256=artifact.graph_digest,
        tile_sha256=artifact.tile_digest,image_sha256=hashlib.sha256(package.image.payload).hexdigest(),
        geometry=package.descriptor.geometry.policy,
        checked_host_samples_ms=host,checked_host_median_ms=statistics.median(host),
        resident_dispatch_window_samples_ms=device,resident_dispatch_window_median_ms=statistics.median(device))


def main():
    if not nvidia_cuda_host_ready():raise RuntimeError("requires owning CUDA host")
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,driver_version,compute_cap",
        "--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or gpu.split(",")[-1].strip()!="12.0":
        raise RuntimeError("requires one exact SM120 GPU")
    rows=[run_case(shape,dtype,order) for dtype in ("fp16","bf16")
        for shape in ((16,16,8),(17,35,19),(64,256,64),(128,1024,64),(256,1024,128))
        for order in ("col_major","row_major")]
    sources=["src/compiler/programming_model/lib/PMPasses.cpp",
        "src/compiler/programming_model/ir/ScheduleDialect.cpp",
        "python/tessera/compiler/scheduled_matmul.py","python/tessera/compiler/nvidia_native.py",
        "python/tessera/runtime.py","benchmarks/nvidia/benchmark_row_major_b_schedule.py"]
    packet=dict(schema="tessera.nvidia.row_major_b_schedule.v1",gpu=gpu,rows=rows,
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        production_package_support="static unfused SM120 fp16/BF16 with fp32 output",
        scope="native Graph/Schedule/Tile/Target/PTX -> checked host and resident ABI; named RHS tensor producer edge remains open",
        timing_scope="matched storage values; host staging wall time and resident CUDA-event dispatch windows remain separate; no layout/format default promotion")
    Path(os.environ["TESSERA_ROW_B_SCHEDULE_PACKET"]).write_text(json.dumps(packet,indent=2,sort_keys=True)+"\n")
    print("verified",len(rows),"native row/column RHS package cases")


if __name__=="__main__":
    main()
