"""Matched bounded row/column RHS compiler packages and host ownership."""
from pathlib import Path
import hashlib
import json
import os
import statistics
import subprocess
import time
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler import nvidia_tensor_lhs as lhs
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.nvidia_tensor_rhs import NvidiaNormRhsProgram
from tests.device.nvidia.test_lhs_tensor_jit import _storage,_oracle
from tests.device.nvidia.test_bounded_lhs_jit import (
    bounded_rms as rms_lhs,bounded_layer as layer_lhs,bounded_softmax as softmax_lhs,
    bounded_rms_fused as rms_lhs_fused,bounded_layer_fused as layer_lhs_fused,
    bounded_softmax_fused as softmax_lhs_fused,
)


def record():
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or gpu.split(",")[-1].strip()!="12.0":
        raise RuntimeError("exact SM120 host required")
    from tessera.compiler.prepared_nvidia_lhs import clear_portable_lhs_owners
    clear_portable_lhs_owners()
    rows=[]
    for dtype in ("fp16","bf16"):
        for kind,plain,fused_fn in (("rmsnorm",rms_lhs,rms_lhs_fused),
                                   ("layernorm",layer_lhs,layer_lhs_fused),
                                   ("softmax",softmax_lhs,softmax_lhs_fused)):
            for fused in (False,True):
                storage=_storage(dtype)
                rng=np.random.default_rng(120606)
                capacity=(128,1024,64)
                rhs_order=os.environ["TESSERA_DYNAMIC_RHS_ORDER"]
                if rhs_order not in {"C","F"}:raise ValueError("RHS order must be C/F")
                x=(rng.normal(size=capacity[:2])*.2).astype(storage)
                b=np.array(rng.normal(size=capacity[1:])*.2,dtype=storage,order="F")
                bias=(rng.normal(size=capacity[2])*.2).astype(np.float32)
                residual=(rng.normal(size=(capacity[0],capacity[2]))*.2).astype(np.float32)
                selected=fused_fn if fused else plain
                function=ts.jit(target="nvidia_sm120",shape_bounds={"M":128,"N":64,"K":1024},
                    rhs_storage_order="row_major" if rhs_order=="C" else "col_major")(selected._fn)
                args=(x[:17,:35],np.array(b[:35,:19],order=rhs_order),bias[:19],residual[:17,:19]) if fused else (x[:17,:35],np.array(b[:35,:19],order=rhs_order))
                started=time.perf_counter_ns()
                first=function(*args)
                compile_ms=(time.perf_counter_ns()-started)/1e6
                np.testing.assert_allclose(first,_oracle(x[:17,:35],np.array(b[:35,:19],order=rhs_order),kind,
                    bias[:19] if fused else None,residual[:17,:19] if fused else None),rtol=.015,atol=.015)
                program=function._nvidia_lhs_last_program
                artifact=rt.RuntimeArtifact.from_json(lhs.runtime_artifact(program).to_json())
                edge=program.edge
                packages=(edge.producer,edge.consumer)
                for m,k,n in ((128,1024,64),(17,35,19),(1,1,1),(63,511,31)):
                    source=x[:m,:k]
                    rhs=b[:k,:n]
                    ab,ar=bias[:n],residual[:m,:n]
                    args=(source,rhs,ab,ar) if fused else (source,rhs)
                    oracle=_oracle(source,rhs,kind,ab if fused else None,ar if fused else None)
                    initial=function(*args)
                    np.testing.assert_allclose(initial,oracle,rtol=.015,atol=.015)
                    wall=[]
                    for _ in range(3):
                        started=time.perf_counter_ns()
                        actual=function(*args)
                        wall.append((time.perf_counter_ns()-started)/1e6)
                        np.testing.assert_allclose(actual,oracle,rtol=.015,atol=.015)
                        if function._nvidia_lhs_last_program is not program:
                            raise RuntimeError("public shape change replaced the bounded program")
                    receipt={"output":actual,"component_receipts":function._nvidia_lhs_last_receipts}
                    binding=receipt["component_receipts"][0].get(
                        "native_call_binding","portable_checked_descriptor")
                    stats=None
                    if binding=="prepared_cpp_dynamic_tensor_matmul":
                        owners=list(function._nvidia_lhs_prepared_calls.values())
                        if len(owners)!=1:raise RuntimeError("public dynamic owner identity ambiguous")
                        stats=list(owners[0].scratch_stats())
                    replay=rt.launch(artifact,args)
                    if not replay.get("ok"):raise RuntimeError(replay)
                    np.testing.assert_array_equal(actual,replay["output"])
                    with NvidiaDeviceSession() as session:
                        ds=session.upload(np.ascontiguousarray(source))
                        db=session.upload(np.array(rhs,order=rhs_order),layout="strided")
                        middle=session.empty((m,k),storage)
                        output=session.empty((m,n),np.float16 if fused else np.float32,layout="strided")
                        pa={edge.producer_input_name:ds,edge.intermediate_name:middle}
                        dims={"Rows":m,"Columns":k,"K":k}
                        pa.update({s.name:dims[s.name] for s in packages[0].descriptor.scalars})
                        ca={edge.consumer_input_name:middle.view(0,(m,k),middle.dtype,layout="strided"),
                            edge.consumer_rhs_name:db,edge.output_name:output,
                            "M":m,"N":n,"K":k,"LDA":k,"LDB":n if rhs_order=="C" else k,"LDD":n}
                        if fused:
                            for name,value in edge._epilogue_inputs(ab,ar,m,n).items():
                                ca[name]=session.upload(value,layout=edge._binding(packages[1],name,"input").layout)
                        for package,values in zip(packages,(pa,ca),strict=True):
                            r=rt.launch(NvidiaNormRhsProgram.runtime_artifact(package),values,stream=session.stream)
                            if not r.get("ok"):raise RuntimeError(r)
                        session.synchronize()
                        np.testing.assert_allclose(session.download(output),oracle,rtol=.015,atol=.015)
                        samples={role:[rt._nvidia_native_descriptor_resident_device_latency(
                            p.image,p.descriptor,values,stream=session.stream,warmup=20,reps=100)
                            for _ in range(3)]
                            for role,p,values in zip(("producer","consumer"),packages,(pa,ca),strict=True)}
                        session.synchronize()
                        np.testing.assert_allclose(session.download(output),oracle,rtol=.015,atol=.015)
                    rows.append(dict(dtype=dtype,kind=kind,fused=fused,capacity_mkn=list(capacity),
                        active_mkn=[m,k,n],rhs_storage_order=packages[1].descriptor.provenance["b_layout"],cold_public_wall_ms=compile_ms,
                        public_binding=binding,owner_scratch_stats=stats,
                        public_warm_wall_samples_ms=wall,public_warm_wall_median_ms=statistics.median(wall),
                        component_event_dispatch_samples_ms=samples,
                        component_event_dispatch_median_ms={name:statistics.median(v) for name,v in samples.items()},
                        max_abs_error=float(np.max(np.abs(receipt["output"].astype(np.float64)-oracle))),
                        image_digests=[p.image.image_digest for p in packages],
                        schedule_digests=[p.descriptor.provenance["schedule_digest"] for p in packages],
                        abi_ids=[p.descriptor.abi_id for p in packages],
                        contract_digest=artifact.metadata["native_program"]["contract_digest"]))
                    print("verified",dtype,kind,fused,(m,k,n),flush=True)
                function.close_native_storage()
    sources=("python/tessera/compiler/nvidia_tensor_lhs.py","python/tessera/compiler/jit.py",
             "python/tessera/compiler/prepared_nvidia_lhs.py","python/tessera/compiler/nvidia_native.py",
             "python/tessera/compiler/scheduled_matmul.py","src/compiler/programming_model/lib/PMPasses.cpp",
             "python/tessera/compiler/prepared_nvidia_matmul.py",
             "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/matmul_prepared.cpp",
             "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.h",
             "python/tessera/runtime.py","tests/device/nvidia/test_lhs_tensor_jit.py",
             "tests/device/nvidia/test_prepared_dynamic_lhs_owner.py",
             "python/tessera/compiler/bounded_nvidia_lhs.py","tests/device/nvidia/test_bounded_lhs_jit.py",
             "benchmarks/nvidia/benchmark_dynamic_row_lhs.py","tests/device/nvidia/test_dynamic_row_lhs.py",
             "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp","tests/device/nvidia/test_dynamic_lhs_frontend.py")
    return dict(schema="tessera.nvidia.dynamic_rhs_lhs.v1",gpu=gpu,rows=rows,
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        runtime_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        scope="Ordinary bounded M/N/K JIT and portable replay; static capacity-specialized producer, dynamic native strided consumer. Native dynamic owner selected by public prepared control; no general composed AD.",
        timing_scope="Warm public wall follows an untimed correctness-gated call; it includes route-specific admission/binding/packing/transfers/two launches/readback/cleanup. CUDA event windows include device dispatch gaps; cold public trace/compile wall reported separately.")


if __name__=="__main__":
    Path(os.environ["TESSERA_DYNAMIC_ROW_LHS_PACKET"]).write_text(json.dumps(record(),indent=2,sort_keys=True)+"\n")
