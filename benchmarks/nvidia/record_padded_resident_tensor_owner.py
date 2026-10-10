"""Same-image resident A/B; public and device dispatch windows stay separate."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import numpy as np
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_native_padded_resident_tensor_owner import PitchedRhs
from tests.device.nvidia.test_lhs_tensor_jit import (
    rms_lhs,layer_lhs,softmax_lhs,rms_lhs_fused,layer_lhs_fused,softmax_lhs_fused,
    _storage,_oracle)


def artifact(package):
    return rt.RuntimeArtifact(metadata={"target":"nvidia_sm120"},
        native_image=package.image,launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir,target_ir=package.target_ir)


def record():
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or gpu.split(",")[-1].strip()!="12.0":
        raise RuntimeError("exact SM120 required")
    rows=[]
    for shape in ((17,35,19),(128,1024,64)):
      for kind,plain,fused_fn in (("rmsnorm",rms_lhs,rms_lhs_fused),
                                 ("layernorm",layer_lhs,layer_lhs_fused),
                                 ("softmax",softmax_lhs,softmax_lhs_fused)):
       for dtype in ("fp16","bf16"):
        for fused in (False,True):
         for order in ("C","F"):
            m,k,n=shape
            rng=np.random.default_rng(120711)
            storage=_storage(dtype)
            dynamic=True
            capacity=(128,1024,64) if dynamic else shape
            x=(rng.normal(size=(capacity[0],capacity[1]))*.2).astype(storage)
            b=np.array(rng.normal(size=capacity[1:])*.2,dtype=storage,order=order)
            bias=(rng.normal(size=capacity[2])*.2).astype(np.float32)
            residual=(rng.normal(size=(capacity[0],capacity[2]))*.2).astype(np.float32)
            fn=fused_fn if fused else plain
            args=(x,b,bias,residual) if fused else (x,b)
            program=fn.compile_native_lhs_matmul(*args,dynamic_axes=("M","N","K") if dynamic else (),
                rhs_storage_order="row_major" if order == "C" else "col_major").edge
            source=np.ascontiguousarray(x[:m,:k])
            right=np.array(b[:k,:n],order=order)
            ab,ar=np.ascontiguousarray(bias[:n]),np.ascontiguousarray(residual[:m,:n])
            expected=_oracle(source,right,kind,ab if fused else None,ar if fused else None)
            with NvidiaDeviceSession() as session, program.prepare_resident() as owner:
                ds=session.upload(source)
                width=np.dtype(storage).itemsize
                if order=="C":
                    backing=np.full((k+1,n+7),7,dtype=storage,order="C")
                    backing[1:,:n]=right
                    strides=((n+7)*width,width)
                    offset=(n+7)*width
                else:
                    backing=np.full((k+7,n+1),7,dtype=storage,order="F")
                    backing[:k,1:]=right
                    strides=(width,(k+7)*width)
                    offset=(k+7)*width
                allocation=session.upload(backing,layout="row_major" if order=="C" else "col_major")
                db=PitchedRhs(allocation,(k,n),strides,offset)
                edge=session.empty((m,k),storage)
                output=session.empty((m,n),np.float16 if fused else np.float32,
                                     layout="strided" if dynamic else "row_major")
                consumer_edge=edge.view(0,(m,k),storage,layout="strided") if dynamic else edge
                extra={}
                if fused:
                    for name,value in program._epilogue_inputs(ab,ar,m,n).items():
                        extra[name]=session.upload(value,layout=program._binding(program.consumer,name,"input").layout)
                buffers=[ds,db]+[extra[binding.name] for binding in
                    sorted(program.consumer.descriptor.buffers,key=lambda item:item.ordinal)[2:-1]]+[output]
                packages=(program.producer,program.consumer)
                artifacts=tuple(artifact(p) for p in packages)
                pa={program.producer_input_name:ds,program.intermediate_name:edge}
                scalar={"Rows":m,"Columns":k,"K":k}
                pa.update({s.name:scalar[s.name] for s in packages[0].descriptor.scalars})
                ca={program.consumer_input_name:consumer_edge,program.consumer_rhs_name:db,
                    program.output_name:output,**extra,"M":m,"N":n,"K":k}
                if dynamic:ca.update(LDA=k,LDB=n+7 if order=="C" else k+7,LDD=n)
                def control():
                    for package,values in zip(artifacts,(pa,ca),strict=True):
                        receipt=rt.launch(package,values,stream=session.stream)
                        if not receipt.get("ok"):raise RuntimeError(receipt)
                    if session.synchronize()!=0:raise RuntimeError("control completion failed")
                def candidate():
                    owner.invoke(buffers,consumer_edge,stream=session.stream)
                for launch in (control,candidate):
                    launch()
                    np.testing.assert_allclose(output.numpy(),expected,rtol=.015,atol=.015)
                windows={"control":[],"native":[]}
                for window in range(6):
                    arms=(("control",control),("native",candidate))
                    if window%2:arms=arms[::-1]
                    for name,launch in arms:
                        start=time.perf_counter_ns()
                        for _ in range(8):launch()
                        windows[name].append((time.perf_counter_ns()-start)/8e6)
                        np.testing.assert_allclose(output.numpy(),expected,rtol=.015,atol=.015)
                # Component timing stays independent from synchronous A/B wall windows.
                component={role:[rt._nvidia_native_descriptor_resident_device_latency(
                    package.image,package.descriptor,values,stream=session.stream,warmup=10,reps=50)
                    for _ in range(3)] for role,package,values in
                    zip(("producer","consumer"),packages,(pa,ca),strict=True)}
                np.testing.assert_allclose(output.numpy(),expected,rtol=.015,atol=.015)
                np.testing.assert_array_equal(allocation.numpy(),backing)
                rows.append(dict(shape_mkn=list(shape),kind=kind,dtype=dtype,fused=fused,
                    dynamic=dynamic,rhs_order=order,rhs_stride_bytes=list(strides),rhs_offset_bytes=offset,wall_samples_ms=windows,
                    wall_medians_ms={arm:statistics.median(samples) for arm,samples in windows.items()},
                    component_device_dispatch_samples_ms=component,
                    image_digests=[p.image.image_digest for p in packages],
                    descriptor_digests=[p.descriptor.descriptor_digest for p in packages],
                    max_abs_error=float(np.max(np.abs(output.numpy().astype(np.float64)-expected))),
                    correctness="independent_oracle_before_and_after_each_arm"))
                print("verified",kind,dtype,shape,"fused",fused,"order",order,flush=True)
    repo=Path(__file__).resolve().parents[2]
    files=[Path(__file__),repo/"tests/device/nvidia/test_native_padded_resident_tensor_owner.py",repo/"python/tessera/compiler/resident_nvidia_tensor.py",
        repo/"python/tessera/compiler/nvidia_native.py",
        repo/"src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/matmul_prepared.cpp",
        repo/"src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.h"]
    return dict(gpu=gpu,rows=rows,schema="tessera.nvidia.padded_resident_native_sequence.v1",
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        runtime_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        timing_scope="Same-image resident synchronized wall calls include Python binding/validation and native submission/completion. Component CUDA event dispatch windows are separate; no isolated-kernel or host-wall speedup inference.",
        host_interference="Concurrent host/device activity is not isolated; results characterize this run and do not promote a selector.")


if __name__=="__main__":
    Path(os.environ["TESSERA_RESIDENT_OWNER_PACKET"]).write_text(json.dumps(record(),indent=2)+"\n")
