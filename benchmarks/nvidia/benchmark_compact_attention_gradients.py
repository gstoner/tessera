#!/usr/bin/env python3
"""Compact native attention gradients: matched ABI, allocation and timing proof."""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import numpy as np
from benchmarks.record_device_ring_protocol import Device
from benchmarks.nvidia.benchmark_jit_attention_vjp import function, reference, download
from benchmarks.nvidia.benchmark_attention_argument_order import function as ordered_function, ORDERS
from benchmarks.nvidia.benchmark_checkpoint_bias_gradient import reference as bias_reference, runtime
from benchmarks.nvidia.benchmark_attention_gradient_activity import paired_timings
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession

class CompletedFrameBuffer:
    """A frame-owned view handed to the timing stream after a context fence."""
    tessera_layout = "row_major"

    def __init__(self, owner, value, stream):
        owner._ready()
        owner.check(owner.sync())
        self.owner, self.value, self.stream = owner, value, stream

    @property
    def dtype(self):
        return np.dtype(self.value.__cuda_array_interface__["typestr"])

    @property
    def shape(self):
        return tuple(self.value.__cuda_array_interface__["shape"])

    @property
    def __cuda_array_interface__(self):
        self.owner._ready()
        interface = self.value.__cuda_array_interface__.copy()
        interface["stream"] = self.stream
        return interface


def prepare(device, frame, cotangent, gradients):
    native = frame._frame
    ptrs = [cotangent.ptr, *(x.pointer.value for x in native._saved[:4])]
    if native._has_bias:
        ptrs.append(native._bias.pointer.value)
    ptrs.append(native._saved[4].pointer.value)
    ptrs.extend(x.__cuda_array_interface__["data"][0] for x in gradients)
    args = [ct.c_void_p(x) for x in ptrs] + [ct.c_int64(x) for x in native._backward_scalars]
    params = (ct.c_void_p*len(args))(*(ct.cast(ct.pointer(x),ct.c_void_p) for x in args))
    shapes = (*native.shapes[:3], *((native._bias_shape,) if native._has_bias_gradient else ()))
    launch_roles = native._gradient_roles if native._gradient_launch == "packed_v1" else range(len(shapes))
    elements = sum(math.prod(shapes[i]) for i in launch_roles)
    threads = native._backward_threads
    blocks = (elements+threads-1)//threads
    def launch():
        device.check(device.launch(native._functions[1],blocks,1,1,threads,1,1,0,None,params,None))
    resources = {"block_threads":threads,"launch_blocks":blocks,"gradient_elements":elements}
    for name, attr in (("registers",4),("local_bytes",3),("shared_bytes",1)):
        val=ct.c_int()
        device.check(device.attribute(ct.byref(val),attr,native._functions[1]))
        resources[name]=val.value
    active=ct.c_int()
    device.check(device.occupancy(ct.byref(active),native._functions[1],threads,0))
    resources["active_blocks_per_sm"]=active.value
    return launch,resources,args,params

def run_case(device, shape, causal, wrt, *, repetitions=32, directory=None, bias_shape=None, nonfinite=None):
    b,hq,hkv,sq,sk,d,dv=shape
    rng=np.random.default_rng(120_510)
    values=[(rng.normal(size=s)*.3).astype(np.float32) for s in
            ((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv))]
    seed=(rng.normal(size=(b,hq,sq,dv))*.3).astype(np.float32)
    bias=(rng.normal(size=bias_shape)*.3).astype(np.float32) if bias_shape else None
    if bias is None:
        output,lse,expected=reference(*values,seed,causal)
        frontend=values
        fn=function(wrt,causal)
        role_names=("q","k","v")
        frontend_order=role_names
    else:
        output,lse,expected=bias_reference(*values,bias,seed,causal)
        axes=tuple(i for i,(n,logical) in enumerate(zip(bias_shape,(b,hq,sq,sk),strict=True)) if n==1 and n!=logical)
        expected=(*expected[:3],expected[3].sum(axis=axes,keepdims=True))
        frontend_order=ORDERS["k_bias_v_q"]
        table=dict(zip(("q","k","v","bias"),(*values,bias),strict=True))
        frontend=[table[x] for x in frontend_order]
        fn=ordered_function("k_bias_v_q",wrt,causal)
        role_names=("q","k","v","bias")
    active=tuple(role_names.index(x) for x in wrt)
    if nonfinite:
        if active != (2,):
            raise ValueError("nonfinite primal V proof requires V-only")
        values[2].fill(np.inf if nonfinite=="inf" else np.nan)
    compiler=Path(os.environ["TESSERA_OPT"])
    programs={name:fn.compile_native_attention_vjp(*frontend,compiler=compiler,compact_gradients=compact,compact_launch=launch,compact_threads=threads)
              for name,compact,launch,threads in (("complete",False,"packed_v1",128),("compact",True,"packed_v1",128),("compact_logical",True,"logical_v1",128),("compact_t64",True,"packed_v1",64),("compact_logical_t64",True,"logical_v1",64))}
    assert all(program.pair.forward.image.payload == programs["complete"].pair.forward.image.payload for program in programs.values())
    allocations,resources,errors={}, {}, []
    native_samples={name:[] for name in programs}
    prepared_samples={name:[] for name in programs}
    saved_states={}
    host_samples={name:[] for name in programs}
    with NvidiaDeviceSession() as session:
        resident=[session.upload(x) for x in frontend]
        cotangent=session.upload(seed)
        session.synchronize()
        frames,raw_results,handles,owners={}, {}, {}, []
        try:
            for name,program in programs.items():
                frame=program.capture(*resident)
                frames[name]=frame
                native=frame._frame
                if not nonfinite:
                    np.testing.assert_allclose(download(session,frame.primal),output,atol=4e-5,rtol=4e-5)
                    np.testing.assert_allclose(download(session,native._saved[4]),lse,atol=4e-5,rtol=4e-5)
                saved_states[name]=(download(session,frame.primal),download(session,native._saved[4]))
                before=len(native.buffers)
                alloc=native.alloc
                calls=[]
                def tracked(pointer,nbytes):
                    calls.append(int(nbytes))
                    return alloc(pointer,nbytes)
                native.alloc=tracked
                try:
                    requested=frame.backward(cotangent)
                finally:
                    native.alloc=alloc
                shapes=(*native.shapes[:3], *((native._bias_shape,) if bias is not None else ()))
                physical=tuple(range(len(shapes))) if name=="complete" else tuple(sorted(active))
                expected_bytes=[seed.nbytes,*(math.prod(shapes[i])*4 for i in physical)]
                assert calls==expected_bytes and len(native.buffers)-before==len(physical)
                assert native._gradient_roles==physical
                allocations[name]={"native_allocations_bytes":calls,
                    "retained_gradient_buffers":len(physical),"retained_gradient_bytes":sum(calls[1:]),
                    "physical_roles":list(physical)}
                for value,index in zip(requested,active,strict=True):
                    host=download(session,value)
                    np.testing.assert_allclose(host,expected[index],atol=4e-5,rtol=4e-5)
                    errors.append(float(np.max(np.abs(host-expected[index]))))
                full=native.backward(cotangent)
                raw_results[name]=full
                for value,index in zip(full,physical,strict=True):
                    host=download(session,value)
                    if index in active:
                        np.testing.assert_allclose(host,expected[index],atol=4e-5,rtol=4e-5)
                    else:
                        np.testing.assert_array_equal(host,np.zeros_like(host))
                launch,res,args,params=prepare(device,frame,cotangent,full)
                handles[name],resources[name]=launch,res
                owners.append((args,params))
            # Mutate caller-owned sources after both captures; saved generation
            # ownership must keep backward independent of these writes.
            clear=ct.CDLL("libcuda.so.1").cuMemsetD32_v2
            clear.argtypes=[ct.c_uint64,ct.c_uint,ct.c_size_t]
            clear.restype=ct.c_int
            for value,host in zip(resident,frontend,strict=True):
                device.check(clear(value.ptr,0,host.size))
            session.synchronize()
            timing=paired_timings(device,handles,repetitions)
            native_args={}
            for name,frame in frames.items():
                native=frame._frame
                pkg=programs[name].pair.backward
                buffers=[cotangent,*native._saved[:4]]
                if native._has_bias:
                    buffers.append(native._bias)
                buffers.extend((native._saved[4],*raw_results[name]))
                args=dict(zip((x.name for x in pkg.descriptor.buffers),(CompletedFrameBuffer(native,value,session.stream) for value in buffers),strict=True))
                scalar_values=shape+(bias_shape if pkg.descriptor.provenance["bias_shape"] else ())
                args.update(zip((x.name for x in pkg.descriptor.scalars),scalar_values,strict=True))
                native_args[name]=args
            for trial in range(5):
                for name in (tuple(frames) if trial%2==0 else tuple(reversed(frames))):
                    pkg=programs[name].pair.backward
                    native_samples[name].append(rt._nvidia_native_descriptor_resident_device_latency(
                        pkg.image,pkg.descriptor,native_args[name],stream=session.stream,
                        warmup=4,reps=max(256,repetitions)))

            for name,results in raw_results.items():
                for value,index in zip(results,allocations[name]["physical_roles"],strict=True):
                    host=download(session,value)
                    if index in active:
                        np.testing.assert_allclose(host,expected[index],atol=4e-5,rtol=4e-5)
                    else:
                        np.testing.assert_array_equal(host,np.zeros_like(host))
            for trial in range(5):
                for name in (tuple(frames) if trial%2==0 else tuple(reversed(frames))):
                    start=time.perf_counter_ns()
                    result=frames[name].backward(cotangent)
                    host_samples[name].append((time.perf_counter_ns()-start)/1e6)
                    for value,index in zip(result,active,strict=True):
                        np.testing.assert_allclose(download(session,value),expected[index],atol=4e-5,rtol=4e-5)
            # Retained result storage survives repeated calls until frame close.
            # Private captured source allocations differ from caller allocations.
            native=frames["compact"]._frame
            for i,name in enumerate(frontend_order):
                role=role_names.index(name)
                private=native._bias if role==3 else native._saved[role]
                assert private.pointer.value != resident[i].ptr
            for multiplier in (2,1):
                direction=session.upload(seed*multiplier)
                result=frames["compact"].backward(direction)
                for value,index in zip(result,active,strict=True):
                    np.testing.assert_allclose(download(session,value),expected[index]*multiplier,atol=4e-5,rtol=4e-5)
        finally:
            for frame in frames.values():
                frame.close()
            for frame in frames.values():
                try:
                    frame.backward(cotangent)
                except ValueError:
                    pass
                else:
                    raise AssertionError("closed attention frame executed backward")
    # Common checked host launcher consumes the compact descriptor, with no
    # derivative reconstruction. Nonfinite V leaves unrelated O/LSE oracle use
    # outside this check; the resident native forward above proves that case.
    checked_receipt=None
    if not nonfinite:
        for candidate in ("compact","compact_logical","compact_t64","compact_logical_t64"):
            pkg=programs[candidate].pair.backward
            outputs=[np.full(expected[i].shape,np.nan,np.float32) for i in sorted(active)]
            physical_inputs=[seed,*values,output.astype(np.float32)]
            if bias is not None:
                physical_inputs.append(bias)
            physical_inputs.append(lse.astype(np.float32))
            args=dict(zip((x.name for x in pkg.descriptor.buffers),(*physical_inputs,*outputs),strict=True))
            scalar_values=shape+(bias_shape if pkg.descriptor.provenance["bias_shape"] else ())
            args.update(zip((x.name for x in pkg.descriptor.scalars),scalar_values,strict=True))
            checked_receipt=rt.launch(runtime(pkg),args)
            assert checked_receipt["ok"] and checked_receipt["execution_kind"]=="native_gpu",checked_receipt
            for value,index in zip(outputs,sorted(active),strict=True):
                np.testing.assert_allclose(value,expected[index],atol=4e-5,rtol=4e-5)
            # The same compact ABI executes on caller-owned resident storage via
            # the checked launcher and explicit producer-stream validation.
            with NvidiaDeviceSession() as session:
                resident_args={x.name:session.upload(value) for x,value in zip(pkg.descriptor.buffers,(*physical_inputs,*outputs),strict=True)}
                scalars=dict(zip((x.name for x in pkg.descriptor.scalars),scalar_values,strict=True))
                result=rt._submit_nvidia_sm120_native(pkg.image,pkg.descriptor,resident_args,scalars,stream=session.stream)
                session.synchronize()
                assert result is not None
                for binding,index in zip([x for x in pkg.descriptor.buffers if x.direction=="output"],sorted(active),strict=True):
                    np.testing.assert_allclose(download(session,resident_args[binding.name]),expected[index],atol=4e-5,rtol=4e-5)
    # Precompute native pointer/scalar launch arguments before event recording.
    # This independent C++ control excludes Python and repeated dispatcher ABI
    # parsing, while still including CUDA driver submission gaps.
    prepared_args={}
    for name,program in programs.items():
        pkg=program.pair.backward
        saved_output,saved_lse=saved_states[name]
        inputs=[seed,*values,saved_output]
        if bias is not None:
            inputs.append(bias)
        inputs.append(saved_lse)
        physical=allocations[name]["physical_roles"]
        outputs=[np.full(expected[i].shape,np.nan,np.float32) for i in physical]
        args=dict(zip((x.name for x in pkg.descriptor.buffers),(*inputs,*outputs),strict=True))
        scalar_values=shape+(bias_shape if pkg.descriptor.provenance["bias_shape"] else ())
        args.update(zip((x.name for x in pkg.descriptor.scalars),scalar_values,strict=True))
        receipt=rt.launch(runtime(pkg),args)
        assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
        prepared_args[name]=(args,outputs,physical)
    for trial in range(5):
        for name in (tuple(programs) if trial%2==0 else tuple(reversed(programs))):
            pkg=programs[name].pair.backward
            args,outputs,physical=prepared_args[name]
            prepared_samples[name].append(rt._nvidia_native_descriptor_device_latency(
                pkg.image,pkg.descriptor,args,warmup=4,reps=max(256,repetitions)))
            receipt=rt.launch(runtime(pkg),args)
            assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
            for value,index in zip(outputs,physical,strict=True):
                if index in active:
                    np.testing.assert_allclose(value,expected[index],atol=4e-5,rtol=4e-5)
                else:
                    np.testing.assert_array_equal(value,np.zeros_like(value))
    label="_".join(map(str,shape))+"_"+str(causal)+"_"+"_".join(wrt)+"_"+str(bias_shape)+"_"+str(nonfinite)
    hashes={}
    for name,program in programs.items():
        pkg=program.pair.backward
        hashes[name]=hashlib.sha256(pkg.image.payload).hexdigest()
        if directory:
            for suffix,text in ((".mlir",pkg.tile_ir),(".target.mlir",pkg.target_ir),(".ptx",pkg.image.payload.decode())):
                (directory/(label+"_"+name+suffix)).write_text(text)
            (directory/(label+"_"+name+".descriptor.json")).write_text(json.dumps(pkg.descriptor.to_dict(),indent=2))
    dispatch={name:statistics.median(samples) for name,samples in timing.items()}
    native_medians={name:statistics.median(samples) for name,samples in native_samples.items()}
    wall={name:statistics.median(samples) for name,samples in host_samples.items()}
    prepared_medians={name:statistics.median(samples) for name,samples in prepared_samples.items()}
    return dict(shape=list(shape),causal=causal,wrt=list(wrt),frontend_order=list(frontend_order),
        bias_shape=list(bias_shape) if bias_shape else [],nonfinite_primal_v=nonfinite,
        max_abs_error=max(errors),allocations=allocations,resources=resources,image_sha256=hashes,
        device_dispatch_samples_ms=timing,device_dispatch_medians_ms=dispatch,
        native_resident_event_samples_ms=native_samples,native_resident_event_medians_ms=native_medians,
        native_prepared_event_samples_ms=prepared_samples,native_prepared_event_medians_ms=prepared_medians,
        native_event_ratio_complete_over_compact=native_medians["complete"]/native_medians["compact"],
        checked_backward_wall_samples_ms=host_samples,checked_backward_wall_medians_ms=wall,
        dispatch_ratio_complete_over_compact=dispatch["complete"]/dispatch["compact"],
        wall_ratio_complete_over_compact=wall["complete"]/wall["compact"],
        checked_host_and_resident_passed=checked_receipt is not None,
        repeated_backward_result_order_and_private_capture="passed")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repetitions",type=int,default=32)
    args=parser.parse_args()
    if args.repetitions<=0:
        raise ValueError("repetitions must be positive")
    gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,compute_cap,driver_version","--format=csv,noheader"],text=True).strip()
    if len(gpu.splitlines())!=1 or "RTX 5070" not in gpu or gpu.split(",")[2].strip()!="12.0":
        raise RuntimeError("compact recorder requires owning RTX 5070 / sm_120")
    device=Device("nvidia")
    directory=args.output.parent/"artifacts"
    directory.mkdir(parents=True,exist_ok=True)
    rows=[]
    for shape in ((1,2,1,3,5,4,3),(1,2,1,8,129,8,6)):
        for causal in (False,True):
            for wrt in (("q",),("k",),("v",),("v","q"),("k","q"),("q","k","v")):
                rows.append(run_case(device,shape,causal,wrt,repetitions=args.repetitions,directory=directory))
                print("verified",shape,causal,wrt,flush=True)
    biased=[]
    for physical in ((2,4,3,5),(1,4,1,1)):
        for causal in (False,True):
            for wrt in (("bias",),("v",),("bias","q","v")):
                biased.append(run_case(device,(2,4,2,3,5,4,3),causal,wrt,repetitions=args.repetitions,directory=directory,bias_shape=physical))
                print("verified bias",physical,causal,wrt,flush=True)
    nonfinite=[run_case(device,(1,2,1,3,5,4,3),causal,("v",),repetitions=args.repetitions,directory=directory,nonfinite=kind)
        for causal in (False,True) for kind in ("inf","nan")]
    sources=("src/transforms/lib/AutodiffPairedPass.cpp","src/compiler/programming_model/lib/NativeCheckpoint.h",
        "src/compiler/ir/TileOps.cpp","src/compiler/codegen/tessera_gpu_backend_NVIDIA/lib/Conversion/NVIDIALowering.cpp",
        "src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp",
        "python/tessera/compiler/scheduled_checkpoint.py","python/tessera/compiler/nvidia_native.py",
        "python/tessera/compiler/native_attention_program.py","python/tessera/compiler/resident_attention.py",
        "python/tessera/compiler/compact_attention_contract.py","python/tessera/compiler/jit.py","python/tessera/runtime.py",
        "benchmarks/nvidia/benchmark_compact_attention_gradients.py")
    fingerprints={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources}
    for name in ("TESSERA_OPT","TESSERA_NVIDIA_OPT","TESSERA_NVIDIA_PTX_LAUNCH_LIB"):
        path=Path(os.environ[name])
        fingerprints[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    packet=dict(device=gpu,architecture="sm_120",rows=rows,biased_rows=biased,nonfinite_rows=nonfinite,
        repetitions=args.repetitions,source_and_binary_sha256=fingerprints,
        timing_scope="Python dispatch and C++ resident submission CUDA-event windows separate; neither asserted to exclude all host submission gaps; allocating/synchronous backward wall separate",
        route="frontend -> native paired AD -> Graph -> Schedule -> Tile -> NVIDIA Target/LLVM/NVVM -> PTX + checked compact ABI")
    args.output.write_text(json.dumps(packet,indent=2)+"\n")
    print("wrote",args.output,flush=True)

if __name__=="__main__":
    main()
