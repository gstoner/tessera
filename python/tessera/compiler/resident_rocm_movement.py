"""Checked native-owned resident storage for compiled static ROCm movement."""
from __future__ import annotations
import ctypes as ct
import os
import threading
import time
import weakref
import math
import numpy as np
from .prepared_rocm_movement import HostView

def views(arrays):
    result=(HostView*len(arrays))()
    for view,array in zip(result,arrays,strict=True):
        if not isinstance(array,np.ndarray) or array.ndim>4:
            raise TypeError("resident movement requires host tensor arrays of rank at most four")
        view.data,view.bytes,view.rank=array.ctypes.data,array.nbytes,array.ndim
        view.dtype=1 if array.dtype==np.dtype("float32") else 2 if array.dtype==np.dtype("int32") else 0
        for axis,(extent,stride) in enumerate(zip(array.shape,array.strides,strict=True)):
            view.shape[axis],view.strides[axis]=extent,stride
    return result

class ResidentMovementResult:
    """Generation-checked output; materialize before reusing its owner."""
    def __init__(self,owner,generation):
        self.owner,self.generation=owner,generation
    def to_host(self):
        return self.owner.read(self.generation)

class ResidentMovementCall:
    def __init__(self,prepared,*,consumer=None):
        if not prepared._finalizer.alive:raise ValueError("prepared movement is closed")
        self.prepared,self.lib=prepared,prepared._lib
        self.consumer=consumer
        self.consumer_artifact_hash=consumer.artifact_hash if consumer is not None else None
        if consumer is not None:
            validate_softmax_consumer(prepared,consumer)
        self.pid=os.getpid()
        self.lock=threading.RLock()
        self.handle=0
        self.closed=False
        self.artifact_hash=prepared._sealed_artifact_hash
        if self.artifact_hash!=prepared.artifact.artifact_hash:
            raise ValueError("resident movement artifact differs from sealed preparation")
        signatures={
            "prepare":[ct.c_uint64,ct.POINTER(ct.c_uint64)],
            "upload":[ct.c_uint64,ct.POINTER(HostView),ct.c_size_t],
            "invoke":[ct.c_uint64,ct.POINTER(ct.c_uint64),ct.POINTER(ct.c_float)],
            "read":[ct.c_uint64,ct.c_uint64,ct.POINTER(HostView)],
            "close":[ct.c_uint64],
        }
        for suffix,signature in signatures.items():
            fn=getattr(self.lib,"tessera_rocm_movement_resident_"+suffix)
            fn.argtypes=signature;fn.restype=ct.c_int
        handle=ct.c_uint64()
        if consumer is None:
            rc=self.lib.tessera_rocm_movement_resident_prepare(prepared._handle,ct.byref(handle))
        else:
            prepare=self.lib.tessera_rocm_movement_resident_prepare_softmax
            prepare.argtypes=[ct.c_uint64,ct.c_void_p,ct.c_size_t,ct.c_char_p,
                              ct.c_int64,ct.c_int64,ct.POINTER(ct.c_uint64)]
            prepare.restype=ct.c_int
            invoke=self.lib.tessera_rocm_movement_resident_invoke_softmax
            invoke.argtypes=[ct.c_uint64,ct.POINTER(ct.c_uint64),
                             ct.POINTER(ct.c_float),ct.POINTER(ct.c_float)]
            invoke.restype=ct.c_int
            payload=ct.create_string_buffer(consumer.native_image.payload)
            shape=prepared.output_shape
            rc=prepare(prepared._handle,payload,len(consumer.native_image.payload),
                       consumer.launch_descriptor.entry_symbol.encode(),
                       math.prod(shape[:-1]),shape[-1],ct.byref(handle))
        self.handle=handle.value
        if rc:
            if self.handle:self.lib.tessera_rocm_movement_resident_close(self.handle)
            raise RuntimeError(f"native resident preparation failed rc={rc}")
        self._finalizer=weakref.finalize(self,self.lib.tessera_rocm_movement_resident_close,self.handle)
    def _ready(self):
        if self.pid!=os.getpid():raise ValueError("resident movement cannot cross fork")
        if self.closed:raise ValueError("resident movement is closed")
    def upload(self,ordered):
        self._ready()
        with self.lock:
            self._ready()
            if len(ordered)!=len(self.prepared.input_positions):
                raise ValueError("resident movement input arity differs")
            arrays=tuple(ordered[i] for i in self.prepared.input_positions)
            metadata=views(arrays)
            rc=self.lib.tessera_rocm_movement_resident_upload(self.handle,metadata,2)
            if rc:raise RuntimeError(f"native resident upload failed rc={rc}")
    def execute(self,*,download=True):
        self._ready()
        if type(download) is not bool:raise TypeError("resident download must be boolean")
        with self.lock:
            self._ready()
            start=time.perf_counter_ns()
            generation=ct.c_uint64();elapsed=ct.c_float()
            consumer_elapsed=ct.c_float()
            if self.consumer is None:
                rc=self.lib.tessera_rocm_movement_resident_invoke(self.handle,ct.byref(generation),ct.byref(elapsed))
            else:
                rc=self.lib.tessera_rocm_movement_resident_invoke_softmax(
                    self.handle,ct.byref(generation),ct.byref(elapsed),ct.byref(consumer_elapsed))
            if rc:raise RuntimeError(f"native resident invocation failed rc={rc}")
            result=ResidentMovementResult(self,generation.value)
            output=result.to_host() if download else result
            wall=(time.perf_counter_ns()-start)/1e6
            from tessera import runtime as rt
            kernel=elapsed.value+consumer_elapsed.value
            rt._last_profile=rt.RuntimeProfile(launch_overhead_ms=wall,kernel_elapsed_ms=kernel)
            receipt=dict(self.prepared.receipt_fields,native_call_binding="resident_cpp_movement",
                         output=output,elapsed_ms=wall,kernel_elapsed_ms=kernel,
                         generation=generation.value,residency="native_owned")
            if self.consumer is not None:
                receipt.update(native_call_binding="resident_cpp_paged_softmax",
                    producer_kernel_elapsed_ms=elapsed.value,
                    consumer_kernel_elapsed_ms=consumer_elapsed.value,
                    consumer_artifact_hash=self.consumer_artifact_hash)
            return output,receipt
    def read(self,generation):
        self._ready()
        with self.lock:
            self._ready()
            output=np.empty(self.prepared.output_shape,np.float32)
            metadata=views((output,))
            rc=self.lib.tessera_rocm_movement_resident_read(self.handle,generation,ct.byref(metadata[0]))
            if rc:raise RuntimeError(f"native resident read failed rc={rc}")
            return output
    def close(self):
        if self.pid!=os.getpid():raise ValueError("resident movement cannot cross fork")
        with self.lock:
            if self.closed:return
            rc=self.lib.tessera_rocm_movement_resident_close(self.handle)
            if rc:raise RuntimeError(f"native resident close failed rc={rc}")
            self._finalizer.detach()
            self.closed=True;self.handle=0
    def __enter__(self):
        self._ready();return self
    def __exit__(self,*exc):
        self.close()

def validate_softmax_consumer(prepared,artifact):
    """Seal the existing f32 row-softmax descriptor to the producer extent."""
    from .native_artifact import (BufferBinding,ShapeGuard,ScalarArgument,
                                  LaunchGeometry,OrderingSemantics,WorkspaceRequirement)
    from .rocm_native import GFX_PAGED_KV_F32_ABI,GFX_SOFTMAX_F32_ABI
    producer=prepared.artifact
    image,descriptor=artifact.native_image,artifact.launch_descriptor
    if (producer.launch_descriptor.abi_id!=GFX_PAGED_KV_F32_ABI
            or image is None or descriptor is None):
        raise ValueError("resident edge requires native paged read and softmax packages")
    descriptor.validate_image(image)
    if (image.target!=producer.native_image.target
            or image.architecture!=producer.native_image.architecture
            or image.binary_format!="hsaco" or descriptor.abi_id!=GFX_SOFTMAX_F32_ABI):
        raise ValueError("resident softmax requires matching owning architecture and f32 ABI")
    shape=prepared.output_shape
    if len(descriptor.buffers)!=2:raise ValueError("softmax needs exactly two bindings")
    names=tuple(b.name for b in descriptor.buffers)
    expected=tuple(BufferBinding(i,name,"input" if i==0 else "output","fp32",
                                len(shape),"row_major",4) for i,name in enumerate(names))
    guards=tuple(ShapeGuard(name,axis,"eq",extent)
                 for name in names for axis,extent in enumerate(shape))
    p=descriptor.provenance
    if (descriptor.buffers!=expected or len(set(names))!=2
            or descriptor.shape_guards!=guards
            or descriptor.scalars!=(ScalarArgument(2,"Rows","int64"),ScalarArgument(3,"K","int64"))
            or descriptor.geometry!=LaunchGeometry(policy=image.architecture+"_softmax_workgroup_per_row_256")
            or descriptor.ordering!=OrderingSemantics(ordered_submission=True,residency="none",synchronization=("completion",))
            or descriptor.workspace!=WorkspaceRequirement()
            or descriptor.dynamic_local_memory_bytes or descriptor.dynamic_local_memory_expression
            or p.get("family")!="softmax" or p.get("kind")!="softmax" or p.get("axis")!=-1
            or p.get("storage")!="f32" or p.get("accum")!="f32" or p.get("keepdims") is not False
            or p.get("exp_mode")!="accurate" or p.get("ftz") is not False
            or tuple(p.get("shape",()))!=shape or tuple(p.get("output_shape",()))!=shape
            or p.get("rows")!=math.prod(shape[:-1]) or p.get("columns")!=shape[-1]):
        raise ValueError("resident softmax descriptor differs from the producer extent/ABI")
