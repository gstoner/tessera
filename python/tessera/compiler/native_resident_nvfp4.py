"""Host marshalling for native-owned compiler NVFP4 resident programs."""
from __future__ import annotations
import ctypes as C
from typing import Any
import os
import threading
import weakref
import copy
import numpy as np
import ml_dtypes
from .rocm_nvfp4_resident import _activation_inputs
from .rocm_nvfp4_ingest_native import _checked_inputs


class NativeResidentNVFP4:
    def __init__(self, program, codes, scales, globals_, a, a_scale, *, reuse=False):
        self.program=copy.deepcopy(program)
        self.program.validate()
        program=self.program
        self.capacity_m,self.n,self.k=program.consumer.m,program.consumer.n,program.consumer.k
        from .rocm_nvfp4_resident import _native_nvfp4_plan
        self.bounded_rows=(program.native_plan_json is not None and
            _native_nvfp4_plan(program.native_plan_json)["schema"].endswith(".v2"))
        self.m=self._active_rows(a)
        offsets=tuple(program.ingest.native.descriptor.provenance["row_offsets"])
        arrays=[*_checked_inputs(self.n,self.k,offsets,codes,scales,globals_),
                *_activation_inputs(self.m,self.k,a,a_scale)]
        # Preparation consumes contiguous copies; native code snapshots them
        # before returning and retains every asynchronous upload owner.
        arrays=[np.array(value,copy=True,order="C") for value in arrays]
        from tessera import runtime as rt
        if rt._rocm_live_arch()!="gfx1201":
            raise RuntimeError("native resident NVFP4 requires exact gfx1201")
        lib=rt._load_rocm_native_movement_runtime()
        if lib is None or not hasattr(lib,"tessera_rocm_nvfp4_prepare"):
            raise RuntimeError("native resident NVFP4 requires the matching runtime")
        self.lib: C.CDLL = lib
        signatures: dict[str, list[Any]] = {
            "prepare":[C.POINTER(C.c_void_p),C.POINTER(C.c_size_t),C.POINTER(C.c_char_p),
                       C.POINTER(C.c_int64),C.POINTER(C.c_uint),C.POINTER(C.c_void_p),
                       C.POINTER(C.c_size_t),C.POINTER(C.c_uint64)],
            "update":[C.c_uint64,C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t],
            "invoke":[C.c_uint64,C.c_int,C.c_int,C.POINTER(C.c_uint64),C.POINTER(C.c_float)],
            "read":[C.c_uint64,C.c_int,C.c_uint64,C.c_void_p,C.c_size_t],
            "close":[C.c_uint64],
            "graph":[C.c_uint64,C.c_int,C.c_int,C.POINTER(C.c_uint64),
                     C.POINTER(C.c_uint64),C.POINTER(C.c_float)],
        }
        signatures["update_inputs"]=[C.c_uint64,C.POINTER(C.c_void_p),C.POINTER(C.c_size_t)]
        if self.bounded_rows:
            signatures["prepare_rows"]=signatures["prepare"][:-1]+[C.c_int64,signatures["prepare"][-1]]
            signatures["update_rows"]=[C.c_uint64,C.c_int64,*signatures["update"][1:]]
            signatures["update_inputs_rows"]=[C.c_uint64,C.c_int64,*signatures["update_inputs"][1:]]
            signatures["frame_stats"]=[C.c_uint64,C.POINTER(C.c_int64),C.POINTER(C.c_int64),
                                       C.POINTER(C.c_uint64),C.POINTER(C.c_uint64)]
        if reuse:
            signatures["prepare_cached"]=signatures["prepare"]+[C.POINTER(C.c_int)]
            signatures["release_cached"]=[C.c_uint64]
            if self.bounded_rows:
                signatures["prepare_cached_rows"]=signatures["prepare_rows"]+[C.POINTER(C.c_int)]
        for name,types in signatures.items():
            fn=getattr(self.lib,"tessera_rocm_nvfp4_"+name)
            fn.argtypes=types;fn.restype=C.c_int
        self.pid=os.getpid();self.lock=threading.RLock()
        self.closed=False;self.handle=0;self.generation=0
        packages=(program.ingest.native,program.storage.native,program.consumer.package)
        blobs=[C.create_string_buffer(p.image.payload) for p in packages]
        images=(C.c_void_p*3)(*[C.cast(b,C.c_void_p).value for b in blobs])
        sizes=(C.c_size_t*3)(*[len(p.image.payload) for p in packages])
        entries=(C.c_char_p*3)(*[p.descriptor.entry_symbol.encode() for p in packages])
        dimensions=(C.c_int64*4)(self.capacity_m,self.n,self.k,len(offsets)-1)
        geometry=(C.c_uint*18)(*[v for p in packages
            for v in (*p.descriptor.geometry.grid,*p.descriptor.geometry.workgroup)])
        inputs=(C.c_void_p*5)(*[a.ctypes.data for a in arrays])
        lengths=(C.c_size_t*5)(*[a.nbytes for a in arrays])
        handle=C.c_uint64()
        self._release=(self.lib.tessera_rocm_nvfp4_release_cached if reuse
                       else self.lib.tessera_rocm_nvfp4_close)
        hit=C.c_int()
        name="prepare_cached" if reuse else "prepare"
        if self.bounded_rows:name+="_rows"
        parameters: list[Any]=[images,sizes,entries,dimensions,geometry,inputs,lengths]
        if self.bounded_rows:parameters.append(self.m)
        parameters.append(C.byref(handle))
        if reuse:parameters.append(C.byref(hit))
        rc=getattr(self.lib,"tessera_rocm_nvfp4_"+name)(*parameters)
        self.native_cache_hit=bool(hit.value)
        self.handle=handle.value
        # Register even partial preparation: a failed completion cannot discard
        # the only handle to native allocations and module leases.
        self._finalizer=weakref.finalize(self,self._release,self.handle)
        if rc:
            if self.handle:
                cleanup=self._release(self.handle)
                if cleanup==0:self._finalizer.detach();self.handle=0
            raise RuntimeError(f"native NVFP4 preparation failed rc={rc}")
        self._specs={
            "packed":(5,(self.n,self.k//2),np.uint8),
            "exponents":(6,(self.k//32,self.n),np.uint8),
            "stats":(7,(self.n,self.k//32,2),np.float64),
            "fragment":(8,(self.n,self.k//2),np.uint8),
            "plane":(9,(self.k//32+1,self.n),np.uint8),
            "output":(10,(self.m,self.n),ml_dtypes.bfloat16),
        }
    def _active_rows(self,a):
        value=np.asarray(a)
        if (value.ndim!=2 or value.shape[1]!=self.k or value.shape[0]<=0
                or value.shape[0]>self.capacity_m
                or (not self.bounded_rows and value.shape[0]!=self.capacity_m)):
            raise ValueError("native NVFP4 activation rows exceed the declared capacity")
        return int(value.shape[0])
    def _set_rows(self,rows):
        self.m=rows
        self._specs["output"]=(10,(rows,self.n),ml_dtypes.bfloat16)
    def frame_stats(self):
        self._ready()
        if not self.bounded_rows:raise ValueError("native row capacity receipt requires a bounded program")
        with self.lock:
            self._ready()
            capacity=C.c_int64();active=C.c_int64();size=C.c_uint64();count=C.c_uint64()
            rc=self.lib.tessera_rocm_nvfp4_frame_stats(self.handle,C.byref(capacity),C.byref(active),
                                                    C.byref(size),C.byref(count))
            if rc:raise RuntimeError(f"native NVFP4 capacity receipt failed rc={rc}")
            return dict(capacity_m=capacity.value,active_m=active.value,
                        allocation_bytes=size.value,allocation_count=count.value)
    def _ready(self):
        if os.getpid()!=self.pid:raise RuntimeError("native NVFP4 cannot cross fork")
        if self.closed:raise RuntimeError("native NVFP4 session is closed")
    def _invoke(self,stage,repeats=1,*,timed=False):
        self._ready()
        if type(repeats) is not int or repeats<=0 or repeats>1048576:
            raise ValueError("native repetitions require bounded positive integers")
        with self.lock:
            self._ready()
            generation=C.c_uint64();elapsed=C.c_float()
            rc=self.lib.tessera_rocm_nvfp4_invoke(self.handle,stage,repeats,
                                                C.byref(generation),C.byref(elapsed) if timed else None)
            if rc:raise RuntimeError(f"native NVFP4 invocation failed rc={rc}")
            self.generation=generation.value
            return elapsed.value
    def convert(self):self._invoke(0)
    def store(self):self._invoke(1)
    def ingest(self):self._invoke(3)
    def launch_matmul(self):self._invoke(2)
    def run_combined(self):self._invoke(4)
    def update_activations(self,a,a_scale):
        self._ready()
        rows=self._active_rows(a)
        a,scale=_activation_inputs(rows,self.k,a,a_scale)
        a,scale=(np.array(x,copy=True,order="C") for x in (a,scale))
        with self.lock:
            self._ready()
            name="update_rows" if self.bounded_rows else "update"
            parameters: list[Any]=[self.handle]
            if self.bounded_rows:parameters.append(rows)
            parameters.extend((a.ctypes.data,a.nbytes,scale.ctypes.data,scale.nbytes))
            rc=getattr(self.lib,"tessera_rocm_nvfp4_"+name)(*parameters)
            if rc:raise RuntimeError(f"native NVFP4 update failed rc={rc}")
            self._set_rows(rows)
    def update_inputs(self,codes,scales,globals_,a,a_scale):
        self._ready()
        offsets=tuple(self.program.ingest.native.descriptor.provenance["row_offsets"])
        rows=self._active_rows(a)
        values=[*_checked_inputs(self.n,self.k,offsets,codes,scales,globals_),
                *_activation_inputs(rows,self.k,a,a_scale)]
        values=[np.array(value,copy=True,order="C") for value in values]
        pointers=(C.c_void_p*5)(*[value.ctypes.data for value in values])
        sizes=(C.c_size_t*5)(*[value.nbytes for value in values])
        with self.lock:
            self._ready()
            name="update_inputs_rows" if self.bounded_rows else "update_inputs"
            parameters: list[Any]=[self.handle]
            if self.bounded_rows:parameters.append(rows)
            parameters.extend((pointers,sizes))
            rc=getattr(self.lib,"tessera_rocm_nvfp4_"+name)(*parameters)
            if rc:raise RuntimeError(f"native NVFP4 full input update failed rc={rc}")
            self._set_rows(rows)

    def _download(self,name):
        self._ready()
        with self.lock:
            self._ready()
            slot,shape,dtype=self._specs[name]
            output=np.empty(shape,dtype)
            rc=self.lib.tessera_rocm_nvfp4_read(self.handle,slot,self.generation,
                                              output.ctypes.data,output.nbytes)
            if rc:raise RuntimeError(f"native NVFP4 read failed rc={rc}")
        return output
    def read_output(self):return self._download("output")
    def conversion_diagnostics(self):
        return {name:self._download(name) for name in ("packed","exponents","stats")}
    def storage_diagnostics(self):
        return {name:self._download(name) for name in ("fragment","plane")}
    def diagnostics(self):
        return {name:self._download(name) for name in self._specs if name!="output"}
    def measure(self,stage,*,samples=3,repeats=10):
        if type(samples) is not int or samples<=0:
            raise ValueError("native timing requires positive samples")
        stages={"converter":0,"storage":1,"consumer":2,"ingest":3,"combined":4}
        if stage not in stages:raise ValueError("unknown native timing stage")
        return [self._invoke(stages[stage],repeats,timed=True) for _ in range(samples)]
    def _graph(self,stage,repeats,*,timed):
        self._ready()
        if type(repeats) is not int or repeats<=0 or repeats>4096:
            raise ValueError("native graph repeats require integers in [1,4096]")
        with self.lock:
            self._ready()
            generation=C.c_uint64();nodes=C.c_uint64();elapsed=C.c_float()
            rc=self.lib.tessera_rocm_nvfp4_graph(self.handle,stage,repeats,
                C.byref(generation),C.byref(nodes),C.byref(elapsed) if timed else None)
            if rc:raise RuntimeError(f"native NVFP4 graph failed rc={rc}")
            self.generation=generation.value
            return dict(stage=stage,repeats=repeats,graph_nodes=nodes.value,
                        host_graph_submissions=1,window_ms=elapsed.value,
                        per_iteration_ms=elapsed.value/repeats)
    def run_conversion_graph(self):
        self._graph(0,1,timed=False)
    def run_storage_graph(self):
        self._graph(1,1,timed=False)
    def run_combined_graph(self):
        self._graph(4,1,timed=False)
    def launch_matmul_graph(self):
        self._graph(2,1,timed=False)
    def measure_graph(self,stage,*,samples=3,repeats=128):
        if type(samples) is not int or samples<=0:
            raise ValueError("native graph timing requires positive samples")
        stages={"converter":0,"storage":1,"consumer":2,"ingest":3,"combined":4}
        if stage not in stages:raise ValueError("unknown native graph stage")
        result=[]
        for _ in range(samples):
            sample=self._graph(stages[stage],repeats,timed=True)
            sample["stage"]=stage
            result.append(sample)
        return result
    def close(self):
        if os.getpid()!=self.pid:raise RuntimeError("native NVFP4 cannot cross fork")
        with self.lock:
            if self.closed:return
            rc=self._release(self.handle)
            if rc:raise RuntimeError(f"native NVFP4 close failed rc={rc}")
            self._finalizer.detach();self.handle=0;self.closed=True
    def __enter__(self):self._ready();return self
    def __exit__(self,*exc):self.close()
