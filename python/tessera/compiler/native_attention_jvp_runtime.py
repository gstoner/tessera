"""Runtime consumer of the pinned compiler-owned saved-LSE JVP program."""
from __future__ import annotations

def execute_unprepared(metadata,args):
    import numpy as np
    from .native_attention_program import NativeAttentionJVPProgram
    from .emit.nvidia_cuda import NvidiaDeviceSession, CudaOwnedDeviceBuffer
    program=NativeAttentionJVPProgram.from_json(
        metadata["program_json"],expected_digest=metadata["program_digest"])
    count=len(program.input_indices)
    names=[f"primal_{i}" for i in range(count)]+[
        f"tangent_{program.input_indices[i]}" for i in program.active]
    if metadata.get("arg_names")!=names:
        raise ValueError("attention JVP launch names disagree with native activity")
    if len(args)!=len(names):
        raise ValueError("attention JVP launch arity disagrees")
    values=tuple(np.asarray(x) for x in args)
    physical_shapes=program.pair.forward.descriptor.provenance["shape"]
    b,hq,hkv,sq,sk,d,dv=physical_shapes
    policy=program.pair.forward.descriptor.provenance
    bias_shape=tuple(policy.get("bias_shape",())) or (b,hq,sq,sk)
    physical=((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv)) + (
        (bias_shape,) if policy.get("bias",False) else ())
    frontend=[None]*count
    for i,index in enumerate(program.input_indices):
        frontend[index]=physical[i]
    expected=tuple(frontend)+tuple(physical[i] for i in program.active)
    if any(x.dtype!=np.float32 or x.shape!=shape
           for x,shape in zip(values,expected,strict=True)):
        raise ValueError("attention JVP host storage disagrees with native contract")
    # Native session owns uploads. Capture owns its private O/LSE generation;
    # downloads complete before either owner releases its allocations.
    with NvidiaDeviceSession() as session:
        resident=[session.upload(x) for x in values]
        with program.capture(*resident[:count]) as frame:
            result=frame.jvp(*resident[count:])
            def download(value):
                if isinstance(value,tuple):
                    return tuple(download(item) for item in value)
                interface=value.__cuda_array_interface__
                view=CudaOwnedDeviceBuffer(session,int(interface["data"][0]),
                    tuple(interface["shape"]),np.float32,
                    int(np.prod(interface["shape"]))*4,owns=False)
                return session.download(view)
            return download(frame.primal),download(result)

# CUDA ownership, module retention, arena layout and call binding are native.
# The Python LRU only retains bounded registration handles for immutable pins.
import ctypes as ct
from typing import Any
import os
from collections import OrderedDict
import tempfile
import threading
from pathlib import Path

_cache: OrderedDict[tuple[int, threading.Thread, str, str], PreparedAttentionJVP] = OrderedDict()
_cache_lock=threading.RLock()
_CACHE_LIMIT=24

class PreparedAttentionJVP:
    def __init__(self,metadata):
        from .native_attention_program import NativeAttentionJVPProgram
        self.identity=(metadata["program_digest"],metadata["program_json"])
        self.program=NativeAttentionJVPProgram.from_json(
            metadata["program_json"],expected_digest=metadata["program_digest"])
        program=self.program
        self.count=len(program.input_indices)
        self.biased=program.pair.forward.descriptor.provenance.get("bias",False)
        self.saved_lse=program.saved_lse
        self.names=tuple(f"primal_{i}" for i in range(self.count))+tuple(
            f"tangent_{program.input_indices[i]}" for i in program.active)
        b,hq,hkv,sq,sk,d,dv=program.pair.forward.descriptor.provenance["shape"]
        self.bias_shape=tuple(program.pair.forward.descriptor.provenance.get("bias_shape",())) or (b,hq,sq,sk)
        physical=((b,hq,sq,d),(b,hkv,sk,d),(b,hkv,sk,dv)) + (
            (self.bias_shape,) if self.biased else ())
        frontend=[None]*self.count
        for i,index in enumerate(program.input_indices):frontend[index]=physical[i]
        self.shapes=tuple(frontend)+tuple(physical[i] for i in program.active)
        self.output_shape=(b,hq,sq,dv)
        self.lse_shape=(b,hq,sq)
        self.handle=0
        self.closed=False
        self.pid=os.getpid()
        self.lock=threading.RLock()
        self.lib: ct.CDLL | None = None
        self.last_device_ms=None

    def _library(self) -> ct.CDLL:
        lib = self.lib
        if lib is None:
            raise RuntimeError("native prepared attention runtime has not been initialized")
        return lib

    def _check(self,status):
        if status:
            reason=self._library().tessera_nvidia_attention_jvp_last_error()
            raise RuntimeError(reason.decode() if reason else "native prepared attention failed")

    def _prepare(self):
        from tessera.runtime import _load_nvidia_ptx_launch
        lib=_load_nvidia_ptx_launch()
        if lib is None:raise RuntimeError("native PTX runtime unavailable")
        self.lib=lib
        P,S,I,U=ct.c_void_p,ct.c_size_t,ct.c_int,ct.c_uint64
        prepare=(lib.tessera_nvidia_attention_jvp_prepare_lse if self.saved_lse else
                 lib.tessera_nvidia_attention_jvp_prepare_bias if self.biased
                 else lib.tessera_nvidia_attention_jvp_prepare)
        signature: list[Any] = [
            P,S,ct.c_char_p,P,S,ct.c_char_p,ct.c_char_p,ct.c_char_p,
            ct.POINTER(ct.c_int64)] + (
                [ct.POINTER(ct.c_int64)] if self.biased or self.saved_lse else []) + [
            ct.POINTER(I),ct.POINTER(I),S,ct.POINTER(U)]
        prepare.argtypes=signature
        prepare.restype=I
        lib.tessera_nvidia_attention_jvp_invoke.argtypes=[
            U,ct.POINTER(P),ct.POINTER(S),S,ct.POINTER(P),ct.POINTER(S),ct.POINTER(ct.c_float)]
        lib.tessera_nvidia_attention_jvp_invoke.restype=I
        lib.tessera_nvidia_attention_jvp_close.argtypes=[U]
        lib.tessera_nvidia_attention_jvp_close.restype=I
        lib.tessera_nvidia_attention_jvp_last_error.argtypes=[]
        lib.tessera_nvidia_attention_jvp_last_error.restype=ct.c_char_p
        program=self.program
        if program.program_digest!=self.identity[0]:
            raise ValueError("prepared attention program differs from pinned identity")
        forward=ct.create_string_buffer(program.pair.forward.image.payload)
        tangent=ct.create_string_buffer(program.tangent.image)
        dims=(ct.c_int64*7)(*program.pair.forward.descriptor.provenance["shape"])
        mapping=(I*self.count)(*program.input_indices)
        roles=(I*len(program.active))(*program.active)
        handle=U()
        bias=(((ct.c_int64*4)(*self.bias_shape) if self.biased else None),) if self.biased or self.saved_lse else ()
        with tempfile.TemporaryDirectory(prefix="tessera-prepared-attention-sizer-") as directory:
            path=Path(directory)/"sizer.so";path.write_bytes(program.tangent.host_library)
            self._check(prepare(
                forward,len(program.pair.forward.image.payload),
                program.pair.forward.descriptor.entry_symbol.encode(),
                tangent,len(program.tangent.image),program.tangent.entry.encode(),
                str(path).encode(),program.tangent.sizer.encode(),dims,*bias,mapping,roles,
                len(program.active),ct.byref(handle)))
        self.handle=handle.value

    def invoke(self,metadata,args):
        import numpy as np
        if self.pid!=os.getpid():raise ValueError("prepared attention cannot cross fork")
        with self.lock:
            if self.closed:raise ValueError("prepared attention is closed")
            self.last_device_ms=None
            if (metadata["program_digest"],metadata["program_json"])!=self.identity:
                raise ValueError("prepared attention launch differs from pinned identity")
            if tuple(metadata.get("arg_names",()))!=self.names:
                raise ValueError("attention JVP launch names disagree with native activity")
            if len(args)!=len(self.shapes):raise ValueError("attention JVP launch arity disagrees")
            resident=any(hasattr(value,"__cuda_array_interface__") for value in args)
            if resident:
                from .resident_nvidia_tensor import cuda_frontend_specs
                if not all(hasattr(value,"__cuda_array_interface__") for value in args):
                    raise ValueError("attention JVP requires all resident roots or all host roots")
                specs=cuda_frontend_specs(args,ranks=(4,))
                if any(dtype!=np.dtype("float32") or shape!=expected
                       for (shape,dtype),expected in zip(specs,self.shapes,strict=True)):
                    raise ValueError("attention JVP resident storage disagrees with native contract")
                values=tuple(args)
            else:
                values=tuple(np.asarray(x) for x in args)
                if any(x.dtype!=np.float32 or x.shape!=shape
                       for x,shape in zip(values,self.shapes,strict=True)):
                    raise ValueError("attention JVP host storage disagrees with native contract")
                values=tuple(np.ascontiguousarray(x) for x in values)
            if not self.handle:self._prepare()
            output_shapes=(self.output_shape,self.lse_shape,self.output_shape,self.lse_shape) if self.saved_lse else (self.output_shape,)*2
            outputs=tuple(np.empty(shape,np.float32) for shape in output_shapes)
            if resident:
                import math
                interfaces=tuple(value.__cuda_array_interface__ for value in values)
                pointers=(ct.c_void_p*len(values))(*(item["data"][0] for item in interfaces))
                lengths=(ct.c_size_t*len(values))(*(math.prod(shape)*dtype.itemsize for shape,dtype in specs))
                streams=(ct.c_uint64*len(values))(*(item["stream"] for item in interfaces))
            else:
                pointers=(ct.c_void_p*len(values))(*(x.ctypes.data for x in values))
                lengths=(ct.c_size_t*len(values))(*(x.nbytes for x in values))
            destinations=(ct.c_void_p*len(outputs))(*(x.ctypes.data for x in outputs))
            sizes=(ct.c_size_t*len(outputs))(*(x.nbytes for x in outputs))
            times=(ct.c_float*2)()
            if resident:
                lib=self._library()
                try:
                    invoke=(lib.tessera_nvidia_attention_jvp_invoke_lse_resident_ordered if self.saved_lse else
                            lib.tessera_nvidia_attention_jvp_invoke_resident_ordered)
                except AttributeError as exc:
                    raise RuntimeError("native attention runtime lacks ordered resident JVP ABI") from exc
                resident_signature: list[Any] = [
                    ct.c_uint64,ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_size_t),ct.c_size_t,
                    ct.POINTER(ct.c_uint64),ct.c_size_t,ct.POINTER(ct.c_void_p),
                    ct.POINTER(ct.c_size_t)] + ([ct.c_size_t] if self.saved_lse else []) + [ct.POINTER(ct.c_float)]
                invoke.argtypes=resident_signature
                invoke.restype=ct.c_int
                self._check(invoke(self.handle,pointers,lengths,len(values),streams,len(values),
                                   destinations,sizes,*((len(outputs),) if self.saved_lse else ()),times))
            else:
                invoke=(self._library().tessera_nvidia_attention_jvp_invoke_lse if self.saved_lse else
                        self._library().tessera_nvidia_attention_jvp_invoke)
                host_signature: list[Any] = [ct.c_uint64,ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_size_t),
                                 ct.c_size_t,ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_size_t)] + (
                                 [ct.c_size_t] if self.saved_lse else []) + [ct.POINTER(ct.c_float)]
                invoke.argtypes=host_signature
                invoke.restype=ct.c_int
                self._check(invoke(self.handle,pointers,lengths,len(values),destinations,sizes,
                                   *((len(outputs),) if self.saved_lse else ()),times))
            self.last_device_ms=tuple(times)
            return ((outputs[0],outputs[1]),(outputs[2],outputs[3])) if self.saved_lse else outputs

    def close(self):
        if self.pid!=os.getpid():raise ValueError("prepared attention cannot cross fork")
        with self.lock:
            if self.closed:return
            if self.handle:self._check(self._library().tessera_nvidia_attention_jvp_close(self.handle))
            self.handle=0;self.closed=True

def prepared(metadata):
    key=(os.getpid(),threading.current_thread(),metadata["program_digest"],metadata["program_json"])
    with _cache_lock:
        owner=_cache.get(key)
        if owner is None or owner.closed:
            owner=PreparedAttentionJVP(metadata);_cache[key]=owner
        _cache.move_to_end(key)
        while len(_cache)>_CACHE_LIMIT:
            _,retired=_cache.popitem(last=False);retired.close()
        return owner

def clear_prepared():
    with _cache_lock:
        while _cache:
            _,owner=_cache.popitem();owner.close()

def execute(metadata,args):
    return prepared(metadata).invoke(metadata,args)


def _forget_inherited_owners():
    # Inherited locks may belong to vanished threads. Drop registrations without
    # attempting CUDA retirement; native PID guards require a fresh exec.
    global _cache,_cache_lock
    _cache=OrderedDict()
    _cache_lock=threading.RLock()

if hasattr(os,"register_at_fork"):
    os.register_at_fork(after_in_child=_forget_inherited_owners)
