"""Native synchronous saved-LSE reverse owner, imported once per product pin."""
from __future__ import annotations

import ctypes as ct
from collections import OrderedDict
import os
import threading

_PROCESS = os.getpid()
_CACHE_LIMIT = 24
_cache: OrderedDict[tuple[threading.Thread, str, str], PreparedAttentionVJP] = OrderedDict()
_lock = threading.RLock()


class PreparedAttentionVJP:
    def __init__(self, metadata):
        from .native_attention_program import NativeAttentionVJPProgram
        from .resident_attention import checkpoint_shapes
        self.identity = (metadata["program_digest"], metadata["program_json"])
        self.program = NativeAttentionVJPProgram.from_json(
            metadata["program_json"], expected_digest=metadata["program_digest"]
        )
        dims, physical = checkpoint_shapes(self.program.pair)
        self.dims = dims
        b, hq, _, sq, sk, _, _ = dims
        provenance = self.program.pair.forward.descriptor.provenance
        self.bias_shape = (tuple(provenance.get("bias_shape", ())) or (b, hq, sq, sk)) if provenance.get("bias") else None
        physical_inputs = (*physical[:3], *((self.bias_shape,) if self.bias_shape else ()))
        frontend = [None] * len(physical_inputs)
        for role, index in enumerate(self.program.input_indices):
            frontend[index] = physical_inputs[role]
        self.names = tuple(f"primal_{i}" for i in range(len(frontend))) + ("cotangent",)
        self.shapes = (*frontend, physical[3])
        self.output_shapes = tuple(physical_inputs[i] for i in self.program.active)
        self.handle = 0
        self.closed = False
        self.pid = os.getpid()
        self.lock = threading.RLock()
        self.lib: ct.CDLL | None = None
        self.last_device_ms = None

    def _library(self) -> ct.CDLL:
        lib = self.lib
        if lib is None:
            raise RuntimeError("native prepared reverse runtime has not been initialized")
        return lib

    def _check(self, status):
        if status:
            reason = self._library().tessera_nvidia_attention_vjp_last_error()
            raise RuntimeError(reason.decode() if reason else "native prepared reverse failed")

    def _prepare(self):
        from tessera.runtime import _load_nvidia_ptx_launch
        lib = _load_nvidia_ptx_launch()
        if lib is None:
            raise RuntimeError("native PTX runtime unavailable")
        self.lib = lib
        P, S, I, U, L = ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_uint64, ct.c_int64
        lib.tessera_nvidia_attention_vjp_prepare.argtypes = [
            P, S, ct.c_char_p, P, S, ct.c_char_p,
            ct.POINTER(L), ct.POINTER(L), ct.POINTER(I), ct.POINTER(I), S, ct.POINTER(U),
        ]
        lib.tessera_nvidia_attention_vjp_prepare.restype = I
        lib.tessera_nvidia_attention_vjp_invoke.argtypes = [
            U, ct.POINTER(P), ct.POINTER(S), S, ct.POINTER(P), ct.POINTER(S), S, ct.POINTER(ct.c_float),
        ]
        lib.tessera_nvidia_attention_vjp_invoke.restype = I
        lib.tessera_nvidia_attention_vjp_close.argtypes = [U]
        lib.tessera_nvidia_attention_vjp_close.restype = I
        lib.tessera_nvidia_attention_vjp_last_error.argtypes = []
        lib.tessera_nvidia_attention_vjp_last_error.restype = ct.c_char_p
        program = self.program
        if program.program_digest != self.identity[0]:
            raise ValueError("prepared reverse differs from pinned identity")
        forward, backward = program.pair.forward, program.pair.backward
        fimage = ct.create_string_buffer(forward.image.payload)
        bimage = ct.create_string_buffer(backward.image.payload)
        dims = (L * 7)(*self.dims)
        bias = (L * 4)(*self.bias_shape) if self.bias_shape else None
        mapping = (I * len(program.input_indices))(*program.input_indices)
        roles = (I * len(program.active))(*program.active)
        handle = U()
        self._check(lib.tessera_nvidia_attention_vjp_prepare(
            fimage, len(forward.image.payload), forward.descriptor.entry_symbol.encode(),
            bimage, len(backward.image.payload), backward.descriptor.entry_symbol.encode(),
            dims, bias, mapping, roles, len(program.active), ct.byref(handle),
        ))
        self.handle = handle.value

    def invoke(self, metadata, args):
        import numpy as np
        if self.pid != os.getpid():
            raise ValueError("prepared reverse cannot cross fork")
        with self.lock:
            if self.closed:
                raise ValueError("prepared reverse is closed")
            self.last_device_ms = None
            if (metadata["program_digest"], metadata["program_json"]) != self.identity:
                raise ValueError("prepared reverse differs from pinned identity")
            if tuple(metadata.get("arg_names", ())) != self.names or len(args) != len(self.shapes):
                raise ValueError("attention VJP launch names/arity differ from native contract")
            values = tuple(np.asarray(x) for x in args)
            if any(x.dtype != np.float32 or x.shape != shape
                   for x, shape in zip(values, self.shapes, strict=True)):
                raise ValueError("attention VJP host storage differs from native contract")
            values = tuple(np.ascontiguousarray(x) for x in values)
            if not self.handle:
                self._prepare()
            outputs = tuple(np.empty(shape, np.float32) for shape in self.output_shapes)
            pointers = (ct.c_void_p * len(values))(*(x.ctypes.data for x in values))
            sizes = (ct.c_size_t * len(values))(*(x.nbytes for x in values))
            destinations = (ct.c_void_p * len(outputs))(*(x.ctypes.data for x in outputs))
            lengths = (ct.c_size_t * len(outputs))(*(x.nbytes for x in outputs))
            times = (ct.c_float * 2)()
            self._check(self._library().tessera_nvidia_attention_vjp_invoke(
                self.handle, pointers, sizes, len(values), destinations, lengths, len(outputs), times,
            ))
            self.last_device_ms = tuple(times)
            return outputs

    def close(self):
        if self.pid != os.getpid():
            raise ValueError("prepared reverse cannot cross fork")
        with self.lock:
            if self.closed:
                return
            if self.handle:
                self._check(self._library().tessera_nvidia_attention_vjp_close(self.handle))
            self.handle = 0
            self.closed = True


def prepared(metadata):
    if os.getpid() != _PROCESS:
        raise ValueError("prepared reverse service cannot cross fork")
    key = (threading.current_thread(), metadata["program_digest"], metadata["program_json"])
    with _lock:
        owner = _cache.get(key)
        if owner is None or owner.closed:
            owner = PreparedAttentionVJP(metadata)
            _cache[key] = owner
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_LIMIT:
            _, retired = _cache.popitem(last=False)
            retired.close()
        return owner


def clear_prepared():
    if os.getpid() != _PROCESS:
        raise ValueError("prepared reverse service cannot cross fork")
    with _lock:
        while _cache:
            _, owner = _cache.popitem()
            owner.close()


def execute(metadata, args):
    return prepared(metadata).invoke(metadata, args)


def _forget_inherited():
    global _cache, _lock
    _cache = OrderedDict()
    _lock = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_inherited)
