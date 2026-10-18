"""Native ownership for compiler-outlined RMSNorm to saved-LSE attention."""
from __future__ import annotations

import ctypes as ct
import math
import os
import threading
import time
import weakref

import numpy as np

from .prepared_attention_forward import input_signature, _retire


class PreparedSourceAttention:
    def __init__(self, program, *, argument_names=("input_0", "input_1", "input_2")):
        from .portable_source_attention import seal, restore
        program.validate()
        artifacts = program.scheduled_artifacts()
        packages = program.package_members(pipeline_name="tessera-nvidia-pipeline-sm120")
        data = seal(program, artifacts, packages, argument_names)
        self._initialize(program, *restore(data))

    @classmethod
    def from_manifest(cls, data):
        from .portable_source_attention import restore
        validated = restore(data)
        owner = cls.__new__(cls)
        owner._initialize(None, *validated)
        return owner

    def _initialize(self, program, data, plan, packages, dims):
        import json
        self.program, self.plan = program, plan
        self._portable_text = json.dumps(data, sort_keys=True, allow_nan=False)
        self.argument_names = tuple(data["argument_names"])
        self.norm, self.attention = packages
        self.dims = dims
        self.cooperative = self.norm.descriptor.provenance["schedule"] == "cooperative_128"
        self.storage = {"f16": 2, "bf16": 3}[plan["buffers"][3]["storage"]]
        self.roles = tuple(plan["role_indices"])
        self.output_order = tuple(index - 4 for index in plan["outputs"])
        self.shapes = tuple(tuple(plan["buffers"][index]["shape"]) for index in self.roles)
        self.output_shapes = tuple(tuple(plan["buffers"][index]["shape"]) for index in (4, 5))
        self.pid, self.closed, self.handle = os.getpid(), False, 0
        self.lock = threading.RLock()
        self.lib: ct.CDLL | None = None
        self._finalizer = None

    def manifest(self):
        import json
        return json.loads(self._portable_text)

    def runtime_artifact(self):
        from .source_attention_runtime import runtime_artifact
        return runtime_artifact(self)

    def _library(self) -> ct.CDLL:
        if self.lib is None:
            raise RuntimeError("native source attention runtime is uninitialized")
        return self.lib

    def _check(self, status):
        if status:
            error = self._library().tessera_nvidia_attention_forward_last_error()
            raise RuntimeError(error.decode() if error else "native source attention failed")

    def _prepare(self):
        from tessera.runtime import _load_nvidia_ptx_launch
        self.lib = _load_nvidia_ptx_launch()
        if self.lib is None or not hasattr(self.lib, "tessera_nvidia_norm_attention_prepare"):
            raise RuntimeError("native composed attention owner unavailable")
        P, S, L, U, I = ct.c_void_p, ct.c_size_t, ct.c_int64, ct.c_uint64, ct.c_int
        self.lib.tessera_nvidia_attention_forward_last_error.argtypes = []
        self.lib.tessera_nvidia_attention_forward_last_error.restype = ct.c_char_p
        self.lib.tessera_nvidia_norm_attention_prepare.argtypes = [
            P, S, ct.c_char_p, I, P, S, ct.c_char_p, ct.POINTER(L), I, ct.POINTER(U)]
        self.lib.tessera_nvidia_norm_attention_prepare.restype = I
        self.lib.tessera_nvidia_attention_forward_invoke.argtypes = [
            U, ct.POINTER(P), ct.POINTER(S), S, ct.POINTER(U), S,
            ct.POINTER(P), ct.POINTER(S), S, ct.POINTER(ct.c_float)]
        self.lib.tessera_nvidia_attention_forward_invoke.restype = I
        self.lib.tessera_nvidia_attention_forward_close.argtypes = [U]
        self.lib.tessera_nvidia_attention_forward_close.restype = I
        self.lib.tessera_nvidia_norm_attention_last_times.argtypes = [
            U, ct.POINTER(ct.c_float), ct.POINTER(ct.c_float)]
        self.lib.tessera_nvidia_norm_attention_last_times.restype = I
        norm = ct.create_string_buffer(self.norm.image.payload)
        attention = ct.create_string_buffer(self.attention.image.payload)
        handle = U()
        self._check(self.lib.tessera_nvidia_norm_attention_prepare(
            norm, len(self.norm.image.payload), self.norm.descriptor.entry_symbol.encode(),
            self.cooperative, attention, len(self.attention.image.payload),
            self.attention.descriptor.entry_symbol.encode(), (L * 7)(*self.dims),
            self.storage, ct.byref(handle)))
        self.handle = handle.value
        self._finalizer = weakref.finalize(self, _retire, self.lib, self.handle, self.pid)

    def __call__(self, *roots):
        if self.pid != os.getpid():
            raise ValueError("native source attention cannot cross fork")
        with self.lock:
            if self.closed:
                raise ValueError("native source attention is closed")
            if len(roots) != 3:
                raise ValueError("native source attention requires three original roots")
            values = tuple(roots[index] for index in self.roles)
            signature = input_signature(values)
            dtype_name = "float16" if self.storage == 2 else "bfloat16"
            if signature != tuple((dtype_name, shape) for shape in self.shapes):
                raise ValueError("native source attention input shape or storage differs")
            resident = hasattr(values[0], "__cuda_array_interface__")
            if not resident:
                values = tuple(np.ascontiguousarray(value) for value in values)
            if self.storage == 3:
                import ml_dtypes
                dtype = ml_dtypes.bfloat16
            else:
                dtype = np.float16
            outputs = (np.empty(self.output_shapes[0], dtype),
                       np.empty(self.output_shapes[1], np.float32))
            start = time.perf_counter()
            if not self.handle:
                self._prepare()
            if resident:
                interfaces = tuple(value.__cuda_array_interface__ for value in values)
                pointers = (ct.c_void_p * 3)(*(item["data"][0] for item in interfaces))
                streams = (ct.c_uint64 * 3)(*(item["stream"] for item in interfaces))
            else:
                pointers = (ct.c_void_p * 3)(*(value.ctypes.data for value in values))
                streams = None
            elapsed, prefix, attention = ct.c_float(), ct.c_float(), ct.c_float()
            self._check(self._library().tessera_nvidia_attention_forward_invoke(
                self.handle, pointers, (ct.c_size_t * 3)(
                    *(math.prod(shape) * 2 for shape in self.shapes)), 3,
                streams, 3 if resident else 0,
                (ct.c_void_p * 2)(*(value.ctypes.data for value in outputs)),
                (ct.c_size_t * 2)(*(value.nbytes for value in outputs)), 2,
                ct.byref(elapsed)))
            self._check(self._library().tessera_nvidia_norm_attention_last_times(
                self.handle, ct.byref(prefix), ct.byref(attention)))
            receipt = {
                "ok": True, "execution_kind": "native_gpu",
                "native_call_binding": "prepared_cpp_norm_attention",
                "source_hash": self.plan["source_hash"],
                "member_images": [self.norm.image.image_digest, self.attention.image.image_digest],
                "private_intermediate": "native_owned_normalized_q",
                "norm_device_ms": prefix.value, "attention_device_ms": attention.value,
                "program_device_ms": elapsed.value,
                "completed_host_ms": (time.perf_counter() - start) * 1000,
                "host_preparation": "native_ordered_resident_snapshot" if resident
                                    else "native_pinned_host_snapshot",
            }
            return tuple(outputs[index] for index in self.output_order), receipt

    def close(self):
        if self.pid != os.getpid():
            raise ValueError("native source attention cannot cross fork")
        with self.lock:
            if self.closed:
                return
            if self.handle:
                self._check(self._library().tessera_nvidia_attention_forward_close(self.handle))
            if self._finalizer is not None:
                self._finalizer.detach()
            self.handle, self.closed = 0, True

    def __enter__(self):
        if self.closed:
            raise ValueError("native source attention is closed")
        return self

    def __exit__(self, *exc):
        self.close()
