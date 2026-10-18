"""Typed metadata and lifetime binding for native CUDA tensor result generations."""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping
import ctypes as ct
import os
import threading
import weakref
from typing import Any
import numpy as np

from .prepared_nvidia_matmul import HostView


def bind(lib):
    signatures = {
        "tessera_nvidia_tensor_program_reserve_result": [ct.c_uint64, ct.POINTER(ct.c_uint64)],
        "tessera_nvidia_tensor_program_submit_result": [
            ct.c_uint64, ct.c_uint64, ct.POINTER(HostView), ct.c_size_t,
            ct.POINTER(ct.c_uint64), ct.c_size_t],
        "tessera_nvidia_tensor_result_query": [
            ct.c_uint64, ct.c_size_t, ct.POINTER(HostView),
            ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_uint64)],
        "tessera_nvidia_tensor_result_read": [ct.c_uint64, ct.c_size_t, ct.POINTER(HostView)],
        "tessera_nvidia_tensor_result_record_consumer": [ct.c_uint64, ct.c_uint64],
        "tessera_nvidia_tensor_result_close": [ct.c_uint64],
    }
    if any(not hasattr(lib, name) for name in signatures):
        raise RuntimeError("native CUDA result placement requires an updated provider")
    for name, args in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = ct.c_int


class _Batch:
    def __init__(self, lib, handle):
        self.lib, self.handle, self.pid = lib, handle, os.getpid()
        self.lock = threading.RLock()
        self.finalizer = weakref.finalize(self, lib.tessera_nvidia_tensor_result_close, handle)

    def check(self, code):
        if code:
            reason = self.lib.tessera_nvidia_matmul_last_error()
            raise RuntimeError(reason.decode() if reason else "native tensor result failed")

    def guard(self):
        if os.getpid() != self.pid:
            raise ValueError("native tensor result cannot cross fork")
        if not self.finalizer.alive:
            raise ValueError("native tensor result is closed")

    def query(self, slot):
        self.guard()
        view, stream, context = HostView(), ct.c_uint64(), ct.c_uint64()
        self.check(self.lib.tessera_nvidia_tensor_result_query(
            self.handle, slot, ct.byref(view), ct.byref(stream), ct.byref(context)))
        return view, stream.value, context.value

    def record(self, stream):
        self.guard()
        value = stream.value if isinstance(stream, ct.c_void_p) else stream
        if type(value) is not int or not 0 <= value < 2**64:
            raise ValueError("native tensor result consumer requires a CUDA stream handle")
        self.check(self.lib.tessera_nvidia_tensor_result_record_consumer(self.handle, value or 1))


class NativeTensorResult:
    """Read-only FP32 device tensor owned independently of its prepared program.

    Call to_host() for an explicit completed host copy. External asynchronous
    consumers must record_consumer(stream) after enqueueing their last read.
    """
    def __init__(self, batch, slot):
        self._batch = batch
        self._slot = slot

    def _guard(self):
        if self._batch is None:
            raise ValueError("native tensor result is closed")
        self._batch.guard()
        return self._batch

    @property
    def __cuda_array_interface__(self):
        batch = self._guard()
        with batch.lock:
            self._guard()
            view, stream, _ = batch.query(self._slot)
            return {"version": 3, "shape": tuple(view.shape), "strides": tuple(view.strides),
                    "typestr": np.dtype("float32").str, "data": (int(view.data), True),
                    "stream": stream}

    @property
    def shape(self):
        return self.__cuda_array_interface__["shape"]

    @property
    def strides(self):
        return self.__cuda_array_interface__["strides"]

    @property
    def dtype(self):
        self._guard()
        return np.dtype("float32")

    @property
    def ndim(self):
        self._guard()
        return 2

    @property
    def nbytes(self):
        batch = self._guard()
        with batch.lock:
            self._guard()
            return batch.query(self._slot)[0].bytes

    def __array__(self, dtype=None, copy=None):
        raise TypeError("native device result requires an explicit to_host() transfer")

    def to_host(self):
        from .prepared_nvidia_tensor_jvp import _view
        batch = self._guard()
        with batch.lock:
            self._guard()
            shape = tuple(batch.query(self._slot)[0].shape)
            output = np.empty(shape, dtype=np.float32)
            view = _view(output)
            batch.check(batch.lib.tessera_nvidia_tensor_result_read(
                batch.handle, self._slot, ct.byref(view)))
            return output

    def record_consumer(self, stream):
        batch = self._guard()
        with batch.lock:
            self._guard()
            batch.record(stream)

    def close(self):
        batch = self._batch
        if batch is None:
            return
        batch.guard()
        with batch.lock:
            batch.guard()
            self._batch = None


def execute(owner, values):
    """Bind host or resident roots to a native program; return private result slots."""
    from .prepared_nvidia_tensor_jvp import _view
    from .resident_nvidia_tensor import cuda_frontend_specs, ordered_resident_views
    owner._guard()
    if len(values) != 3:
        raise ValueError("native tensor results require three explicit roots")
    plan = owner.plan
    resident = any(hasattr(value, "__cuda_array_interface__") for value in values)
    if resident:
        if not all(hasattr(value, "__cuda_array_interface__") for value in values):
            raise ValueError("native result roots must use one storage placement")
        specs = cuda_frontend_specs(values, ranks=(2,))
        for (shape, dtype), expected in zip(specs, plan["buffers"][:3], strict=True):
            storage = {"f16": "float16", "bf16": "bfloat16"}[expected["storage"]]
            if list(shape) != expected["shape"] or dtype.name != storage:
                raise ValueError("native result root shape/storage differs")
        roots = tuple(values)
        views, streams = ordered_resident_views(roots, None, writable_from=3)
        producers = (ct.c_uint64 * 3)(*streams)
        count = 3
    else:
        if any(not isinstance(value, np.ndarray) for value in values):
            raise ValueError("native host result roots require ndarrays")
        for value, expected in zip(values, plan["buffers"][:3], strict=True):
            storage = {"f16": "float16", "bf16": "bfloat16"}[expected["storage"]]
            if (list(value.shape) != expected["shape"] or value.dtype.name != storage
                    or not value.dtype.isnative):
                raise ValueError("native result root shape/storage differs")
        roots = tuple(np.ascontiguousarray(value) for value in values)
        views = (HostView * 3)(*(_view(value) for value in roots))
        producers = None
        count = 0
    lib = owner.lib
    bind(lib)
    handle = ct.c_uint64()
    owner._check(lib.tessera_nvidia_tensor_program_reserve_result(owner.handle, ct.byref(handle)))
    batch = _Batch(lib, handle.value)
    try:
        owner._check(lib.tessera_nvidia_tensor_program_submit_result(
            owner.handle, handle.value, views, 3, producers, count))
        pair = (NativeTensorResult(batch, 0), NativeTensorResult(batch, 1))
        for value in pair:
            _ = value.__cuda_array_interface__
    except BaseException:
        batch.finalizer()
        raise
    owner.receipt.update(result_placement="cuda", result_ownership="native_generation",
                         input_binding="cuda_resident_roots" if resident else "host_roots",
                         root_uploads=0 if resident else 3)
    return pair


def _results(value, seen=None):
    if isinstance(value, NativeTensorResult):
        return [value]
    from tessera.runtime import NativeBufferValue
    if seen is None:
        seen = set()
    if id(value) in seen:
        return []
    if isinstance(value, NativeBufferValue):
        seen.add(id(value))
        return _results(value.value, seen)
    if isinstance(value, (tuple, list, Mapping)):
        seen.add(id(value))
        children = value.values() if isinstance(value, Mapping) else value
        return [result for child in children for result in _results(child, seen)]
    return []


@contextmanager
def input_scope(values: Any, stream=None):
    """Pin generations through native consumption and retain explicit stream completion."""
    results = _results(values)
    batches = sorted({result._guard() for result in results}, key=lambda batch: batch.handle)
    acquired = []
    try:
        for batch in batches:
            batch.lock.acquire()
            acquired.append(batch)
        for result in results:
            result._guard()
        yield
    finally:
        try:
            if stream is not None:
                for batch in acquired:
                    batch.record(stream)
        finally:
            for batch in reversed(acquired):
                batch.lock.release()
