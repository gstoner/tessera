"""Run a packaged native GPU storage kernel from host arrays.

Shared by the Clifford and EBM device routes (2026-09-16): device buffers are
allocated per call through the bound package's own driver handle, inputs are
copied in, the binding launches on the caller's current context, outputs are
copied back into new host arrays, and every buffer is freed before returning.
Correctness evidence only -- per-call transfers are not a performance path.
"""
from __future__ import annotations

import ctypes as ct
import threading
from types import SimpleNamespace

import numpy as np

from .native_gpu_tensor import TensorSpec

_CONTEXT_LOCK = threading.Lock()
_CONTEXTS: dict[str, bool] = {}
_DTYPES = {"fp32": (np.float32, "<f4"), "int64": (np.int64, "<i8"), "int32": (np.int32, "<i4"), "fp64": (np.float64, "<f8")}


def ensure_device_context(backend: str) -> None:
    """Make device 0 current for this process once (the bound package launches
    on the caller's active context and never creates one)."""
    with _CONTEXT_LOCK:
        if _CONTEXTS.get(backend):
            return
        cuda = backend == "nvidia"
        driver = ct.CDLL("libcuda.so.1" if cuda else "libamdhip64.so")
        P = ct.c_void_p

        def call(name, types, *args):
            fn = getattr(driver, name)
            fn.argtypes, fn.restype = types, ct.c_int
            status = fn(*args)
            if status:
                raise RuntimeError(f"native host program: {name} failed with status {status}")
        if cuda:
            call("cuInit", [ct.c_uint], 0)
            context = P()
            call("cuDevicePrimaryCtxRetain", [ct.POINTER(P), ct.c_int], ct.byref(context), 0)
            call("cuCtxSetCurrent", [P], context)
        else:
            call("hipInit", [ct.c_uint], 0)
            call("hipSetDevice", [ct.c_int], 0)
        _CONTEXTS[backend] = True


class HostArrayProgram:
    """One bound native tensor call, runnable from host arrays; the spec order
    is the ABI order (inputs first, then writable outputs)."""
    def __init__(self, binding, name: str = "program"):
        self.binding, self.name = binding, name
        self.package = binding.package
        self.bound = binding.package.bind()
        binding._bound = self.bound
        cuda = self.package.backend == "nvidia"
        P, S = ct.c_void_p, ct.c_size_t

        def bind(cu, hip, types):
            fn = getattr(self.bound._driver, cu if cuda else hip)
            fn.argtypes, fn.restype = types, ct.c_int
            return fn
        self._alloc = bind("cuMemAlloc_v2", "hipMalloc", [ct.POINTER(P), S])
        self._free = bind("cuMemFree_v2", "hipFree", [P])
        self._to_device = bind("cuMemcpyHtoD_v2", "hipMemcpyHtoD", [P, P, S])
        self._to_host = bind("cuMemcpyDtoH_v2", "hipMemcpyDtoH", [P, P, S])
        self.specs = tuple(s for s in binding.specs if isinstance(s, TensorSpec))
        self.inputs = tuple(s for s in self.specs if not s.writable)
        self.outputs = tuple(s for s in self.specs if s.writable)
        self._lock = threading.RLock()

    def run(self, *arrays):
        """Return the output array (or a tuple of them, in ABI order)."""
        if len(arrays) != len(self.inputs):
            raise ValueError(f"{self.name} takes {len(self.inputs)} operand(s)")
        host: list[np.ndarray] = []
        for spec, value in zip(self.inputs, arrays, strict=True):
            kind, _ = _DTYPES[spec.dtype]
            array = np.ascontiguousarray(np.asarray(value, dtype=kind))
            if array.shape != tuple(spec.shape):
                raise ValueError(f"{self.name} admits {spec.name} of shape {tuple(spec.shape)}")
            host.append(array)
        outs = [np.empty(tuple(spec.shape), _DTYPES[spec.dtype][0]) for spec in self.outputs]
        pointers: list[ct.c_void_p] = []
        with self._lock:
            try:
                views = []
                for spec, value in zip(self.specs, [*host, *outs], strict=True):
                    pointer = ct.c_void_p()
                    self.bound._check(self._alloc(ct.byref(pointer), max(value.nbytes, 4)))
                    pointers.append(pointer)
                    views.append(SimpleNamespace(__cuda_array_interface__=dict(
                        version=3, shape=value.shape, typestr=_DTYPES[spec.dtype][1], data=(pointer.value, False))))
                for value, pointer in zip(host, pointers[:len(host)], strict=True):
                    self.bound._check(self._to_device(pointer, value.ctypes.data, value.nbytes))
                self.binding(*views, 1)
                self.bound._check(self.bound._sync())
                for value, pointer in zip(outs, pointers[len(host):], strict=True):
                    self.bound._check(self._to_host(value.ctypes.data, pointer, value.nbytes))
            finally:
                for pointer in reversed(pointers):
                    if pointer.value:
                        self._free(pointer)
        return outs[0] if len(outs) == 1 else tuple(outs)

    def close(self):
        self.bound.close()
