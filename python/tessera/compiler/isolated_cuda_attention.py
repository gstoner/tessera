"""Resident CUDA attention (the saved-LSE checkpoint pair) in a replaceable
process; no device pointers cross IPC.

The worker retains the device's primary context, uploads the captured Q/K/V,
captures ``pair`` into a ``ResidentAttentionTape`` (saved O and LSE stay
device-resident in the child), admits itself through a zero/nonzero VJP
probe against the numpy reference of the checkpoint contract, and then
relays host-array cotangents in and host-array gradients out. Twin of
``isolated_rocm_attention`` over the same ``IsolatedAttentionTape`` body.
"""
from __future__ import annotations

import ctypes as ct
import os

import numpy as np

from .isolated_attention import IsolatedAttentionTape, _UNCERTAIN, serve
from .resident_rocm_attention import prepare_attention_cotangent

__all__ = ["IsolatedCUDAAttentionTape", "reference_checkpoint_backward", "_UNCERTAIN"]


def reference_checkpoint_backward(do, q, k, v, *, scale, causal):
    """The checkpoint pair's backward in numpy: end-aligned causal mask, GQA
    by head grouping, f64 accumulation. Mirrors the sm_120 pair's device test."""
    do, q, k, v = (np.asarray(x, np.float64) for x in (do, q, k, v))
    b, hq, sq, d = q.shape
    hkv, sk = k.shape[1], k.shape[2]
    group = hq // hkv
    kk = np.repeat(k, group, axis=1)
    vv = np.repeat(v, group, axis=1)
    scores = np.matmul(q, kk.swapaxes(-1, -2)) * scale
    if causal:
        legal = np.arange(sk)[None, :] <= np.arange(sq)[:, None] + max(sk - sq, 0)
        scores = np.where(legal, scores, -np.inf)
    lse = np.logaddexp.reduce(scores, axis=-1)
    p = np.exp(scores - lse[..., None])
    dp = do @ vv.swapaxes(-1, -2)
    ds = p * (dp - (p * dp).sum(axis=-1, keepdims=True))
    dq = ds @ kk * scale
    dk = (ds.swapaxes(-1, -2) @ q * scale).reshape(b, hkv, group, sk, d).sum(axis=2)
    dv = (p.swapaxes(-1, -2) @ do).reshape(b, hkv, group, sk, v.shape[-1]).sum(axis=2)
    return tuple(x.astype(np.float32) for x in (dq, dk, dv))


class IsolatedCUDAAttentionTape(IsolatedAttentionTape):
    """Host-array owner of one resident CUDA attention generation."""

    def __init__(self, pair, q, k, v, *, device=0, timeout_seconds=30.0):
        self.pair = pair
        arrays = []
        for name, value in (("q", q), ("k", k), ("v", v)):
            array = np.array(value, dtype=np.float32, copy=True, order='C')
            if array.ndim != 4 or not np.all(np.isfinite(array)):
                raise ValueError(f"isolated attention requires a finite rank-4 f32 {name}")
            array.setflags(write=False)
            arrays.append(array)
        self._q, self._k, self._v = arrays
        b, hq, sq, _ = self._q.shape
        self._shape, self._dtype = (b, hq, sq, self._v.shape[-1]), np.dtype(np.float32)
        super().__init__(device=device, timeout_seconds=timeout_seconds)

    def _worker_target(self):
        return _worker

    def _payload(self):
        return (self.pair, self._q, self._k, self._v, self.device)

    def _ready_token(self):
        return self.pair.contract_digest

    def _prepare_cotangent(self, cotangent, casting):
        return prepare_attention_cotangent(cotangent, self._shape, self._dtype, casting)

    def _replacement_kwargs(self):
        return dict(pair=self.pair, q=self._q, k=self._k, v=self._v)


class _Driver:
    """The few libcuda calls the worker needs, bound once."""

    def __init__(self, device):
        self.lib = ct.CDLL('libcuda.so.1')
        P, S = ct.c_void_p, ct.c_size_t

        def bind(name, args):
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, ct.c_int
            return fn
        init = bind('cuInit', [ct.c_uint])
        get = bind('cuDeviceGet', [ct.POINTER(ct.c_int), ct.c_int])
        retain = bind('cuDevicePrimaryCtxRetain', [ct.POINTER(P), ct.c_int])
        current = bind('cuCtxSetCurrent', [P])
        self.alloc = bind('cuMemAlloc_v2', [ct.POINTER(P), S])
        self.free = bind('cuMemFree_v2', [P])
        self.htod = bind('cuMemcpyHtoD_v2', [P, P, S])
        self.dtoh = bind('cuMemcpyDtoH_v2', [P, P, S])
        self.sync = bind('cuCtxSynchronize', [])
        self.check(init(0))
        handle = ct.c_int()
        self.check(get(ct.byref(handle), device))
        context = P()
        self.check(retain(ct.byref(context), handle))
        self.check(current(context))

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f'attention worker CUDA status {status}')


class _DeviceArray:
    """A cuMemAlloc'd f32 array with a CUDA array interface."""

    def __init__(self, driver, shape):
        self.driver, self.shape = driver, tuple(int(d) for d in shape)
        self.nbytes = 4 * int(np.prod(self.shape))
        self.pointer = ct.c_void_p()
        driver.check(driver.alloc(ct.byref(self.pointer), self.nbytes))

    def upload(self, host):
        host = np.ascontiguousarray(host, np.float32)
        if host.shape != self.shape:
            raise ValueError('attention worker upload shape disagrees')
        self.driver.check(self.driver.htod(self.pointer, host.ctypes.data_as(ct.c_void_p), self.nbytes))
        return self

    def free(self):
        if self.pointer.value:
            self.driver.free(self.pointer)
            self.pointer = ct.c_void_p()

    @property
    def __cuda_array_interface__(self):
        return {'version': 3, 'shape': self.shape, 'typestr': '<f4', 'data': (int(self.pointer.value), False)}


def _download(driver, view):
    interface = view.__cuda_array_interface__
    host = np.empty(interface['shape'], np.float32)
    driver.check(driver.dtoh(host.ctypes.data_as(ct.c_void_p), ct.c_void_p(interface['data'][0]), host.nbytes))
    return host


def _run_backward(driver, tape, cotangent, shape):
    device_cotangent = _DeviceArray(driver, shape).upload(cotangent)
    try:
        gradients = tape.backward(device_cotangent)
        driver.check(driver.sync())
        return tuple(_download(driver, g) for g in gradients)
    finally:
        device_cotangent.free()


def _check_health(driver, tape, pair, q, k, v, shape):
    zero = _run_backward(driver, tape, np.zeros(shape, np.float32), shape)
    if not all(np.all(value == 0) for value in zero):
        raise RuntimeError('attention zero-VJP health probe failed')
    provenance = pair.forward.descriptor.provenance
    cotangent = np.ones(shape, np.float32)
    expected = reference_checkpoint_backward(cotangent, q, k, v, scale=float(provenance['scale']), causal=bool(provenance['causal']))
    actual = _run_backward(driver, tape, cotangent, shape)
    if len(actual) != 3 or not all(np.allclose(a, e, rtol=3e-4, atol=3e-5, equal_nan=False) for a, e in zip(actual, expected, strict=True)):
        raise RuntimeError('attention nonzero-VJP health probe failed')


def _worker(connection, pair, q, k, v, device):
    try:
        driver = _Driver(device)
        shape = (q.shape[0], q.shape[1], q.shape[2], v.shape[-1])
        inputs = [_DeviceArray(driver, array.shape).upload(array) for array in (q, k, v)]
        try:
            with pair.capture(*inputs) as tape:
                _check_health(driver, tape, pair, q, k, v, shape)
                serve(connection, tape, pair.contract_digest,
                      lambda t, value: _run_backward(driver, t, value, shape))
        finally:
            for array in inputs:
                array.free()
    except BaseException as error:
        try:
            connection.send(('error', type(error).__name__, str(error)))
            connection.close()
        finally:
            os._exit(1)
