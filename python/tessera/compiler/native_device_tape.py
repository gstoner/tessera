"""Owned CUDA/HIP snapshots for persistent, recomputed native reverse products.

Frames may nest for allocation lifetime management. This is not higher-order AD
or a lowering of control-flow tapes; derivatives still come from the compiler.
"""
from __future__ import annotations

import ctypes as ct
import threading


class _Buffer:
    def __init__(self, frame, shape):
        self.frame, self.shape = frame, tuple(shape)
        if any(type(dim) is not int or dim <= 0 for dim in self.shape):
            raise ValueError('native tape requires positive static allocation extents')
        self.pointer = ct.c_void_p()
        self.nbytes = 4
        for dim in shape:
            if self.nbytes > ((1 << 63) - 1) // dim:
                raise ValueError('native tape allocation byte extent overflows')
            self.nbytes *= dim
        frame.check(frame.alloc(ct.byref(self.pointer), self.nbytes))
        frame.buffers.append(self)

    @property
    def __cuda_array_interface__(self):
        if self.frame.closed or not self.pointer.value:
            raise ValueError('native tape allocation is closed')
        return dict(version=3, shape=self.shape, typestr='<f4', strides=None,
                    data=(self.pointer.value, False))


class NativeDeviceTape:
    """A persistent snapshot with frame-owned primal and per-backward results."""
    def __init__(self, pair, value):
        if pair.contract['mode'] != 'reverse' or getattr(pair.package, 'backend', None) not in ('nvidia', 'rocm'):
            raise ValueError('device tape requires a CUDA/HIP reverse pair')
        self.pair = pair
        self.closed = False
        self.buffers = []
        self.children = []
        self._lock = threading.RLock()
        self.identity = pair.package.binding_digest
        specs = pair.binding.specs
        if [s.name for s in specs] != ['arg0', 'arg1', 'primal', 'derivative', 'n']:
            raise ValueError('device tape requires a single-input native reverse product')
        self.width = specs[-1].minimum
        if pair.binding._bound is None:
            pair.binding._bound = pair.package.bind()
        native = pair.binding._bound
        self.check = native._check
        self.driver = native._driver
        cuda = pair.package.backend == 'nvidia'
        P, S = ct.c_void_p, ct.c_size_t
        def bind(cu, hip, args):
            fn = getattr(self.driver, cu if cuda else hip)
            fn.argtypes, fn.restype = args, ct.c_int
            return fn
        self.alloc = bind('cuMemAlloc_v2', 'hipMalloc', [ct.POINTER(P), S])
        self.free = bind('cuMemFree_v2', 'hipFree', [P])
        self.sync = native._sync
        self.current = bind('cuCtxGetCurrent', 'hipGetDevice', [ct.POINTER(P if cuda else ct.c_int)])
        self.context_type = P if cuda else ct.c_int
        self.context = self.context_type()
        self.check(self.current(ct.byref(self.context)))
        self.copy = bind('cuMemcpyDtoD_v2', 'hipMemcpyDtoD', [P, P, S])
        self.zero = bind('cuMemsetD8_v2', 'hipMemset', [P, ct.c_ubyte if cuda else ct.c_int, S])
        try:
            self._input = _Buffer(self, specs[0].shape)
            self.cotangent = _Buffer(self, specs[1].shape)
            self.primal = _Buffer(self, specs[2].shape)
            scratch = _Buffer(self, specs[3].shape)
            # Reuse the production adapter's dtype, shape, bounds and device
            # checks before copying any caller-provided address.
            raw, _, _, _, _ = pair.binding._resident(value, self.cotangent, self.primal, scratch, self.width)
            self.check(self.sync())
            self.check(self.copy(self._input.pointer, P(raw[0]), self._input.nbytes))
            self.check(self.zero(self.cotangent.pointer, 0, self.cotangent.nbytes))
            pair(self._input, self.cotangent, self.primal, scratch, self.width)
        except BaseException:
            self.close()
            raise

    def _ready(self):
        if self.closed:
            raise ValueError('native tape is closed')
        current = self.context_type()
        self.check(self.current(ct.byref(current)))
        if current.value != self.context.value:
            raise ValueError('native tape requires its owning device context')
        if self.pair.package.binding_digest != self.identity:
            raise ValueError('native tape package identity changed')

    def backward(self, cotangent):
        """Recompute from the immutable snapshot; each result has its own storage."""
        with self._lock:
            self._ready()
            start = len(self.buffers)
            try:
                primal = _Buffer(self, self.primal.shape)
                derivative = _Buffer(self, self._input.shape)
                self.pair(self._input, cotangent, primal, derivative, self.width)
            except BaseException:
                self.check(self.sync())
                while len(self.buffers) > start:
                    buffer = self.buffers[-1]
                    self.check(self.free(buffer.pointer))
                    buffer.pointer = ct.c_void_p()
                    self.buffers.pop()
                raise
            return derivative

    def child(self, pair, value):
        """Nest an invocation's storage lifetime, without asserting higher AD order."""
        with self._lock:
            self._ready()
            child = pair.capture(value)
            self.children.append(child)
            return child

    def close(self):
        with self._lock:
            if self.closed:
                return
            self._ready()
            for child in self.children:
                child.close()
            self.check(self.sync())
            while self.buffers:
                buffer = self.buffers[-1]
                self.check(self.free(buffer.pointer))
                buffer.pointer = ct.c_void_p()
                self.buffers.pop()
            self.closed = True

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self, *_exc):
        self.close()
