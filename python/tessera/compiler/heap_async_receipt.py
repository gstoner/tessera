"""Bounded private device statuses copied into pinned host receipts.

A receipt is recycled only after pin failure or proven unpin completion. Pools
retain both allocations through uncertain enqueue/copy/event outcomes.
"""
import ctypes as ct
import numpy as np
from .native_device_tape import _Buffer
from .native_reader_retirement import _record


class HeapReceipt:
    def __init__(self, pool):
        self.pool, self.host, self.ticket = pool, ct.c_void_p(), None
        self.free_attempt_failed = False
        pool._heap_receipts.append(self)
        cuda = pool.collect_program.package.backend == 'nvidia'
        driver = pool.native._driver
        def bind(cu, hip, args):
            fn = getattr(driver, cu if cuda else hip)
            fn.argtypes, fn.restype = args, ct.c_int
            return fn
        P, S, U = ct.c_void_p, ct.c_size_t, ct.c_uint
        allocate = bind('cuMemHostAlloc', 'hipHostMalloc', [ct.POINTER(P), S, U])
        self.free = bind('cuMemFreeHost', 'hipHostFree', [P])
        self.copy = bind('cuMemcpyDtoHAsync_v2', 'hipMemcpyDtoHAsync', [P, P, S, P])
        try:
            pool.check(allocate(ct.byref(self.host), 24, 0))
            self.values = np.ctypeslib.as_array(ct.cast(self.host, ct.POINTER(ct.c_int64)), shape=(3,))
            self.device = _Buffer(pool, (3,), 'int64')
        except BaseException:
            pool._poison_objects()
            raise

    def enqueue_copy(self, stream):
        pool = self.pool
        pool.check(self.copy(self.host, self.device.pointer, 24, ct.c_void_p(stream)))
        tickets: list = []
        try:
            _record(pool.native, stream, (pool, self), tickets)
        finally:
            if tickets:
                self.ticket = tickets[-1]

    def poll(self):
        if self.ticket is None or not self.ticket.poll():
            return None
        return tuple(int(v) for v in self.values)

    def close_host(self):
        if self.free_attempt_failed:
            raise RuntimeError('uncertain pinned-host free requires process teardown')
        if self.host.value:
            self.free_attempt_failed = True
            self.pool.check(self.free(self.host))
            self.host = ct.c_void_p()
            self.free_attempt_failed = False
