"""Process-owned gated heap with explicit death-confirmed teardown recovery."""
import ctypes as ct
import math
import multiprocessing as mp
import os
import threading
import time
import numpy as np
from .native_driver_isolation import DriverIsolationLease, IsolationRecovery
from .native_isolated_ann import _ProcessBoundary

_UNCERTAIN: set["IsolatedHeapPool"] = set()
_WORKER_SLOTS = threading.BoundedSemaphore(8)


def _heap_worker(channel, dimensions, options):
    try:
        from .resident_gated_pool import ResidentGatedPool
        from .native_device_tape import _Buffer
        cuda = options['backend'] == 'nvidia'
        driver = ct.CDLL('libcuda.so.1' if cuda else 'libamdhip64.so')
        P = ct.c_void_p
        def call(name, types, *args):
            fn = getattr(driver, name)
            fn.argtypes, fn.restype = types, ct.c_int
            if fn(*args):
                raise RuntimeError('isolated heap device initialization failed')
        context, stream = P(), P()
        if cuda:
            call('cuInit', [ct.c_uint], 0)
            call('cuDevicePrimaryCtxRetain', [ct.POINTER(P), ct.c_int], ct.byref(context), 0)
            call('cuCtxSetCurrent', [P], context)
        else:
            call('hipInit', [ct.c_uint], 0)
            call('hipSetDevice', [ct.c_int], 0)
        call('cuStreamCreate' if cuda else 'hipStreamCreateWithFlags', [ct.POINTER(P), ct.c_uint], ct.byref(stream), 1)
        pool = ResidentGatedPool(*dimensions, stream=stream.value, **options)
        data = _Buffer(pool, (dimensions[1],), 'int8')
        channel.send(('ready',))
        while True:
            op, value = channel.recv()
            if op == 'close':
                ticket = pool.close_async()
                while not ticket.poll():
                    time.sleep(.001)
                channel.send(('closed',))
                return
            if op == 'allocate':
                pool.check(pool._upload(data.pointer, P(value.ctypes.data), value.nbytes))
                result: object = tuple(int(x) for x in pool._status_values(pool.allocate(stream.value, data, len(value))))
            elif op == 'inspect':
                result = pool.inspect_metadata(stream.value)
            elif op == 'mark':
                pool.begin_mark(stream.value)
                pool.mark_step(stream.value, dimensions[0]).wait()
                receipt = pool.finish_mark_async(stream.value)
                while not receipt.poll():
                    time.sleep(.001)
                result = True
            else:
                raise ValueError('unknown isolated heap operation')
            channel.send(('result', result))
    except BaseException as error:
        try:
            channel.send(('error', type(error).__name__, str(error)))
        except BaseException:
            pass
        os._exit(1)  # Never retry an uncertain driver action during cleanup.
    finally:
        channel.close()


class IsolatedHeapPool:
    def __init__(self, slots, width, references, *, timeout_seconds=30., **options):
        from .gpu_heap_collection import emit_pool
        emit_pool(slots, width, 'atomic_allocate', payload_dtype='int8', references=references)
        if options.get('backend') not in ('nvidia', 'rocm'):
            raise ValueError('isolated heap requires CUDA or HIP')
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('isolated heap requires finite positive timeout')
        self.width, self.timeout = width, timeout_seconds
        self._lock = threading.RLock()
        self._pending = None
        self._recovery = None
        self.closed = self.failed = False
        self._slot = _WORKER_SLOTS
        if not self._slot.acquire(blocking=False):
            raise ValueError('isolated heap worker capacity exhausted')
        self._slot_owned = True
        parent = child = None
        try:
            ctx = mp.get_context('spawn')
            parent, child = ctx.Pipe()
            self.channel = parent
            self.process = ctx.Process(target=_heap_worker, args=(child, (slots, width, references), options), daemon=True)
            self.process.start()
        except BaseException:
            if parent is not None:
                parent.close()
            if child is not None:
                child.close()
            self._release_slot()
            raise
        child.close()
        self.lease = DriverIsolationLease(_ProcessBoundary(self.process), context_identity=f'heap-{self.process.pid}',
                                          timeout_seconds=min(timeout_seconds, 5.))
        try:
            if not parent.poll(timeout_seconds) or parent.recv() != ('ready',):
                raise RuntimeError('isolated heap startup failed')
        except BaseException:
            self._poison()
            self.lease.recover()
            parent.close()
            _UNCERTAIN.discard(self)
            self._release_slot()
            raise

    def _release_slot(self):
        if getattr(self, '_slot_owned', False):
            self._slot_owned = False
            self._slot.release()

    def _poison(self):
        self.failed = True
        self.lease.mark_uncertain()
        _UNCERTAIN.add(self)

    def submit(self, op, value=None):
        with self._lock:
            if self.closed or self.failed or self._pending is not None:
                raise ValueError('isolated heap is unavailable or has a pending request')
            if op == 'allocate':
                value = np.array(value, copy=True, order='C')
                if value.dtype != np.int8 or value.shape != (self.width,):
                    raise ValueError('isolated heap allocation requires its exact int8 payload width')
            elif op not in ('inspect', 'mark', 'close') or value is not None:
                raise ValueError('invalid isolated heap request')
            self._pending = (op, time.monotonic() + self.timeout)
            try:
                self.channel.send((op, value))
            except BaseException:
                self._poison()
                raise

    def poll(self):
        with self._lock:
            if self.failed or self.closed:
                raise ValueError('isolated heap is closed or uncertain')
            if self._pending is None:
                raise ValueError('no pending isolated heap request')
            op, deadline = self._pending
            try:
                if not self.channel.poll():
                    if time.monotonic() >= deadline:
                        raise TimeoutError('isolated heap request timed out')
                    return None
                result = self.channel.recv()
                if result[0] == 'error':
                    raise RuntimeError(str(result))
                if op == 'close':
                    if result != ('closed',):
                        raise RuntimeError('invalid heap close receipt')
                    # Receipt alone is not process-death evidence.
                    self._poison()
                    return 'close_receipt'
                if result[0] != 'result':
                    raise RuntimeError('invalid isolated heap result')
                self._pending = None
                return result[1]
            except BaseException:
                self._poison()
                raise

    def recover_async(self):
        with self._lock:
            if self.closed:
                return self._recovery
            self._poison()
            if self._recovery is None:
                self._recovery = IsolationRecovery.submit(self.lease, owner=self)
            return self._recovery

    def poll_recovery(self):
        with self._lock:
            if self._recovery is None or not self._recovery.poll():
                return False
            if not self.lease.reusable:
                raise RuntimeError('heap worker death is unconfirmed')
            self.channel.close()
            self.closed = True
            self._pending = None
            _UNCERTAIN.discard(self)
            self._release_slot()
            return True
