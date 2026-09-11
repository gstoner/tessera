"""Nonmoving pool with object payload leases and incremental metadata marking.

One metadata writer is stream-ordered. Payload readers may overlap marking and
logical retirement, but never reuse. Optional private receipts support polled
admission/unpin; legacy admission and pool teardown remain synchronous.
Metadata writers remain serialized; no racing metadata atomics are claimed.
"""
import ctypes as ct
import threading
from types import SimpleNamespace
import numpy as np
from .resident_object_pool import ResidentObjectPool, _UNCERTAIN_POOLS
from .native_device_tape import _Buffer
from .native_reader_retirement import _BorrowedView, _record, _stream
from .gpu_heap_collection import materialize_pool


class ResidentIncrementalPool(ResidentObjectPool):
    def __init__(self, slots, width, references, *, stream, **options):
        self._pending_finalization = None
        self._teardown = None
        self._object_readers = []
        self._heap_receipts = []
        self._free_receipts = []
        self._object_uncertain = False
        self._mark_active = False
        self._incremental_bindings = {}
        super().__init__(slots, width, references, stream=stream, **options)
        try:
            self._pins = _Buffer(self, (slots,), 'int64')
            zeros = np.zeros(slots, np.int64)
            self.check(self._upload(self._pins.pointer, ct.c_void_p(zeros.ctypes.data), zeros.nbytes))
            name = 'cuMemcpyDtoH_v2' if self.collect_program.package.backend == 'nvidia' else 'hipMemcpyDtoH'
            self._download = getattr(self.native._driver, name)
            self._download.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
            self._download.restype = ct.c_int
        except BaseException:
            self.close()
            raise

    def _ready(self, *, recovery=False):
        teardown = getattr(self, '_teardown', None)
        if teardown is not None and teardown.worker != threading.get_ident():
            raise ValueError('pool teardown has been admitted')
        if getattr(self, '_pending_finalization', None) is not None and not recovery:
            raise ValueError('poll pending mark finalization before further pool operations')
        if self._object_uncertain and not recovery:
            raise RuntimeError('uncertain object pin operation requires pool close')
        super()._ready(recovery=recovery)
        for binding in self._incremental_bindings.values():
            if binding._bound is not None:
                for ticket in list(binding._bound._pending):
                    ticket.poll()

    def _poison_objects(self):
        self._object_uncertain = True
        if self not in _UNCERTAIN_POOLS:
            _UNCERTAIN_POOLS.append(self)

    def _binding(self, mode):
        if mode not in self._incremental_bindings:
            slots, width, refs = self._dimensions
            self._incremental_bindings[mode] = materialize_pool(
                slots, width, mode, payload_dtype='int8', references=refs, **self._options).bind()
        return self._incremental_bindings[mode]

    def _status_values(self, ticket):
        ticket.wait()
        values = np.empty(3, np.int64)
        self.check(self._download(ct.c_void_p(values.ctypes.data), self._status.pointer, values.nbytes))
        return values

    def _pin_args(self, slot, generation):
        return (self._state, self._roots, self._edges, self._pins, self._status, slot, generation, 1)

    def _drain_object_readers(self, stream):
        # Poll before enqueue: an unfinished reader must not stall the metadata
        # stream (and hence unrelated object admission). Unknown completion stays pinned.
        for reader in list(self._object_readers):
            if isinstance(reader, _AsyncObjectReader):
                reader._progress(stream)
                continue
            if reader.active or reader.completion is None or not reader.completion.poll():
                continue
            binding = self._binding('unpin')
            attempted = False
            try:
                with self._access().write(stream):
                    attempted = True
                    submission = binding.submit(stream, *self._pin_args(reader.slot, reader.generation))
                if self._status_values(submission.ticket)[0] != 0:
                    raise RuntimeError('native unpin disagrees with retained generation')
            except BaseException:
                # An uncertain decrement cannot be retried: it may already have run.
                if attempted:
                    self._poison_objects()
                raise
            self._object_readers.remove(reader)

    def prepare_readers(self, count=1):
        """Preallocate bounded private receipt storage and native pin/unpin bindings."""
        if type(count) is not int or not 1 <= count <= 256:
            raise ValueError('receipt capacity must be 1..256')
        from .heap_async_receipt import HeapReceipt
        with self._lock:
            self._ready()
            self._binding('pin')
            self._binding('unpin')
            while len(self._heap_receipts) < count:
                self._free_receipts.append(HeapReceipt(self))

    def _take_receipt(self):
        if not self._free_receipts:
            self.prepare_readers(len(self._heap_receipts) + 1)
        return self._free_receipts.pop()

    def _submit_pin_receipt(self, mode, stream, slot, generation, receipt):
        binding = self._binding(mode)
        attempted = False
        try:
            with self._access().write(stream):
                attempted = True
                binding.submit(stream, self._state, self._roots, self._edges, self._pins,
                               receipt.device, slot, generation, 1)
                receipt.enqueue_copy(stream)
        except BaseException:
            if attempted:
                self._poison_objects()
            raise

    def begin_read_object(self, stream, slot, generation):
        """Enqueue admission; poll before entering the payload reader scope.

        Warm storage/bindings with prepare_readers to avoid lazy setup. Enqueue
        and poll add no event/stream/context wait. Driver call latency is not bounded.
        """
        reader = _AsyncObjectReader(self, stream, slot, generation)
        with self._lock:
            self._ready()
            self._drain_object_readers(stream)
            if len(self._object_readers) >= 256:
                raise ValueError('object reader admission budget exhausted')
            reader.receipt = self._take_receipt()
            self._object_readers.append(reader)
            try:
                self._submit_pin_receipt('pin', reader.stream, slot, generation, reader.receipt)
            except BaseException:
                if not self._object_uncertain:
                    self._object_readers.remove(reader)
                    self._free_receipts.append(reader.receipt)
                raise
            return reader

    def poll_object_readers(self, stream):
        """Advance receipt-driven cleanup without waiting for unfinished readers."""
        with self._lock:
            self._ready()
            self._drain_object_readers(_stream(stream))
            return not self._object_readers

    def read_object(self, stream, slot, generation):
        return _ObjectReader(self, stream, slot, generation)

    def allocate(self, stream, payload, length):
        with self._lock:
            self._ready()
            self._drain_object_readers(stream)
            if self._mark_active:
                binding = self._binding('allocate_marked')
                with self._access().write(stream):
                    result = binding.submit(stream, self._state, self._roots, self._edges,
                                            self._payload, payload, self._status, length, self._marks, 1)
                return result.ticket
            return super().allocate(stream, payload, length)

    def set_graph(self, stream, roots, edges):
        with self._lock:
            self._ready()
            if not self._mark_active:
                return super().set_graph(stream, roots, edges)
            binding = self._binding('graph_incremental')
            with self._access().write(stream):
                result = binding.submit(stream, self._state, self._roots, self._edges,
                                        roots, edges, self._marks, self._status, 1)
            return result.ticket

    def _mark_call(self, mode, stream, *extra):
        binding = self._binding(mode)
        with self._access().write(stream):
            result = binding.submit(stream, self._state, self._roots, self._edges,
                                    self._marks, self._status, *extra, 1)
        return result.ticket

    def begin_mark(self, stream):
        with self._lock:
            self._ready()
            if self._mark_active:
                raise ValueError('incremental marking is already active')
            ticket = self._mark_call('mark_begin', stream)
            if self._status_values(ticket)[0] != 0:
                raise ValueError('invalid graph at mark begin')
            self._mark_active = True
            return ticket

    def mark_step(self, stream, budget=1):
        with self._lock:
            self._ready()
            if not self._mark_active:
                raise ValueError('begin incremental marking first')
            return self._mark_call('mark_step', stream, budget)

    def finish_mark_async(self, stream):
        """Enqueue private final-mark status; poll before further mutations."""
        from .heap_finalization import MarkFinalization
        with self._lock:
            self._ready()
            if not self._mark_active:
                raise ValueError('begin incremental marking first')
            binding = self._binding('retire_marked')
            receipt = self._take_receipt()
            attempted = False
            try:
                with self._access().write(stream):
                    attempted = True
                    binding.submit(stream, self._state, self._roots, self._edges,
                                   self._marks, receipt.device, 1)
                    receipt.enqueue_copy(stream)
            except BaseException:
                if attempted:
                    self._poison_objects()
                else:
                    self._free_receipts.append(receipt)
                raise
            ticket = MarkFinalization(self, receipt)
            self._pending_finalization = ticket
            return ticket

    def close_async(self):
        """Transfer destruction to a bounded worker; no driver latency bound."""
        from .heap_finalization import PoolTeardown
        with self._lock:
            self._ready()
            if any(reader.active for reader in self._object_readers):
                raise ValueError('object readers must close before pool close')
            if self._access()._active:
                raise ValueError('metadata readers must close before pool close')
            return PoolTeardown.submit(self)

    def finish_mark(self, stream):
        with self._lock:
            self._ready()
            if not self._mark_active:
                raise ValueError('begin incremental marking first')
            ticket = self._mark_call('retire_marked', stream)
            if self._status_values(ticket)[0] != 0:
                raise ValueError('marking is incomplete or graph closure is invalid')
            self._mark_active = False
            return ticket

    def reclaim_retired(self, stream):
        with self._lock:
            self._ready()
            if self._mark_active:
                raise ValueError('finish marking before reclamation')
            self._drain_object_readers(stream)
            return self._mark_call('reclaim_pinned', stream, self._pins)

    def collect(self, stream):
        raise ValueError('incremental pools require begin_mark/mark_step/finish_mark')

    def retire_unreachable(self, stream):
        raise ValueError('incremental pools retire through finish_mark')

    def begin_collection(self, stream, *, marker_stream):
        raise ValueError('incremental pools use begin_mark instead of snapshot collection')

    def finish_collection(self, stream, *, sweep_budget=None):
        raise ValueError('incremental pools use finish_mark instead of snapshot collection')

    def wait(self):
        with self._lock:
            if any(reader.active for reader in self._object_readers):
                raise ValueError('object readers must close before pool wait')
            for reader in self._object_readers:
                if reader.completion is not None:
                    reader.completion.wait()
            super().wait()

    def close(self):
        with self._lock:
            teardown = getattr(self, "_teardown", None)
            if teardown is not None and teardown.worker != threading.get_ident():
                raise ValueError("pool teardown has been admitted")
            if self.closed:
                return
            if any(reader.active for reader in self._object_readers):
                raise ValueError('object readers must close before pool close')
            try:
                pending = getattr(self, '_pending_finalization', None)
                if pending is not None:
                    pending.receipt.ticket.wait()
                    try:
                        pending.poll()
                    except ValueError:
                        pass  # A completed incomplete-mark result does not prevent destruction.
                for reader in self._object_readers:
                    if reader.completion is not None:
                        reader.completion.wait()
                self.check(self.native._sync())
                for receipt in self._heap_receipts:
                    receipt.close_host()
                for binding in self._incremental_bindings.values():
                    binding.close()
                super().close()
            except BaseException:
                self._poison_objects()
                raise
            self._object_readers.clear()
            if self in _UNCERTAIN_POOLS:
                _UNCERTAIN_POOLS.remove(self)


class _ObjectReader:
    def __init__(self, pool, stream, slot, generation):
        if (type(slot) is not int or not 0 <= slot < pool._dimensions[0] or
                type(generation) is not int or not 1 <= generation < (1 << 31)):
            raise ValueError('object reader requires a bounded slot and generation')
        self.pool, self.stream = pool, _stream(stream)
        self.slot, self.generation = slot, generation
        self.active = self.used = False
        self.completion = None

    def __enter__(self):
        pool = self.pool
        with pool._lock:
            pool._ready()
            if self.used:
                raise ValueError('object reader is single-use')
            pool._drain_object_readers(self.stream)
            if len(pool._object_readers) >= 256:
                raise ValueError('object reader admission budget exhausted')
            binding = pool._binding('pin')
            self.used = True
            pool._object_readers.append(self)
            attempted = False
            try:
                with pool._access().write(self.stream):
                    attempted = True
                    submission = binding.submit(self.stream, *pool._pin_args(self.slot, self.generation))
                status = pool._status_values(submission.ticket)
            except BaseException:
                if attempted:
                    pool._poison_objects()
                else:
                    pool._object_readers.remove(self)
                raise
            if status[0] != 0:
                pool._object_readers.remove(self)
                raise ValueError('object handle is stale, retired or pin capacity is exhausted')
            self.active = True
            pointer = pool._payload.pointer.value + self.slot * pool._dimensions[1]
            view = SimpleNamespace(pointer=ct.c_void_p(pointer), __cuda_array_interface__={
                'version': 3, 'shape': (int(status[1]),), 'typestr': '|i1', 'data': (pointer, True)})
            return _BorrowedView(self, view)

    def __exit__(self, *exc):
        with self.pool._lock:
            tickets: list = []
            try:
                _record(self.pool.native, self.stream, (self.pool,), tickets)
            finally:
                self.active = False
                if tickets:
                    self.completion = tickets[-1]


class _AsyncObjectReader(_ObjectReader):
    def __init__(self, *args):
        super().__init__(*args)
        self.receipt = None
        self._admitted = False
        self._unpin_started = False
        self._released = False
        self._cancelled = False
        self._error = None
        self.length = None

    def _release(self):
        self.pool._object_readers.remove(self)
        self.pool._free_receipts.append(self.receipt)
        self._released = True

    def _poll_admission(self):
        if self._admitted:
            return True
        if self._released:
            return False
        assert self.receipt is not None
        status = self.receipt.poll()
        if status is None:
            return False
        if status[0] == 2:
            self._error = 'object handle is stale, retired or pin capacity is exhausted'
            self._release()
            return False
        if status[0] != 0:
            self.pool._poison_objects()
            raise RuntimeError('unknown native pin status')
        self.length = status[1]
        self._admitted = True
        return True

    def poll(self):
        with self.pool._lock:
            self.pool._ready()
            ready = self._poll_admission()
            if self._error:
                raise ValueError(self._error)
            if self._cancelled or self._released:
                raise ValueError('asynchronous reader is cancelled or retired')
            return ready

    def __enter__(self):
        with self.pool._lock:
            if self.used:
                raise ValueError('object reader is single-use')
            if not self.poll():
                raise ValueError('object admission is pending; poll before entering')
            self.used = self.active = True
            pointer = self.pool._payload.pointer.value + self.slot * self.pool._dimensions[1]
            view = SimpleNamespace(pointer=ct.c_void_p(pointer), __cuda_array_interface__={
                'version': 3, 'shape': (self.length,), 'typestr': '|i1', 'data': (pointer, True)})
            return _BorrowedView(self, view)

    def cancel(self):
        with self.pool._lock:
            self.pool._ready()
            if self.active:
                raise ValueError('close the active payload scope before cancellation')
            self._cancelled = True
            self._progress(self.stream)

    def _progress(self, stream):
        if self._released or not self._poll_admission() or self.active:
            return
        assert self.receipt is not None
        if self._cancelled and self.completion is None:
            self.completion = self.receipt.ticket
        if self.completion is None or not self.completion.poll():
            return
        if not self._unpin_started:
            # A pre-submission refusal leaves the request retryable. An
            # uncertain enqueue poisons the pool and cannot submit twice.
            self.pool._submit_pin_receipt('unpin', stream, self.slot, self.generation, self.receipt)
            self._unpin_started = True
        assert self.receipt is not None
        status = self.receipt.poll()
        if status is None:
            return
        if status[0] != 0:
            self.pool._poison_objects()
            raise RuntimeError('native unpin disagrees with retained generation')
        self._release()
