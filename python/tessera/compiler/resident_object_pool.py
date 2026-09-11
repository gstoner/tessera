"""Resident bounded object records with stream-ordered collection and readers.

Payloads are opaque bytes with a caller-owned schema; graph edges are explicit
(index,generation) pairs. Discovery accepts quiescent builtin containers and plain instance dictionaries.
"""

from __future__ import annotations

import ctypes as ct
import threading
import numpy as np
from .gpu_heap_collection import materialize_pool
from .native_device_tape import _Buffer
from .native_reader_retirement import _record, _stream
from .native_gpu_tensor import TensorSubmission
from .native_gpu_storage import NativeSubmission
from .native_stream_epoch import NativeStreamEpoch

_UNCERTAIN_POOLS: list[ResidentObjectPool] = []


class ResidentObjectPool:
    def __init__(self, slots, width, references, *, stream, **options):
        stream = _stream(stream)
        self._options = options
        self._dimensions = slots, width, references
        self.allocate_program = materialize_pool(
            slots, width, "allocate", payload_dtype="int8", references=references, **options
        )
        self.collect_program = materialize_pool(
            slots, width, "collect", payload_dtype="int8", references=references, **options
        )
        self.graph_program = materialize_pool(
            slots, width, "graph", payload_dtype="int8", references=references, **options
        )
        self._graph = self.graph_program.bind()
        self._allocate, self._collect = self.allocate_program.bind(), self.collect_program.bind()
        self._collect._bound = self.collect_program.package.bind()
        self.native = self._collect._bound
        self._lock, self.closed = threading.RLock(), False
        self.buffers: list[_Buffer] = []
        self.check = self.native._check
        driver, cuda = self.native._driver, self.collect_program.package.backend == "nvidia"
        P, S = ct.c_void_p, ct.c_size_t

        def bind(cu, hip, args):
            fn = getattr(driver, cu if cuda else hip)
            fn.argtypes, fn.restype = args, ct.c_int
            return fn

        self.alloc = bind("cuMemAlloc_v2", "hipMalloc", [ct.POINTER(P), S])
        self.free = bind("cuMemFree_v2", "hipFree", [P])
        self._copy_async = bind("cuMemcpyDtoDAsync_v2", "hipMemcpyDtoDAsync", [P, P, S, P])
        self.context_type = P if cuda else ct.c_int
        self.current = bind("cuCtxGetCurrent", "hipGetDevice", [ct.POINTER(self.context_type)])
        self.context = self.context_type()
        self.check(self.current(ct.byref(self.context)))
        self._upload = bind("cuMemcpyHtoD_v2", "hipMemcpyHtoD", [P, P, S])
        self.epoch = None
        self._collection = None
        self._collection_uncertain = False
        self._snapshot = None
        self._marker = self._seeded = None
        try:
            self._state = _Buffer(self, (slots, 3), "int64")
            self._roots = _Buffer(self, (slots,), "int64")
            self._edges = _Buffer(self, (slots, 2 * references), "int64")
            self._payload = _Buffer(self, (slots, width), "int8")
            self._marks = _Buffer(self, (slots,), "int64")
            self._status = _Buffer(self, (3,), "int64")
            for buffer in self.buffers:
                data = np.zeros(buffer.shape, dtype=buffer.typestr)
                if buffer is self._edges:
                    data[:, ::2] = -1
                self.check(self._upload(buffer.pointer, ct.c_void_p(data.ctypes.data), data.nbytes))
            pending: list[NativeSubmission] = []
            _record(self.native, stream, (self,), pending)
            self.epoch = NativeStreamEpoch(
                self,
                self.native,
                (self._state, self._roots, self._edges, self._payload, self._status),
                TensorSubmission(pending[0], ()),
            )
        except BaseException:
            try:
                self.close()
            except BaseException:
                _UNCERTAIN_POOLS.append(self)
            raise

    def _ready(self, *, recovery=False):
        if self.closed:
            raise ValueError("resident object pool is closed")
        if self._collection_uncertain and not recovery:
            raise RuntimeError("uncertain collection requires explicit wait or close")
        context = self.context_type()
        self.check(self.current(ct.byref(context)))
        if context.value != self.context.value:
            raise ValueError("object pool requires its owning context")
        for binding in (self._allocate, self._collect, self._graph, self._marker, self._seeded):
            if binding is not None and binding._bound is not None:
                for ticket in list(binding._bound._pending):
                    ticket.poll()

    def _access(self):
        self._ready()
        if self.epoch is None:
            raise ValueError("object pool has not initialized its stream epoch")
        return self.epoch

    def read(self, stream):
        return self._access().read(stream)

    def allocate(self, stream, payload, length):
        with self._access().write(stream):
            result = self._allocate.submit(
                stream, self._state, self._roots, self._edges, self._payload, payload, self._status, length, 1
            )
        return result.ticket

    def collect(self, stream):
        with self._lock, self._access().write(stream):
            if self._collection is not None:
                raise ValueError("finish the active snapshot collection first")
            result = self._collect.submit(stream, self._state, self._roots, self._edges, self._marks, self._status, 1)
        return result.ticket

    def set_graph(self, stream, roots, edges):
        with self._access().write(stream):
            result = self._graph.submit(stream, self._state, self._roots, self._edges, roots, edges, 1)
        return result.ticket

    @classmethod
    def from_objects(cls, *roots, stream, **options):
        from .object_discovery import discover_objects

        snapshot = discover_objects(*roots)
        slots = len(snapshot.payloads)
        width = max(map(len, snapshot.payloads))
        references = max(1, max(map(len, snapshot.edges)))
        pool = cls(slots, width, references, stream=stream, **options)
        try:
            state = np.zeros((slots, 3), np.int64)
            roots_array = np.zeros(slots, np.int64)
            edges = np.tile(np.array([-1, 0] * references, np.int64), (slots, 1))
            payload = np.zeros((slots, width), np.int8)
            for i, data in enumerate(snapshot.payloads):
                state[i] = [1, len(data), 1]
                payload[i, : len(data)] = np.frombuffer(data, np.int8)
                for j, target in enumerate(snapshot.edges[i]):
                    edges[i, 2 * j : 2 * j + 2] = [target, 1]
            roots_array[list(snapshot.roots)] = 1
            with pool._lock:
                pool.wait()
                with pool._access().write(stream):
                    for buffer, array in zip(
                        (pool._state, pool._roots, pool._edges, pool._payload),
                        (state, roots_array, edges, payload),
                        strict=True,
                    ):
                        pool.check(pool._upload(buffer.pointer, ct.c_void_p(array.ctypes.data), array.nbytes))
            return pool
        except BaseException:
            try:
                pool.close()
            except BaseException:
                _UNCERTAIN_POOLS.append(pool)
            raise

    def begin_collection(self, stream, *, marker_stream):
        """Snapshot under the writer epoch; mark immutable copies independently."""
        marker_stream = _stream(marker_stream)
        with self._lock:
            self._ready()
            if self._collection is not None:
                raise ValueError("snapshot collection is already active")
            if self._snapshot is None:
                slots, width, references = self._dimensions
                self._marker = materialize_pool(
                    slots, width, "mark", payload_dtype="int8", references=references, **self._options
                ).bind()
                self._seeded = materialize_pool(
                    slots, width, "collect_seeded", payload_dtype="int8", references=references, **self._options
                ).bind()
                start = len(self.buffers)
                try:
                    self._snapshot = tuple(
                        _Buffer(self, b.shape, "int64")
                        for b in (self._state, self._roots, self._edges, self._marks, self._status)
                    )
                except BaseException:
                    # Snapshot allocations have never been submitted to a GPU.
                    while len(self.buffers) > start:
                        buffer = self.buffers[-1]
                        self.check(self.free(buffer.pointer))
                        buffer.pointer = ct.c_void_p()
                        self.buffers.pop()
                    raise
            state, roots, edges, marks, status = self._snapshot
            with self._access().write(stream):
                for source, dest in zip((self._state, self._roots, self._edges), (state, roots, edges), strict=True):
                    self.check(self._copy_async(dest.pointer, source.pointer, source.nbytes, ct.c_void_p(stream)))
            assert self.epoch is not None and self._marker is not None
            self.epoch._submission.ticket.wait_on(marker_stream)
            try:
                self._collection = self._marker.submit(marker_stream, state, roots, edges, marks, status, 1)
            except BaseException:
                self._collection_uncertain = True
                raise
            return self._collection.ticket

    def finish_collection(self, stream):
        """Remark current roots/edges plus snapshot survivors, then sweep."""
        with self._lock:
            self._ready()
            if self._collection is None:
                raise ValueError("no snapshot collection is active")
            with self._access().write(stream):
                self._collection.ticket.wait_on(stream)
                assert self._snapshot is not None and self._seeded is not None
                marks, status = self._snapshot[-2:]
                try:
                    result = self._seeded.submit(
                        stream, self._state, self._roots, self._edges, self._marks, self._status, marks, status, 1
                    )
                except BaseException:
                    self._collection_uncertain = True
                    raise
                self._collection = None
            return result.ticket

    def wait(self):
        """Prove completion explicitly, including an event-record failure."""
        with self._lock:
            self._ready(recovery=True)
            if self._collection_uncertain:
                self.check(self.native._sync())
                self._collection_uncertain = False
            self._access().wait()
            if self._collection is not None:
                self._collection.ticket.wait()

    def close(self):
        with self._lock:
            if self.closed:
                return
            self._ready(recovery=True)
            if self._collection_uncertain:
                self.check(self.native._sync())
                self._collection_uncertain = False
            if self.epoch is not None:
                self.epoch.wait()
            if self._collection is not None:
                self._collection.ticket.wait()
            self.check(self.native._sync())
            while self.buffers:
                buffer = self.buffers[-1]
                self.check(self.free(buffer.pointer))
                buffer.pointer = ct.c_void_p()
                self.buffers.pop()
            if self._marker is not None:
                self._marker.close()
            if self._seeded is not None:
                self._seeded.close()
            self._graph.close()
            self._allocate.close()
            self._collect.close()
            self.closed = True

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self, *exc):
        self.close()
