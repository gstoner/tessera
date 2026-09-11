"""All admitted metadata operations share a native gate; epochs retain ownership.

Live full-pool pointers and legacy snapshot/import paths are intentionally not
admitted. Metadata inspection copies under the gate; payload readers use pins.
"""
import ctypes as ct
import numpy as np
from .resident_incremental_pool import ResidentIncrementalPool
from .native_device_tape import _Buffer
from .gpu_heap_collection import materialize_pool
from .heap_barrier_contract import ATOMIC_MODES


class _GatedBinding:
    def __init__(self, pool, binding):
        self.pool, self.binding = pool, binding

    def __getattr__(self, name):
        return getattr(self.binding, name)

    def submit(self, stream, *args):
        return self.binding.submit(stream, *args[:-1], self.pool._gate, args[-1])


class ResidentGatedPool(ResidentIncrementalPool):
    def __init__(self, slots, width, references, *, stream, **options):
        super().__init__(slots, width, references, stream=stream, **options)
        try:
            self._gate = _Buffer(self, (1,), 'int64')
            zero = np.zeros(1, np.int64)
            self.check(self._upload(self._gate.pointer, ct.c_void_p(zero.ctypes.data), zero.nbytes))
            self._inspection = [_Buffer(self, shape, 'int64') for shape in
                                ((slots, 3), (slots,), (slots, 2 * references), (slots,))]
        except BaseException:
            self.close()
            raise

    def _binding(self, mode):
        if mode not in ATOMIC_MODES:
            raise ValueError('metadata operation has no gated producer')
        if mode not in self._incremental_bindings:
            slots, width, refs = self._dimensions
            binding = materialize_pool(slots, width, 'atomic_' + mode, payload_dtype='int8',
                                       references=refs, **self._options).bind()
            self._incremental_bindings[mode] = _GatedBinding(self, binding)
        return self._incremental_bindings[mode]

    def allocate(self, stream, payload, length):
        with self._lock:
            self._ready()
            self._drain_object_readers(stream)
            mode = 'allocate_marked' if self._mark_active else 'allocate'
            args: tuple = (self._state, self._roots, self._edges, self._payload, payload, self._status, length)
            if self._mark_active:
                args += (self._marks,)
            with self._access().write(stream):
                return self._binding(mode).submit(stream, *args, 1).ticket

    def set_graph(self, stream, roots, edges):
        with self._lock:
            self._ready()
            mode = 'graph_incremental' if self._mark_active else 'graph_checked'
            args: tuple = (self._state, self._roots, self._edges, roots, edges)
            if self._mark_active:
                args += (self._marks,)
            with self._access().write(stream):
                return self._binding(mode).submit(stream, *args, self._status, 1).ticket

    def inspect_metadata(self, stream):
        """Return checked host copies, never live metadata pointers."""
        with self._lock:
            self._ready()
            with self._access().write(stream):
                ticket = self._binding('inspect').submit(stream, self._state, self._roots, self._edges,
                    self._marks, self._status, *self._inspection, 1).ticket
            if self._status_values(ticket)[0] != 0:
                raise ValueError('metadata inspection gate busy')
            result = []
            for buffer in self._inspection:
                array = np.empty(buffer.shape, np.int64)
                self.check(self._download(ct.c_void_p(array.ctypes.data), buffer.pointer, array.nbytes))
                result.append(array)
            return tuple(result)

    def read(self, stream):
        raise ValueError('gated pool exposes copied inspect_metadata or pinned payload readers')

    def snapshot(self, stream):
        raise ValueError('gated pool requires copied inspect_metadata')

    @classmethod
    def from_objects(cls, *args, **kwargs):
        raise ValueError('gated pool requires native allocation and graph publication')
