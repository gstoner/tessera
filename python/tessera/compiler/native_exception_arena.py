"""C-compatible exception storage with scoped ABI readers and reusable payloads."""
from __future__ import annotations

import ctypes as ct
import threading


class _Node(ct.Structure):
    _fields_ = [("kind", ct.c_uint32), ("payload_offset", ct.c_uint32),
                ("payload_size", ct.c_uint32), ("cause", ct.c_int32),
                ("context", ct.c_int32), ("generation", ct.c_uint64)]


class ExceptionHeapABI:
    """Pins arena storage until close, which must follow native completion.

    Borrowed addresses must not escape this lease. The strong arena reference
    also keeps backing storage alive when the caller drops its arena reference.
    """
    def __init__(self, arena):
        self._arena = arena

    def _value(self, name):
        arena = self._arena
        if arena is None:
            raise ValueError("exception ABI reader is closed")
        with arena._lock:
            if name == 'nodes':
                return ct.addressof(arena._nodes)
            if name == 'payload':
                return ct.addressof(arena._payload)
            return len(arena._nodes if name == 'capacity' else arena._payload)

    nodes = property(lambda self: self._value('nodes'))
    payload = property(lambda self: self._value('payload'))
    capacity = property(lambda self: self._value('capacity'))
    payload_capacity = property(lambda self: self._value('payload_capacity'))

    def close(self):
        arena = self._arena
        if arena is not None:
            with arena._lock:
                if self._arena is not None:
                    arena._readers -= 1
                    self._arena = None

    def __enter__(self):
        self._value('nodes')
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


class NativeExceptionArena:
    """Handles survive growth; read leases exclude allocation and collection."""
    def __init__(self, capacity: int = 16, payload_capacity: int = 1024):
        if type(capacity) is not int or type(payload_capacity) is not int or capacity < 1 or payload_capacity < 1:
            raise ValueError("native exception arena requires positive capacities")
        self._nodes = (_Node * capacity)()
        self._payload = (ct.c_ubyte * payload_capacity)()
        self._live: dict[int, tuple[int, tuple[int, ...]]] = {}
        self._roots: set[int] = set()
        self._next = 0
        self._payload_used = 0
        self._generation = 1
        self._free: list[int] = []
        self._payload_free: list[tuple[int, int]] = []
        self._readers = 0
        self._lock = threading.RLock()

    @property
    def abi(self) -> ExceptionHeapABI:
        with self._lock:
            self._readers += 1
            return ExceptionHeapABI(self)

    def _writable(self):
        if self._readers:
            raise RuntimeError("exception arena mutation requires completed ABI readers")

    def _payload_fits(self, size):
        return self._payload_used + size <= len(self._payload) or any(n >= size for _, n in self._payload_free)

    def _take_payload(self, size):
        if not size:
            return 0
        for i, (offset, length) in enumerate(self._payload_free):
            if length >= size:
                if length == size:
                    self._payload_free.pop(i)
                else:
                    self._payload_free[i] = (offset + size, length - size)
                return offset
        offset = self._payload_used
        self._payload_used += size
        return offset

    def allocate(self, kind: int, payload: bytes, *, edges: tuple[int, ...] = (), root: bool = False) -> int:
        with self._lock:
            self._writable()
            if type(kind) is not int or not 0 <= kind <= 0xffffffff or not isinstance(payload, bytes):
                raise ValueError("invalid native exception object")
            if any(type(edge) is not int or edge not in self._live for edge in edges) or len(edges) > 2:
                raise ValueError("native exception edges must name live handles")
            exhausted = (not self._free and self._next == len(self._nodes)) or not self._payload_fits(len(payload))
            if exhausted and not edges:
                self.collect()
            if (not self._free and self._next == len(self._nodes)) or not self._payload_fits(len(payload)):
                self._grow(max(self._next + (not self._free), len(self._nodes)),
                           max(self._payload_used + len(payload), len(self._payload)))
            handle = self._free.pop() if self._free else self._next
            offset = self._take_payload(len(payload))
            if payload:
                ct.memmove(ct.addressof(self._payload) + offset, payload, len(payload))
            padded = edges + (-1,) * (2 - len(edges))
            self._nodes[handle] = _Node(kind, offset, len(payload), padded[0], padded[1], self._generation)
            self._live[handle] = (self._generation, edges)
            if handle == self._next:
                self._next += 1
            if root:
                self._roots.add(handle)
            return handle

    def release(self, handle: int) -> None:
        with self._lock:
            self._roots.discard(handle)

    def collect(self) -> int:
        with self._lock:
            self._writable()
            reachable: set[int] = set()
            pending = list(self._roots)
            while pending:
                handle = pending.pop()
                if handle in reachable or handle not in self._live:
                    continue
                reachable.add(handle)
                pending.extend(self._live[handle][1])
            dead = set(self._live) - reachable
            for handle in dead:
                node = self._nodes[handle]
                if node.payload_size:
                    self._payload_free.append((node.payload_offset, node.payload_size))
                self._nodes[handle] = _Node()
                del self._live[handle]
                self._free.append(handle)
            spans: list[tuple[int, int]] = []
            for offset, size in sorted(self._payload_free):
                if spans and spans[-1][0] + spans[-1][1] == offset:
                    previous, length = spans.pop()
                    spans.append((previous, length + size))
                else:
                    spans.append((offset, size))
            if spans and sum(spans[-1]) == self._payload_used:
                self._payload_used = spans.pop()[0]
            self._payload_free = spans
            if dead and not self._live:
                self._roots.clear(); self._free.clear(); self._next = 0; self._payload_used = 0
                self._payload_free.clear()
                self._generation += 1
            return len(dead)

    def _grow(self, capacity: int, payload_capacity: int) -> None:
        self._writable()
        if capacity > len(self._nodes):
            nodes = (_Node * max(capacity, 2 * len(self._nodes)))()
            ct.memmove(nodes, self._nodes, ct.sizeof(self._nodes))
            self._nodes = nodes
        if payload_capacity > len(self._payload):
            payload = (ct.c_ubyte * max(payload_capacity, 2 * len(self._payload)))()
            ct.memmove(payload, self._payload, self._payload_used)
            self._payload = payload
