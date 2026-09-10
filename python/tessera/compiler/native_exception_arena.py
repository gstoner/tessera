"""C-compatible exception object arena with explicit roots and collection."""
from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass


class _Node(ct.Structure):
    _fields_ = [("kind", ct.c_uint32), ("payload_offset", ct.c_uint32),
                ("payload_size", ct.c_uint32), ("cause", ct.c_int32),
                ("context", ct.c_int32), ("generation", ct.c_uint64)]


@dataclass(frozen=True)
class ExceptionHeapABI:
    nodes: int
    payload: int
    capacity: int
    payload_capacity: int


class NativeExceptionArena:
    """Growable native storage; handles stay stable until their generation dies."""
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

    @property
    def abi(self) -> ExceptionHeapABI:
        return ExceptionHeapABI(ct.addressof(self._nodes), ct.addressof(self._payload),
                                len(self._nodes), len(self._payload))

    def allocate(self, kind: int, payload: bytes, *, edges: tuple[int, ...] = (), root: bool = False) -> int:
        if type(kind) is not int or kind < 0 or not isinstance(payload, bytes):
            raise ValueError("invalid native exception object")
        if any(edge not in self._live for edge in edges) or len(edges) > 2:
            raise ValueError("native exception edges must name live handles")
        exhausted = (not self._free and self._next == len(self._nodes)) or self._payload_used + len(payload) > len(self._payload)
        if exhausted and not edges:
            self.collect()
        exhausted = (not self._free and self._next == len(self._nodes)) or self._payload_used + len(payload) > len(self._payload)
        if exhausted:
            self._grow(max(self._next + 1, len(self._nodes) * 2),
                       max(self._payload_used + len(payload), len(self._payload) * 2))
        handle = self._free.pop() if self._free else self._next
        offset = self._payload_used
        if payload:
            ct.memmove(ct.addressof(self._payload) + offset, payload, len(payload))
        padded = edges + (-1,) * (2 - len(edges))
        self._nodes[handle] = _Node(kind, offset, len(payload), padded[0], padded[1], self._generation)
        self._live[handle] = (self._generation, edges)
        if handle == self._next:
            self._next += 1
        self._payload_used += len(payload)
        if root:
            self._roots.add(handle)
        return handle

    def release(self, handle: int) -> None:
        self._roots.discard(handle)

    def collect(self) -> int:
        reachable: set[int] = set()
        pending = list(self._roots)
        while pending:
            handle = pending.pop()
            if handle in reachable or handle not in self._live:
                continue
            reachable.add(handle)
            pending.extend(self._live[handle][1])
        dead = set(self._live) - reachable
        if not dead:
            return 0
        for handle in dead:
            del self._live[handle]
            self._free.append(handle)
        if not self._live:
            self._roots.clear(); self._free.clear(); self._next = 0; self._payload_used = 0
            self._generation += 1
        return len(dead)

    def _grow(self, capacity: int, payload_capacity: int) -> None:
        nodes = (_Node * capacity)(); payload = (ct.c_ubyte * payload_capacity)()
        ct.memmove(nodes, self._nodes, ct.sizeof(self._nodes))
        ct.memmove(payload, self._payload, self._payload_used)
        self._nodes, self._payload = nodes, payload
