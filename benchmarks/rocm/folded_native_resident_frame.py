"""Explicit shared resident roots for matched native benchmark arms."""
from __future__ import annotations
import hashlib


class FoldedNativeFramePool:
    """Own rotating frames until every engine lease has closed."""

    def __init__(self, factory):
        self._factory = factory
        self._signature = None
        self._copies = None
        self._leases = 0
        self._closed = False
        self.report = {"shared": True, "leases": 0, "closed": False}

    def acquire(self, arrays, count):
        if self._closed:
            raise RuntimeError("resident frame pool is closed")
        signature = (count, tuple(
            (id(a), tuple(a.shape), a.dtype.str, tuple(a.strides)) for a in arrays
        ))
        if self._signature is not None and signature != self._signature:
            raise ValueError("shared resident roots or copy count differ")
        if self._copies is None:
            copies = self._factory(arrays, count)
            self._copies = copies
            self._signature = signature
            pointers = tuple(tuple(int(p.value) for p in c.device) for c in copies)
            self.report["pointer_identity_sha256"] = hashlib.sha256(
                repr(pointers).encode()
            ).hexdigest()
            self.report["copies"] = count
        self._leases += 1
        self.report["leases"] = self._leases
        return _FrameLease(self, self._copies)

    def close(self):
        if self._closed:
            return
        if self._leases:
            raise RuntimeError("cannot free resident frames with live engine leases")
        errors = []
        for copy in self._copies or ():
            try:
                copy.close()
            except Exception as exc:
                errors.append(exc)
        self._closed = True
        self.report["closed"] = True
        if errors:
            raise errors[0]


class _FrameLease:
    def __init__(self, pool, copies):
        self._pool = pool
        self.copies = copies
        self._closed = False

    def close(self):
        if not self._closed:
            self._closed = True
            self._pool._leases -= 1
            self._pool.report["leases"] = self._pool._leases
