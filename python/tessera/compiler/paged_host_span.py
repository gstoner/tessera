"""Checked positive-stride host storage for a read-only f32 page tensor.

This module supplies memory facts to the native ABI; it emits no kernel or IR.
"""
from __future__ import annotations

import numpy as np


def checked_page_span(array: np.ndarray) -> tuple[int, tuple[int, ...]]:
    """Return physical bytes and element strides after proving backing capacity."""
    if not isinstance(array, np.ndarray) or array.ndim != 4 or array.dtype != np.dtype("float32"):
        raise ValueError("strided pages require a rank-four native f32 array")
    if any(extent <= 0 for extent in array.shape):
        raise ValueError("strided pages require positive extents")
    if any(stride <= 0 or stride % 4 for stride in array.strides):
        raise ValueError("strided pages require positive whole-element strides")
    strides = tuple(stride // 4 for stride in array.strides)
    span = 4 + sum((extent-1)*stride for extent, stride in zip(array.shape, array.strides, strict=True))
    if span > 2**63-1:
        raise ValueError("strided page physical span exceeds the signed native extent")

    # Follow real NumPy ownership rather than trusting as_strided's claimed
    # extent. Its DummyArray is a view wrapper, not an allocation certificate.
    owner = array
    seen: set[int] = set()
    while isinstance(owner, np.ndarray) and not owner.flags.owndata:
        if id(owner) in seen:
            raise ValueError("cyclic page allocation ownership")
        seen.add(id(owner))
        base = owner.base
        if type(base).__name__ == "DummyArray" and type(base).__module__.startswith("numpy.lib."):
            base = base.base
        if isinstance(base, np.ndarray):
            owner = base
            continue
        try:
            buffer = memoryview(base)
            if not buffer.contiguous:
                raise ValueError("page backing buffer is not contiguous")
            backing = np.frombuffer(buffer, dtype=np.uint8)
        except (TypeError, BufferError, ValueError) as exc:
            raise ValueError("page backing allocation capacity cannot be proved") from exc
        lower, capacity = backing.ctypes.data, backing.nbytes
        break
    else:
        if not owner.flags.c_contiguous and not owner.flags.f_contiguous:
            raise ValueError("page owner allocation capacity requires contiguous backing")
        lower, capacity = owner.ctypes.data, owner.nbytes

    address = array.ctypes.data
    if address % 4 or address < lower or span > capacity or address-lower > capacity-span:
        raise ValueError("page physical span exceeds the backing allocation capacity")
    return span, strides
