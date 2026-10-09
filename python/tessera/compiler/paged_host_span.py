"""Checked positive-stride host storage for a read-only f32 page tensor.

This module supplies memory facts to the native ABI; it emits no kernel or IR.
"""
from __future__ import annotations

import numpy as np


def checked_page_span(array: np.ndarray) -> tuple[int, tuple[int, ...]]:
    """Return physical bytes and element strides after proving backing capacity."""
    if not isinstance(array, np.ndarray) or array.ndim != 4 or array.dtype != np.dtype("float32"):
        raise ValueError("strided pages require a rank-four native f32 array")
    span, byte_strides = checked_host_span(array, label="page")
    return span, tuple(stride // 4 for stride in byte_strides)


def checked_host_span(array: np.ndarray, *, label: str = "tensor") -> tuple[int, tuple[int, ...]]:
    """Prove positive host byte strides against the actual owning allocation."""
    if not isinstance(array, np.ndarray) or not 2 <= array.ndim <= 32:
        raise ValueError(f"strided {label} storage requires rank 2 through 32")
    itemsize = array.dtype.itemsize
    if any(extent <= 0 for extent in array.shape):
        raise ValueError(f"strided {label}s require positive extents")
    if any(stride <= 0 or stride % itemsize for stride in array.strides):
        raise ValueError(f"strided {label}s require positive whole-element strides")
    strides = tuple(array.strides)
    span = itemsize + sum((extent-1)*stride for extent, stride in zip(array.shape, array.strides, strict=True))
    if span > 2**63-1:
        raise ValueError(f"strided {label} physical span exceeds the signed native extent")

    # Follow real NumPy ownership rather than trusting as_strided's claimed
    # extent. Its DummyArray is a view wrapper, not an allocation certificate.
    owner = array
    seen: set[int] = set()
    while isinstance(owner, np.ndarray) and not owner.flags.owndata:
        if id(owner) in seen:
            raise ValueError(f"cyclic {label} allocation ownership")
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
                raise ValueError(f"{label} backing buffer is not contiguous")
            backing = np.frombuffer(buffer, dtype=np.uint8)
        except (TypeError, BufferError, ValueError) as exc:
            raise ValueError(f"{label} backing allocation capacity cannot be proved") from exc
        lower, capacity = backing.ctypes.data, backing.nbytes
        break
    else:
        if any(stride < 0 for stride in owner.strides):
            raise ValueError(f"{label} owner allocation capacity requires positive backing")
        # NumPy may allocate a dense permutation (for example stack of views)
        # that is neither C nor F contiguous. OWNDATA certifies the allocation;
        # its actual positive-stride span, not logical nbytes, bounds storage.
        lower = owner.ctypes.data
        capacity = owner.itemsize + sum(
            (extent - 1) * stride for extent, stride in
            zip(owner.shape, owner.strides, strict=True)
        )

    address = array.ctypes.data
    if address % itemsize or address < lower or span > capacity or address-lower > capacity-span:
        raise ValueError(f"{label} physical span exceeds the backing allocation capacity")
    return span, strides
