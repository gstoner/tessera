"""Immutable frontend FP32 splats shared by AST and abstract tracing."""
from dataclasses import dataclass
import struct

import numpy as np


@dataclass(frozen=True)
class FP32Splat:
    shape: tuple[int,...]
    bits: int

    @property
    def value(self):
        return struct.unpack("<f",self.bits.to_bytes(4,"little"))[0]

    def array(self):
        return np.full(self.shape,self.value,dtype=np.float32)

    def attributes(self,ty):
        return f"value = dense<{self.value!r}> : {ty}"


def capture(value):
    """Only explicitly read-only, finite, bit-identical FP32 arrays are literals.

    A value snapshot, rather than an ndarray reference or truncated repr, owns
    cache identity. Mutable arrays remain runtime operands, not hidden weights.
    """
    if (not isinstance(value,np.ndarray) or value.flags.writeable
            or value.dtype != np.dtype("float32") or not value.size):
        return None
    raw=np.array(value,copy=True,order="C").view(np.uint32).reshape(-1)
    first=int(raw[0])
    if not np.all(raw==raw[0]) or first & 0x7f800000 == 0x7f800000:
        return None
    return FP32Splat(tuple(value.shape),first)
