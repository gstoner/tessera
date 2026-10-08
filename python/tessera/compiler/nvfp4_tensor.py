"""Logical NVFP4 metadata over caller-owned packed host storage."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class NVFP4Tensor:
    """Low nibble first; two E2M1 values per byte along packed_axis.

    This binding performs no conversion, allocation or kernel construction.
    Scales and numerical policy remain Graph attributes/operands.
    """
    storage: np.ndarray
    shape: tuple[int, ...]
    packed_axis: int

    @property
    def dtype(self) -> str:
        return "nvfp4"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if (not isinstance(self.shape, tuple) or len(self.shape) < 2
                or any(type(d) is not int or not 0 < d < 2**63 for d in self.shape)
                or type(self.packed_axis) is not int
                or not 0 <= self.packed_axis < len(self.shape)):
            raise ValueError("NVFP4 requires positive logical matrix dimensions with an optional leading batch prefix and a packing axis")
        if (not isinstance(self.storage, np.ndarray) or self.storage.dtype != np.uint8
                or not self.storage.flags.c_contiguous):
            raise ValueError("NVFP4 requires compact uint8 host storage")
        physical = list(self.shape)
        physical[self.packed_axis] = physical[self.packed_axis] // 2 + physical[self.packed_axis] % 2
        if self.storage.shape != tuple(physical):
            raise ValueError("NVFP4 packed buffer shape differs from its logical tensor")

    def __array__(self, dtype=None, copy=None):
        raise TypeError("NVFP4 packed storage is not an ordinary numerical ndarray")
