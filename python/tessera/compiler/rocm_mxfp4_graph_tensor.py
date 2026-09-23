"""Model-owned HIP buffers for the opt-in gfx1201 MXFP4 graph route.

This is deliberately separate from the Metal-only public ``DeviceTensor`` ABI.
The graph borrows a stable pointer; the model retains allocation ownership.
"""
from __future__ import annotations

import ctypes
from math import prod
from typing import Any

import numpy as np


class ROCMGraphTensor:
    """Contiguous, model-owned allocation that cannot be freed while borrowed.

    The caller must order any device-side writes on the graph's stream before
    replay. A failed final synchronization quarantines the allocation because
    it is not safe to infer that outstanding device users have completed.
    """

    def __init__(
        self, shape: tuple[int, ...], dtype: Any, *, hip: Any | None = None,
    ) -> None:
        from tessera import runtime as rt

        if not shape or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise ValueError("ROCm graph tensor requires positive dimensions")
        resolved_dtype = np.dtype(dtype)
        bf16 = rt._bfloat16_dtype()
        accepted = {np.dtype(np.float32)}
        if bf16 is not None:
            accepted.add(np.dtype(bf16))
        if resolved_dtype not in accepted:
            raise TypeError("ROCm graph tensor requires FP32 or BF16 storage")
        if rt._rocm_live_arch() != "gfx1201":
            raise RuntimeError("ROCm graph tensor requires selected gfx1201")
        loaded_hip = hip if hip is not None else rt._load_hip_for_launch()
        if loaded_hip is None or loaded_hip.hipInit(0) != 0:
            raise RuntimeError("ROCm graph tensor requires a usable HIP context")
        self._hip: Any = loaded_hip
        self._hip.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._hip.hipFree.argtypes = [ctypes.c_void_p]
        ordinal = ctypes.c_int()
        if self._hip.hipGetDevice(ctypes.byref(ordinal)) != 0:
            raise RuntimeError("ROCm graph tensor cannot identify selected device")
        self.device_ordinal = ordinal.value
        self.shape = tuple(shape)
        self.dtype = resolved_dtype
        self.nbytes = prod(shape) * resolved_dtype.itemsize
        self._pointer = ctypes.c_void_p()
        self._borrowers = 0
        self._closed = False
        self._uncertain = False
        if self._hip.hipMalloc(ctypes.byref(self._pointer), self.nbytes) != 0:
            raise RuntimeError("ROCm graph tensor allocation failed")
        if not self._pointer.value:
            raise RuntimeError("ROCm graph tensor allocation returned null")

    @property
    def pointer(self) -> int:
        if self._closed or self._uncertain or self._pointer.value is None:
            raise RuntimeError("ROCm graph tensor pointer is unavailable")
        return self._pointer.value

    @property
    def borrowers(self) -> int:
        return self._borrowers

    def borrow(
        self, shape: tuple[int, ...], dtype: Any, device_ordinal: int,
    ) -> int:
        if self._borrowers:
            raise RuntimeError("ROCm graph tensor already has a live graph borrower")
        if self.shape != shape or self.dtype != np.dtype(dtype):
            raise ValueError("ROCm graph tensor shape/dtype disagrees with graph ABI")
        if self.device_ordinal != device_ordinal:
            raise ValueError("ROCm graph tensor belongs to another HIP device")
        pointer = self.pointer
        self._borrowers += 1
        return pointer

    def release(self, *, synchronized: bool) -> None:
        if self._borrowers <= 0:
            raise RuntimeError("ROCm graph tensor has no active borrower")
        self._borrowers -= 1
        if not synchronized:
            self._uncertain = True

    def close(self) -> None:
        if self._closed:
            return
        if self._borrowers or self._uncertain:
            raise RuntimeError("ROCm graph tensor cannot free a borrowed or uncertain pointer")
        if self._hip.hipFree(self._pointer) != 0:
            raise RuntimeError("ROCm graph tensor hipFree failed")
        self._pointer.value = None
        self._closed = True

    def __enter__(self) -> ROCMGraphTensor:
        _ = self.pointer
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["ROCMGraphTensor"]
