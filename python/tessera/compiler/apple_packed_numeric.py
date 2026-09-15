"""Explicit Metal4.1 packed-buffer numerical execution (no MX/matrix admission).

Storage is E4M3FN, E5M2 or low-nibble-first E2M1. Arithmetic is fp32 after
Metal unpack. NaN sign/payload preservation is not promised. Packing uses
nearest-even; E4M3/FP4 saturate overflow and E5M2 permits infinity.
"""
from __future__ import annotations

import ctypes as ct

import numpy as np


def evaluate(codes: np.ndarray, values: np.ndarray, dtype: str) -> tuple[np.ndarray, np.ndarray]:
    """Return decode/add-one/mul-two/div-two planes and independently packed values.

    This bounded explicit API performs a synchronous GPU submission with a
    five-second wait bound. Failure raises; it never substitutes CPU execution.
    It does not enable generic microscaling, tensor or matrix routes.
    """
    kinds = {'fp8_e4m3': 0, 'fp8_e5m2': 1, 'fp4_e2m1': 2}
    if dtype not in kinds:
        raise ValueError('unsupported Metal packed dtype')
    if not isinstance(values, np.ndarray) or values.dtype != np.float32 or values.ndim != 1:
        raise ValueError('values must be a rank-one fp32 array')
    n = values.size
    if n == 0 or n > 1048576 or n % 8:
        raise ValueError('count must be a positive multiple of eight, at most 1048576')
    size = n // 2 if kinds[dtype] == 2 else n
    if not isinstance(codes, np.ndarray) or codes.dtype != np.uint8 or codes.shape != (size,):
        raise ValueError('codes must contain exactly the packed uint8 storage')
    from .._apple_gpu_dispatch import apple_gpu_runtime

    lib = apple_gpu_runtime()
    if lib is None or not hasattr(lib, 'tessera_apple_gpu_packed_numeric_status'):
        raise RuntimeError('SDK27 packed numeric runtime unavailable')
    run = lib.tessera_apple_gpu_packed_numeric_status
    run.argtypes = [ct.c_int32, ct.c_void_p, ct.c_void_p, ct.c_int32, ct.c_void_p, ct.c_void_p]
    run.restype = ct.c_int32
    codes = np.ascontiguousarray(codes)
    values = np.ascontiguousarray(values)
    output = np.empty((4, n), np.float32)
    packed = np.empty(size, np.uint8)
    if run(kinds[dtype], codes.ctypes.data, values.ctypes.data, n,
           output.ctypes.data, packed.ctypes.data) != 1:
        raise RuntimeError('Metal packed numeric submission failed; no CPU fallback')
    return output, packed
