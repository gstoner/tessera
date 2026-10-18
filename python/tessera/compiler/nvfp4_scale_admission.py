"""Checked native encoded-scale metadata; no GPU or tensor arithmetic."""
import ctypes as C
import numpy as np


def native_scale_bounds(library):
    function = getattr(library, "tessera_rocm_program_nvfp4_scale_bounds", None)
    if function is None:
        return None
    function.argtypes = [C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64,
                         C.POINTER(C.c_int64), C.c_uint64,
                         C.POINTER(C.c_uint8), C.c_uint64]
    function.restype = C.c_int

    def bounds(scales, offsets):
        compact = np.ascontiguousarray(scales)
        boundaries = (C.c_int64 * len(offsets))(*offsets)
        result = (C.c_uint8 * (2 * (len(offsets) - 1)))()
        status = function(C.c_void_p(compact.ctypes.data), compact.nbytes,
                          *compact.shape, boundaries, len(offsets), result, len(result))
        if status == 2:
            raise ValueError("NVFP4 expansion scales must be finite and nonnegative")
        if status:
            raise RuntimeError(f"NVFP4 native scale bounds status {status}")
        return tuple((int(result[i]), int(result[i + 1])) for i in range(0, len(result), 2))
    return bounds
