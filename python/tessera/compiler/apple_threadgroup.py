"""Apple-owned dynamic threadgroup slot contract for the tiled MSL ABI."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ThreadgroupSlot:
    index: int
    name: str
    element_type: str
    element_bytes: int

    def declaration(self) -> str:
        if type(self.index) is not int or not 0 <= self.index < 31 or not self.name.isidentifier():
            raise ValueError('invalid Metal threadgroup slot')
        if (self.element_type, self.element_bytes) not in (('float', 4), ('uchar', 1)):
            raise ValueError('unsupported Metal threadgroup storage type')
        return f'threadgroup {self.element_type}* {self.name} [[threadgroup({self.index})]]'

    def length(self, elements: int, *, device_limit: int, static_bytes: int = 0) -> int:
        self.declaration()
        if any(type(v) is not int for v in (elements, device_limit, static_bytes)) or elements <= 0 or static_bytes < 0:
            raise ValueError('invalid Metal threadgroup extent')
        size = elements * self.element_bytes
        if size % 16:
            raise ValueError('legacy Metal tiled ABI requires a 16-byte aligned dynamic extent')
        if size + static_bytes > device_limit:
            raise ValueError('Metal static plus dynamic threadgroup memory exceeds device limit')
        return size


TILED_SCORES = ThreadgroupSlot(0, 'tg_scores', 'float', 4)


def tiled_threadgroup_length(region, columns: int, device_limit: int) -> int:
    # The existing MSL producer reserves one 32-float reduction scratch array.
    return TILED_SCORES.length(columns, device_limit=device_limit,
                               static_bytes=128 if region.reduction is not None else 0)


def tiled_threadgroup_available(region, columns: int) -> bool:
    from tessera.runtime import _load_apple_gpu_runtime
    runtime = _load_apple_gpu_runtime()
    query = getattr(runtime, 'tessera_apple_gpu_max_threadgroup_memory_length', None)
    if query is None:
        return False
    import ctypes
    query.argtypes, query.restype = [], ctypes.c_int64
    try:
        tiled_threadgroup_length(region, columns, int(query()))
        return True
    except ValueError:
        return False
