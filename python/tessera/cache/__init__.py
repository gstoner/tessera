"""KV-cache handle abstraction (Phase B2).

Today the storage is a single contiguous numpy buffer per K/V — the
``page_size`` parameter is recorded for forward-compat with paged backends
(Phase E will land block quantization + rolling-window state). The handle
type is opaque to op signatures, so the same Python code works once
backends start lowering caches as first-class state.

Ergonomic entry point — all the user normally needs:

::

    cache = tessera.cache.KVCacheHandle(num_heads=8, head_dim=64, max_seq=2048)
    cache = tessera.ops.kv_cache_append(cache, k, v)   # functional update
    k_slice, v_slice = tessera.ops.kv_cache_read(cache, 0, 64)
"""

from __future__ import annotations

from .handle import KVCacheHandle
from .latent import LatentKVCacheHandle
from .memory_state import MemoryStateHandle
from .mla_block_paged import MLABlockPagedCache, MLABlockPagedCacheError
from .mla_paged import MLAPagedDecoder
from .resident_decode import ResidentMLADecoder
from .resident_kv import (ResidentLatentKVCache, ResidentBlockPagedKVCache,
                          ResidentBlockPagedKVCacheError)
from .tiered import (TieredKVCache, TieredKVCacheError, StageStats,
                     TieredStats, lookahead_attention_tiered)

__all__ = [
    "TieredKVCache",
    "TieredKVCacheError",
    "StageStats",
    "TieredStats",
    "lookahead_attention_tiered",
    "KVCacheHandle",
    "LatentKVCacheHandle",
    "MemoryStateHandle",
    "MLAPagedDecoder",
    "MLABlockPagedCache",
    "MLABlockPagedCacheError",
    "ResidentMLADecoder",
    "ResidentLatentKVCache",
    "ResidentBlockPagedKVCache",
    "ResidentBlockPagedKVCacheError",
]
