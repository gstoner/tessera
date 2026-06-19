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
from .paged_kv import (PagedKVState, PageTier, KVKind, KVGeometry,
                       PageTableEntry, as_paged_kv_state, paged_attention,
                       latent_paged_kv, quantized_tail_paged_kv)
from .memory_state import MemoryStateHandle
from .mla_block_paged import MLABlockPagedCache, MLABlockPagedCacheError
from .mla_paged import MLAPagedDecoder
from .resident_decode import ResidentMLADecoder
from .resident_kv import (ResidentLatentKVCache, ResidentBlockPagedKVCache,
                          ResidentBlockPagedKVCacheError)
from .ssm_state import SSMStateHandle
from .delta_state import DeltaNetStateHandle
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
    "PagedKVState",
    "PageTier",
    "KVKind",
    "KVGeometry",
    "PageTableEntry",
    "as_paged_kv_state",
    "paged_attention",
    "latent_paged_kv",
    "quantized_tail_paged_kv",
    "MemoryStateHandle",
    "SSMStateHandle",
    "DeltaNetStateHandle",
    "MLAPagedDecoder",
    "MLABlockPagedCache",
    "MLABlockPagedCacheError",
    "ResidentMLADecoder",
    "ResidentLatentKVCache",
    "ResidentBlockPagedKVCache",
    "ResidentBlockPagedKVCacheError",
]
