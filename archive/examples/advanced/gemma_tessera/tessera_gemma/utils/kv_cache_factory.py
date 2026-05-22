"""
tessera_gemma/utils/kv_cache_factory.py

Convenience factory for building a list of per-layer PagedKVCache objects.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from ..kernels.kv_cache_tessera import PagedKVCache


def make_kv_caches(
    num_layers: int,
    batch: int,
    kv_heads: int,
    head_dim: int,
    *,
    page_size: int = 128,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> List[PagedKVCache]:
    """Create one fresh PagedKVCache per transformer layer.

    Args:
        num_layers: Number of decoder layers.
        batch:      Batch size.
        kv_heads:   Number of KV attention heads.
        head_dim:   Per-head feature dimension.
        page_size:  Tokens per KV page.
        device:     Target device.
        dtype:      Float dtype for KV tensors.

    Returns:
        List[PagedKVCache] of length num_layers.
    """
    return [
        PagedKVCache(
            batch=batch,
            kv_heads=kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            device=device,
            dtype=dtype,
        )
        for _ in range(num_layers)
    ]
