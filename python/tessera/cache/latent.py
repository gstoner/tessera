"""Multi-Latent Attention paged KV cache (Theme 5).

DeepSeek's MLA caches the *compressed* latent vector ``c`` (typically
shape ``[seq, latent_dim]``) instead of the full per-head K and V tensors
(``[seq, num_heads, head_dim]`` each). The K and V are reconstructed at
read time via :func:`tessera.ops.latent_kv_expand_k` /
:func:`tessera.ops.latent_kv_expand_v`. With ``latent_dim ≪
num_heads * head_dim`` this gives the ~93% KV-cache memory reduction
DeepSeek reports.

This module ships the Python-side latent-cache value type. Per-backend
target kernels (FlashMLA on Hopper / Blackwell, absorb-K fusion) are
deferred to Phase G — the Python op surface unblocks the
``examples/advanced/mla/`` end-to-end path today.

API contract:

    cache = tessera.cache.LatentKVCacheHandle(latent_dim=128, max_seq=2048)
    cache.append(c)                      # functional append; returns self
    c_slice = cache.read(start, end)     # paged read
    cache.evict_oldest(n)                # rolling-window decode
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


_DTYPE_MAP = {
    "fp16": np.float16,
    "bf16": np.float32,  # bf16 stored as fp32 in the numpy reference
    "fp32": np.float32,
    "fp64": np.float64,
}


@dataclass
class LatentKVCacheHandle:
    """Opaque paged latent-KV cache.

    Stores a ``[max_seq, latent_dim]`` compressed buffer. K and V are
    reconstructed on demand via the expand ops; this handle never
    materializes full K/V — that's the memory win.

    For decoupled-RoPE MLA (the standard variant) callers usually keep a
    second tiny cache of shape ``[max_seq, rope_dim]`` for the positional
    slice. Two ``LatentKVCacheHandle`` instances do the job — set
    ``latent_dim=rope_dim`` for the second one.

    Parameters
    ----------
    latent_dim
        Width of the compressed latent vector. Typically 64 / 128 / 256
        for production MLA configurations (vs.
        ``num_heads × head_dim`` ≈ 1k–8k for full K/V).
    max_seq
        Cache capacity in tokens.
    dtype
        Storage dtype: ``"fp16"`` / ``"bf16"`` / ``"fp32"`` / ``"fp64"``.
    page_size
        Page size in tokens. Recorded for forward-compat with paged
        backends; today the storage is a single contiguous buffer.
    auto_evict
        When ``True``, append past ``max_seq`` shifts the trailing
        window down (sliding window). When ``False`` (default), it
        raises.
    """

    latent_dim: int
    max_seq: int
    dtype: str = "fp32"
    page_size: int = 128
    auto_evict: bool = False

    current_seq: int = field(default=0, init=False)
    latents: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self):
        if self.latent_dim <= 0 or self.max_seq <= 0:
            raise ValueError("latent_dim / max_seq must be positive")
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        np_dtype = _DTYPE_MAP.get(self.dtype, np.float32)
        self.latents = np.zeros((self.max_seq, self.latent_dim), dtype=np_dtype)

    # ------------------------------------------------------------------
    # Read-only metadata
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """Logical ``(seq, latent_dim)`` of currently-populated cache."""
        return (self.current_seq, self.latent_dim)

    @property
    def num_pages(self) -> int:
        return (self.current_seq + self.page_size - 1) // self.page_size

    @property
    def is_full(self) -> bool:
        return self.current_seq >= self.max_seq

    # ------------------------------------------------------------------
    # Mutation — functional surface
    # ------------------------------------------------------------------

    def append(self, c) -> "LatentKVCacheHandle":
        """Append a ``(n_new, latent_dim)`` chunk of compressed latents.

        Behavior on overflow mirrors :class:`KVCacheHandle`:
          * ``auto_evict=False`` (default): raises ``ValueError``.
          * ``auto_evict=True``: shifts the trailing window down so the
            new chunk fits.
        """
        if hasattr(c, "_data"):
            c = c._data
        c_arr = np.asarray(c)
        if c_arr.ndim == 1:
            c_arr = c_arr.reshape(1, -1)
        if c_arr.ndim != 2 or c_arr.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent chunk must have shape (seq, {self.latent_dim}); "
                f"got {c_arr.shape}"
            )
        n_new = c_arr.shape[0]
        if self.current_seq + n_new > self.max_seq:
            if not self.auto_evict:
                raise ValueError(
                    f"LatentKVCacheHandle.append would exceed "
                    f"max_seq={self.max_seq}: current={self.current_seq}, "
                    f"appending={n_new}"
                )
            if n_new > self.max_seq:
                raise ValueError(
                    f"chunk size {n_new} exceeds max_seq={self.max_seq}; "
                    f"cannot fit even after eviction"
                )
            keep = self.max_seq - n_new
            shift_start = self.current_seq - keep
            self.latents[:keep] = self.latents[shift_start:self.current_seq]
            self.latents[keep:self.current_seq] = 0
            self.current_seq = keep

        end = self.current_seq + n_new
        np_dtype = _DTYPE_MAP.get(self.dtype, np.float32)
        self.latents[self.current_seq:end] = c_arr.astype(np_dtype, copy=False)
        self.current_seq = end
        return self

    def evict_oldest(self, n: int) -> "LatentKVCacheHandle":
        """Drop the oldest ``n`` tokens, shifting the remainder to the front."""
        if n < 0:
            raise ValueError("evict_oldest n must be non-negative")
        if n == 0:
            return self
        n = min(n, self.current_seq)
        keep = self.current_seq - n
        if keep > 0:
            self.latents[:keep] = self.latents[n:self.current_seq]
        self.latents[keep:self.current_seq] = 0
        self.current_seq = keep
        return self

    def read(self, start: int, end: Optional[int] = None) -> np.ndarray:
        """Read a slice of compressed latents as a ``(end - start, latent_dim)``
        array. With one arg ``start``, returns the single-token chunk
        ``[start, start+1)`` for decode-style usage."""
        if end is None:
            end = start + 1
        if start < 0 or end > self.current_seq or start > end:
            raise ValueError(
                f"latent cache slice [{start}, {end}) out of range "
                f"[0, {self.current_seq})"
            )
        return np.asarray(self.latents[start:end])

    def __repr__(self) -> str:
        return (
            f"LatentKVCacheHandle(latent_dim={self.latent_dim}, "
            f"max_seq={self.max_seq}, current_seq={self.current_seq}, "
            f"dtype={self.dtype!r}, auto_evict={self.auto_evict})"
        )
