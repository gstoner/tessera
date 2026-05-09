"""KVCacheHandle — paged opaque value type for KV state.

Storage layout (v1):
    K, V are pre-allocated numpy arrays of shape ``(max_seq, num_heads, head_dim)``.
    ``current_seq`` tracks the high-water mark; reads/writes index into the
    leading time axis.

The ``page_size`` parameter is recorded but not yet used to physically page
the buffer — Phase E (rolling-window state, block quantization) will add the
real paging logic. Keeping the parameter on the public surface today means
user code doesn't need to change when paging lands.

Functional update semantics:
    ``append(k, v)`` returns a new handle reference whose ``current_seq``
    advanced by ``k.shape[0]``. The underlying numpy buffers are reused
    (handles share storage) — that mirrors the eventual GPU semantics where
    a single device-resident pool is updated in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# Tessera dtype string → numpy dtype
_DTYPE_MAP = {
    "fp16": np.float16,
    "fp32": np.float32,
    "fp64": np.float64,
    "bf16": np.float32,  # bf16 stored as fp32 in the numpy reference path
}


def _to_np_dtype(dtype: str):
    if dtype not in _DTYPE_MAP:
        raise ValueError(
            f"Unknown KV-cache dtype {dtype!r}. Valid: {sorted(_DTYPE_MAP)}"
        )
    return _DTYPE_MAP[dtype]


def _coerce_kv_input(name: str, arr: np.ndarray, num_heads: int, head_dim: int) -> np.ndarray:
    """Validate + reshape an incoming K/V chunk to ``(seq, num_heads, head_dim)``."""
    if hasattr(arr, "_data") and not isinstance(arr, np.ndarray):
        # DistributedArray / Parameter / similar
        arr = np.asarray(arr)
    arr = np.asarray(arr)

    if arr.ndim == 3:
        # Already (seq, num_heads, head_dim)
        if arr.shape[1:] != (num_heads, head_dim):
            raise ValueError(
                f"{name} shape {arr.shape} does not match cache "
                f"(num_heads={num_heads}, head_dim={head_dim})"
            )
        return arr

    if arr.ndim == 2 and arr.shape[1] == num_heads * head_dim:
        # (seq, num_heads*head_dim) — packed; reshape
        return arr.reshape(arr.shape[0], num_heads, head_dim)

    raise ValueError(
        f"{name} must be (seq, num_heads, head_dim) or (seq, num_heads*head_dim); "
        f"got shape {arr.shape}"
    )


@dataclass
class KVCacheHandle:
    """Opaque handle to a paged KV cache.

    All public attributes are read-only metadata. Mutations go through
    :func:`tessera.ops.kv_cache_append`, :func:`tessera.ops.kv_cache_prune`,
    or the equivalent methods on this handle.
    """

    num_heads: int
    head_dim: int
    max_seq: int
    dtype: str = "fp32"
    page_size: int = 128

    # Mutable state — kept off the dataclass-generated `__init__` via field(...)
    current_seq: int = field(default=0, init=False)
    keys: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    values: np.ndarray = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self):
        if self.num_heads <= 0 or self.head_dim <= 0 or self.max_seq <= 0:
            raise ValueError(
                "num_heads / head_dim / max_seq must all be positive"
            )
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")

        np_dtype = _to_np_dtype(self.dtype)
        # Pre-allocate the full (max_seq, num_heads, head_dim) buffer per K/V.
        self.keys = np.zeros(
            (self.max_seq, self.num_heads, self.head_dim), dtype=np_dtype
        )
        self.values = np.zeros(
            (self.max_seq, self.num_heads, self.head_dim), dtype=np_dtype
        )

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int]:
        """Logical (seq, num_heads, head_dim) of currently-populated cache."""
        return (self.current_seq, self.num_heads, self.head_dim)

    @property
    def num_pages(self) -> int:
        return (self.current_seq + self.page_size - 1) // self.page_size

    @property
    def is_full(self) -> bool:
        return self.current_seq >= self.max_seq

    # ------------------------------------------------------------------
    # Mutation — functional surface (returns self for chaining)
    # ------------------------------------------------------------------

    def append(self, k, v) -> "KVCacheHandle":
        """Append a chunk of (seq, num_heads, head_dim) tokens to the cache."""
        k_arr = _coerce_kv_input("k", k, self.num_heads, self.head_dim)
        v_arr = _coerce_kv_input("v", v, self.num_heads, self.head_dim)
        if k_arr.shape[0] != v_arr.shape[0]:
            raise ValueError(
                f"k and v must have matching sequence length; got "
                f"{k_arr.shape[0]} vs {v_arr.shape[0]}"
            )
        n_new = k_arr.shape[0]
        if self.current_seq + n_new > self.max_seq:
            raise ValueError(
                f"KVCacheHandle.append would exceed max_seq={self.max_seq}: "
                f"current={self.current_seq}, appending={n_new}"
            )
        np_dtype = _to_np_dtype(self.dtype)
        end = self.current_seq + n_new
        self.keys[self.current_seq:end] = k_arr.astype(np_dtype, copy=False)
        self.values[self.current_seq:end] = v_arr.astype(np_dtype, copy=False)
        self.current_seq = end
        return self

    def read(self, start: int, end: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """Read a slice of the cache as ``(K_slice, V_slice)`` numpy views.

        With one positional arg ``start``, returns the chunk
        ``[start, start+1)`` (single-token read) — a common decode pattern.
        With both ``start`` and ``end``, returns ``[start, end)``.
        """
        if end is None:
            end = start + 1
        if not (0 <= start <= self.current_seq):
            raise IndexError(
                f"KVCacheHandle.read: start={start} out of range "
                f"[0, {self.current_seq}]"
            )
        if not (start <= end <= self.current_seq):
            raise IndexError(
                f"KVCacheHandle.read: end={end} out of range "
                f"[{start}, {self.current_seq}]"
            )
        return self.keys[start:end], self.values[start:end]

    def prune(self, max_entries: int) -> "KVCacheHandle":
        """Drop everything before the trailing ``max_entries`` tokens.

        Common for sliding-window decoding. Mutates ``self`` and returns it.
        """
        if max_entries < 0:
            raise ValueError("max_entries must be non-negative")
        if max_entries >= self.current_seq:
            return self
        start = self.current_seq - max_entries
        # Shift the trailing window into [0, max_entries)
        self.keys[:max_entries] = self.keys[start:self.current_seq]
        self.values[:max_entries] = self.values[start:self.current_seq]
        # Zero out the now-stale region for hygiene
        self.keys[max_entries:self.current_seq] = 0
        self.values[max_entries:self.current_seq] = 0
        self.current_seq = max_entries
        return self

    def __repr__(self) -> str:
        return (
            f"KVCacheHandle(num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"max_seq={self.max_seq}, dtype={self.dtype!r}, "
            f"page_size={self.page_size}, current_seq={self.current_seq})"
        )


__all__ = ["KVCacheHandle"]
