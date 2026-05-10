"""Reference learned-memory primitives for standalone model research.

These functions are CPU/NumPy reference contracts for Titans/Atlas-style
memory. They are intentionally functional: writes and evictions return a new
table, making them suitable for later compiler state-effect typing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _unwrap(value):
    return value._data if hasattr(value, "_data") else value


@dataclass(frozen=True)
class MemoryTable:
    """Batched key/value memory table.

    `keys` has shape `(entries, key_dim)` and `values` has shape
    `(entries, value_dim...)`. Metadata arrays are carried positionally with the
    same leading `entries` dimension.
    """

    keys: np.ndarray
    values: np.ndarray
    metadata: dict[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        keys = np.asarray(_unwrap(self.keys))
        values = np.asarray(_unwrap(self.values))
        if keys.ndim != 2:
            raise ValueError("MemoryTable.keys must have shape (entries, key_dim)")
        if values.ndim < 1:
            raise ValueError("MemoryTable.values must have at least one dimension")
        if keys.shape[0] != values.shape[0]:
            raise ValueError("MemoryTable keys and values must share the entries dimension")
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "values", values)
        if self.metadata is not None:
            normalized = {name: np.asarray(_unwrap(value)) for name, value in self.metadata.items()}
            for name, value in normalized.items():
                if value.shape[:1] != keys.shape[:1]:
                    raise ValueError(f"metadata[{name!r}] must share the entries dimension")
            object.__setattr__(self, "metadata", normalized)

    @property
    def size(self) -> int:
        return int(self.keys.shape[0])


@dataclass(frozen=True)
class MemoryReadResult:
    values: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    scores: np.ndarray


def _as_table(memory: MemoryTable | tuple[np.ndarray, np.ndarray]) -> MemoryTable:
    if isinstance(memory, MemoryTable):
        return memory
    keys, values = memory
    return MemoryTable(keys=np.asarray(_unwrap(keys)), values=np.asarray(_unwrap(values)))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def memory_read(
    memory: MemoryTable | tuple[np.ndarray, np.ndarray],
    query: np.ndarray,
    *,
    top_k: int = 1,
    normalize: bool = True,
) -> MemoryReadResult:
    """Read from memory with top-k similarity and normalized weighted values."""

    table = _as_table(memory)
    query_arr = np.asarray(_unwrap(query))
    single_query = query_arr.ndim == 1
    if single_query:
        query_arr = query_arr[None, :]
    if query_arr.ndim != 2:
        raise ValueError("query must have shape (key_dim,) or (batch, key_dim)")
    if query_arr.shape[-1] != table.keys.shape[-1]:
        raise ValueError("query key_dim must match MemoryTable.keys key_dim")
    if table.size == 0:
        raise ValueError("cannot read from an empty MemoryTable")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    k = min(int(top_k), table.size)
    scores = query_arr @ table.keys.T
    partition = np.argpartition(-scores, kth=k - 1, axis=-1)[:, :k]
    top_scores = np.take_along_axis(scores, partition, axis=-1)
    order = np.argsort(-top_scores, axis=-1)
    indices = np.take_along_axis(partition, order, axis=-1)
    top_scores = np.take_along_axis(top_scores, order, axis=-1)
    weights = _softmax(top_scores, axis=-1) if normalize else np.ones_like(top_scores)
    if not normalize:
        weights = weights / k
    gathered_values = table.values[indices]
    read_values = np.sum(gathered_values * weights[(...,) + (None,) * (table.values.ndim - 1)], axis=1)
    if single_query:
        return MemoryReadResult(
            values=read_values[0],
            indices=indices[0],
            weights=weights[0],
            scores=top_scores[0],
        )
    return MemoryReadResult(values=read_values, indices=indices, weights=weights, scores=top_scores)


def memory_write(
    memory: MemoryTable | tuple[np.ndarray, np.ndarray],
    keys: np.ndarray,
    values: np.ndarray,
    *,
    max_entries: int | None = None,
) -> MemoryTable:
    """Append key/value rows and optionally evict oldest rows to `max_entries`."""

    table = _as_table(memory)
    new_keys = np.asarray(_unwrap(keys))
    new_values = np.asarray(_unwrap(values))
    if new_keys.ndim == 1:
        new_keys = new_keys[None, :]
    if new_values.shape[:1] != new_keys.shape[:1]:
        new_values = np.asarray(new_values)[None, ...]
    if new_keys.ndim != 2 or new_keys.shape[-1] != table.keys.shape[-1]:
        raise ValueError("keys must have shape (entries, key_dim)")
    if new_values.shape[1:] != table.values.shape[1:]:
        raise ValueError("values must match MemoryTable.values trailing dimensions")

    keys_out = np.concatenate([table.keys, new_keys], axis=0)
    values_out = np.concatenate([table.values, new_values], axis=0)
    metadata = None
    if table.metadata is not None:
        metadata = {}
        for name, value in table.metadata.items():
            fill_shape = (new_keys.shape[0],) + value.shape[1:]
            metadata[name] = np.concatenate([value, np.zeros(fill_shape, dtype=value.dtype)], axis=0)
    written = MemoryTable(keys_out, values_out, metadata)
    if max_entries is not None and written.size > max_entries:
        written = memory_evict(written, keep_last=max_entries)
    return written


def memory_evict(
    memory: MemoryTable | tuple[np.ndarray, np.ndarray],
    *,
    indices: Iterable[int] | np.ndarray | None = None,
    keep_last: int | None = None,
    max_entries: int | None = None,
) -> MemoryTable:
    """Evict explicit indices, or keep the newest `keep_last`/`max_entries` rows."""

    table = _as_table(memory)
    if sum(value is not None for value in (indices, keep_last, max_entries)) != 1:
        raise ValueError("provide exactly one of indices, keep_last, or max_entries")
    if indices is not None:
        drop = np.asarray(list(indices), dtype=np.int64)
        mask = np.ones(table.size, dtype=bool)
        mask[drop] = False
    else:
        keep = keep_last if keep_last is not None else max_entries
        if keep is None or keep < 0:
            raise ValueError("keep count must be >= 0")
        start = max(table.size - int(keep), 0)
        mask = np.zeros(table.size, dtype=bool)
        mask[start:] = True
    metadata = None
    if table.metadata is not None:
        metadata = {name: value[mask] for name, value in table.metadata.items()}
    return MemoryTable(keys=table.keys[mask], values=table.values[mask], metadata=metadata)


__all__ = [
    "MemoryReadResult",
    "MemoryTable",
    "memory_evict",
    "memory_read",
    "memory_write",
]
