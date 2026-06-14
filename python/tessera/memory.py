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
    abstained: "bool | np.ndarray" = False
    """Whether the read abstained because no entry cleared ``abstain_below``.

    Scalar ``bool`` for a single query; a ``(batch,)`` bool array for a batched
    query.  Defaults to ``False`` so existing callers (and the autodiff
    constructors that build this result positionally by keyword) are unaffected.
    When a row abstains, its ``values`` are filled with ``NaN`` so a stale hit
    can never be mistaken for a real retrieval — the LongMemEval "answer is not
    in memory → abstain" contract.
    """


def _as_table(memory: MemoryTable | tuple[np.ndarray, np.ndarray]) -> MemoryTable:
    if isinstance(memory, MemoryTable):
        return memory
    keys, values = memory
    return MemoryTable(keys=np.asarray(_unwrap(keys)), values=np.asarray(_unwrap(values)))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _recency_signal(table: MemoryTable, recency_key: str | None) -> np.ndarray:
    """The per-entry recency ordering used to break score ties.

    ``None`` means *insertion order* — the bank appends newest-last, so the row
    index is the recency rank.  A ``recency_key`` reads an explicit metadata
    column (e.g. ``"version"`` / ``"timestamp"`` / ``"session"``)."""
    if recency_key is None:
        return np.arange(table.size)
    if table.metadata is None or recency_key not in table.metadata:
        raise ValueError(
            f"recency_key {recency_key!r} not found in MemoryTable.metadata"
        )
    return np.asarray(table.metadata[recency_key])


def memory_read(
    memory: MemoryTable | tuple[np.ndarray, np.ndarray],
    query: np.ndarray,
    *,
    top_k: int = 1,
    normalize: bool = True,
    abstain_below: float | None = None,
    prefer_recent: bool = False,
    recency_key: str | None = None,
) -> MemoryReadResult:
    """Read from memory with top-k similarity and normalized weighted values.

    When ``abstain_below`` is given, a query whose best similarity score (the
    same score used to rank entries) falls below the threshold *abstains*: its
    ``values`` are filled with ``NaN`` and the result's ``abstained`` flag is set
    (a scalar ``bool`` for a single query, a ``(batch,)`` bool array otherwise).
    Scores are raw ``query·keyᵀ`` similarities, so normalize keys/queries
    upstream if you want the threshold to behave like a cosine floor.

    When ``prefer_recent`` is set (or a ``recency_key`` metadata column is
    named), entries are ranked by ``(score desc, recency desc)`` so that an
    exact-key tie resolves to the *newest* write — the version-aware /
    knowledge-update retrieval that plain similarity top-k cannot express. The
    recency signal is insertion order by default, or the named metadata column.
    """

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

    if prefer_recent or recency_key is not None:
        # Rank by (score desc, recency desc): lexsort's LAST key is primary, so
        # ``(-recency, -score)`` sorts ascending -score (score desc) then breaks
        # ties on ascending -recency (recency desc).
        recency = _recency_signal(table, recency_key)
        recency_b = np.broadcast_to(recency, scores.shape)
        ranked = np.lexsort((-recency_b, -scores), axis=-1)
        indices = ranked[:, :k]
        top_scores = np.take_along_axis(scores, indices, axis=-1)

    weights = _softmax(top_scores, axis=-1) if normalize else np.ones_like(top_scores)
    if not normalize:
        weights = weights / k
    gathered_values = table.values[indices]
    read_values = np.sum(gathered_values * weights[(...,) + (None,) * (table.values.ndim - 1)], axis=1)

    abstained_mask = np.zeros(query_arr.shape[0], dtype=bool)
    if abstain_below is not None:
        abstained_mask = top_scores.max(axis=-1) < float(abstain_below)  # (nq,) bool
        if abstained_mask.any():
            read_values = read_values.copy()
            read_values[abstained_mask] = np.nan

    if single_query:
        return MemoryReadResult(
            values=read_values[0],
            indices=indices[0],
            weights=weights[0],
            scores=top_scores[0],
            abstained=bool(abstained_mask[0]),
        )
    return MemoryReadResult(
        values=read_values,
        indices=indices,
        weights=weights,
        scores=top_scores,
        abstained=abstained_mask if abstain_below is not None else False,
    )


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


# ─────────────────────────────────────────────────────────────────────────────
# Sprint D — vmap-axis registry for stateful memory primitives.
#
# The memory bank arg is shared state — `vmap` should NOT add a batch axis
# to it.  The query / keys / values / scores args ARE batchable.
#
# Per-primitive axis map: tuple-keyed by op name, value is a tuple of
# `int | None | "state"` annotations matching the positional-arg slots.
#   - int N    : this arg's batch dimension is axis N
#   - None     : this arg is unbatched (broadcast through)
#   - "state"  : this arg is shared state — never replicate, never split
#
# Backends consult `vmap_axis_for(name)` before falling back to uniform-axis
# default semantics.
# ─────────────────────────────────────────────────────────────────────────────


_VMAP_AXIS_MAP: dict[str, tuple] = {
    # (memory_or_handle, query)                 batched: query at axis 0
    "memory_read":  ("state", 0),
    # (memory_or_handle, keys, values, scores)  batched: all writes at axis 0
    "memory_write": ("state", 0, 0, 0),
    # (memory_or_handle, n_or_indices)          neither batched
    "memory_evict": ("state", None),
}


def vmap_axis_for(op_name: str) -> tuple | None:
    """Return the per-arg vmap-axis tuple for ``op_name``, or ``None`` if
    no override is registered (in which case the caller falls back to the
    uniform-default vmap semantics)."""
    return _VMAP_AXIS_MAP.get(op_name)


def register_vmap_axis(op_name: str, axes: tuple) -> None:
    """Register the per-arg axis tuple for ``op_name``.

    Each axis slot is one of:
      - ``int`` — that arg is batched along the given axis
      - ``None`` — that arg is unbatched (broadcast through)
      - the string ``"state"`` — that arg is shared state; never replicate
        or split during vmap

    Overwrites any existing entry for ``op_name``.
    """
    _VMAP_AXIS_MAP[op_name] = tuple(axes)


__all__ = [
    "MemoryReadResult",
    "MemoryTable",
    "memory_evict",
    "memory_read",
    "memory_write",
    # Sprint D — vmap axis registry
    "vmap_axis_for",
    "register_vmap_axis",
]
