"""Sprint D — Persistent memory-state handle for Titans/Atlas-style banks.

Today the `tessera.memory.memory_read/write/evict` reference accepts a
plain `MemoryTable` (frozen dataclass).  This file adds a stateful
``MemoryStateHandle`` analogous to ``KVCacheHandle``:

  - opaque to ops (so the same Python code works once backends lower
    learned memory as first-class state);
  - functional read; in-place-with-cow write/evict (matching
    `tessera.state.STATE_COLLECTION_SPECS["memory_state"].mutable=True`);
  - optional ``MemoryShardSpec`` for SPMD partitioning of the bank;
  - ``checkpoint()`` / ``restore()`` round-trip via the S12 state-dict
    contract.

The handle does NOT replace ``MemoryTable`` — the reference primitives
still accept tables.  The handle is the *production-ready persistent
state ABI* the audit doc names; backends register lowering for the handle
type, not for the table.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..sharding import MemoryShardSpec


def _to_array(value: Any) -> np.ndarray:
    if hasattr(value, "_data"):
        value = value._data
    return np.asarray(value)


@dataclass
class MemoryStateHandle:
    """Opaque persistent handle for a Titans/Atlas-style memory bank.

    Parameters
    ----------
    capacity : int
        Maximum number of entries in the bank.  ``write()`` past capacity
        triggers eviction via the configured policy.
    key_dim : int
        Length of each key vector.
    value_dim : tuple[int, ...]
        Trailing dims of each value vector (single-vector → ``(value_dim,)``;
        multi-vector → e.g., ``(num_heads, head_dim)``).
    dtype : str
        Canonical storage dtype for keys + values.  Default ``"fp32"``.
    shard_spec : MemoryShardSpec | None
        Optional partitioning spec — keys are hashed/bucketed to shards
        when this is supplied.
    eviction : str
        Eviction policy: ``"score"`` (lowest learned score), ``"lru"``,
        ``"fifo"``, ``"oldest"``.  Overrides ``shard_spec.eviction`` when
        both are present.

    Attributes (read-only)
    ----------------------
    keys : np.ndarray            shape (size, key_dim)
    values : np.ndarray          shape (size, *value_dim)
    metadata : dict[str, np.ndarray]  side-table aligned on entries axis
    size : int                   current number of live entries
    capacity : int               max entries
    """

    capacity: int
    key_dim: int
    value_dim: tuple[int, ...]
    dtype: str = "fp32"
    shard_spec: MemoryShardSpec | None = None
    eviction: str = "score"

    _keys: np.ndarray = field(init=False, repr=False)
    _values: np.ndarray = field(init=False, repr=False)
    _metadata: dict[str, np.ndarray] = field(init=False, repr=False)
    _size: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        from ..dtype import canonicalize_dtype

        canon = canonicalize_dtype(self.dtype, allow_planned_gated=True)
        object.__setattr__(self, "dtype", canon)
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")
        if self.key_dim <= 0:
            raise ValueError(f"key_dim must be positive, got {self.key_dim}")
        if isinstance(self.value_dim, int):  # type: ignore[unreachable]
            object.__setattr__(self, "value_dim", (self.value_dim,))  # type: ignore[unreachable]
        if self.eviction not in {"lru", "fifo", "score", "oldest"}:
            raise ValueError(
                f"eviction must be 'lru'|'fifo'|'score'|'oldest', "
                f"got {self.eviction!r}"
            )
        # Numpy backing — never bf16 directly (numpy lacks), fall back to f32.
        np_dtype = {
            "bf16": np.float32, "fp16": np.float16, "fp32": np.float32,
            "fp64": np.float64,
            "fp8_e4m3": np.float32, "fp8_e5m2": np.float32,
            "fp6_e2m3": np.float32, "fp6_e3m2": np.float32,
            "fp4_e2m1": np.float32, "nvfp4": np.float32,
            "int8": np.int8, "int16": np.int16,
            "int32": np.int32, "int64": np.int64,
            "uint8": np.uint8, "bool": np.bool_,
        }.get(canon, np.float32)
        self._keys = np.zeros((self.capacity, self.key_dim), dtype=np_dtype)
        self._values = np.zeros(
            (self.capacity,) + tuple(self.value_dim), dtype=np_dtype
        )
        # Side-table metadata: per-entry write step, score, age (for LRU).
        self._metadata = {
            "step":  np.full(self.capacity, -1, dtype=np.int64),
            "score": np.zeros(self.capacity, dtype=np.float32),
            "age":   np.zeros(self.capacity, dtype=np.int64),
        }

    # ── Read-only views ─────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size >= self.capacity

    @property
    def keys(self) -> np.ndarray:
        """Live keys (no padding rows)."""
        return self._keys[: self._size]

    @property
    def values(self) -> np.ndarray:
        return self._values[: self._size]

    @property
    def metadata(self) -> dict[str, np.ndarray]:
        return {k: v[: self._size] for k, v in self._metadata.items()}

    # ── Functional read (matches tessera.memory.memory_read) ────────────

    def read(
        self,
        query: np.ndarray,
        *,
        top_k: int = 1,
        normalize: bool = True,
        temperature: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Top-k weighted read.  Returns ``(values, indices, weights, scores)``.

        Matches `tessera.memory.memory_read(top_k=..., normalize=...)`
        signature.  ``temperature`` divides the query before scoring (so
        lower temperature → sharper attention).  Backends lower this to a
        fused attention-like kernel.
        """
        from ..memory import memory_read

        q = np.asarray(query) / max(float(temperature), 1e-12)
        result = memory_read(
            (self.keys, self.values),
            q,
            top_k=top_k,
            normalize=normalize,
        )
        return result.values, result.indices, result.weights, result.scores

    # ── In-place-with-COW mutation ──────────────────────────────────────

    def write(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        *,
        scores: np.ndarray | None = None,
        step: int | None = None,
    ) -> "MemoryStateHandle":
        """Append new entries to the bank.  Triggers eviction when full.

        Returns the same handle (mutated in place — matches
        ``STATE_COLLECTION_SPECS['memory_state'].mutable=True``).  For
        functional-only callers, take a snapshot via ``.clone()`` first.
        """
        keys_arr = _to_array(keys)
        vals_arr = _to_array(values)
        if keys_arr.ndim == 1:
            keys_arr = keys_arr[np.newaxis, :]
        if vals_arr.ndim == len(self.value_dim):
            vals_arr = vals_arr[np.newaxis, ...]
        n = keys_arr.shape[0]
        if n == 0:
            return self

        # Eviction: free up enough slots if needed.
        slots_needed = max(0, (self._size + n) - self.capacity)
        if slots_needed > 0:
            self._evict_n(slots_needed)

        start = self._size
        end = start + n
        self._keys[start:end] = keys_arr.astype(self._keys.dtype, copy=False)
        self._values[start:end] = vals_arr.astype(self._values.dtype, copy=False)
        if scores is not None:
            self._metadata["score"][start:end] = np.asarray(
                scores, dtype=self._metadata["score"].dtype
            )
        if step is not None:
            self._metadata["step"][start:end] = int(step)
        self._metadata["age"][:end] += 1  # increment age for everyone alive
        self._metadata["age"][start:end] = 0  # fresh entries start at age 0
        self._size = end
        return self

    def evict(self, n: int = 1) -> "MemoryStateHandle":
        """Evict ``n`` entries using the handle's configured policy."""
        self._evict_n(int(n))
        return self

    def _evict_n(self, n: int) -> None:
        if n <= 0 or self._size == 0:
            return
        n = min(n, self._size)
        if self.eviction == "fifo":
            keep_idx = np.arange(n, self._size)
        elif self.eviction == "lru" or self.eviction == "oldest":
            # Highest age = least recently used.
            ages = self._metadata["age"][: self._size]
            evict_idx = np.argsort(-ages)[:n]
            mask = np.ones(self._size, dtype=bool)
            mask[evict_idx] = False
            keep_idx = np.nonzero(mask)[0]
        elif self.eviction == "score":
            scores = self._metadata["score"][: self._size]
            evict_idx = np.argsort(scores)[:n]  # lowest scores evicted
            mask = np.ones(self._size, dtype=bool)
            mask[evict_idx] = False
            keep_idx = np.nonzero(mask)[0]
        else:
            raise ValueError(f"unknown eviction policy {self.eviction!r}")
        keep_n = keep_idx.size
        self._keys[:keep_n] = self._keys[keep_idx]
        self._values[:keep_n] = self._values[keep_idx]
        for k, v in self._metadata.items():
            v[:keep_n] = v[keep_idx]
        # Wipe stale rows so we don't carry phantom data.
        self._keys[keep_n:] = 0
        self._values[keep_n:] = 0
        for k, v in self._metadata.items():
            v[keep_n:] = 0
        self._metadata["step"][keep_n:] = -1
        self._size = keep_n

    def clone(self) -> "MemoryStateHandle":
        """Deep copy — useful for the functional `tape()` path."""
        return copy.deepcopy(self)

    # ── S12 checkpoint round-trip ───────────────────────────────────────

    def checkpoint(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for ``tessera.checkpoint.save_state``.

        Maps cleanly onto ``STATE_COLLECTION_SPECS["memory_state"]`` (a
        mutable, persistent state collection).
        """
        return {
            "capacity": int(self.capacity),
            "key_dim": int(self.key_dim),
            "value_dim": tuple(int(d) for d in self.value_dim),
            "dtype": self.dtype,
            "eviction": self.eviction,
            "shard_spec": _shard_spec_to_dict(self.shard_spec),
            "keys": np.array(self.keys, copy=True),
            "values": np.array(self.values, copy=True),
            "metadata": {k: np.array(v, copy=True) for k, v in self.metadata.items()},
            "size": int(self._size),
        }

    @classmethod
    def restore(cls, state: Mapping[str, Any]) -> "MemoryStateHandle":
        """Inverse of ``checkpoint()``."""
        handle = cls(
            capacity=int(state["capacity"]),
            key_dim=int(state["key_dim"]),
            value_dim=tuple(int(d) for d in state["value_dim"]),
            dtype=str(state.get("dtype", "fp32")),
            shard_spec=_dict_to_shard_spec(state.get("shard_spec")),
            eviction=str(state.get("eviction", "score")),
        )
        size = int(state.get("size", len(state["keys"])))
        keys = np.asarray(state["keys"])
        values = np.asarray(state["values"])
        handle._keys[:size] = keys[:size]
        handle._values[:size] = values[:size]
        for k, v in dict(state.get("metadata", {})).items():
            arr = np.asarray(v)
            handle._metadata[k][:size] = arr[:size]
        handle._size = size
        return handle

    # ── Sharded ownership lookup ────────────────────────────────────────

    def shard_for_key(self, key: np.ndarray, mesh) -> int:
        """Resolve the owning shard for a single key.

        Returns 0 when no ``shard_spec`` is configured (replicated bank).
        """
        if self.shard_spec is None:
            return 0
        return self.shard_spec.shard_owner(np.asarray(key), mesh)


def _shard_spec_to_dict(spec: MemoryShardSpec | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    return {
        "mesh_axis": spec.mesh_axis,
        "mode": spec.mode,
        "eviction": spec.eviction,
        "persistence": spec.persistence,
        "bucket_fn": spec.bucket_fn,
    }


def _dict_to_shard_spec(d: Mapping[str, Any] | None) -> MemoryShardSpec | None:
    if not d:
        return None
    return MemoryShardSpec(
        mesh_axis=str(d["mesh_axis"]),
        mode=str(d.get("mode", "key_hash")),
        eviction=str(d.get("eviction", "score")),
        persistence=str(d.get("persistence", "persistent")),
        bucket_fn=d.get("bucket_fn"),
    )


__all__ = ["MemoryStateHandle"]
