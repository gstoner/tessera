"""Device-resident memory bank — the resident half of ``resident_state_handle``
and ``kv_cache_append_read``.

The long-memory benchmarks (RULER / LongMemEval / MemoryArena / AMA-Bench) punish
re-deriving resident state: each decode step appends a new entry and scores a
query against the whole bank, and a naive path re-uploads the bank every step.
``ResidentBank`` keeps the bank's keys on-device and never re-uploads them:

* **read-residency** — the keys stay resident; a query scores against them via
  the encode-session ``bmm_enc`` lane.  Per-read traffic is ``O(query)``, not
  ``O(bank)``.
* **append-residency** — a new entry is written into the resident buffer at its
  byte offset via the ``ts_dev_upload_at`` device symbol (the ``kv_cache_append_
  read`` primitive).  Per-append traffic is ``O(entry)``, not ``O(bank)`` — no
  full re-upload.  Construct with ``capacity=`` to enable; ``append`` then writes
  at the tail and ``read`` scores the first ``current_seq`` rows.

Keys are stored ``(capacity, key_dim)`` row-major so an append is a single
contiguous offset-write; scoring uses ``bmm_enc(keys, queryᵀ, M=current_seq)``.
When the encode-session runtime is unavailable (non-Darwin / no Metal), the bank
falls back to a numpy reference with identical semantics, so callers and tests
are portable.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ResidentBank:
    """A read- and (optionally) append-resident key/value bank.

    ``keys`` is ``(n, key_dim)`` and ``values`` is ``(n, value_dim...)``.  Without
    ``capacity`` the bank is fixed (read-only residency).  With ``capacity >= n``
    the bank is appendable up to ``capacity`` rows; ``append`` offset-writes the
    new entry without re-uploading the bank.
    """

    def __init__(self, keys: np.ndarray, values: np.ndarray, *,
                 capacity: int | None = None):
        keys = np.ascontiguousarray(np.asarray(keys, dtype=np.float32))
        if keys.ndim != 2:
            raise ValueError("keys must have shape (n, key_dim)")
        values = np.asarray(values)
        if values.shape[0] != keys.shape[0]:
            raise ValueError("keys and values must share the leading (n) dim")
        n, self.key_dim = int(keys.shape[0]), int(keys.shape[1])
        self.appendable = capacity is not None
        self.capacity = max(int(capacity) if capacity is not None else n, n)
        self.current_seq = n

        # host value store, sized to capacity (gather is host-side after top-k)
        vtail = values.shape[1:] if values.ndim > 1 else (1,)
        self._values = np.empty((self.capacity,) + vtail, dtype=values.dtype)
        if n:
            self._values[:n] = values.reshape((n,) + vtail)

        # telemetry — what residency buys: the bank uploads once, appends and
        # reads upload only O(entry)/O(query), never the bank.
        self.uploads = 1
        self.bank_bytes = int(keys.nbytes)         # initial upload
        self.append_bytes = 0
        self.read_query_bytes = 0
        self.recompute_upload_bytes = 0            # a recompute re-uploads the bank per read
        self.reads = 0

        self._native = False
        self._agb: Any = None
        self._keys_dev: Any = None
        self._keys_ref: np.ndarray | None = None
        try:                                       # pragma: no cover - device-gated
            import tessera.apple_gpu_batched as agb
            if agb.session_available() and self.capacity > 0:
                self._agb = agb
                self._keys_dev = agb.device_empty(self.capacity * self.key_dim * 4)
                if n:
                    self._keys_dev.upload_at(keys, 0)
                self._native = True
        except Exception:                          # noqa: BLE001 - reference fallback
            self._native = False
        if not self._native:
            ref = np.zeros((self.capacity, self.key_dim), np.float32)
            if n:
                ref[:n] = keys
            self._keys_ref = ref

    # ── append-residency ─────────────────────────────────────────────────────
    def append(self, key: np.ndarray, value: np.ndarray) -> int:
        """Append one ``(key, value)`` to the resident bank without re-uploading
        it.  Returns the new entry's index."""
        if not self.appendable:
            raise ValueError("ResidentBank is fixed; construct with capacity= to append")
        if self.current_seq >= self.capacity:
            raise ValueError(f"ResidentBank is full (capacity {self.capacity})")
        k = np.ascontiguousarray(np.asarray(key, np.float32).reshape(self.key_dim))
        idx = self.current_seq
        offset = idx * self.key_dim * 4
        if self._native:                           # pragma: no cover - device-gated
            self._keys_dev.upload_at(k, offset)
        else:
            assert self._keys_ref is not None
            self._keys_ref[idx] = k
        self._values[idx] = np.asarray(value).reshape(self._values.shape[1:])
        self.append_bytes += int(k.nbytes)         # O(key_dim), not O(bank)
        self.current_seq += 1
        return idx

    # ── scoring ──────────────────────────────────────────────────────────────
    def _score(self, query: np.ndarray) -> np.ndarray:
        cur = self.current_seq
        self.read_query_bytes += self.key_dim * 4
        self.recompute_upload_bytes += cur * self.key_dim * 4   # bank a recompute re-sends
        self.reads += 1
        if self._native:                           # pragma: no cover - device-gated
            agb = self._agb
            qcol = np.ascontiguousarray(query.reshape(self.key_dim, 1).astype(np.float32))
            q_dev = agb.device_tensor(qcol)
            try:
                with agb.batched_session() as s:
                    sc = agb.bmm_enc(s, self._keys_dev, q_dev, batch=1,
                                     M=cur, N=1, K=self.key_dim)
                scores = sc.download(np.float32, (cur, 1))[:, 0]
                sc.free()
            finally:
                q_dev.free()
            return scores
        assert self._keys_ref is not None
        return self._keys_ref[:cur] @ query.reshape(self.key_dim).astype(np.float32)

    def read(self, query: np.ndarray, *, top_k: int = 1
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score ``query`` against the resident bank and return
        ``(values, indices, scores)`` for the top-k entries (descending)."""
        if self.current_seq == 0:
            raise ValueError("cannot read from an empty ResidentBank")
        k = max(1, min(int(top_k), self.current_seq))
        scores = self._score(np.asarray(query))
        order = np.argsort(scores)[::-1][:k]
        return self._values[order], order.astype(np.int64), scores[order]

    # ── telemetry ────────────────────────────────────────────────────────────
    @property
    def values(self) -> np.ndarray:
        return self._values[:self.current_seq]

    def resident_bytes(self) -> int:
        """Bytes the resident path actually uploaded: the bank once + per-append
        entries + per-read queries."""
        return self.bank_bytes + self.append_bytes + self.read_query_bytes

    def recompute_equivalent_bytes(self) -> int:
        """Bytes a recompute path would have uploaded (re-sends the bank per read)."""
        return self.recompute_upload_bytes

    def telemetry(self) -> dict[str, Any]:
        resident = self.resident_bytes()
        recompute = self.recompute_upload_bytes
        return {
            "n": self.current_seq,
            "key_dim": self.key_dim,
            "appendable": self.appendable,
            "reads": self.reads,
            "execution": "metal_runtime" if self._native else "reference",
            "bank_uploads": self.uploads,
            "append_bytes": self.append_bytes,
            "resident_upload_bytes": resident,
            "recompute_upload_bytes": recompute,
            "upload_reduction_x": round(recompute / max(resident, 1), 2),
        }

    def free(self) -> None:
        if self._native and self._keys_dev is not None:   # pragma: no cover - device-gated
            self._keys_dev.free()
            self._keys_dev = None

    def __enter__(self) -> "ResidentBank":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.free()


__all__ = ["ResidentBank"]
