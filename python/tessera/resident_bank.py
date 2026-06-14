"""Device-resident memory bank — the read side of ``resident_state_handle``.

The long-memory benchmarks (RULER / LongMemEval / MemoryArena / AMA-Bench) punish
re-deriving resident state: each decode step scores a query against the whole
bank, and a naive path re-uploads the bank every step. ``ResidentBank`` keeps the
bank's keys on-device (a single upload) and scores each query against the
resident keys via the encode-session ``bmm_enc`` lane — so per-read traffic is
``O(query)``, not ``O(bank)``.

This lands read-residency: the bank stays on the GPU across decode steps. The
remaining follow-on is *incremental on-device append* (writing a new entry into
the resident buffer without a full re-upload), which needs a device offset-write
symbol — tracked as ``kv_cache_append_read`` in
``benchmarks.long_memory_core.MEMORY_PRIMITIVE_GAPS``.

When the encode-session runtime is unavailable (non-Darwin / no Metal), the bank
falls back to a numpy reference with identical semantics, so callers and tests
are portable.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ResidentBank:
    """A read-resident key/value bank.

    ``keys`` is ``(n, key_dim)`` and ``values`` is ``(n, value_dim...)``. The keys
    are uploaded to the device once; ``read`` scores a query against them without
    re-uploading. Values are gathered on the host after the (tiny, data-dependent)
    top-k selection. Free the device memory with :meth:`free` or use as a context
    manager.
    """

    def __init__(self, keys: np.ndarray, values: np.ndarray):
        keys = np.ascontiguousarray(np.asarray(keys, dtype=np.float32))
        if keys.ndim != 2:
            raise ValueError("keys must have shape (n, key_dim)")
        self.values = np.asarray(values)
        if self.values.shape[0] != keys.shape[0]:
            raise ValueError("keys and values must share the leading (n) dim")
        self.n, self.key_dim = int(keys.shape[0]), int(keys.shape[1])

        # telemetry — what residency buys: the bank uploads once; reads upload
        # only the query, never the bank.
        self.uploads = 1
        self.bank_bytes = int(keys.nbytes)
        self.read_query_bytes = 0
        self.reads = 0

        self._native = False
        self._agb: Any = None
        self._kT: Any = None
        self._keys_ref: np.ndarray | None = None
        try:                                              # pragma: no cover - device-gated
            import tessera.apple_gpu_batched as agb
            if agb.session_available():
                # resident transpose (d, n) for query·keysᵀ scoring
                self._agb = agb
                self._kT = agb.device_tensor(np.ascontiguousarray(keys.T))
                self._native = True
        except Exception:                                 # noqa: BLE001 - reference fallback
            self._native = False
        if not self._native:
            self._keys_ref = keys

    # ── scoring ──────────────────────────────────────────────────────────────
    def _score(self, query: np.ndarray) -> np.ndarray:
        q = np.ascontiguousarray(query.reshape(1, self.key_dim).astype(np.float32))
        self.read_query_bytes += int(q.nbytes)            # query only — not the bank
        self.reads += 1
        if self._native:                                  # pragma: no cover - device-gated
            agb = self._agb
            q_dev = agb.device_tensor(q)
            try:
                with agb.batched_session() as s:
                    sc = agb.bmm_enc(s, q_dev, self._kT, batch=1, M=1,
                                     N=self.n, K=self.key_dim)
                scores = sc.download(np.float32, (1, self.n))[0]
                sc.free()
            finally:
                q_dev.free()
            return scores
        assert self._keys_ref is not None        # set in __init__ when not native
        return (q @ self._keys_ref.T)[0]

    def read(self, query: np.ndarray, *, top_k: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score ``query`` against the resident bank and return
        ``(values, indices, scores)`` for the top-k entries (descending)."""
        if self.n == 0:
            raise ValueError("cannot read from an empty ResidentBank")
        k = max(1, min(int(top_k), self.n))
        scores = self._score(np.asarray(query))
        order = np.argsort(scores)[::-1][:k]
        return self.values[order], order.astype(np.int64), scores[order]

    # ── residency-vs-recompute traffic accounting ────────────────────────────
    def recompute_equivalent_bytes(self) -> int:
        """Bytes a *recompute* path would have uploaded — it re-sends the whole
        bank every read. Compare against ``resident_bytes`` for the win."""
        return self.reads * self.bank_bytes

    def resident_bytes(self) -> int:
        """Bytes the resident path actually uploaded: the bank once + per-read
        queries."""
        return self.bank_bytes + self.read_query_bytes

    def telemetry(self) -> dict[str, Any]:
        recompute = self.recompute_equivalent_bytes()
        resident = self.resident_bytes()
        return {
            "n": self.n,
            "key_dim": self.key_dim,
            "reads": self.reads,
            "execution": "metal_runtime" if self._native else "reference",
            "bank_uploads": self.uploads,
            "resident_upload_bytes": resident,
            "recompute_upload_bytes": recompute,
            "upload_reduction_x": round(recompute / max(resident, 1), 2),
        }

    def free(self) -> None:
        if self._native and self._kT is not None:         # pragma: no cover - device-gated
            self._kT.free()
            self._kT = None

    def __enter__(self) -> "ResidentBank":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.free()


__all__ = ["ResidentBank"]
