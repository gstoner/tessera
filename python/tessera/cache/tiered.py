"""TieredKVCache — CPU cold-pool ↔ GPU-resident KV tiering (the LSA tier).

This is the substrate that closes the LSA "FlashMemory" gap (Gap 1, see
``docs/audit/domain/archive/lsa_scope.md``): without it, ``lookahead_sparse_
attention`` is honestly just *lookahead-periodic sparse attention*; with it, the
selected historical KV chunks are physically *staged* from a host cold pool into
a bounded **GPU-resident** working set, exactly the periodic-lookahead memory
hierarchy the policy is for.

Storage model
-------------
* **Cold pool** — every KV page lives in host (numpy) memory:
  ``K_cold / V_cold`` shaped ``(num_pages, page_size, num_heads, head_dim)``.
  A *page* is the staging granularity and equals the LSA *block* (so block
  selection maps 1:1 onto pages).
* **Resident set** — a bounded ``resident_capacity`` of pages held in
  device-resident :class:`~tessera.runtime.DeviceTensor` pools (GPU-resident on
  Apple unified memory; a portable numpy pool otherwise). A page is *resident*
  iff it currently occupies a resident slot.

Host↔device staging ABI
-----------------------
* :meth:`stage` copies cold pages → resident slots (host→device), LRU-evicting
  when over capacity, with byte accounting.
* :meth:`evict` returns resident pages to cold-only.
* :meth:`gather` reads K/V for a token set, *requiring* their pages resident —
  enforcing the tiering discipline (a real backend's DMA + residency manager).

The data movement is real numpy/DeviceTensor copies; on unified memory the
"upload" is a write through the resident buffer's view (the GPU sees it with no
discrete copy), which is the same zero-copy staging ``ResidentLatentKVCache``
uses. The page table, residency bound, LRU eviction, and byte accounting are the
honest, testable core of the ABI a discrete-memory backend would implement with
``cudaMemcpyAsync`` / a Metal blit.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import runtime as R


@dataclass
class StageStats:
    """Accounting for one :meth:`TieredKVCache.stage` call."""

    requested: int = 0          # pages asked for
    already_resident: int = 0   # of those, already in the resident set
    staged: int = 0             # cold → resident copies performed
    evicted: int = 0            # resident → cold evictions to make room
    bytes_staged: int = 0       # K+V bytes copied host → device


@dataclass
class TieredStats:
    """Lifetime accounting for the cache."""

    stage_calls: int = 0
    pages_staged: int = 0
    pages_evicted: int = 0
    bytes_staged: int = 0
    gather_tokens: int = 0


class TieredKVCacheError(RuntimeError):
    """Raised when a tiering invariant is violated (e.g. gather of a cold page)."""


@dataclass
class TieredKVCache:
    """Host cold-pool + bounded GPU-resident working set for KV.

    Parameters
    ----------
    num_heads, head_dim
        KV geometry.
    max_seq
        Capacity in tokens (rounded up to a whole number of pages).
    page_size
        Tokens per page = LSA block size. Must divide the sequences LSA selects
        over for the 1:1 block↔page mapping.
    resident_capacity
        Maximum number of *pages* held resident at once. Defaults to all pages
        (no eviction). Set below ``num_pages`` to force real cold/hot tiering.
    dtype, eviction
        Storage dtype (numpy reference path) and resident eviction policy
        (``"lru"`` or ``"fifo"``).
    """

    num_heads: int
    head_dim: int
    max_seq: int
    page_size: int = 128
    resident_capacity: int | None = None
    dtype: str = "fp32"
    eviction: str = "lru"

    current_seq: int = field(default=0, init=False)
    stats: TieredStats = field(default_factory=TieredStats, init=False)

    def __post_init__(self) -> None:
        if self.num_heads <= 0 or self.head_dim <= 0 or self.max_seq <= 0:
            raise ValueError("num_heads / head_dim / max_seq must be positive")
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        if self.eviction not in ("lru", "fifo"):
            raise ValueError("eviction must be 'lru' or 'fifo'")
        self._num_pages = (self.max_seq + self.page_size - 1) // self.page_size
        cap = self._num_pages if self.resident_capacity is None else int(self.resident_capacity)
        if cap <= 0:
            raise ValueError("resident_capacity must be positive")
        self._cap = min(cap, self._num_pages)
        H, D, P = self.num_heads, self.head_dim, self.page_size

        # Cold pool — host numpy, holds every page.
        self._k_cold = np.zeros((self._num_pages, P, H, D), dtype=np.float32)
        self._v_cold = np.zeros((self._num_pages, P, H, D), dtype=np.float32)

        # Resident pools — device-resident if available, else numpy fallback.
        self._k_pool_dt = R.DeviceTensor.empty((self._cap, P, H, D), np.float32)
        self._v_pool_dt = R.DeviceTensor.empty((self._cap, P, H, D), np.float32)
        if self._k_pool_dt is not None and self._v_pool_dt is not None:
            self._resident_device = True
            self._k_pool = self._k_pool_dt.numpy()  # unified-memory view
            self._v_pool = self._v_pool_dt.numpy()
        else:
            self._resident_device = False
            self._k_pool = np.zeros((self._cap, P, H, D), dtype=np.float32)
            self._v_pool = np.zeros((self._cap, P, H, D), dtype=np.float32)

        # Residency bookkeeping.
        self._page_to_slot: "OrderedDict[int, int]" = OrderedDict()  # insertion = LRU order
        self._free_slots: list[int] = list(range(self._cap))

    # ------------------------------------------------------------------
    # Geometry / introspection
    # ------------------------------------------------------------------

    @property
    def num_pages(self) -> int:
        return self._num_pages

    @property
    def resident_capacity_pages(self) -> int:
        return self._cap

    @property
    def device_resident(self) -> bool:
        """True when the resident pool is a real device buffer (not the numpy fallback)."""
        return self._resident_device

    def page_of(self, token: int) -> int:
        return int(token) // self.page_size

    def resident_pages(self) -> set[int]:
        return set(self._page_to_slot)

    def is_resident(self, page_id: int) -> bool:
        return page_id in self._page_to_slot

    def filled_pages(self) -> int:
        return (self.current_seq + self.page_size - 1) // self.page_size

    # ------------------------------------------------------------------
    # Write path (host cold pool)
    # ------------------------------------------------------------------

    def write(self, k: Any, v: Any) -> "TieredKVCache":
        """Append ``(seq, num_heads, head_dim)`` K/V tokens to the cold pool."""
        k = np.asarray(k._data if hasattr(k, "_data") else k, dtype=np.float32)
        v = np.asarray(v._data if hasattr(v, "_data") else v, dtype=np.float32)
        if k.shape != v.shape:
            raise ValueError("k and v must share shape")
        if k.ndim != 3 or k.shape[1:] != (self.num_heads, self.head_dim):
            raise ValueError(
                f"k/v must be (seq, {self.num_heads}, {self.head_dim}); got {k.shape}")
        n = k.shape[0]
        if self.current_seq + n > self.max_seq:
            raise ValueError(
                f"write would exceed max_seq={self.max_seq}: "
                f"current={self.current_seq}, appending={n}")
        for i in range(n):
            tok = self.current_seq + i
            pg, off = divmod(tok, self.page_size)
            self._k_cold[pg, off] = k[i]
            self._v_cold[pg, off] = v[i]
        self.current_seq += n
        return self

    # ------------------------------------------------------------------
    # Staging ABI
    # ------------------------------------------------------------------

    def _page_bytes(self) -> int:
        return int(self._k_cold[0].nbytes + self._v_cold[0].nbytes)

    def _evict_one(self) -> int:
        """Evict the LRU/FIFO resident page, returning the freed slot."""
        victim_page, slot = next(iter(self._page_to_slot.items()))
        del self._page_to_slot[victim_page]
        self.stats.pages_evicted += 1
        return slot

    def stage(self, page_ids) -> StageStats:
        """Ensure ``page_ids`` are resident (cold → device), LRU-evicting as needed."""
        s = StageStats()
        want = []
        seen: set[int] = set()
        for pg in page_ids:
            pg = int(pg)
            if pg < 0 or pg >= self._num_pages:
                raise IndexError(f"page {pg} out of range [0, {self._num_pages})")
            if pg in seen:
                continue
            seen.add(pg)
            want.append(pg)
        s.requested = len(want)
        if len(want) > self._cap:
            raise TieredKVCacheError(
                f"cannot stage {len(want)} pages into a {self._cap}-page resident set")
        self.stats.stage_calls += 1
        for pg in want:
            if pg in self._page_to_slot:
                self._page_to_slot.move_to_end(pg)  # LRU touch
                s.already_resident += 1
                continue
            if self._free_slots:
                slot = self._free_slots.pop()
            else:
                slot = self._evict_one()
                s.evicted += 1
            # Cold (host) → resident (device) copy — the staging upload.
            self._k_pool[slot] = self._k_cold[pg]
            self._v_pool[slot] = self._v_cold[pg]
            self._page_to_slot[pg] = slot
            s.staged += 1
            s.bytes_staged += self._page_bytes()
        self.stats.pages_staged += s.staged
        self.stats.bytes_staged += s.bytes_staged
        return s

    def evict(self, page_ids) -> int:
        """Return ``page_ids`` to cold-only. Returns the count actually evicted."""
        n = 0
        for pg in page_ids:
            pg = int(pg)
            slot = self._page_to_slot.pop(pg, None)
            if slot is not None:
                self._free_slots.append(slot)
                n += 1
        self.stats.pages_evicted += n
        return n

    # ------------------------------------------------------------------
    # Read path (resident only)
    # ------------------------------------------------------------------

    def gather(self, token_indices, *, require_resident: bool = True):
        """Read ``(K, V)`` for ``token_indices`` from the **resident** pools.

        With ``require_resident=True`` (default) a token whose page is not
        resident raises :class:`TieredKVCacheError` — the discipline a real
        device gather enforces. With ``False`` the page is auto-staged first.
        """
        idx = np.asarray(token_indices, dtype=np.int64).reshape(-1)
        K = np.empty((idx.shape[0], self.num_heads, self.head_dim), np.float32)
        V = np.empty((idx.shape[0], self.num_heads, self.head_dim), np.float32)
        for i, tok_np in enumerate(idx):
            tok = int(tok_np)
            if not (0 <= tok < self.current_seq):
                raise IndexError(f"token {tok} out of range [0, {self.current_seq})")
            pg, off = divmod(tok, self.page_size)
            if pg not in self._page_to_slot:
                if require_resident:
                    raise TieredKVCacheError(
                        f"token {tok} is on cold page {pg}; stage it before gather")
                self.stage([pg])
            slot = self._page_to_slot[pg]
            if self.eviction == "lru":
                self._page_to_slot.move_to_end(pg)
            K[i] = self._k_pool[slot, off]
            V[i] = self._v_pool[slot, off]
        self.stats.gather_tokens += int(idx.shape[0])
        return K, V

    # ------------------------------------------------------------------
    # Indexer keys (compressed per-page summaries for memory_index_select)
    # ------------------------------------------------------------------

    def block_keys(self) -> np.ndarray:
        """Mean-pool each *filled* page into a summary key ``(num_filled, H, D)``.

        These feed ``memory_index_select`` as the per-block indexer keys. Only
        fully or partially filled pages up to ``current_seq`` are returned.
        """
        nfilled = self.filled_pages()
        if nfilled == 0:
            return np.zeros((0, self.num_heads, self.head_dim), np.float32)
        keys = np.zeros((nfilled, self.num_heads, self.head_dim), np.float32)
        for pg in range(nfilled):
            lo = pg * self.page_size
            hi = min(lo + self.page_size, self.current_seq)
            keys[pg] = self._k_cold[pg, : hi - lo].mean(axis=0)
        return keys

    def free(self) -> None:
        if self._k_pool_dt is not None:
            self._k_pool_dt.free()
        if self._v_pool_dt is not None:
            self._v_pool_dt.free()
        self._page_to_slot.clear()

    def __repr__(self) -> str:
        return (
            f"TieredKVCache(heads={self.num_heads}, head_dim={self.head_dim}, "
            f"max_seq={self.max_seq}, page_size={self.page_size}, "
            f"resident={self._cap}/{self._num_pages} pages, "
            f"device_resident={self._resident_device}, current_seq={self.current_seq})"
        )


def lookahead_attention_tiered(
    query,
    cache: "TieredKVCache",
    *,
    window_size: int,
    threshold: float = 0.5,
    causal: bool = True,
    tau: int = 64,
    indexer_keys=None,
    scale: float | None = None,
):
    """Lookahead sparse attention over a :class:`TieredKVCache` (Gap 1).

    This is the tiered realization of ``lookahead_sparse_attention``: the
    sigmoid-threshold selector (:func:`tessera.lsa.memory_index_select`) chooses
    which historical *pages* matter, those pages are **staged cold→resident**,
    and the per-query footprint (causal local window ∪ selected pages) is
    attended reading **only from the resident pools**. The local-window pages of
    each query are always part of its footprint, so a query's working set is
    bounded and the result is independent of ``resident_capacity`` (smaller
    capacity ⇒ more cold↔resident traffic, identical output).

    The page size equals the LSA block size (1:1 block↔page), so this is
    numerically equal to ``lsa.lookahead_sparse_attention`` on the reconstructed
    full K/V — the tiering is a residency optimization, not a math change.

    Parameters
    ----------
    query
        ``(S_q, num_heads, head_dim)`` — single sequence, per-token. Typically
        ``S_q == cache.current_seq``.
    Returns
    -------
    (out, stats)
        ``out`` shaped ``(S_q, num_heads, head_dim)`` and a fresh
        :class:`StageStats`-style snapshot of the staging traffic this call did.
    """
    from .. import lsa as _lsa

    q = np.asarray(query._data if hasattr(query, "_data") else query, dtype=np.float64)
    if q.ndim != 3 or q.shape[1:] != (cache.num_heads, cache.head_dim):
        raise ValueError(
            f"query must be (S_q, {cache.num_heads}, {cache.head_dim}); got {q.shape}")
    S_q, H, D = q.shape
    block_size = cache.page_size
    if tau <= 0 or window_size <= 0:
        raise ValueError("tau and window_size must be positive")
    sc = float(scale) if scale is not None else 1.0 / np.sqrt(D)

    keys = indexer_keys if indexer_keys is not None else cache.block_keys()
    keys = np.asarray(keys, dtype=np.float64)  # (num_blocks, H, D)
    num_blocks = keys.shape[0]

    # Per-head selection: reshape to the memory_index_select (B=1, H, *, Dk) layout.
    keys_bhnd = np.transpose(keys, (1, 0, 2))[None]     # (1, H, num_blocks, D)
    q_bhsd = np.transpose(q, (1, 0, 2))[None]           # (1, H, S_q, D)
    mask = _lsa.memory_index_select(
        keys_bhnd, q_bhsd, block_size=block_size, threshold=threshold,
        causal=causal, scale=scale, fallback_local=True).mask[0]  # (H, S_q, num_blocks)

    before = cache.stats.bytes_staged
    pages_before = cache.stats.pages_staged
    out = np.zeros((S_q, H, D), dtype=np.float64)
    for sq in range(S_q):
        if causal:
            local_lo = max(0, sq - window_size + 1)
            local_hi = sq
        else:
            half = window_size // 2
            local_lo = max(0, sq - half)
            local_hi = min(cache.current_seq - 1, sq + half)
        for h in range(H):
            toks = set(range(local_lo, local_hi + 1))
            for blk in np.flatnonzero(mask[h, sq]):
                st = int(blk) * block_size
                toks.update(range(st, min(st + block_size, cache.current_seq)))
            if causal:
                toks = {t for t in toks if t <= sq}
            footprint = np.array(sorted(toks), dtype=np.int64)
            # Stream the footprint through the resident window: gather stages each
            # cold page just-in-time (cold→resident, LRU-evicting), so the peak
            # residency is bounded by capacity and the result is independent of
            # resident_capacity (≥1). Selection drives which pages get staged.
            K_g, V_g = cache.gather(footprint, require_resident=False)
            k_h = K_g[:, h, :].astype(np.float64)
            v_h = V_g[:, h, :].astype(np.float64)
            s = (q[sq, h] @ k_h.T) * sc
            s -= s.max()
            w = np.exp(s)
            w /= w.sum()
            out[sq, h] = w @ v_h

    call_stats = StageStats(
        staged=cache.stats.pages_staged - pages_before,
        bytes_staged=cache.stats.bytes_staged - before,
    )
    return out, call_stats


__all__ = [
    "TieredKVCache",
    "TieredKVCacheError",
    "StageStats",
    "TieredStats",
    "lookahead_attention_tiered",
]
