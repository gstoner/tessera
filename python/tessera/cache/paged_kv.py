"""PagedKVState — the unifying KV cache ABI (Workstream A).

The problem this closes: Tessera grew two KV substrates that never shared a
contract.

  * :class:`~tessera.cache.handle.KVCacheHandle` is contiguous — its own
    docstring says ``page_size`` is *recorded but not used to physically page*.
  * :class:`~tessera.cache.tiered.TieredKVCache` is genuinely paged — it has a
    ``stage``/``gather``/``evict`` page-table ABI with host↔resident tiering.

Nothing let an attention op consume *either* one uniformly, so paging stayed an
implementation detail rather than an ABI. ``PagedKVState`` is that ABI: a
``runtime_checkable`` protocol describing what an attention consumer needs from
*any* KV state —

    page table  ·  per-page tier  ·  quantization  ·  block sharing  ·  trim

The contiguous handle satisfies it degenerately (one tier, all-resident, one
block per page); the tiered/MLA substrates satisfy it directly. The
PagedAttentionLoweringPass (next slice) reads this protocol to insert
prefetch → gather → dequant stages — see ``docs/audit/roadmap/CONTRACT_PASS_PLAN.md``.

This module is the *contract*. It is deliberately non-invasive: rather than edit
the two large cache classes, :func:`as_paged_kv_state` wraps either one in a thin
adapter that exposes the protocol. Adapters add no storage — they read through to
the underlying cache.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np


class PageTier(enum.Enum):
    """Where a logical KV page physically lives.

    The attention consumer uses this to decide whether a page must be *staged*
    (host → resident) or *gathered + dequantized* before the kernel runs.
    """

    RESIDENT = "resident"   # device-resident; gather is a direct read
    HOST = "host"           # host/cold pool; must be staged before gather
    OFFLOAD = "offload"     # spilled further (disk/remote); stage is multi-hop


class KVKind(enum.Enum):
    """The storage family of a KV state — the ShadowKV heterogeneity axis.

    All kinds share this one protocol; the lowering pass branches on ``kind`` only
    to pick the dequant/expand stage (e.g. latent caches need an expand matmul).
    """

    FULL = "full"                   # full per-head K/V
    LATENT = "latent"               # MLA compressed latent (expand on read)
    LOW_RANK = "low_rank"           # low-rank factored K/V
    QUANTIZED_TAIL = "quantized_tail"  # hot fp window + quantized cold tail


@dataclass(frozen=True)
class KVGeometry:
    """Static shape facts an attention consumer needs to allocate its kernel."""

    num_heads: int
    head_dim: int
    max_seq: int
    page_size: int
    dtype: str = "fp32"


@dataclass(frozen=True)
class PageTableEntry:
    """One logical page's placement.

    ``page_id`` is the logical page index (token range
    ``[page_id*page_size, (page_id+1)*page_size)``). ``tier`` is where it lives.
    ``shared_with`` lists other sequence ids that alias this physical page (prefix
    sharing); empty for unshared pages.
    """

    page_id: int
    tier: PageTier
    shared_with: tuple[int, ...] = ()


@runtime_checkable
class PagedKVState(Protocol):
    """The KV ABI an attention op consumes, regardless of physical layout.

    A conforming object exposes its geometry, a page table with per-page tiers,
    its quantization (bits or ``None``), and a ``gather`` that returns dense
    ``(K, V)`` for an arbitrary token-index set — staging/dequantizing as needed.
    """

    kind: KVKind

    def kv_geometry(self) -> KVGeometry: ...

    def seq_len(self) -> int: ...

    def quant_bits(self) -> int | None: ...

    def page_table(self) -> list[PageTableEntry]: ...

    def tier(self, page_id: int) -> PageTier: ...

    def gather(self, token_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return dense fp32 ``(K, V)`` of shape ``(n, num_heads, head_dim)``.

        The consumer's contract: this method performs whatever staging
        (host→resident) and dequantization the underlying state requires, so the
        caller always sees a contiguous dense slice.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Adapters — make the two existing substrates satisfy the protocol structurally,
# without editing their large class bodies.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _ContiguousPagedKV:
    """Adapter: a contiguous ``KVCacheHandle`` as a degenerate paged state.

    Every page is RESIDENT (the buffer is one contiguous device-resident pool),
    there is no sharing, and ``gather`` indexes the buffer directly — dequantizing
    when the handle is quantized.
    """

    handle: Any  # KVCacheHandle (avoid import cycle)
    kind: KVKind = KVKind.FULL

    def kv_geometry(self) -> KVGeometry:
        h = self.handle
        return KVGeometry(h.num_heads, h.head_dim, h.max_seq, h.page_size, h.dtype)

    def seq_len(self) -> int:
        return int(self.handle.current_seq)

    def quant_bits(self) -> int | None:
        return self.handle.quantize_bits

    def page_table(self) -> list[PageTableEntry]:
        h = self.handle
        ps = h.page_size
        n_pages = (int(h.current_seq) + ps - 1) // ps if h.current_seq else 0
        return [PageTableEntry(p, PageTier.RESIDENT) for p in range(n_pages)]

    def tier(self, page_id: int) -> PageTier:
        return PageTier.RESIDENT

    def gather(self, token_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        h = self.handle
        idx = np.asarray(token_indices, dtype=np.int64).reshape(-1)
        if idx.size and (idx.min() < 0 or idx.max() >= h.current_seq):
            raise IndexError(
                f"gather token index out of range [0, {h.current_seq})")
        k = np.asarray(h.keys)[idx]
        v = np.asarray(h.values)[idx]
        if h.quantize_bits is not None:
            from .. import ops as _ops
            scale = h._scales[:, idx]
            k, v = _ops.dequantize_kv(k, v, scale)
        return np.asarray(k, np.float32), np.asarray(v, np.float32)


@dataclass
class _TieredPagedKV:
    """Adapter: a ``TieredKVCache`` — the genuinely paged, host↔resident state."""

    cache: Any  # TieredKVCache
    kind: KVKind = KVKind.FULL

    def kv_geometry(self) -> KVGeometry:
        c = self.cache
        return KVGeometry(c.num_heads, c.head_dim, c.max_seq, c.page_size, c.dtype)

    def seq_len(self) -> int:
        return int(self.cache.current_seq)

    def quant_bits(self) -> int | None:
        return getattr(self.cache, "quantize_bits", None)

    def page_table(self) -> list[PageTableEntry]:
        c = self.cache
        ps = c.page_size
        n_pages = (int(c.current_seq) + ps - 1) // ps if c.current_seq else 0
        resident = set(c._page_to_slot)
        return [
            PageTableEntry(
                p, PageTier.RESIDENT if p in resident else PageTier.HOST)
            for p in range(n_pages)
        ]

    def tier(self, page_id: int) -> PageTier:
        return (PageTier.RESIDENT if page_id in self.cache._page_to_slot
                else PageTier.HOST)

    def gather(self, token_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        # The tiered cache's own gather enforces the residency discipline. We
        # auto-stage cold pages in residency-bounded *waves* — the prefetch stage
        # the lowering pass owns — so a gather touching more pages than the
        # resident set can hold still succeeds (stage a batch, gather it, repeat).
        idx = np.asarray(token_indices, dtype=np.int64).reshape(-1)
        c = self.cache
        ps = c.page_size
        cap = c._cap
        H, D = c.num_heads, c.head_dim
        out_k = np.empty((idx.shape[0], H, D), np.float32)
        out_v = np.empty((idx.shape[0], H, D), np.float32)

        # Group the gather's positions by page, preserving original positions so
        # we can scatter results back in caller order.
        page_to_positions: dict[int, list[int]] = {}
        for pos, tok in enumerate(idx.tolist()):
            page_to_positions.setdefault(tok // ps, []).append(pos)

        pages = sorted(page_to_positions)
        for start in range(0, len(pages), cap):
            wave = pages[start:start + cap]
            c.stage(wave)
            wave_positions = [p for pg in wave for p in page_to_positions[pg]]
            wk, wv = c.gather(idx[wave_positions], require_resident=True)
            for slot, pos in enumerate(wave_positions):
                out_k[pos] = wk[slot]
                out_v[pos] = wv[slot]
        return out_k, out_v


def paged_attention(
    Q: np.ndarray,
    kv_state: Any,
    *,
    scale: float | None = None,
    causal: bool = False,
    token_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Attention that consumes a :class:`PagedKVState` instead of dense K/V.

    This is the first consumer of the unifying ABI — the proof the contract is
    not an orphan. It reads the page table, gathers the requested tokens (staging
    + dequantizing through the protocol), and runs reference attention per head.

    ``Q`` is ``(num_heads, q_len, head_dim)``. When ``token_indices`` is ``None``
    the full logical sequence ``[0, seq_len)`` is attended. Returns
    ``(num_heads, q_len, head_dim)``.

    The lowering pass (next slice) will replace the eager gather with staged
    prefetch/gather/dequant stages on the backend; the numerics defined here are
    the oracle those stages must reproduce.
    """
    state = as_paged_kv_state(kv_state)
    Q = np.asarray(Q._data if hasattr(Q, "_data") else Q, dtype=np.float32)
    if Q.ndim != 3:
        raise ValueError(f"Q must be (num_heads, q_len, head_dim); got {Q.shape}")

    n = state.seq_len()
    idx = list(range(n)) if token_indices is None else list(token_indices)
    # gather → (S, num_heads, head_dim); transpose to per-head (num_heads, S, hd).
    gk, gv = state.gather(idx)
    K = np.transpose(gk, (1, 0, 2))
    V = np.transpose(gv, (1, 0, 2))

    d = Q.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(d)
    scores = np.matmul(Q, K.swapaxes(-1, -2)) * scale
    if causal:
        q_len, k_len = scores.shape[-2], scores.shape[-1]
        mask = np.triu(np.ones((q_len, k_len), dtype=bool),
                       k=1 + max(k_len - q_len, 0))
        scores = np.where(mask, -np.inf, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return np.matmul(weights, V)


def as_paged_kv_state(cache: Any, *, kind: KVKind | None = None) -> PagedKVState:
    """Wrap any supported KV cache in a :class:`PagedKVState` adapter.

    Already-conforming objects are returned unchanged. ``KVCacheHandle`` and
    ``TieredKVCache`` get the contiguous / tiered adapters respectively.
    """
    if isinstance(cache, PagedKVState) and not _is_raw_handle(cache):
        return cache

    cls_name = type(cache).__name__
    if cls_name == "KVCacheHandle":
        return _ContiguousPagedKV(cache, kind or KVKind.FULL)
    if cls_name == "TieredKVCache":
        return _TieredPagedKV(cache, kind or KVKind.FULL)

    raise TypeError(
        f"{cls_name} does not (yet) satisfy PagedKVState. Supported: "
        f"KVCacheHandle, TieredKVCache, or any object implementing the protocol.")


def _is_raw_handle(cache: Any) -> bool:
    # KVCacheHandle/TieredKVCache may structurally pass runtime_checkable in the
    # future; route them through the explicit adapters until they do natively.
    return type(cache).__name__ in {"KVCacheHandle", "TieredKVCache"}


__all__ = [
    "PageTier",
    "KVKind",
    "KVGeometry",
    "PageTableEntry",
    "PagedKVState",
    "as_paged_kv_state",
    "paged_attention",
]
