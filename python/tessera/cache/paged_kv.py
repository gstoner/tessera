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
import subprocess
import time
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


@dataclass(frozen=True)
class PagedKVBufferABI:
    """Stable physical-page ABI presented to device backends.

    K/V pages use ``[physical_page, page_offset, head, dim]`` (``PLHD``), while
    ``page_table[logical_page]`` names the physical page. Attention kernels must
    not infer physical placement from a logical token index. Version 1 is f32;
    later quantized/latent variants can add explicit descriptors without
    changing the logical-to-physical indirection.
    """

    k_pages: np.ndarray
    v_pages: np.ndarray
    page_table: np.ndarray
    logical_length: int
    abi_version: int = 1
    layout: str = "PLHD"

    def __post_init__(self) -> None:
        k = np.asarray(self.k_pages)
        v = np.asarray(self.v_pages)
        table = np.asarray(self.page_table)
        if self.abi_version != 1 or self.layout != "PLHD":
            raise ValueError("unsupported paged KV buffer ABI version/layout")
        if (k.dtype != np.float32 or v.dtype != np.float32
                or k.ndim != 4 or k.shape != v.shape):
            raise ValueError("paged KV ABI requires equal rank-4 f32 K/V pages")
        if table.dtype != np.int32 or table.ndim != 1 or table.size < 1:
            raise ValueError("paged KV ABI page_table must be non-empty rank-1 i32")
        if np.any(table < 0) or np.any(table >= k.shape[0]):
            raise ValueError("paged KV ABI page_table references an invalid physical page")
        capacity = int(table.size) * int(k.shape[1])
        if self.logical_length < 0 or self.logical_length > capacity:
            raise ValueError("paged KV ABI logical_length exceeds table capacity")

    @property
    def page_size(self) -> int:
        return int(self.k_pages.shape[1])

    def gather(self, token_indices: Any) -> tuple[np.ndarray, np.ndarray]:
        """Reference gather through the page table, preserving caller order."""
        idx = np.asarray(token_indices, dtype=np.int64).reshape(-1)
        if idx.size and (idx.min() < 0 or idx.max() >= self.logical_length):
            raise IndexError(
                f"paged KV token index outside [0, {self.logical_length})")
        logical_page, offset = np.divmod(idx, self.page_size)
        physical_page = self.page_table[logical_page]
        return (np.asarray(self.k_pages[physical_page, offset], np.float32),
                np.asarray(self.v_pages[physical_page, offset], np.float32))


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
        # Delegate to the handle's public fancy-index gather — it owns the
        # quant-scale layout, so the adapter never reaches into ``_scales``.
        k, v = self.handle.gather(token_indices)
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
        resident = c.resident_pages()
        return [
            PageTableEntry(
                p, PageTier.RESIDENT if p in resident else PageTier.HOST)
            for p in range(n_pages)
        ]

    def tier(self, page_id: int) -> PageTier:
        return (PageTier.RESIDENT if self.cache.is_resident(page_id)
                else PageTier.HOST)

    def gather(self, token_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        # The tiered cache's own gather enforces the residency discipline. We
        # auto-stage cold pages in residency-bounded *waves* — the prefetch stage
        # the lowering pass owns — so a gather touching more pages than the
        # resident set can hold still succeeds (stage a batch, gather it, repeat).
        idx = np.asarray(token_indices, dtype=np.int64).reshape(-1)
        c = self.cache
        ps = c.page_size
        cap = c.resident_capacity_pages
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


@dataclass
class _LatentPagedKV:
    """Adapter: an MLA latent cache (the LATENT kind, ShadowKV #6).

    Stores only the compressed ``[S, latent_dim]`` latents; ``gather`` reconstructs
    per-head K/V via the expand projections ``W_k``/``W_v`` — the dequant/expand
    stage the lowering pass owns. ~93% of the memory of a full K/V cache.
    """

    handle: Any                # LatentKVCacheHandle
    w_k: np.ndarray            # (latent_dim, num_heads*head_dim)
    w_v: np.ndarray
    num_heads: int
    head_dim: int
    kind: KVKind = KVKind.LATENT

    def kv_geometry(self) -> KVGeometry:
        h = self.handle
        return KVGeometry(self.num_heads, self.head_dim, h.max_seq,
                          h.page_size, h.dtype)

    def seq_len(self) -> int:
        return int(self.handle.current_seq)

    def quant_bits(self) -> int | None:
        return None

    def page_table(self) -> list[PageTableEntry]:
        h = self.handle
        ps = h.page_size
        n = (int(h.current_seq) + ps - 1) // ps if h.current_seq else 0
        return [PageTableEntry(p, PageTier.RESIDENT) for p in range(n)]

    def tier(self, page_id: int) -> PageTier:
        return PageTier.RESIDENT

    def gather(self, token_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(token_indices, dtype=np.int64).reshape(-1)
        lat = np.asarray(self.handle.latents, np.float32)[idx]   # (m, latent_dim)
        m = lat.shape[0]
        k = (lat @ self.w_k).reshape(m, self.num_heads, self.head_dim)
        v = (lat @ self.w_v).reshape(m, self.num_heads, self.head_dim)
        return np.asarray(k, np.float32), np.asarray(v, np.float32)


@dataclass
class _QuantizedTailPagedKV:
    """Adapter: a hot fp window + an int8-quantized cold tail (QUANTIZED_TAIL).

    Recent tokens ``[split, S)`` stay fp32 (the hot window the model attends most
    precisely); older tokens ``[0, split)`` are stored int8 with per-token scales
    and dequantized on ``gather`` — the quantized-tail memory tradeoff under the
    same ABI.
    """

    k_hot: np.ndarray          # (S-split, H, hd) fp32
    v_hot: np.ndarray
    k_tail_q: np.ndarray       # (split, H, hd) int8
    v_tail_q: np.ndarray
    tail_scale_k: np.ndarray   # (split, 1, 1) fp32
    tail_scale_v: np.ndarray
    split: int
    page_size: int = 128
    dtype: str = "fp32"
    kind: KVKind = KVKind.QUANTIZED_TAIL

    @property
    def _S(self) -> int:
        return self.split + self.k_hot.shape[0]

    def kv_geometry(self) -> KVGeometry:
        H, hd = self.k_hot.shape[1], self.k_hot.shape[2]
        return KVGeometry(H, hd, self._S, self.page_size, self.dtype)

    def seq_len(self) -> int:
        return self._S

    def quant_bits(self) -> int | None:
        return 8

    def page_table(self) -> list[PageTableEntry]:
        ps = self.page_size
        n = (self._S + ps - 1) // ps if self._S else 0
        return [PageTableEntry(p, PageTier.RESIDENT) for p in range(n)]

    def tier(self, page_id: int) -> PageTier:
        return PageTier.RESIDENT

    def gather(self, token_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(token_indices, dtype=np.int64).reshape(-1)
        H, hd = self.k_hot.shape[1], self.k_hot.shape[2]
        out_k = np.empty((idx.shape[0], H, hd), np.float32)
        out_v = np.empty((idx.shape[0], H, hd), np.float32)
        for pos, tok in enumerate(idx.tolist()):
            if tok >= self.split:                       # hot fp window
                out_k[pos] = self.k_hot[tok - self.split]
                out_v[pos] = self.v_hot[tok - self.split]
            else:                                        # cold int8 tail → dequant
                out_k[pos] = self.k_tail_q[tok].astype(np.float32) * self.tail_scale_k[tok]
                out_v[pos] = self.v_tail_q[tok].astype(np.float32) * self.tail_scale_v[tok]
        return out_k, out_v


def latent_paged_kv(handle: Any, w_k: np.ndarray, w_v: np.ndarray, *,
                    num_heads: int, head_dim: int) -> PagedKVState:
    """Wrap an MLA ``LatentKVCacheHandle`` + expand projections as a PagedKVState."""
    w_k = np.asarray(w_k, np.float32)
    w_v = np.asarray(w_v, np.float32)
    if w_k.shape[1] != num_heads * head_dim or w_v.shape[1] != num_heads * head_dim:
        raise ValueError(
            f"expand weights must map latent_dim → {num_heads}*{head_dim}; "
            f"got w_k {w_k.shape}, w_v {w_v.shape}")
    return _LatentPagedKV(handle, w_k, w_v, num_heads, head_dim)


def _quantize_per_token_int8(M: np.ndarray):
    """Symmetric int8 per-token (axis 0) quantize → (q int8, scale (n,1,1))."""
    absmax = np.maximum(np.abs(M).reshape(M.shape[0], -1).max(axis=1), 1e-12)
    scale = (absmax / 127.0).reshape(-1, 1, 1)
    q = np.round(M / scale).clip(-127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def quantized_tail_paged_kv(K_full: np.ndarray, V_full: np.ndarray, *,
                            hot_window: int, page_size: int = 128) -> PagedKVState:
    """Build a QUANTIZED_TAIL state from full fp K/V: keep the most-recent
    ``hot_window`` tokens fp32, quantize the older tail to int8."""
    K_full = np.asarray(K_full, np.float32)
    V_full = np.asarray(V_full, np.float32)
    S = K_full.shape[0]
    split = max(0, S - hot_window)
    k_tail_q, sk = _quantize_per_token_int8(K_full[:split]) if split else (
        np.zeros((0,) + K_full.shape[1:], np.int8), np.zeros((0, 1, 1), np.float32))
    v_tail_q, sv = _quantize_per_token_int8(V_full[:split]) if split else (
        np.zeros((0,) + V_full.shape[1:], np.int8), np.zeros((0, 1, 1), np.float32))
    return _QuantizedTailPagedKV(
        k_hot=K_full[split:], v_hot=V_full[split:],
        k_tail_q=k_tail_q, v_tail_q=v_tail_q, tail_scale_k=sk, tail_scale_v=sv,
        split=split, page_size=page_size)


def materialize_paged_kv_abi(kv_state: Any) -> PagedKVBufferABI:
    """Expose any :class:`PagedKVState` through the stable device-page ABI.

    A backend-native state may provide ``paged_kv_abi()`` and retain its own
    physical placement. Generic adapters are packed once at this boundary; the
    attention implementation still consumes only pages + page table and never
    reaches into a cache's contiguous storage.
    """
    state = as_paged_kv_state(kv_state)
    native = getattr(state, "paged_kv_abi", None)
    if callable(native):
        abi = native()
        if not isinstance(abi, PagedKVBufferABI):
            raise TypeError("paged_kv_abi() must return PagedKVBufferABI")
        return abi

    geometry = state.kv_geometry()
    n = state.seq_len()
    page_size = int(geometry.page_size)
    logical_pages = max(1, (n + page_size - 1) // page_size)
    shape = (logical_pages, page_size,
             int(geometry.num_heads), int(geometry.head_dim))
    k_pages = np.zeros(shape, np.float32)
    v_pages = np.zeros(shape, np.float32)
    if n:
        k, v = state.gather(range(n))
        flat_k = k_pages.reshape(-1, geometry.num_heads, geometry.head_dim)
        flat_v = v_pages.reshape(-1, geometry.num_heads, geometry.head_dim)
        flat_k[:n] = np.asarray(k, np.float32)
        flat_v[:n] = np.asarray(v, np.float32)
    return PagedKVBufferABI(
        k_pages=k_pages, v_pages=v_pages,
        page_table=np.arange(logical_pages, dtype=np.int32),
        logical_length=n)


def _reference_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         scale: float, causal: bool) -> np.ndarray:
    """Per-head numpy attention. Q (H,q,d), K/V (H,S,d) → (H,q,d)."""
    if K.shape[0] != Q.shape[0]:
        if K.shape[0] < 1 or Q.shape[0] % K.shape[0]:
            raise ValueError(
                "attention query heads must be divisible by KV heads; got "
                f"Q={Q.shape[0]} KV={K.shape[0]}")
        ratio = Q.shape[0] // K.shape[0]
        K = np.repeat(K, ratio, axis=0)
        V = np.repeat(V, ratio, axis=0)
    scores = np.matmul(Q, K.swapaxes(-1, -2)) * scale
    if causal:
        q_len, k_len = scores.shape[-2], scores.shape[-1]
        mask = np.triu(np.ones((q_len, k_len), dtype=bool),
                       k=1 + max(k_len - q_len, 0))
        scores = np.where(mask, -np.inf, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    return np.matmul(w, V)


def paged_attention(
    Q: np.ndarray,
    kv_state: Any,
    *,
    scale: float | None = None,
    causal: bool = False,
    token_indices: Sequence[int] | None = None,
    backend: str = "reference",
    return_execution: bool = False,
) -> Any:
    """Attention that consumes a :class:`PagedKVState` instead of dense K/V.

    The consumer of the unifying ABI: it reads the page table, gathers the
    requested tokens (staging + dequantizing through the protocol — the
    prefetch/gather/dequant stages), then runs attention.

    ``Q`` is ``(num_heads, q_len, head_dim)``. ``token_indices=None`` attends the
    full logical sequence ``[0, seq_len)``. Returns ``(num_heads, q_len, head_dim)``.

    ``backend``:
      * ``"reference"`` (default) — numpy attention over the gathered K/V.
      * ``"apple_gpu"`` — the gathered (staged + dequantized) dense K/V feeds the
        shipped fused matmul→softmax→matmul Metal kernel **per head**
        (``run_fused_attention``). This is #8's native execution path: gather is
        the staging stage; the contraction runs on ``metal_runtime``.
      * ``"rocm"`` — the same gathered dense K/V feeds the compiler-generated FA-2
        WMMA forward kernel on gfx1151 (the folded ``(num_heads, S, head_dim)``
        batch is exactly the lane's ``[..., S, D]`` contract). Gather is the
        staging stage; the fused attention runs on ``native_gpu``.
      * ``"nvidia"`` — size-dispatched CUDA execution over the stable page-table
        ABI: a direct single-launch paged-attention kernel for short decode, or a
        staged device-resident gather/contraction plan for longer contexts. D2
        warm-starts the measured crossover; neither route stages through host.

    With ``return_execution=True`` returns ``(out, execution_kind)`` where
    ``execution_kind`` is the native provenance token (``"metal_runtime"`` for
    apple_gpu, ``"native_gpu"`` for rocm) only if the native kernel genuinely ran
    — else ``"reference"`` (a silent fallback can never overclaim the native rung,
    the signal the cross-substrate oracle gates on).
    """
    state = as_paged_kv_state(kv_state)
    Q = np.asarray(Q._data if hasattr(Q, "_data") else Q, dtype=np.float32)
    if Q.ndim != 3:
        raise ValueError(f"Q must be (num_heads, q_len, head_dim); got {Q.shape}")

    n = state.seq_len()
    idx = list(range(n)) if token_indices is None else list(token_indices)
    d = Q.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(d)

    if backend == "rocm":
        # ROCm consumes the stable physical-page ABI. Its HIP gather follows the
        # logical→physical table; attention never assumes logical pages are
        # contiguous or identity-mapped.
        abi = materialize_paged_kv_abi(state)
        out, exe = _paged_attention_rocm(
            Q, abi, idx, float(scale), causal)
        return (out, exe) if return_execution else out
    if backend == "nvidia":
        abi = materialize_paged_kv_abi(state)
        out, exe = _paged_attention_nvidia(
            Q, abi, idx, float(scale), causal)
        return (out, exe) if return_execution else out

    # Other backends still consume dense K/V today. Their gather remains behind
    # PagedKVState rather than reaching into a cache's physical representation.
    gk, gv = state.gather(idx)
    K = np.transpose(gk, (1, 0, 2))
    V = np.transpose(gv, (1, 0, 2))
    if backend == "apple_gpu":
        out, exe = _paged_attention_apple_gpu(Q, K, V, float(scale), causal)
    elif backend == "reference":
        out, exe = _reference_attention(Q, K, V, float(scale), causal), "reference"
    else:
        raise ValueError(
            f"unknown backend {backend!r}; use 'reference', 'apple_gpu', 'rocm', "
            "or 'nvidia'")

    return (out, exe) if return_execution else out


def _paged_attention_apple_gpu(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                               scale: float, causal: bool) -> tuple[np.ndarray, str]:
    """Run the gathered per-head attention on the Apple GPU fused kernel.

    Loops heads through the shipped ``matmul→softmax→matmul`` Metal kernel (the
    same one the MLA-E2E proof uses). Reports ``metal_runtime`` only if every head
    ran natively; any fallback demotes the whole call to ``reference`` so the
    provenance signal can never overclaim.
    """
    from ..compiler.fusion import run_fused_attention, AttentionRegion

    H, q_len, d = Q.shape
    Dv = V.shape[-1]
    region = AttentionRegion(scale=scale, causal=causal)
    out = np.empty((H, q_len, Dv), np.float32)
    all_native = True
    for h in range(H):
        o_h, exe_h = run_fused_attention(region, Q[h], K[h], V[h])
        out[h] = o_h
        if exe_h != "metal_runtime":
            all_native = False
    return out, ("metal_runtime" if all_native else "reference")


_rocm_paged_attention_route_cache: dict[tuple[Any, ...], str] = {}
_rocm_paged_attention_route_evidence: dict[tuple[Any, ...], dict[str, Any]] = {}


def _rocm_paged_attention_corpus_winner(
    q_heads: int, kv_heads: int, q_len: int, tokens: int, dim: int,
    page_size: int,
) -> str | None:
    """Warm-start the production route from committed gfx1151 wall timing."""
    from ..compiler.emit import autotune as at
    from ..compiler.emit.kernel_emitter import SpecPolicy, bucket_key
    cache = at.MeasureCache()
    at.load_corpus(cache=cache)
    record = cache.get((
        "rocm:gfx1151", "rocm", "paged_kv_decode",
        bucket_key((q_len, q_heads, kv_heads, tokens, dim, page_size),
                   SpecPolicy.BUCKET),
        "f32", at.TIMING_END_TO_END))
    if record is not None and record.winner in {"gather_fa", "direct"}:
        return record.winner
    return None


def _paged_attention_rocm(
    Q: np.ndarray, abi: PagedKVBufferABI, token_indices: Sequence[int],
    scale: float, causal: bool, *, _force_measure: bool = False,
) -> tuple[np.ndarray, str]:
    """Read stable physical pages, then run the compiled ROCm FA-2 WMMA kernel.

    The first HIP launch follows ``abi.page_table`` to gather arbitrary logical
    tokens. The resulting ``(num_heads,S,head_dim)`` staging tensor then feeds
    the current FlashAttention ABI. This explicit boundary is intentionally
    replaceable by a fused paged-attention kernel without changing the cache ABI.

    Reports ``native_gpu`` only when the kernel genuinely fired. A host without a
    usable lane (no GPU / tessera-opt) or a shape the WMMA kernel cannot take
    (head_dim not a multiple of 16) demotes the whole call to ``reference`` — the
    provenance token can never overclaim.
    """
    from .. import runtime as rt

    idx = np.asarray(token_indices, dtype=np.int64).reshape(-1)

    def reference() -> np.ndarray:
        gk, gv = abi.gather(idx)
        return _reference_attention(
            Q, np.transpose(gk, (1, 0, 2)), np.transpose(gv, (1, 0, 2)),
            scale, causal)

    head_dim = int(Q.shape[-1])
    # The WMMA kernel needs head_dim a positive multiple of 16 and a usable lane;
    # anything else is a clean demotion to the numpy reference, not an error (the
    # reference is still correct, we just didn't earn the native rung).
    if (head_dim <= 0 or head_dim % 16 != 0
            or not rt._rocm_compiled_flash_attn_available()):
        return reference(), "reference"

    try:
        from ..compiler.emit.rocm_hip import (
            run_paged_attention_direct_f32,
            run_paged_kv_cache_read_f32,
        )

        def execute(route: str) -> tuple[np.ndarray, float | None, float]:
            if route == "direct":
                _, device_ms, _ = run_paged_attention_direct_f32(
                    Q, abi.k_pages, abi.v_pages, abi.page_table, idx,
                    scale=float(scale), causal=causal)
                out, _, wall_ms = run_paged_attention_direct_f32(
                    Q, abi.k_pages, abi.v_pages, abi.page_table, idx,
                    scale=float(scale), causal=causal, reps=1)
                return out, device_ms, wall_ms

            def gather_fa(reps: int) -> tuple[np.ndarray, float | None, float]:
                start = time.perf_counter()
                gk, k_ms = run_paged_kv_cache_read_f32(
                    abi.k_pages, abi.page_table, idx, return_device_ms=True,
                    reps=reps)
                gv, v_ms = run_paged_kv_cache_read_f32(
                    abi.v_pages, abi.page_table, idx, return_device_ms=True,
                    reps=reps)
                K = np.transpose(gk, (1, 0, 2))
                V = np.transpose(gv, (1, 0, 2))
                # Dense FA's causal mask is anchored at query 0. Paged decode
                # needs the right-aligned T-Q offset, expressed as bias here.
                bias = None
                if causal:
                    q_len, tokens = int(Q.shape[1]), int(idx.size)
                    limit = np.arange(q_len)[:, None] + max(tokens - q_len, 0)
                    bias = np.where(
                        np.arange(tokens)[None, :] <= limit, 0.0,
                        -np.inf).astype(np.float32)
                result, fa_ms = rt._rocm_flash_attn_forward_o(
                    Q.astype(np.float16), K.astype(np.float16),
                    V.astype(np.float16), float(scale), 0, bias=bias,
                    _timed_reps=reps)
                out = np.asarray(result, np.float32).reshape(Q.shape)
                device_parts = (k_ms, v_ms, fa_ms)
                device_ms = (sum(float(part) for part in device_parts)
                             if all(part is not None and float(part) > 0
                                    for part in device_parts) else None)
                return out, device_ms, \
                    (time.perf_counter() - start) * 1e3

            _, device_ms, _ = gather_fa(20)  # also warms code + allocations
            out, _, wall_ms = gather_fa(1)
            return out, device_ms, wall_ms

        _, page_size, kv_heads, _ = abi.k_pages.shape
        key = (int(Q.shape[0]), int(kv_heads), int(Q.shape[1]), int(idx.size),
               head_dim, int(page_size), bool(causal))
        selected = (None if _force_measure else
                    _rocm_paged_attention_route_cache.get(key))
        if selected is None and not _force_measure:
            selected = _rocm_paged_attention_corpus_winner(
                int(Q.shape[0]), int(kv_heads), int(Q.shape[1]), int(idx.size),
                head_dim, int(page_size))
            if selected is not None:
                _rocm_paged_attention_route_cache[key] = selected
        if selected is not None:
            # Cached/corpus routes execute once; timing loops are paid only on a
            # genuine miss or by the recorder, never on every serving request.
            if selected == "direct":
                out, _, _ = run_paged_attention_direct_f32(
                    Q, abi.k_pages, abi.v_pages, abi.page_table, idx,
                    scale=float(scale), causal=causal, reps=1)
            else:
                gk = run_paged_kv_cache_read_f32(
                    abi.k_pages, abi.page_table, idx, reps=1)
                gv = run_paged_kv_cache_read_f32(
                    abi.v_pages, abi.page_table, idx, reps=1)
                K = np.transpose(gk, (1, 0, 2))
                V = np.transpose(gv, (1, 0, 2))
                bias = None
                if causal:
                    q_len, tokens = int(Q.shape[1]), int(idx.size)
                    limit = np.arange(q_len)[:, None] + max(tokens - q_len, 0)
                    bias = np.where(
                        np.arange(tokens)[None, :] <= limit, 0.0,
                        -np.inf).astype(np.float32)
                result = rt._rocm_flash_attn_forward_o(
                    Q.astype(np.float16), K.astype(np.float16),
                    V.astype(np.float16), float(scale), 0, bias=bias)
                out = np.asarray(result, np.float32).reshape(Q.shape)
            return out, "native_gpu"

        oracle = reference()
        results = {route: execute(route) for route in ("gather_fa", "direct")}
        for route, (candidate, _, _) in results.items():
            tolerance = 2e-2 if route == "gather_fa" else 3e-5
            if not np.allclose(candidate, oracle, rtol=tolerance, atol=tolerance):
                raise RuntimeError(
                    f"ROCm paged-attention {route} disagrees with oracle")
        if not np.allclose(results["gather_fa"][0], results["direct"][0],
                           rtol=2e-2, atol=2e-2):
            raise RuntimeError("ROCm paged-attention routes disagree")
        device = {name: value[1] for name, value in results.items()}
        end_to_end = {name: value[2] for name, value in results.items()}
        available_device: dict[str, float] = {
            name: value for name, value in device.items() if value is not None
        }
        device_winner = (min(available_device, key=available_device.__getitem__)
                         if len(available_device) == len(device)
                         else None)
        end_to_end_winner = min(end_to_end, key=end_to_end.__getitem__)
        # Serving chooses full-call latency. Keep the device winner beside it so
        # a future resident ABI can make a deliberate, evidence-backed switch.
        selected = end_to_end_winner
        _rocm_paged_attention_route_cache[key] = selected
        _rocm_paged_attention_route_evidence[key] = {
            "device_ms": device, "end_to_end_ms": end_to_end,
            "device_winner": device_winner,
            "device_timing_status": ("available" if device_winner is not None
                                     else "unavailable"),
            "end_to_end_winner": end_to_end_winner,
            "selected": selected,
        }
        out = results[selected][0]
    except (rt._RocmCompiledUnavailable, RuntimeError, OSError,
            FileNotFoundError, subprocess.CalledProcessError):
        # Lane probed available but couldn't launch here (serialization / driver);
        # fall back rather than fail the paged-attention call.
        return reference(), "reference"
    return out, "native_gpu"


def _paged_attention_nvidia(
    Q: np.ndarray, abi: PagedKVBufferABI, token_indices: Sequence[int],
    scale: float, causal: bool,
) -> tuple[np.ndarray, str]:
    """Consume physical pages through a device-resident CUDA attention plan."""
    idx = np.asarray(token_indices, np.int64)
    try:
        from ..compiler.emit.nvidia_cuda import run_paged_attention_resident_f32
        out, _latency_ms = run_paged_attention_resident_f32(
            Q, abi.k_pages, abi.v_pages, abi.page_table, idx,
            scale=float(scale), causal=causal)
    except (RuntimeError, OSError, FileNotFoundError):
        k, v = abi.gather(idx)
        return _reference_attention(
            Q, np.transpose(k, (1, 0, 2)), np.transpose(v, (1, 0, 2)),
            scale, causal), "reference"
    return np.asarray(out, dtype=np.float32), "native_gpu"


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
    "PagedKVBufferABI",
    "as_paged_kv_state",
    "materialize_paged_kv_abi",
    "paged_attention",
    "latent_paged_kv",
    "quantized_tail_paged_kv",
]
