"""Multi-sequence block-paged MLA cache — vLLM-style paged attention (2026-05-30).

The production-serving follow-on to :class:`MLAPagedDecoder`. Where that handles
one sequence over a contiguous window, this manages **many concurrent sequences**
sharing a single physical block pool — the core idea behind vLLM's PagedAttention.

Physical storage is a pool of fixed-size blocks:
  latent pool ``[num_blocks, block_size, latent_dim]``
  rope pool   ``[num_blocks, block_size, rope_dim]``

Each sequence owns a **block table** — an ordered list of physical block ids that
hold its tokens. Logical token ``i`` lives at physical
``(block_table[i // block_size], i % block_size)``. Blocks are allocated from a
free list on demand and returned when a sequence finishes, so a finished
request's pages are immediately reusable by another — no per-sequence contiguous
reservation, no external fragmentation.

For MLA the per-token footprint is just the compressed latent + shared rope key
(``latent_dim + rope_dim``, shared across all heads), so a block holds far more
context than a per-head K/V block would.

Serving loop::

    pool = MLABlockPagedCache(num_heads=H, nope_dim=dn, rope_dim=dr, v_dim=dv,
                              latent_dim=Dl, Wuk_t=Wuk_t, Wuv=Wuv,
                              num_blocks=1024, block_size=16)
    pool.add_sequence("req-A"); pool.add_sequence("req-B")
    pool.append("req-A", c_kv_A, k_rope_A)        # prefill / decode appends
    pool.append("req-B", c_kv_B, k_rope_B)
    out = pool.decode_batch({"req-A": (qnA, qrA), "req-B": (qnB, qrB)})
    pool.free_sequence("req-A")                    # pages return to the pool

Compute loops per sequence (lengths are ragged); the contribution here is the
block-table memory manager, which is what makes concurrent serving tractable.
Batching same-length sequences through the kernel's ``B>1`` path is a future
optimization.
"""

from __future__ import annotations

from typing import Any, Hashable

import numpy as np

from .mla_paged import absorb_decode_batch, absorb_decode_one

_DTYPE_MAP = {"fp16": np.float16, "bf16": np.float32, "fp32": np.float32,
              "fp64": np.float64}


class _SeqState:
    __slots__ = ("block_table", "length")

    def __init__(self) -> None:
        self.block_table: list[int] = []
        self.length: int = 0


class MLABlockPagedCacheError(RuntimeError):
    """Raised on block-pool exhaustion or invalid sequence operations."""


class MLABlockPagedCache:
    """vLLM-style block-paged MLA cache across concurrent sequences.

    Parameters
    ----------
    num_heads, nope_dim, rope_dim, v_dim, latent_dim
        MLA head geometry. ``rope_dim`` must be even.
    Wuk_t
        Absorbed K up-projection ``[num_heads, nope_dim, latent_dim]``.
    Wuv
        V up-projection ``[num_heads, latent_dim, v_dim]``.
    num_blocks, block_size
        Physical pool geometry — ``num_blocks`` pages of ``block_size`` tokens.
    rope_base, rotation_style, dtype
        RoPE base, ``"interleaved"``/``"half"``, and storage dtype.
    """

    def __init__(self, *, num_heads: int, nope_dim: int, rope_dim: int,
                 v_dim: int, latent_dim: int, Wuk_t: Any, Wuv: Any,
                 num_blocks: int, block_size: int, rope_base: float = 10000.0,
                 rotation_style: str = "interleaved", dtype: str = "fp32") -> None:
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {rope_dim}")
        if rotation_style not in ("interleaved", "half"):
            raise ValueError(f"rotation_style must be interleaved|half, "
                             f"got {rotation_style!r}")
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_heads = int(num_heads)
        self.nope_dim = int(nope_dim)
        self.rope_dim = int(rope_dim)
        self.v_dim = int(v_dim)
        self.latent_dim = int(latent_dim)
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.rope_base = float(rope_base)
        self.rotation_style = rotation_style

        self.Wuk_t = np.ascontiguousarray(Wuk_t, np.float32)
        self.Wuv = np.ascontiguousarray(Wuv, np.float32)
        if self.Wuk_t.shape != (self.num_heads, self.nope_dim, self.latent_dim):
            raise ValueError(
                f"Wuk_t must be [H,nope_dim,latent_dim]="
                f"{(self.num_heads, self.nope_dim, self.latent_dim)}; "
                f"got {self.Wuk_t.shape}")
        if self.Wuv.shape != (self.num_heads, self.latent_dim, self.v_dim):
            raise ValueError(
                f"Wuv must be [H,latent_dim,v_dim]="
                f"{(self.num_heads, self.latent_dim, self.v_dim)}; "
                f"got {self.Wuv.shape}")

        np_dtype = _DTYPE_MAP.get(dtype, np.float32)
        self._latent_pool = np.zeros(
            (self.num_blocks, self.block_size, self.latent_dim), np_dtype)
        self._rope_pool = np.zeros(
            (self.num_blocks, self.block_size, self.rope_dim), np_dtype)
        self._np_dtype = np_dtype
        # Free-block stack (LIFO favors recently-freed warm pages).
        self._free: list[int] = list(range(self.num_blocks - 1, -1, -1))
        self._seqs: dict[Hashable, _SeqState] = {}

    # ------------------------------------------------------------------
    # Pool accounting
    # ------------------------------------------------------------------
    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    @property
    def utilization(self) -> float:
        return self.num_used_blocks / self.num_blocks

    @property
    def num_sequences(self) -> int:
        return len(self._seqs)

    def sequence_length(self, seq_id: Hashable) -> int:
        return self._seq(seq_id).length

    def block_table(self, seq_id: Hashable) -> list[int]:
        """Physical block ids backing ``seq_id`` (copy)."""
        return list(self._seq(seq_id).block_table)

    def cache_bytes_per_token(self) -> int:
        return (self.latent_dim + self.rope_dim) * self._latent_pool.itemsize

    # ------------------------------------------------------------------
    # Sequence lifecycle
    # ------------------------------------------------------------------
    def add_sequence(self, seq_id: Hashable) -> None:
        if seq_id in self._seqs:
            raise MLABlockPagedCacheError(f"sequence {seq_id!r} already exists")
        self._seqs[seq_id] = _SeqState()

    def free_sequence(self, seq_id: Hashable) -> None:
        """Return all of a sequence's blocks to the free pool."""
        st = self._seqs.pop(seq_id, None)
        if st is None:
            raise MLABlockPagedCacheError(f"unknown sequence {seq_id!r}")
        for blk in st.block_table:
            self._latent_pool[blk] = 0
            self._rope_pool[blk] = 0
            self._free.append(blk)

    def _seq(self, seq_id: Hashable) -> _SeqState:
        st = self._seqs.get(seq_id)
        if st is None:
            raise MLABlockPagedCacheError(f"unknown sequence {seq_id!r}")
        return st

    # ------------------------------------------------------------------
    # Append (prefill + per-step) with on-demand block allocation
    # ------------------------------------------------------------------
    def append(self, seq_id: Hashable, c_kv: Any, k_rope: Any) -> None:
        """Append ``n_new`` tokens' latent + shared rope key to ``seq_id``,
        allocating physical blocks from the free pool as needed."""
        st = self._seq(seq_id)
        c_arr = np.asarray(c_kv)
        r_arr = np.asarray(k_rope)
        if c_arr.ndim == 1:
            c_arr = c_arr.reshape(1, -1)
        if r_arr.ndim == 1:
            r_arr = r_arr.reshape(1, -1)
        if c_arr.shape[1] != self.latent_dim or r_arr.shape[1] != self.rope_dim:
            raise ValueError(
                f"expected c_kv [*,{self.latent_dim}] and k_rope "
                f"[*,{self.rope_dim}]; got {c_arr.shape} / {r_arr.shape}")
        if c_arr.shape[0] != r_arr.shape[0]:
            raise ValueError("latent and rope chunks must append the same "
                             "number of tokens")
        n_new = c_arr.shape[0]
        # ensure capacity (allocate trailing blocks)
        needed_blocks = (st.length + n_new + self.block_size - 1) // self.block_size
        while len(st.block_table) < needed_blocks:
            if not self._free:
                raise MLABlockPagedCacheError(
                    f"block pool exhausted: {self.num_blocks} blocks all in use "
                    f"(free a finished sequence to reclaim pages)")
            st.block_table.append(self._free.pop())
        # scatter the new tokens into their physical slots
        c_arr = c_arr.astype(self._np_dtype, copy=False)
        r_arr = r_arr.astype(self._np_dtype, copy=False)
        for t in range(n_new):
            logical = st.length + t
            blk = st.block_table[logical // self.block_size]
            off = logical % self.block_size
            self._latent_pool[blk, off] = c_arr[t]
            self._rope_pool[blk, off] = r_arr[t]
        st.length += n_new

    # ------------------------------------------------------------------
    # Gather + decode
    # ------------------------------------------------------------------
    def _gather(self, st: _SeqState) -> tuple[np.ndarray, np.ndarray]:
        """Materialize a sequence's logical window from its (possibly
        non-contiguous) physical blocks."""
        S = st.length
        c_kv = np.empty((S, self.latent_dim), np.float32)
        k_rope = np.empty((S, self.rope_dim), np.float32)
        for i in range(S):
            blk = st.block_table[i // self.block_size]
            off = i % self.block_size
            c_kv[i] = self._latent_pool[blk, off]
            k_rope[i] = self._rope_pool[blk, off]
        return c_kv, k_rope

    def decode(self, seq_id: Hashable, q_nope: Any, q_rope: Any) -> np.ndarray:
        """Decode one query for ``seq_id`` against its cached window.
        Returns ``[num_heads, v_dim]``."""
        st = self._seq(seq_id)
        if st.length == 0:
            raise MLABlockPagedCacheError(
                f"sequence {seq_id!r} is empty; append before decoding")
        qn = np.ascontiguousarray(q_nope, np.float32)
        qr = np.ascontiguousarray(q_rope, np.float32)
        if qn.shape != (self.num_heads, self.nope_dim):
            raise ValueError(f"q_nope must be {(self.num_heads, self.nope_dim)}; "
                             f"got {qn.shape}")
        if qr.shape != (self.num_heads, self.rope_dim):
            raise ValueError(f"q_rope must be {(self.num_heads, self.rope_dim)}; "
                             f"got {qr.shape}")
        c_kv, k_rope = self._gather(st)
        key_pos = np.arange(st.length)
        return absorb_decode_one(qn, qr, c_kv, k_rope, self.Wuk_t, self.Wuv,
                                 key_pos, st.length - 1, self.rope_base,
                                 self.rotation_style)

    def decode_batch(self, queries: dict) -> dict:
        """Decode a batch of concurrent sequences. ``queries`` maps
        ``seq_id -> (q_nope, q_rope)``; returns ``seq_id -> [num_heads, v_dim]``.

        Sequences are ragged, but those sharing a cached length share RoPE
        positions, so they are **grouped by length and dispatched together**
        (one ``B = group_size`` kernel call per length) instead of looping
        per sequence — a throughput win when many concurrent requests sit at the
        same decode step."""
        # group seq ids by current length
        by_len: dict[int, list] = {}
        for sid, (qn, qr) in queries.items():
            st = self._seq(sid)
            if st.length == 0:
                raise MLABlockPagedCacheError(
                    f"sequence {sid!r} is empty; append before decoding")
            by_len.setdefault(st.length, []).append((sid, qn, qr))

        out: dict = {}
        for L, items in by_len.items():
            G = len(items)
            qn_stack = np.empty((G, self.num_heads, self.nope_dim), np.float32)
            qr_stack = np.empty((G, self.num_heads, self.rope_dim), np.float32)
            c_stack = np.empty((G, L, self.latent_dim), np.float32)
            r_stack = np.empty((G, L, self.rope_dim), np.float32)
            for i, (sid, qn, qr) in enumerate(items):
                qn = np.ascontiguousarray(qn, np.float32)
                qr = np.ascontiguousarray(qr, np.float32)
                if qn.shape != (self.num_heads, self.nope_dim):
                    raise ValueError(
                        f"q_nope for {sid!r} must be "
                        f"{(self.num_heads, self.nope_dim)}; got {qn.shape}")
                if qr.shape != (self.num_heads, self.rope_dim):
                    raise ValueError(
                        f"q_rope for {sid!r} must be "
                        f"{(self.num_heads, self.rope_dim)}; got {qr.shape}")
                c_kv, k_rope = self._gather(self._seq(sid))
                qn_stack[i], qr_stack[i] = qn, qr
                c_stack[i], r_stack[i] = c_kv, k_rope
            res = absorb_decode_batch(qn_stack, qr_stack, c_stack, r_stack,
                                      self.Wuk_t, self.Wuv, np.arange(L), L - 1,
                                      self.rope_base, self.rotation_style)
            for i, (sid, _, _) in enumerate(items):
                out[sid] = np.ascontiguousarray(res[i])
        return out

    def __repr__(self) -> str:
        return (f"MLABlockPagedCache(num_blocks={self.num_blocks}, "
                f"block_size={self.block_size}, "
                f"used={self.num_used_blocks}/{self.num_blocks}, "
                f"sequences={self.num_sequences}, "
                f"latent_dim={self.latent_dim}, rope_dim={self.rope_dim})")
