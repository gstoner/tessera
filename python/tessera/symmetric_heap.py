"""C1 — symmetric-heap placement as a sharding concept.

A *symmetric heap* is the substrate that one-sided collectives are built on.
Iris (AMD) and Mori (ByteByte/distributed) both converge on the same idea: a
distributed allocation where every rank reserves the same number of bytes at the
**same virtual offset** within its own heap.  Because the offset is identical
across ranks, addressing a peer's shard is pure offset arithmetic —

    remote_addr = peer_heap_base + (local_addr - my_heap_base)

— with **no address-translation table** to consult per access.  The peer's base
pointer (``peer_heap_base``) is a tiny per-rank constant that lives in L1, so a
``get`` / ``put`` / ``atomic`` over the heap costs one subtract + one add, not a
table lookup.  That is what makes one-sided RDMA/IPC remote-memory access cheap
enough to be a primitive rather than a message-passing fallback.

Important: it is the **offset** that is symmetric, *not* the base pointer.  Real
systems (NVLink P2P, ROCm IPC handles, RDMA-registered regions) hand each rank a
different virtual base; the invariant they preserve is that a shard placed at
offset ``O`` on rank ``i`` is also at offset ``O`` on rank ``j``.  So this model
deliberately carries a *list* of per-rank bases — ``heap_bases`` — and never
assumes they are equal.

This is a **hardware-free model** today: it describes the placement contract and
does the address arithmetic in pure Python, so the compiler / autotuner / audit
surface can reason about symmetric allocations without any RDMA or IPC bring-up.
Real fabric registration (RDMA verbs, NVLink P2P, ROCm IPC) is Phase G/H.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

# Placement modes for a symmetric shard.
#   replicated  — every rank holds the full bank at the symmetric offset; a
#                 read of global offset O lands on the local copy on any rank.
#   partitioned — each rank owns one contiguous slice of the logical bank; the
#                 global address space is the concatenation of the per-rank
#                 slices in rank order.
SYMMETRIC_MODES = ("replicated", "partitioned")


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@dataclass(frozen=True)
class SymmetricHeap:
    """A symmetric allocation: ``bytes_per_rank`` reserved on every rank.

    Attributes
    ----------
    num_ranks : int
        Number of ranks participating in the symmetric heap (>= 1).
    bytes_per_rank : int
        Bytes each rank reserves at the symmetric offset.  Must be a
        non-negative multiple of ``alignment`` (every rank reserves the same
        amount — that uniformity is what makes the offset symmetric).
    mesh_axis : str | None
        Optional name of the mesh axis the heap is symmetric across (e.g.
        ``"dp"``).  Carried as metadata; not validated against a mesh here so
        this module stays a leaf (callers may cross-check against a
        ``NamedMesh.axis_names``).
    alignment : int
        Allocation alignment in bytes; must be a power of two.  Default 256
        (a comfortable cache-line / RDMA-registration granularity).
    """

    num_ranks: int
    bytes_per_rank: int
    mesh_axis: str | None = None
    alignment: int = 256

    def __post_init__(self) -> None:
        if self.num_ranks < 1:
            raise ValueError(
                f"SymmetricHeap.num_ranks must be >= 1; got {self.num_ranks}")
        if not _is_power_of_two(self.alignment):
            raise ValueError(
                f"SymmetricHeap.alignment must be a power of two; "
                f"got {self.alignment}")
        if self.bytes_per_rank < 0:
            raise ValueError(
                f"SymmetricHeap.bytes_per_rank must be >= 0; "
                f"got {self.bytes_per_rank}")
        if self.bytes_per_rank % self.alignment != 0:
            raise ValueError(
                f"SymmetricHeap.bytes_per_rank={self.bytes_per_rank} must be a "
                f"multiple of alignment={self.alignment}")

    # ── Iris-style address translation ──────────────────────────────────────
    def remote_address(
        self,
        local_offset: int,
        src_rank: int,
        dst_rank: int,
        heap_bases: Sequence[int],
    ) -> int:
        """Resolve the address on ``dst_rank`` of a symmetric-heap location.

        The Iris translation.  ``local_offset`` is an offset into ``src_rank``'s
        heap; ``heap_bases[r]`` is the per-rank virtual base pointer (which may
        differ per rank — only the *offset* is symmetric).  The full local
        address is ``heap_bases[src_rank] + local_offset``; subtracting the src
        base recovers the symmetric offset and adding the dst base lands on the
        peer's copy:

            offset      = local_addr - heap_bases[src_rank]   (== local_offset)
            remote_addr = heap_bases[dst_rank] + offset

        which reduces to ``heap_bases[dst_rank] + local_offset`` — no
        translation table, just two base pointers and a subtract/add.
        """
        if len(heap_bases) != self.num_ranks:
            raise ValueError(
                f"heap_bases length {len(heap_bases)} must equal num_ranks="
                f"{self.num_ranks}")
        self._check_rank(src_rank, "src_rank")
        self._check_rank(dst_rank, "dst_rank")
        self._check_offset(local_offset)
        local_addr = int(heap_bases[src_rank]) + int(local_offset)
        offset = local_addr - int(heap_bases[src_rank])
        return int(heap_bases[dst_rank]) + offset

    # ── Partition helpers ───────────────────────────────────────────────────
    @property
    def total_bytes(self) -> int:
        """Logical size of a partitioned bank — the concatenation of shards."""
        return self.bytes_per_rank * self.num_ranks

    def owner_rank(self, global_offset: int) -> int:
        """Owning rank of a byte offset into the concatenated (partitioned) bank.

        Block layout: rank ``r`` owns ``[r * bytes_per_rank, (r+1) *
        bytes_per_rank)``.
        """
        if self.bytes_per_rank == 0:
            raise ValueError(
                "owner_rank is undefined for a zero-byte symmetric heap")
        if not 0 <= global_offset < self.total_bytes:
            raise ValueError(
                f"global_offset={global_offset} out of range "
                f"[0, {self.total_bytes})")
        return int(global_offset) // self.bytes_per_rank

    def local_slice(self, rank: int) -> tuple[int, int]:
        """The ``[start, stop)`` byte slice of the logical bank owned by ``rank``."""
        self._check_rank(rank, "rank")
        start = rank * self.bytes_per_rank
        return start, start + self.bytes_per_rank

    def global_to_local(self, global_offset: int) -> tuple[int, int]:
        """Map a concatenated-bank offset to ``(rank, local_offset)``."""
        rank = self.owner_rank(global_offset)
        return rank, int(global_offset) - rank * self.bytes_per_rank

    def as_metadata_dict(self) -> dict[str, Any]:
        return {
            "num_ranks": self.num_ranks,
            "bytes_per_rank": self.bytes_per_rank,
            "mesh_axis": self.mesh_axis,
            "alignment": self.alignment,
            "total_bytes": self.total_bytes,
        }

    # ── internal validators ─────────────────────────────────────────────────
    def _check_rank(self, rank: int, label: str) -> None:
        if not 0 <= rank < self.num_ranks:
            raise ValueError(
                f"{label}={rank} out of range [0, {self.num_ranks})")

    def _check_offset(self, local_offset: int) -> None:
        if local_offset < 0 or local_offset >= max(self.bytes_per_rank, 1):
            raise ValueError(
                f"local_offset={local_offset} out of range "
                f"[0, {self.bytes_per_rank})")


@dataclass(frozen=True)
class SymmetricShardSpec:
    """A :class:`SymmetricHeap` paired with a placement mode.

    ``replicated`` — every rank holds the full logical bank at the symmetric
    offset; a one-sided read of any global offset can be served from the local
    copy (and ``put``/``atomic`` to all peers fans out).

    ``partitioned`` — each rank owns one contiguous slice; the logical bank is
    the concatenation of per-rank slices and a remote access targets the owning
    rank computed from the global offset.
    """

    heap: SymmetricHeap
    mode: str = "partitioned"

    def __post_init__(self) -> None:
        if self.mode not in SYMMETRIC_MODES:
            raise ValueError(
                f"SymmetricShardSpec.mode must be one of {SYMMETRIC_MODES}; "
                f"got {self.mode!r}")

    @property
    def is_replicated(self) -> bool:
        return self.mode == "replicated"

    @property
    def logical_bytes(self) -> int:
        """Bytes of the logical bank the spec represents.

        Replicated: one bank's worth (``bytes_per_rank``) — every rank holds
        the same bank.  Partitioned: the concatenation of all shards.
        """
        if self.is_replicated:
            return self.heap.bytes_per_rank
        return self.heap.total_bytes

    def shard_bytes(self, rank: int) -> int:
        """Bytes physically resident on ``rank``.

        Both modes store ``bytes_per_rank`` per rank — replicated holds the
        full bank on every rank, partitioned holds one slice.  Validates the
        rank against the heap either way.
        """
        self.heap._check_rank(rank, "rank")
        return self.heap.bytes_per_rank

    def global_to_rank(self, global_offset: int) -> tuple[int, int]:
        """Map a logical-bank offset to ``(rank, local_offset)``.

        Replicated: any rank serves the read, so by convention rank 0 with the
        offset unchanged.  Partitioned: the owning rank's contiguous slice.
        """
        if self.is_replicated:
            if not 0 <= global_offset < self.heap.bytes_per_rank:
                raise ValueError(
                    f"global_offset={global_offset} out of range "
                    f"[0, {self.heap.bytes_per_rank}) for a replicated bank")
            return 0, int(global_offset)
        return self.heap.global_to_local(global_offset)

    def as_metadata_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "logical_bytes": self.logical_bytes,
            "heap": self.heap.as_metadata_dict(),
        }


__all__ = [
    "SYMMETRIC_MODES",
    "SymmetricHeap",
    "SymmetricShardSpec",
]
