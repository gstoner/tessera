"""Reference lane maps for NVIDIA cooperative-matrix fragment lowering.

The Tile IR never exposes these register assignments.  They live here as an
executable oracle for the NVIDIA lowering, derived from Tessera's on-silicon
``mma.sync.m16n8k16.row.col`` PTX path.  A lowering may use a different load
mechanism (global, shared, or ``ldmatrix``), but it must produce these logical
element pairs for every lane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FragmentRole = Literal["a", "b"]


@dataclass(frozen=True)
class PackedPair:
    """Two contiguous f16 elements carried by one 32-bit fragment register."""

    first: tuple[int, int]
    second: tuple[int, int]


def sm120_m16n8k16_f16_pairs(role: FragmentRole, lane: int) -> tuple[PackedPair, ...]:
    """Return the logical f16 pairs for one ``mma.sync`` lane.

    A coordinates are ``(m, k)`` in row-major storage.  B coordinates are
    ``(k, n)`` in column-major storage.  Each returned pair is contiguous in
    that storage order and therefore corresponds to one ``.b32`` load.

    This is the exact mapping used by the CUDA/PTX execution oracle.  The
    accumulator mapping is deliberately absent: the current canonical MLIR
    f16-accumulator ABI has not yet been executed on hardware, so guessing its
    physical result order would undermine the contract this module protects.
    """
    if not 0 <= lane < 32:
        raise ValueError(f"mma.sync lane must be in [0, 32); got {lane}")
    gid, tig = lane >> 2, lane & 3
    offset = 2 * tig
    if role == "a":
        return (
            PackedPair((gid, offset), (gid, offset + 1)),
            PackedPair((gid + 8, offset), (gid + 8, offset + 1)),
            PackedPair((gid, offset + 8), (gid, offset + 9)),
            PackedPair((gid + 8, offset + 8), (gid + 8, offset + 9)),
        )
    if role == "b":
        return (
            PackedPair((offset, gid), (offset + 1, gid)),
            PackedPair((offset + 8, gid), (offset + 9, gid)),
        )
    raise ValueError(f"sm_120 m16n8k16 f16 supports only A/B packs; got {role!r}")
