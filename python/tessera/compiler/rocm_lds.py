"""tessera.compiler.rocm_lds — software-pipeline depth + arch-keyed LDS
conflict-avoidance layout selection (B2, from the AMD Gluon GEMM tutorial).

Two compiler-visible contracts ported from the AMD Gluon GEMM tutorial (the
layout-explicit Triton dialect):

1.  **Software pipelining** — an N-stage matmul pipeline allocates N LDS
    (shared-memory) buffers so the global→LDS copy of stage ``k+1`` overlaps
    the MFMA compute of stage ``k``.  The tutorial's ``v4`` GEMM is
    double-buffered (2 stages), ``v5`` is triple-buffered (3 stages).
    :class:`SoftwarePipeline` records the depth and the per-tile LDS-buffer
    cost.

2.  **Bank-conflict-free LDS layout** — LDS is banked; two lanes touching the
    same bank in the same cycle serialize.  The Gluon tutorial avoids conflicts
    two ways:

    * an **XOR swizzle** (general) — ``col`` is XOR-ed by a row-derived phase so
      a warp's lanes land in distinct banks (:class:`SwizzledLdsLayout`); and
    * **additive padding** — widen the inner (stored) dimension by a few
      elements so successive rows shift out of the conflicting bank
      (:class:`PaddedLdsLayout`).

    The selection rule is arch-keyed: CDNA 4 (gfx950)'s ``GLOBAL_LOAD_LDS_*``
    direct-to-LDS path requires *consecutive warp-wide writes*, which an XOR
    swizzle breaks up — so AMD prefers **padding** on that path.  Everywhere
    else the general **swizzle** wins.  :func:`select_lds_layout` encodes that.

Pure data — no kernel import.  Leaf-ish module; the only Tessera import is
:mod:`tessera.compiler.rocm_target` for :class:`~tessera.compiler.rocm_target.AMDArch`.
Everything flattens to plain JSON-able dicts via ``as_metadata_dict``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

from tessera.compiler.rocm_target import AMDArch

# ── helpers ──────────────────────────────────────────────────────────────────


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# ── software pipeline ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SoftwarePipeline:
    """An N-stage software pipeline over the matmul K-loop (Gluon GEMM tutorial).

    ``stages`` is the pipeline depth.  An N-stage pipeline allocates N LDS
    buffers so the next stage's global→LDS load overlaps the current stage's
    MFMA compute.  The tutorial's ``v4`` GEMM uses ``stages=2`` (double
    buffering), ``v5`` uses ``stages=3`` (triple buffering).
    """

    stages: int = 2

    def __post_init__(self) -> None:
        if self.stages < 1:
            raise ValueError(
                f"SoftwarePipeline: stages must be >= 1, got {self.stages}"
            )

    def lds_buffers(self, tile_bytes: int) -> int:
        """Total LDS bytes an N-stage pipeline needs for a ``tile_bytes`` tile.

        N-buffered: each of the ``stages`` in flight owns one copy of the tile,
        so the cost is ``stages * tile_bytes``.
        """
        if tile_bytes < 0:
            raise ValueError(
                f"SoftwarePipeline.lds_buffers: tile_bytes must be >= 0, "
                f"got {tile_bytes}"
            )
        return self.stages * tile_bytes

    @property
    def is_double_buffered(self) -> bool:
        return self.stages >= 2

    @property
    def is_triple(self) -> bool:
        return self.stages >= 3

    def as_metadata_dict(self) -> dict[str, Any]:
        return {
            "kind": "software_pipeline",
            "stages": self.stages,
            "is_double_buffered": self.is_double_buffered,
            "is_triple": self.is_triple,
        }


# ── LDS layout strategies ────────────────────────────────────────────────────


@dataclass(frozen=True)
class SwizzledLdsLayout:
    """XOR-swizzle LDS layout (Gluon ``SwizzledSharedLayout``).

    A column index is XOR-ed by a *phase* derived from the row, so a warp's
    lanes (which share a row but vary the column) land in distinct LDS banks and
    never conflict.

    Parameters mirror the Gluon swizzle encoding:

    * ``vec`` — the contiguous vectorization width (elements XOR-swizzled as a
      unit; a power of two, e.g. 8 for a 128-bit f16 load).
    * ``per_phase`` — how many consecutive rows share one phase.
    * ``max_phase`` — number of distinct phases (cycles before the swizzle
      repeats); a power of two so the XOR stays within ``[0, max_phase)``.
    * ``order`` — the (fast, slow) axis order; ``(1, 0)`` = row-major inner dim
      fastest (the default for an MxK / KxN tile).
    """

    vec: int
    per_phase: int
    max_phase: int
    order: tuple[int, int] = (1, 0)

    def __post_init__(self) -> None:
        if self.vec < 1:
            raise ValueError(
                f"SwizzledLdsLayout: vec must be >= 1, got {self.vec}"
            )
        if not _is_power_of_two(self.vec):
            raise ValueError(
                f"SwizzledLdsLayout: vec must be a power of two, got {self.vec}"
            )
        if self.per_phase < 1:
            raise ValueError(
                f"SwizzledLdsLayout: per_phase must be >= 1, got {self.per_phase}"
            )
        if self.max_phase < 1:
            raise ValueError(
                f"SwizzledLdsLayout: max_phase must be >= 1, got {self.max_phase}"
            )
        if not _is_power_of_two(self.max_phase):
            raise ValueError(
                f"SwizzledLdsLayout: max_phase must be a power of two, "
                f"got {self.max_phase}"
            )
        if sorted(self.order) != [0, 1]:
            raise ValueError(
                f"SwizzledLdsLayout: order must be a permutation of (0, 1); "
                f"got {self.order}"
            )

    def swizzled_col(self, row: int, col: int) -> int:
        """XOR-swizzle ``col`` for ``row`` so warp lanes hit distinct banks.

        ``phase = (row // per_phase) % max_phase``;
        ``((col // vec) ^ phase) * vec + (col % vec)`` — i.e. the vector-block
        index ``col // vec`` is XOR-ed by the row phase, then the intra-vector
        offset ``col % vec`` is restored.
        """
        if row < 0 or col < 0:
            raise ValueError(
                f"SwizzledLdsLayout.swizzled_col: row/col must be >= 0, "
                f"got row={row}, col={col}"
            )
        phase = (row // self.per_phase) % self.max_phase
        return ((col // self.vec) ^ phase) * self.vec + (col % self.vec)

    def as_metadata_dict(self) -> dict[str, Any]:
        return {
            "kind": "lds_layout",
            "strategy": "swizzle",
            "vec": self.vec,
            "per_phase": self.per_phase,
            "max_phase": self.max_phase,
            "order": list(self.order),
        }


@dataclass(frozen=True)
class PaddedLdsLayout:
    """Additive-padding LDS layout (Gluon ``PaddedSharedLayout``).

    Widen the stored inner dimension by ``pad_elems`` extra elements so each
    logical row starts at a shifted bank, breaking the conflicting stride.
    Preferred over swizzle on the CDNA 4 ``GLOBAL_LOAD_LDS`` path, which needs
    consecutive warp-wide writes that a swizzle would scatter.

    * ``pad_elems`` — extra elements appended to each row (>= 1; typically a
      small constant chosen so ``inner_dim + pad_elems`` is coprime with the
      bank count).
    * ``inner_dim`` — the logical (unpadded) inner dimension the pad applies to.
    """

    pad_elems: int
    inner_dim: int

    def __post_init__(self) -> None:
        if self.pad_elems < 1:
            raise ValueError(
                f"PaddedLdsLayout: pad_elems must be >= 1, got {self.pad_elems}"
            )
        if self.inner_dim < 1:
            raise ValueError(
                f"PaddedLdsLayout: inner_dim must be >= 1, got {self.inner_dim}"
            )

    def padded_stride(self, cols: int) -> int:
        """Physical row stride for a logical ``cols``-wide row: ``cols + pad_elems``."""
        if cols < 0:
            raise ValueError(
                f"PaddedLdsLayout.padded_stride: cols must be >= 0, got {cols}"
            )
        return cols + self.pad_elems

    def as_metadata_dict(self) -> dict[str, Any]:
        return {
            "kind": "lds_layout",
            "strategy": "pad",
            "pad_elems": self.pad_elems,
            "inner_dim": self.inner_dim,
        }


#: Union of the two conflict-avoidance LDS layout strategies.
LdsLayout = Union[SwizzledLdsLayout, PaddedLdsLayout]


# ── arch-keyed selection ─────────────────────────────────────────────────────


def select_lds_layout(
    arch: AMDArch,
    *,
    global_to_lds: bool,
    vec: int = 8,
    inner_dim: int = 64,
) -> LdsLayout:
    """Pick the conflict-avoidance LDS layout for ``arch`` (Gluon GEMM rule).

    The rule, from the AMD Gluon GEMM tutorial:

    * On **CDNA 4 (gfx950)** with the direct ``GLOBAL_LOAD_LDS_*`` global→LDS
      path (``global_to_lds=True``), the hardware requires *consecutive
      warp-wide writes* into LDS.  An XOR swizzle scatters those writes and
      makes the direct-to-LDS load inefficient, so AMD uses **additive
      padding** there → :class:`PaddedLdsLayout`.
    * Everywhere else — any other arch, or gfx950 when *not* using the
      global→LDS direct path — the general **XOR swizzle** is the better
      conflict-avoidance scheme → :class:`SwizzledLdsLayout`.

    ``vec`` is the swizzle vectorization width (used only on the swizzle path);
    ``inner_dim`` is the logical inner dimension (used only on the pad path).
    """
    if arch is AMDArch.GFX_950 and global_to_lds:
        # CDNA 4 GLOBAL_LOAD_LDS path: pad to keep consecutive warp-wide writes.
        return PaddedLdsLayout(pad_elems=4, inner_dim=inner_dim)
    # General path: XOR swizzle. per_phase/max_phase sized so a 128-bit-class
    # vector tile cycles through distinct banks before repeating.
    return SwizzledLdsLayout(vec=vec, per_phase=1, max_phase=8)


__all__ = [
    "SoftwarePipeline",
    "SwizzledLdsLayout",
    "PaddedLdsLayout",
    "LdsLayout",
    "select_lds_layout",
]
