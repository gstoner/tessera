"""Register-distribution tile layouts as first-class typed objects (Gluon model).

AMD Gluon (the layout-explicit dialect riding on Triton) treats a tile's
*register distribution* — how a tensor's elements are spread across registers,
lanes, warps, and CTAs — as a **typed object on the value**, not an implicit
codegen choice.  Converting between two layouts (``convert_layout``) is then a
real, costed operation: free when the layouts are bit-compatible, and a
shared-memory round-trip otherwise.

This module ports that model as compiler-visible contract metadata — pure data,
no kernel import.  It is a LEAF module (stdlib only, no tessera imports) so the
audit registry, ``op_catalog``, and the runtime can all import it without a
cycle.  Everything flattens to plain JSON-able dicts via ``as_metadata_dict``.

Four layout types:

* :class:`BlockedLayout` — the canonical Gluon register-distribution layout:
  ``size_per_thread`` × ``threads_per_warp`` × ``warps_per_cta`` (with an
  ``order`` permutation giving the fastest-varying axis).  ``block_shape`` is the
  elementwise product — the tile shape one CTA covers.

* :class:`SliceLayout` — a parent layout with one dimension dropped, for 1-D
  offset / index tensors derived from a 2-D layout (Gluon's ``SliceEncoding``).

* :class:`LinearLayout` — the **bit-basis / linear-layout** representation: a
  layout expressed as a linear map over the *bits* of the element index, with
  separate register / lane / warp / block bases.  Because the map is linear
  (XOR over bit-vectors), ``transpose`` / ``reshape`` / ``permute`` / ``split`` /
  ``join`` are all **metadata-only** rewrites of the basis vectors — no data
  movement.  This is the key abstraction that makes ``convert_layout`` between
  two bit-permuted layouts free.

* A cost model — :func:`convert_cost` + :class:`ConvertLayout` — giving the
  data-movement cost of a ``convert_layout`` between two layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

# ── helpers ──────────────────────────────────────────────────────────────────


def _is_permutation(order: tuple[int, ...], rank: int) -> bool:
    """True iff ``order`` is a permutation of ``range(rank)``."""
    return sorted(order) == list(range(rank))


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# ── BlockedLayout ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BlockedLayout:
    """Gluon's register-distribution layout for one CTA's tile.

    ``size_per_thread`` — elements each thread holds in registers, per axis.
    ``threads_per_warp`` — lane grid, per axis (product == warp size).
    ``warps_per_cta`` — warp grid within the CTA, per axis.
    ``order`` — a permutation of ``range(rank)`` giving the memory order
    (``order[0]`` is the fastest-varying axis).

    All three vectors share the same rank.  :attr:`block_shape` is their
    elementwise product — the per-CTA tile shape.
    """

    size_per_thread: tuple[int, ...]
    threads_per_warp: tuple[int, ...]
    warps_per_cta: tuple[int, ...]
    order: tuple[int, ...]

    def __post_init__(self) -> None:
        ranks = {
            len(self.size_per_thread),
            len(self.threads_per_warp),
            len(self.warps_per_cta),
            len(self.order),
        }
        if len(ranks) != 1:
            raise ValueError(
                "BlockedLayout: size_per_thread, threads_per_warp, warps_per_cta "
                f"and order must share one rank; got ranks {sorted(ranks)} for "
                f"{self.size_per_thread}, {self.threads_per_warp}, "
                f"{self.warps_per_cta}, order={self.order}")
        rank = len(self.order)
        for name, vec in (
            ("size_per_thread", self.size_per_thread),
            ("threads_per_warp", self.threads_per_warp),
            ("warps_per_cta", self.warps_per_cta),
        ):
            if any(v <= 0 for v in vec):
                raise ValueError(
                    f"BlockedLayout: {name} must be all-positive; got {vec}")
        if not _is_permutation(self.order, rank):
            raise ValueError(
                f"BlockedLayout: order must be a permutation of range({rank}); "
                f"got {self.order}")

    @property
    def rank(self) -> int:
        return len(self.order)

    @property
    def block_shape(self) -> tuple[int, ...]:
        """Per-CTA tile shape = size_per_thread * threads_per_warp * warps_per_cta."""
        return tuple(
            s * t * w
            for s, t, w in zip(
                self.size_per_thread, self.threads_per_warp, self.warps_per_cta
            )
        )

    @property
    def warp_size(self) -> int:
        return prod(self.threads_per_warp)

    @property
    def warps(self) -> int:
        return prod(self.warps_per_cta)

    def as_metadata_dict(self) -> dict[str, Any]:
        return {
            "kind": "blocked",
            "size_per_thread": list(self.size_per_thread),
            "threads_per_warp": list(self.threads_per_warp),
            "warps_per_cta": list(self.warps_per_cta),
            "order": list(self.order),
            "block_shape": list(self.block_shape),
        }


# ── SliceLayout ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SliceLayout:
    """A parent layout with dimension ``dim`` dropped (Gluon ``SliceEncoding``).

    Used for the 1-D offset / index tensors derived from a 2-D tile layout — the
    row- or column-index vector of a matmul tile, for example.  No data movement:
    a slice is purely a re-projection of the parent's distribution.
    """

    dim: int
    parent: "BlockedLayout | LinearLayout | SliceLayout"

    def __post_init__(self) -> None:
        prank = _layout_rank(self.parent)
        if not (0 <= self.dim < prank):
            raise ValueError(
                f"SliceLayout: dim={self.dim} out of range for a rank-{prank} "
                f"parent layout")

    @property
    def rank(self) -> int:
        return _layout_rank(self.parent) - 1

    @property
    def block_shape(self) -> tuple[int, ...]:
        """Parent block_shape with the sliced dimension removed."""
        parent_shape = _layout_block_shape(self.parent)
        return parent_shape[: self.dim] + parent_shape[self.dim + 1:]

    def as_metadata_dict(self) -> dict[str, Any]:
        return {
            "kind": "slice",
            "dim": self.dim,
            "parent": self.parent.as_metadata_dict(),
            "block_shape": list(self.block_shape),
        }


# ── LinearLayout ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LinearLayout:
    """The bit-basis / linear-layout representation of a tile distribution.

    A linear layout maps each *bit* of the flattened element index to an output
    coordinate via a basis vector.  Bits are partitioned across four hardware
    levels — register (``reg_bases``), lane (``lane_bases``), warp
    (``warp_bases``), and block / CTA (``block_bases``).  Each ``*_bases`` is a
    tuple of basis vectors; every basis vector is an int tuple of length
    ``len(shape)`` (one output coordinate per logical axis).

    Because the map is linear over bit-vectors (XOR composition), the structural
    rewrites below — :meth:`transpose`, :meth:`reshape`, :meth:`permute`,
    :meth:`split`, :meth:`join` — are **metadata-only**: they permute / re-stack
    the basis vectors and return a *new* :class:`LinearLayout` with no data
    movement.  This is what makes a ``convert_layout`` between two layouts that
    differ only by a bit-permutation cost zero.
    """

    reg_bases: tuple[tuple[int, ...], ...]
    lane_bases: tuple[tuple[int, ...], ...]
    warp_bases: tuple[tuple[int, ...], ...]
    block_bases: tuple[tuple[int, ...], ...]
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.shape or any(d <= 0 for d in self.shape):
            raise ValueError(
                f"LinearLayout: shape must be non-empty all-positive; got {self.shape}")
        rank = len(self.shape)
        for name, bases in self._levels():
            for vec in bases:
                if len(vec) != rank:
                    raise ValueError(
                        f"LinearLayout: every {name} basis vector must have length "
                        f"{rank} (== len(shape)); got {vec}")

    # ── introspection ────────────────────────────────────────────────────────
    def _levels(self) -> tuple[tuple[str, tuple[tuple[int, ...], ...]], ...]:
        return (
            ("reg_bases", self.reg_bases),
            ("lane_bases", self.lane_bases),
            ("warp_bases", self.warp_bases),
            ("block_bases", self.block_bases),
        )

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def block_shape(self) -> tuple[int, ...]:
        return self.shape

    @property
    def num_bits(self) -> int:
        """Total number of index bits carried across all four levels."""
        return (
            len(self.reg_bases)
            + len(self.lane_bases)
            + len(self.warp_bases)
            + len(self.block_bases)
        )

    @property
    def shape_bits(self) -> int:
        """log2 of the total element count (the number of bits the layout must
        span to be a bijection over the index space)."""
        total = prod(self.shape)
        return total.bit_length() - 1

    def all_bases(self) -> tuple[tuple[int, ...], ...]:
        return self.reg_bases + self.lane_bases + self.warp_bases + self.block_bases

    def is_valid(self) -> bool:
        """True iff the bases form a bijection over the ``log2(prod(shape))`` bit
        index space.

        Two structural requirements: (1) the total element count must be a power
        of two (so it is bit-addressable), and (2) the basis vectors, read as
        rows of a matrix over GF(2), must be linearly independent and exactly
        ``shape_bits`` of them — i.e. they span the whole index space without
        collapsing two indices onto one coordinate.
        """
        total = prod(self.shape)
        if not _is_power_of_two(total):
            return False
        if self.num_bits != self.shape_bits:
            return False
        return _bases_independent_gf2(self.all_bases())

    # ── free (metadata-only) operations ──────────────────────────────────────
    def transpose(self, perm: tuple[int, ...]) -> "LinearLayout":
        """Reorder the *output coordinate axes* by ``perm`` (free — relabels each
        basis vector's coordinates and the shape).  ``transpose`` is the rank-2
        case of :meth:`permute`."""
        return self.permute(perm)

    def permute(self, perm: tuple[int, ...]) -> "LinearLayout":
        """Permute the output coordinate axes by ``perm`` (free).  Each basis
        vector's components are reordered and ``shape`` follows."""
        if not _is_permutation(perm, self.rank):
            raise ValueError(
                f"LinearLayout.permute: perm must be a permutation of "
                f"range({self.rank}); got {perm}")
        relabel = lambda vec: tuple(vec[p] for p in perm)
        return LinearLayout(
            reg_bases=tuple(relabel(v) for v in self.reg_bases),
            lane_bases=tuple(relabel(v) for v in self.lane_bases),
            warp_bases=tuple(relabel(v) for v in self.warp_bases),
            block_bases=tuple(relabel(v) for v in self.block_bases),
            shape=tuple(self.shape[p] for p in perm),
        )

    def reshape(self, new_shape: tuple[int, ...]) -> "LinearLayout":
        """Reinterpret the layout over ``new_shape`` (free) when the total element
        count is unchanged.

        A reshape collapses the per-axis output coordinates of every basis vector
        into a single flat offset, then re-expands that offset against
        ``new_shape``.  Because the map is linear, this is a pure relabel — no
        data moves.
        """
        if prod(new_shape) != prod(self.shape):
            raise ValueError(
                f"LinearLayout.reshape: total size must be preserved; "
                f"{self.shape} (={prod(self.shape)}) -> {new_shape} "
                f"(={prod(new_shape)})")
        if any(d <= 0 for d in new_shape) or not new_shape:
            raise ValueError(
                f"LinearLayout.reshape: new_shape must be non-empty all-positive; "
                f"got {new_shape}")

        def reproject(vec: tuple[int, ...]) -> tuple[int, ...]:
            flat = _to_flat(vec, self.shape)
            return _from_flat(flat, new_shape)

        return LinearLayout(
            reg_bases=tuple(reproject(v) for v in self.reg_bases),
            lane_bases=tuple(reproject(v) for v in self.lane_bases),
            warp_bases=tuple(reproject(v) for v in self.warp_bases),
            block_bases=tuple(reproject(v) for v in self.block_bases),
            shape=tuple(new_shape),
        )

    def split(self, level: str = "reg_bases") -> "tuple[LinearLayout, LinearLayout]":
        """Split off the top bit of ``level`` into a new single-bit layout (free).

        Returns ``(low, high)`` where ``high`` carries the most-significant basis
        vector of ``level`` and ``low`` carries the rest.  The inverse of
        :meth:`join`.  Useful for peeling a register tile in two.
        """
        bases = dict(self._levels())
        if level not in bases:
            raise ValueError(
                f"LinearLayout.split: level must be one of {list(bases)}; got {level!r}")
        src = bases[level]
        if not src:
            raise ValueError(
                f"LinearLayout.split: {level} has no basis bits to split")
        empty: tuple[tuple[int, ...], ...] = ()
        low = LinearLayout(
            reg_bases=src[:-1] if level == "reg_bases" else self.reg_bases,
            lane_bases=src[:-1] if level == "lane_bases" else self.lane_bases,
            warp_bases=src[:-1] if level == "warp_bases" else self.warp_bases,
            block_bases=src[:-1] if level == "block_bases" else self.block_bases,
            shape=self.shape,
        )
        top: tuple[tuple[int, ...], ...] = (src[-1],)
        high = LinearLayout(
            reg_bases=top if level == "reg_bases" else empty,
            lane_bases=top if level == "lane_bases" else empty,
            warp_bases=top if level == "warp_bases" else empty,
            block_bases=top if level == "block_bases" else empty,
            shape=self.shape,
        )
        return low, high

    def join(self, other: "LinearLayout", level: str = "reg_bases") -> "LinearLayout":
        """Concatenate ``other``'s basis bits below ``self``'s (free).

        The inverse of :meth:`split`: appends ``other``'s bases (at every level)
        below ``self``'s.  ``shape`` must match.  ``level`` names the level
        :meth:`split` peeled from and is validated for symmetry.
        """
        if other.shape != self.shape:
            raise ValueError(
                f"LinearLayout.join: shapes must match; {self.shape} vs {other.shape}")
        if level not in dict(self._levels()):
            raise ValueError(
                f"LinearLayout.join: level must be one of "
                f"{[n for n, _ in self._levels()]}; got {level!r}")
        return LinearLayout(
            reg_bases=self.reg_bases + other.reg_bases,
            lane_bases=self.lane_bases + other.lane_bases,
            warp_bases=self.warp_bases + other.warp_bases,
            block_bases=self.block_bases + other.block_bases,
            shape=self.shape,
        )

    def as_metadata_dict(self) -> dict[str, Any]:
        return {
            "kind": "linear",
            "reg_bases": [list(v) for v in self.reg_bases],
            "lane_bases": [list(v) for v in self.lane_bases],
            "warp_bases": [list(v) for v in self.warp_bases],
            "block_bases": [list(v) for v in self.block_bases],
            "shape": list(self.shape),
            "num_bits": self.num_bits,
            "block_shape": list(self.block_shape),
        }


# ── flat-offset reprojection (linear reshape kernel) ─────────────────────────


def _to_flat(coord: tuple[int, ...], shape: tuple[int, ...]) -> int:
    """Row-major flatten an output coordinate against ``shape``."""
    flat = 0
    for c, d in zip(coord, shape):
        flat = flat * d + c
    return flat


def _from_flat(flat: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Row-major un-flatten a flat offset against ``shape``."""
    coord: list[int] = [0] * len(shape)
    for i in range(len(shape) - 1, -1, -1):
        coord[i] = flat % shape[i]
        flat //= shape[i]
    return tuple(coord)


def _bases_independent_gf2(bases: tuple[tuple[int, ...], ...]) -> bool:
    """True iff the basis vectors are linearly independent over GF(2).

    Each basis vector is flattened to a single bit-pattern (its flat offset in a
    power-of-two-padded index space) and Gaussian-eliminated over GF(2).  Linear
    independence == the layout does not collapse two distinct index bits onto the
    same coordinate, which is the bijection condition.
    """
    rows = [_basis_to_int(v) for v in bases]
    if any(r == 0 for r in rows):
        return False
    rank = 0
    for bit in reversed(range(max((r.bit_length() for r in rows), default=0))):
        pivot = next((r for r in rows if (r >> bit) & 1), 0)
        if not pivot:
            continue
        rank += 1
        rows = [r ^ pivot if (r >> bit) & 1 else r for r in rows]
    return rank == len(bases)


def _basis_to_int(vec: tuple[int, ...]) -> int:
    """Encode a basis vector as one integer by interleaving each coordinate's
    bits into disjoint lanes (so distinct coordinates never alias)."""
    out = 0
    for axis, c in enumerate(vec):
        out |= c << (axis * 32)
    return out


# ── dispatch over the layout union ───────────────────────────────────────────

Layout = "BlockedLayout | SliceLayout | LinearLayout"


def _layout_rank(layout: Any) -> int:
    return int(layout.rank)


def _layout_block_shape(layout: Any) -> tuple[int, ...]:
    return tuple(layout.block_shape)


# ── cost model ────────────────────────────────────────────────────────────────


def _bit_permutation_equivalent(src: LinearLayout, dst: LinearLayout) -> bool:
    """True iff ``src`` and ``dst`` are the same linear layout up to a free
    bit-permutation — same shape, same multiset of basis vectors at each level.

    Two linear layouts related only by reordering basis bits describe the same
    register distribution; a ``convert_layout`` between them is a no-op relabel.
    """
    if src.shape != dst.shape:
        return False
    for (name, a), (_, b) in zip(src._levels(), dst._levels()):
        if sorted(a) != sorted(b):
            return False
    return True


def convert_cost(src: Any, dst: Any) -> int:
    """Data-movement cost of a ``convert_layout`` from ``src`` to ``dst``.

    Returns ``0`` when the conversion is free:
      * the two layouts are structurally identical, or
      * both are :class:`LinearLayout` and differ only by a bit-permutation
        (same shape + same multiset of bases per level).

    Otherwise the conversion must round-trip the tile through shared memory, so
    the cost is the number of elements moved == ``prod(block_shape)``.
    """
    if src == dst:
        return 0
    if isinstance(src, LinearLayout) and isinstance(dst, LinearLayout):
        if _bit_permutation_equivalent(src, dst):
            return 0
    return prod(_layout_block_shape(dst))


@dataclass(frozen=True)
class ConvertLayout:
    """A costed ``convert_layout`` op: source layout, destination layout, and the
    data-movement cost computed by :func:`convert_cost`."""

    src: "BlockedLayout | SliceLayout | LinearLayout"
    dst: "BlockedLayout | SliceLayout | LinearLayout"

    @property
    def cost(self) -> int:
        return convert_cost(self.src, self.dst)

    @property
    def is_free(self) -> bool:
        return self.cost == 0

    def as_metadata_dict(self) -> dict[str, Any]:
        return {
            "kind": "convert_layout",
            "src": self.src.as_metadata_dict(),
            "dst": self.dst.as_metadata_dict(),
            "cost": self.cost,
            "is_free": self.is_free,
        }
