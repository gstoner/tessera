"""Mock-collective halo transport runtime — Sub-4 (2026-05-20).

The C++ ``HaloTransportLowerPass`` emits a triple of
``halo.pack`` / ``halo.transport`` / ``halo.unpack`` ops per (axis, side).
This module provides a Python-side reference implementation of that
triple suitable for testing distributed correctness without NCCL/RCCL.
The Python harness composes the same primitives the runtime adapter
will compose; the numerical contract here is what each backend
implementation must reproduce.

API
---

``halo_pack(field, axis, side, width)`` returns the contiguous ghost
slab (shape: same as field except the ``axis``-th dim becomes ``width``).

``halo_transport_ring(packs, *, axes=None)`` runs a 1D ring exchange:
for each rank `r`, ``packs[r][(axis, side)]`` is sent to rank
``(r + 1) % nranks`` when side == "hi", or rank
``(r - 1) % nranks`` when side == "lo".  Returns ``received[rank]``
dict of received slabs.

``halo_unpack(field, received, axis, side, width)`` writes the received
slab into the field's ghost region.  Convention:
  side == "lo" — the received slab fills the ``[0, width)`` prefix on
                 ``axis`` (treating the existing ``[0, width)`` as the
                 ghost region to be overwritten).
  side == "hi" — the received slab fills the ``[N - width, N)`` suffix.

``halo_exchange_ring(field, mesh_size, axes_widths)`` is the
convenience composition: pack → ring-transport → unpack for each
(axis, width) declared in ``axes_widths``.

The mock collective uses a periodic (wrap-around) ring topology so it
matches the ``peer_rule = "neg1"|"pos1"`` ABI the C++ transport pass
emits.  Real NCCL/RCCL adapters target the same packing convention.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Pack / unpack
# --------------------------------------------------------------------------- #


def halo_pack(field: np.ndarray, *, axis: int, side: str,
              width: int) -> np.ndarray:
    """Gather a ghost slab from one side of one axis.

    For side == "lo", the slab is the ``[0, width)`` slice on ``axis``;
    for side == "hi", it is the ``[N - width, N)`` slice.  This matches
    the convention the C++ pass emits — pack from the boundary to send
    to the neighbour who consumes it on their opposite boundary.
    """
    if width <= 0:
        raise ValueError(f"width must be positive; got {width}")
    if field.shape[axis] < width:
        raise ValueError(
            f"axis {axis} dim {field.shape[axis]} smaller than width {width}"
        )
    slicer: list = [slice(None)] * field.ndim
    if side == "lo":
        slicer[axis] = slice(0, width)
    elif side == "hi":
        slicer[axis] = slice(field.shape[axis] - width, field.shape[axis])
    else:
        raise ValueError(f"side must be 'lo' or 'hi'; got {side!r}")
    return field[tuple(slicer)].copy()


def halo_unpack(field: np.ndarray, slab: np.ndarray, *,
                axis: int, side: str, width: int) -> np.ndarray:
    """Write the received slab into the field's ghost region.

    Returns a NEW array; the input ``field`` is not modified.  This is
    the same functional shape the IR ops use (every op returns a new
    tensor SSA value).
    """
    if slab.shape[axis] != width:
        raise ValueError(
            f"slab dim on axis {axis} = {slab.shape[axis]} does not match "
            f"width {width}"
        )
    out = field.copy()
    slicer: list = [slice(None)] * field.ndim
    if side == "lo":
        slicer[axis] = slice(0, width)
    elif side == "hi":
        slicer[axis] = slice(field.shape[axis] - width, field.shape[axis])
    else:
        raise ValueError(f"side must be 'lo' or 'hi'; got {side!r}")
    out[tuple(slicer)] = slab
    return out


# --------------------------------------------------------------------------- #
# Ring transport (mock collective)
# --------------------------------------------------------------------------- #


def halo_transport_ring(
    packs: Sequence[Mapping[tuple[int, str], np.ndarray]],
) -> list[dict[tuple[int, str], np.ndarray]]:
    """Periodic ring exchange across ``len(packs)`` ranks.

    ``packs[r][(axis, side)]`` is the slab rank ``r`` will *send*.  The
    convention (matching the C++ pass's ``peer_rule``):

      side == "hi"  → send to (r + 1) % nranks, received as side "lo"
      side == "lo"  → send to (r - 1) % nranks, received as side "hi"

    Returns ``received[r][(axis, side)]`` — the slab rank ``r`` should
    *unpack* on side ``side`` of axis ``axis``.
    """
    nranks = len(packs)
    if nranks == 0:
        return []
    received: list[dict] = [dict() for _ in range(nranks)]
    for r in range(nranks):
        for (axis, side), slab in packs[r].items():
            if side == "hi":
                dest_rank = (r + 1) % nranks
                # The receiver sees this slab as their "lo" boundary
                received[dest_rank][(axis, "lo")] = slab
            elif side == "lo":
                dest_rank = (r - 1) % nranks
                received[dest_rank][(axis, "hi")] = slab
            else:
                raise ValueError(f"unknown side {side!r}")
    return received


# --------------------------------------------------------------------------- #
# End-to-end composition
# --------------------------------------------------------------------------- #


def halo_exchange_ring(
    fields: Sequence[np.ndarray],
    *,
    axes_widths: Sequence[tuple[int, int]],
) -> list[np.ndarray]:
    """Pack → ring-transport → unpack for one collective step.

    ``axes_widths`` is a list of ``(axis, width)`` tuples — one per
    spatial axis the halo covers.  For each axis we pack both sides on
    every rank, transport, then unpack.

    Returns the updated per-rank fields with ghost regions populated.
    """
    # Per-rank pack tables.
    packs = []
    for f in fields:
        d = {}
        for axis, w in axes_widths:
            if w <= 0:
                continue
            d[(axis, "lo")] = halo_pack(f, axis=axis, side="lo", width=w)
            d[(axis, "hi")] = halo_pack(f, axis=axis, side="hi", width=w)
        packs.append(d)

    # Transport.
    received = halo_transport_ring(packs)

    # Unpack onto the corresponding ghost region.
    out = []
    for r, f in enumerate(fields):
        cur = f
        for (axis, side), slab in received[r].items():
            # Width is the slab's size on that axis.
            cur = halo_unpack(
                cur, slab, axis=axis, side=side,
                width=slab.shape[axis],
            )
        out.append(cur)
    return out


__all__ = [
    "halo_pack",
    "halo_unpack",
    "halo_transport_ring",
    "halo_exchange_ring",
]
