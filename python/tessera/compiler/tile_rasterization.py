"""Block **rasterization order** — how a 2-D tile grid is walked by a 1-D block id.

A tiled GEMM launches `grid_m x grid_n` blocks. *Which* tile each block id maps
to is free: any bijection is semantically identical, because every tile is
computed exactly once either way. It is also one of the largest levers on cache
behaviour that exists, because it decides which tiles are in flight at the same
time and therefore what stays resident. TileSight (arXiv:2607.22432) reports a
block swizzle moving measured L2 hit rate from 35% to 72% on the same kernel.

Before this module, Tessera had no such knob outside Apple: the MSL GEMM carried
an MLX-inherited hardcoded heuristic
(``apple_gemm_schedules.swizzle_log = 0 if tm <= 3 else 1``), ROCm's "swizzle"
was an unrelated LDS bank-conflict XOR, and NVIDIA/x86 had none. See
``docs/audit/compiler/TILESIGHT_ASSESSMENT.md`` §3.2.

**Why this is safe to add.** A rasterization order is a *permutation of block
ids*, not a change of arithmetic — the same tiles are computed with the same
inputs in the same precision. That makes it the rare optimization with a total,
executable, hardware-free oracle: :func:`is_bijection` enumerates the grid and
proves every tile is hit exactly once. Correctness is provable on CPU today; only
the *benefit* needs silicon. (Same shape of proof as
``matmul_opt_ladder.verify_split_k_equivalence``.)

**Default is identity.** :data:`RasterOrder.ROW_MAJOR` reproduces the existing
``m = bid / grid_n; n = bid % grid_n`` mapping exactly, so wiring this into an
emitter changes nothing until a schedule sets the knob.

Leaf module — stdlib only.
"""

from __future__ import annotations

from enum import Enum


class RasterOrder(str, Enum):
    """How a flat block id is mapped onto a 2-D `(m_tile, n_tile)` grid."""

    #: `m = bid / grid_n; n = bid % grid_n`. The identity — what every Tessera
    #: emitter does today. Consecutive blocks sweep a full N row before advancing
    #: M, so the B panel is re-read once per M row and the A tile is reused
    #: `grid_n` times. Good when `grid_n` is small; poor when it is large.
    ROW_MAJOR = "row_major"

    #: `m = bid % grid_m; n = bid / grid_m`. The transpose; the A/B roles swap.
    COLUMN_MAJOR = "column_major"

    #: Walk in panels of ``group`` M-tiles ("supergrouping"). Consecutive blocks
    #: fill a `group x grid_n` panel column-first, so the concurrent working set
    #: is roughly `group + W/group` tiles rather than `W`. This is the order
    #: Triton exposes as ``GROUP_SIZE_M`` and CUTLASS as a threadblock swizzle,
    #: and it is the one that matters in practice.
    GROUPED_M = "grouped_m"

    #: :data:`GROUPED_M` with the axes exchanged — panels of ``group`` N-tiles.
    #: Use when the N-side operand is the one worth keeping resident.
    GROUPED_N = "grouped_n"


#: The knob's choice set, for the autotuner / ``schedule.knob`` op.
RASTER_ORDER_CHOICES: tuple[str, ...] = tuple(o.value for o in RasterOrder)

#: Group sizes worth sweeping. 1 makes GROUPED_M/N degenerate to ROW/COLUMN_MAJOR,
#: which keeps the identity reachable from every point in the search space.
RASTER_GROUP_CHOICES: tuple[int, ...] = (1, 2, 4, 8, 16)

# NOTE: Z-order / Morton is deliberately absent. It is a clean bijection only on
# power-of-two square grids; on a general grid it needs either a padded launch
# with an out-of-range guard (changing the block count) or an iterative skip
# (not expressible as the closed-form snippet an emitter needs). Grouped-M is
# both the standard industry choice and the one that emits cleanly. Add Morton
# only with a real measurement showing grouped-M leaves something on the table.


class RasterError(ValueError):
    """An invalid rasterization request — bad order, group, or grid."""


def _validate(grid_m: int, grid_n: int, group: int) -> None:
    if grid_m < 1 or grid_n < 1:
        raise RasterError(
            f"grid dims must be >= 1, got grid_m={grid_m}, grid_n={grid_n}")
    if group < 1:
        raise RasterError(f"group must be >= 1, got {group}")


def num_blocks(grid_m: int, grid_n: int) -> int:
    """Total blocks in the grid. Rasterization never changes this — it is a
    permutation, so the launch geometry is identical for every order."""
    _validate(grid_m, grid_n, 1)
    return grid_m * grid_n


def remap(order: RasterOrder | str, bid: int, grid_m: int, grid_n: int, *,
          group: int = 1) -> tuple[int, int]:
    """Map flat block id ``bid`` to its ``(m_tile, n_tile)`` coordinate.

    This is the **reference** implementation; :func:`emit_c` generates the C form
    of the same map, and :func:`is_bijection` proves the map is a permutation.
    Raises :class:`RasterError` on an out-of-range ``bid`` rather than returning a
    silently wrong tile.
    """
    order = RasterOrder(order)
    _validate(grid_m, grid_n, group)
    total = grid_m * grid_n
    if not 0 <= bid < total:
        raise RasterError(
            f"block id {bid} out of range for a {grid_m}x{grid_n} grid "
            f"({total} blocks)")

    if order is RasterOrder.ROW_MAJOR:
        return bid // grid_n, bid % grid_n
    if order is RasterOrder.COLUMN_MAJOR:
        return bid % grid_m, bid // grid_m
    if order is RasterOrder.GROUPED_M:
        return _grouped(bid, grid_m, grid_n, group)
    if order is RasterOrder.GROUPED_N:
        # The transpose: group along N by solving the mirrored problem and
        # swapping the result back.
        n_tile, m_tile = _grouped(bid, grid_n, grid_m, group)
        return m_tile, n_tile
    raise RasterError(f"unhandled rasterization order {order!r}")


def _grouped(bid: int, grid_major: int, grid_minor: int,
             group: int) -> tuple[int, int]:
    """Supergrouped traversal: panels of ``group`` major-axis tiles, each filled
    minor-axis-first.

    The final panel may be short (``grid_major`` need not divide by ``group``);
    ``panel_rows`` handles that, which is what keeps the map a bijection on
    ragged grids — the failure mode a naive ``bid % group`` would introduce.
    """
    per_panel = group * grid_minor
    panel = bid // per_panel
    first_major = panel * group
    panel_rows = min(grid_major - first_major, group)
    within = bid % per_panel
    return first_major + within % panel_rows, within // panel_rows


def is_bijection(order: RasterOrder | str, grid_m: int, grid_n: int, *,
                 group: int = 1) -> bool:
    """**The oracle.** True iff the order maps the `grid_m x grid_n` block ids
    onto the tile set exactly once each — i.e. every tile is computed, none
    twice.

    Total and exact by enumeration (no sampling), and hardware-free: a
    rasterization order can be proven correct on CPU before any GPU sees it.
    Keep grids modest — this is O(grid_m x grid_n).
    """
    _validate(grid_m, grid_n, group)
    seen: set[tuple[int, int]] = set()
    for bid in range(grid_m * grid_n):
        coord = remap(order, bid, grid_m, grid_n, group=group)
        m_tile, n_tile = coord
        if not (0 <= m_tile < grid_m and 0 <= n_tile < grid_n):
            return False
        if coord in seen:
            return False
        seen.add(coord)
    return len(seen) == grid_m * grid_n


def working_set_tiles(order: RasterOrder | str, grid_m: int, grid_n: int, *,
                      group: int = 1, window: int, aligned: bool = True) -> int:
    """Distinct operand panels touched by a wave of ``window`` consecutive
    blocks — the worst case over waves.

    This is a **counting fact about the traversal, not a latency prediction**: it
    says how many A-row and B-column panels a wave of concurrently resident
    blocks references, which is the quantity a cache has to hold. Row-major over
    a wide grid gives ~`window + 1`; grouped-M with group `g` gives ~`g +
    window/g`, minimised near `g = sqrt(window)`.

    ``aligned`` (the default) starts waves at multiples of ``window``, which is
    how a GPU actually dispatches them: blocks `0..W-1` occupy the machine, then
    `W..2W-1`. Sliding the window one block at a time (``aligned=False``) instead
    scores a phase that never occurs, and its worst case is dominated by the
    single straddle across a panel boundary — which washes out most of the
    grouping signal rather than measuring it.

    It is the honest, verifiable part of the swizzle story. Turning it into a hit
    rate needs the tile-reuse-distance model (TILESIGHT_ASSESSMENT §3.1 T1),
    which is not built — do not read this as a latency estimate.

    Cost: O(grid_m x grid_n x window) ``remap`` calls (x ``window`` again when
    ``aligned=False``). An analysis helper for a schedule search, not something to
    call per launch — a 256x256 grid with a 256-block wave is ~16M evaluations.
    """
    _validate(grid_m, grid_n, group)
    if window < 1:
        raise RasterError(f"window must be >= 1, got {window}")
    total = grid_m * grid_n
    window = min(window, total)
    starts = (range(0, total, window) if aligned
              else range(total - window + 1))
    worst = 0
    for start in starts:
        rows: set[int] = set()
        cols: set[int] = set()
        for bid in range(start, min(start + window, total)):
            m_tile, n_tile = remap(order, bid, grid_m, grid_n, group=group)
            rows.add(m_tile)
            cols.add(n_tile)
        worst = max(worst, len(rows) + len(cols))
    return worst


def suggest_group(window: int) -> int:
    """The group size that minimises the grouped-M working set for a wave of
    ``window`` concurrent blocks — the nearest choice in
    :data:`RASTER_GROUP_CHOICES` to ``sqrt(window)``.

    A starting point for a sweep, not a substitute for one: the real optimum
    depends on cache capacity and tile bytes, which this does not know.
    """
    if window < 1:
        raise RasterError(f"window must be >= 1, got {window}")
    target = window ** 0.5
    return min(RASTER_GROUP_CHOICES, key=lambda g: (abs(g - target), g))


# --- emission ----------------------------------------------------------------

def emit_c(order: RasterOrder | str, *, group: int = 1, bid: str = "bid",
           grid_m: str = "gm", grid_n: str = "gn", out_m: str = "mt",
           out_n: str = "nt", indent: str = "  ") -> str:
    """C statements computing ``out_m``/``out_n`` from flat block id ``bid``.

    The emitted text is valid in **both** CUDA and HIP (plain integer arithmetic,
    no vendor intrinsics), so one snippet serves the NVIDIA and ROCm emitters.
    ``grid_m``/``grid_n`` name expressions already in scope.

    The generated code mirrors :func:`remap` exactly — including the short final
    panel — so the reference and the device kernel cannot drift. Row-major emits
    the plain divide/modulo Tessera already uses, making the default a no-op.

    **Repeat-safe.** The grouped forms need temporaries; they are declared inside
    a braced block with a ``_tsr_raster_`` prefix, and only ``out_m``/``out_n``
    escape it. A kernel may therefore emit this more than once (two grids in one
    launch) without redefinition errors, and cannot collide with a variable the
    surrounding kernel already owns. The only names the caller must keep distinct
    across emissions are ``out_m`` and ``out_n``, which the caller chooses.
    """
    order = RasterOrder(order)
    if group < 1:
        raise RasterError(f"group must be >= 1, got {group}")
    i = indent

    # group == 1 makes the grouped orders degenerate to their ungrouped form
    # (see RASTER_GROUP_CHOICES), so they share the simple emission.
    if order is RasterOrder.ROW_MAJOR or (group == 1
                                          and order is RasterOrder.GROUPED_M):
        return (f"{i}int {out_m} = (int)({bid} / {grid_n});\n"
                f"{i}int {out_n} = (int)({bid} % {grid_n});\n")
    if order is RasterOrder.COLUMN_MAJOR or (group == 1
                                             and order is RasterOrder.GROUPED_N):
        return (f"{i}int {out_m} = (int)({bid} % {grid_m});\n"
                f"{i}int {out_n} = (int)({bid} / {grid_m});\n")

    if order is RasterOrder.GROUPED_M:
        major, minor, res_major, res_minor = grid_m, grid_n, out_m, out_n
    elif order is RasterOrder.GROUPED_N:
        major, minor, res_major, res_minor = grid_n, grid_m, out_n, out_m
    else:  # pragma: no cover - RasterOrder is exhaustive above
        raise RasterError(f"unhandled rasterization order {order!r}")

    p = "_tsr_raster_"
    return (
        f"{i}int {out_m}, {out_n};\n"
        f"{i}{{\n"
        f"{i}  const int {p}pp = {group} * (int)({minor});\n"
        f"{i}  const int {p}pn = (int)({bid}) / {p}pp;\n"
        f"{i}  const int {p}first = {p}pn * {group};\n"
        f"{i}  const int {p}rows = (int)({major}) - {p}first < {group}"
        f" ? (int)({major}) - {p}first : {group};\n"
        f"{i}  const int {p}w = (int)({bid}) % {p}pp;\n"
        f"{i}  {res_major} = {p}first + {p}w % {p}rows;\n"
        f"{i}  {res_minor} = {p}w / {p}rows;\n"
        f"{i}}}\n"
    )


__all__ = [
    "RASTER_GROUP_CHOICES",
    "RASTER_ORDER_CHOICES",
    "RasterError",
    "RasterOrder",
    "emit_c",
    "is_bijection",
    "num_blocks",
    "remap",
    "suggest_group",
    "working_set_tiles",
]
