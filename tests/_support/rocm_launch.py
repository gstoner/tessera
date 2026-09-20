"""The launch grid for a ROCm scheduled-matmul kernel, with a coverage proof.

A throughput harness that launches the wrong grid produces **correct results and
wrong numbers**: extra workgroups recompute values that are already right, so
every correctness check passes while the measured rate is divided by the
redundancy. That is not hypothetical -- it inflated every LDS figure recorded on
2026-09-19 by ~4x and sent two conclusions and a shipped default the wrong way
(`docs/backends/rocm/wmma-fragment-layout.md` section 10i).

The trap is that the register body and the LDS body tile differently:

  * the register body covers `macroTileM x macroTileN` per workgroup;
  * the LDS body covers `wavesM * macroTileM x wavesN * macroTileN`, because
    every wave in the group takes its own panel, and it computes
    `gridM = ceil(M / wgM)` itself before indexing with `bidY` / `bidX`.

So the grid cannot be derived from the artifact's macro tile alone. Use
`launch_grid`, which takes the staging and wave shape and **asserts exact
coverage** -- each output computed once, none missed.
"""

from __future__ import annotations

# One rule, one place. `workgroup_tile` lives beside the packager that stamps
# the launch tile into descriptor provenance, so a harness and the production
# launcher cannot disagree about it -- the disagreement is exactly what went
# wrong, and it is invisible in every correctness check.
from tessera.compiler.rocm_native import workgroup_tile  # noqa: F401


def launch_grid(m: int, n: int, macro_m: int, macro_n: int, *, staging: str,
                lds_waves: tuple[int, int] = (1, 1)) -> tuple[int, int]:
    """`(grid_x, grid_y)` covering `m x n` exactly once, or raise.

    The assertion is the point. A grid that is too large still produces the
    right answer, so nothing downstream will notice; a grid that is too small
    leaves outputs untouched, which a correctness check *would* notice but only
    after the timing has already been recorded.
    """
    wm, wn = workgroup_tile(macro_m, macro_n, staging=staging,
                            lds_waves=lds_waves)
    if wm <= 0 or wn <= 0:
        raise ValueError(f"non-positive workgroup tile {wm}x{wn}")
    gy, gx = (m + wm - 1) // wm, (n + wn - 1) // wn
    for axis, extent, tile, grid in (("M", m, wm, gy), ("N", n, wn, gx)):
        covered = grid * tile
        if covered < extent:
            raise AssertionError(
                f"{axis}: grid {grid} x tile {tile} = {covered} does not reach "
                f"{extent}; some outputs would never be written")
        if covered >= extent + tile:
            raise AssertionError(
                f"{axis}: grid {grid} x tile {tile} = {covered} exceeds "
                f"{extent} by a whole tile or more, so at least one workgroup "
                f"is redundant. Results stay CORRECT and every throughput "
                f"figure is divided by the redundancy -- the 2026-09-19 LDS "
                f"measurements failed exactly here. Check whether this is the "
                f"LDS body, whose workgroup tile is waves x macro tile.")
    return gx, gy
