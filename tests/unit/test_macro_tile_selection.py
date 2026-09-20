"""Host-free: the macro tile is selected THROUGH the ranking model, identically.

`rank_candidates` modelled register fit, LDS footprint, bank padding and
pipeline depth and had **no caller** — while the shipped panels were chosen by
a hardcoded band beside it. A model nothing consults cannot be wrong in a way
anyone notices, which is how `split_k_required` came to answer `False` for the
one shape that needed split-K most.

`select_macro_tile` is now its single caller. Two properties have to hold, and
they pull against each other:

1. **The selection is unchanged.** `PMPasses.cpp` is the production authority
   and `verify_matmul_projection` compares `macro_tile_m/n` against the IR it
   emits, so the Python side is a Decision #31 oracle. An oracle that disagrees
   with the thing it checks is worse than none.
2. **The measured facts stay measured.** Which panel wins in which band came
   from Tajasarus and Princess-Luna. The ranking's job is feasibility; the
   measurement's job is preference. Deriving the preference here would swap a
   measured answer for a plausible one.
"""

from __future__ import annotations

import pytest

from tessera.compiler.rocm_target import AMDArch, ROCmTargetProfile, rocm_arch_string
from tessera.compiler.rocm_tiling import select_macro_tile
from tessera.compiler.scheduled_matmul import rocm_gfx1151_panel, rocm_gfx1201_panel

#: Edges chosen to straddle every boundary the two selectors have: the 1024
#: floor, the 2048 band end, the %64 tiling rule, and either side of each.
_EDGES = (16, 63, 64, 256, 1023, 1024, 1025, 1536, 2047, 2048, 4096)


@pytest.mark.parametrize("m", _EDGES)
@pytest.mark.parametrize("n", (16, 256, 1024, 2048, 4096))
@pytest.mark.parametrize("dynamic", (False, True))
def test_gfx1201_panel_matches_its_measured_rule(m: int, n: int, dynamic: bool) -> None:
    """4x4 for every static, fully tiled problem at 1024+; 1x1 otherwise."""
    expected = (64, 64) if (
        not dynamic and m >= 1024 and n >= 1024 and m % 64 == 0 and n % 64 == 0
    ) else (16, 16)
    assert rocm_gfx1201_panel(m, n, dynamic=dynamic) == expected


@pytest.mark.parametrize("m", _EDGES)
@pytest.mark.parametrize("n", (16, 256, 1024, 2048, 4096))
@pytest.mark.parametrize("dynamic", (False, True))
def test_gfx1151_panel_matches_its_measured_band(m: int, n: int, dynamic: bool) -> None:
    """4x4 only inside the fully tiled [1024, 2048) band; the 2x4 elsewhere."""
    in_band = (
        not dynamic and 1024 <= m < 2048 and 1024 <= n < 2048
        and m % 64 == 0 and n % 64 == 0
    )
    assert rocm_gfx1151_panel(m, n, dynamic=dynamic) == ((64, 64) if in_band else (32, 64))


def test_selection_routes_through_the_ranking_model() -> None:
    """The point of the exercise: the panels are not computed beside it."""
    import inspect

    from tessera.compiler import scheduled_matmul

    for fn in (scheduled_matmul.rocm_gfx1201_panel, scheduled_matmul.rocm_gfx1151_panel):
        body = inspect.getsource(fn)
        assert "select_macro_tile" in body, (
            f"{fn.__name__} no longer selects through the ranking model. If the "
            f"selection moved, rank_candidates has no caller again and its model "
            f"stops being falsifiable — which is the state this replaced."
        )


def test_an_infeasible_panel_is_not_selected() -> None:
    """Feasibility is the ranking's half of the split, and it must bite.

    A panel that does not fit the register budget is not a choice however well
    it measured, so the selector must fall back rather than emit a spilling
    tile it was told to prefer.
    """
    profile = ROCmTargetProfile(arch=AMDArch.GFX_1201)
    absurd = select_macro_tile(
        4096, 4096, profile=profile, dynamic=False, dtype="fp32",
        measured_large_panel=(1024, 1024),   # cannot possibly fit
        measured_small_panel=(16, 16))
    assert absurd == (16, 16), (
        "a panel the chip cannot hold was selected because a measurement "
        "preferred it; feasibility must gate preference, not the reverse")


def test_unmeasured_arch_still_selects_without_inventing_occupancy() -> None:
    """dispatch_slots is None for an unmeasured part; selection must not break.

    Split-K declines to conclude there, but the macro tile does not depend on
    occupancy — conflating the two would make every unmeasured arch unusable.
    """
    profile = ROCmTargetProfile(arch=AMDArch.GFX_942)
    assert select_macro_tile(
        2048, 2048, profile=profile, dynamic=False,
        measured_large_panel=(64, 64), measured_small_panel=(16, 16)) == (64, 64)
