"""Riemann surface lifting — branch tracking for multi-valued functions.

Coverage:

  - ``lift_sqrt`` returns the correct principal value at branch 0
    and the negation at branch 1.
  - ``lift_log`` branches differ by ``2πi`` as expected.
  - Walking a path around the origin on the sqrt surface ends on
    the OPPOSITE sheet from where it started (the canonical
    Riemann-surface demonstration).
  - Walking a path that doesn't enclose the origin stays on the
    same sheet.
  - Walking around 0 on log surface increments the branch by ±1.
"""

from __future__ import annotations

import cmath
import math

import pytest

from tessera import riemann_surface as rs


# ---------------------------------------------------------------------------
# lift_sqrt
# ---------------------------------------------------------------------------

def test_lift_sqrt_principal_branch() -> None:
    p = rs.lift_sqrt(4 + 0j, branch=0)
    assert p.branch == 0
    assert abs(p.z - 2) < 1e-12


def test_lift_sqrt_other_branch_negates_principal() -> None:
    p0 = rs.lift_sqrt(4 + 0j, branch=0)
    p1 = rs.lift_sqrt(4 + 0j, branch=1)
    assert p0.branch == 0 and p1.branch == 1
    assert abs(p0.z + p1.z) < 1e-12    # opposite-sign sheets


def test_lift_sqrt_at_minus_one_is_i() -> None:
    p = rs.lift_sqrt(-1 + 0j, branch=0)
    assert abs(p.z - 1j) < 1e-12


def test_lift_sqrt_rejects_invalid_branch() -> None:
    with pytest.raises(ValueError, match="2 sheets"):
        rs.lift_sqrt(1 + 0j, branch=2)


# ---------------------------------------------------------------------------
# lift_log
# ---------------------------------------------------------------------------

def test_lift_log_principal_branch() -> None:
    p = rs.lift_log(1 + 0j, branch=0)
    assert abs(p.z) < 1e-12


def test_lift_log_other_branches_differ_by_2pi_i() -> None:
    p0 = rs.lift_log(2 + 0j, branch=0)
    p1 = rs.lift_log(2 + 0j, branch=1)
    p_minus1 = rs.lift_log(2 + 0j, branch=-1)
    assert abs((p1.z - p0.z) - 2j * math.pi) < 1e-12
    assert abs((p0.z - p_minus1.z) - 2j * math.pi) < 1e-12


def test_lift_log_zero_is_undefined_on_every_branch() -> None:
    with pytest.raises(ValueError, match="undefined"):
        rs.lift_log(0 + 0j)


# ---------------------------------------------------------------------------
# Path-following on the Riemann surface
# ---------------------------------------------------------------------------

def _circle_path(center: complex, radius: float, n_steps: int,
                 *, ccw: bool = True) -> list[complex]:
    """Discrete points on a circle, parameterized by angle."""
    sign = 1.0 if ccw else -1.0
    return [
        center + radius * complex(math.cos(sign * 2 * math.pi * t),
                                   math.sin(sign * 2 * math.pi * t))
        for t in [i / n_steps for i in range(n_steps + 1)]
    ]


def test_sqrt_path_around_origin_flips_sheet() -> None:
    """The canonical Riemann-surface demonstration: walking
    around 0 once carries the sqrt value from the +√ sheet to
    the -√ sheet."""
    path = _circle_path(center=0, radius=1.0, n_steps=200, ccw=True)
    lifted = rs.follow_path_on_riemann_surface(
        "sqrt", path, initial_branch=0,
    )
    # Start: branch 0.  End: branch 1 (single cut crossing).
    assert lifted[0].branch == 0
    assert lifted[-1].branch == 1
    # The final value is the OPPOSITE sign of the principal value
    # at the starting point (which is √1 = 1).
    assert abs(lifted[-1].z - (-1.0)) < 1e-3


def test_sqrt_path_not_enclosing_origin_stays_on_same_sheet() -> None:
    """A small circle around (2 + 0j) doesn't cross the cut —
    the branch index is preserved."""
    path = _circle_path(center=2 + 0j, radius=0.5, n_steps=200, ccw=True)
    lifted = rs.follow_path_on_riemann_surface(
        "sqrt", path, initial_branch=0,
    )
    branches = {p.branch for p in lifted}
    assert branches == {0}


def test_log_path_around_origin_increments_branch() -> None:
    """Walking CCW around 0 once advances log's branch by +1
    (i.e., the log value increases by 2πi)."""
    path = _circle_path(center=0, radius=1.0, n_steps=200, ccw=True)
    lifted = rs.follow_path_on_riemann_surface(
        "log", path, initial_branch=0,
    )
    assert lifted[0].branch == 0
    assert lifted[-1].branch == 1
    # log(1) on branch 1 = 0 + 2πi.
    assert abs(lifted[-1].z - 2j * math.pi) < 1e-2


def test_log_path_around_origin_clockwise_decrements_branch() -> None:
    """CW walk decrements the branch by 1."""
    path = _circle_path(center=0, radius=1.0, n_steps=200, ccw=False)
    lifted = rs.follow_path_on_riemann_surface(
        "log", path, initial_branch=0,
    )
    assert lifted[-1].branch == -1


def test_follow_path_rejects_unknown_function() -> None:
    with pytest.raises(ValueError, match="must be 'sqrt' or 'log'"):
        rs.follow_path_on_riemann_surface("cos", [1 + 0j])


def test_follow_path_empty_returns_empty() -> None:
    out = rs.follow_path_on_riemann_surface("sqrt", [])
    assert out == []
