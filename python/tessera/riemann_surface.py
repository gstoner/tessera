"""``tessera.riemann_surface`` — branch tracking for multi-valued
complex functions (Needham Ch. 12).

A multi-valued function like ``sqrt`` or ``log`` becomes
single-valued on its **Riemann surface**: a many-sheeted covering
of ℂ in which the branch index is tracked alongside the
``z``-coordinate.  Walking a path on the surface near a branch
point (e.g., the origin for ``sqrt``) carries the index from one
sheet to the next.

This module ships a small but real implementation:

  - :class:`RiemannSurfacePoint` — ``(z, branch)`` pair, hashable.
  - :func:`lift_sqrt(z, branch=0)` — pick one of sqrt's two sheets.
  - :func:`lift_log(z, branch=0)` — pick one of log's infinitely
    many sheets (each differs by ``2πi``).
  - :func:`follow_path_on_riemann_surface(f, path, initial)` —
    walk a parameterized path and track branch flips at the cut.

The branch-cut convention matches :func:`tessera.complex.complex_log`
(principal branch cut along the negative real axis,
``arg ∈ (-π, π]``).
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RiemannSurfacePoint:
    """A point on a multi-sheet Riemann surface.

    ``z`` is the underlying complex coordinate; ``branch`` is the
    sheet index (0 for the principal sheet).  Equality compares
    both fields so the same ``z`` on different sheets is treated
    as distinct.
    """
    z: complex
    branch: int

    def __repr__(self) -> str:
        return f"RSP(z={self.z}, branch={self.branch})"


# ─────────────────────────────────────────────────────────────────────────────
# Lifts
# ─────────────────────────────────────────────────────────────────────────────

def lift_sqrt(z: complex, *, branch: int = 0) -> RiemannSurfacePoint:
    """Evaluate ``sqrt(z)`` on its 2-sheet Riemann surface.

    Branches:

      * ``branch=0`` — principal: ``Im(√z) ≥ 0``.  ``√4 = 2``,
        ``√(-1) = i``.
      * ``branch=1`` — second sheet: principal value times ``-1``.

    Returns a :class:`RiemannSurfacePoint` whose ``z`` field is
    the chosen sheet's image of the input.
    """
    if branch not in (0, 1):
        raise ValueError(
            f"sqrt's Riemann surface has 2 sheets; got branch={branch}"
        )
    principal = cmath.sqrt(z)
    image = principal if branch == 0 else -principal
    return RiemannSurfacePoint(z=image, branch=branch)


def lift_log(z: complex, *, branch: int = 0) -> RiemannSurfacePoint:
    """Evaluate ``log(z)`` on its infinitely-sheeted Riemann
    surface.

    Branches differ by ``2πi``: ``log_k(z) = log|z| + i·(arg(z) + 2πk)``.
    Principal: ``branch=0`` ⇒ ``arg ∈ (-π, π]``.
    """
    if z == 0:
        raise ValueError("log is undefined at z=0 on every branch")
    principal = cmath.log(z)
    image = principal + 2j * math.pi * branch
    return RiemannSurfacePoint(z=image, branch=branch)


# ─────────────────────────────────────────────────────────────────────────────
# Path-following with branch-cut detection.
# ─────────────────────────────────────────────────────────────────────────────

def _crosses_negative_real_axis(z_prev: complex, z_curr: complex) -> int:
    """Return +1 / -1 / 0 indicating how the path step from
    ``z_prev`` to ``z_curr`` crosses the principal cut along the
    negative real axis.

    Sign convention: matches what's needed to make ``log(z)``
    continuous across the cut.  When walking CCW around 0,
    ``arg(z)`` increases by 2π and the principal log jumps DOWN
    by 2πi at the cut — so we add +2πi (advance the branch by
    +1) to keep the lifted log continuous.  The cut crossing
    that happens during a CCW loop is from above (Im > 0) to
    below (Im < 0), so that crossing must return +1.

    Returns:

      * +1: ``z_prev.imag > 0``, ``z_curr.imag < 0`` (above-to-below).
      * -1: ``z_prev.imag < 0``, ``z_curr.imag > 0`` (below-to-above).
      * 0: doesn't cross.
    """
    if z_prev.real >= 0 or z_curr.real >= 0:
        return 0
    if z_prev.imag == 0 or z_curr.imag == 0:
        return 0
    if z_prev.imag > 0 and z_curr.imag < 0:
        return +1
    if z_prev.imag < 0 and z_curr.imag > 0:
        return -1
    return 0


def follow_path_on_riemann_surface(
    f_name: str,
    path: Sequence[complex],
    *,
    initial_branch: int = 0,
) -> list[RiemannSurfacePoint]:
    """Walk a discrete path of ``z`` values and lift each step
    onto the Riemann surface of ``f_name``, flipping the branch
    each time the path crosses the principal cut.

    Parameters
    ----------
    f_name
        One of ``"sqrt"`` or ``"log"``.
    path
        Discrete sequence of complex ``z`` values forming the
        path.
    initial_branch
        Branch index at the first point of the path.

    Returns
    -------
    list of :class:`RiemannSurfacePoint`
        One entry per point in ``path``.  The branch index can
        differ between consecutive entries: for ``sqrt`` it
        toggles (mod 2) on each cut crossing; for ``log`` it
        increments / decrements by the cut-crossing direction.

    The canonical demonstration: walking a small circle around 0
    crosses the cut once and flips the sqrt branch — i.e., starting
    at +√z and returning to -√z.  See ``test_sqrt_path_around_origin``.
    """
    if f_name not in ("sqrt", "log"):
        raise ValueError(
            f"follow_path_on_riemann_surface: f_name must be "
            f"'sqrt' or 'log'; got {f_name!r}"
        )
    if not path:
        return []
    branch = initial_branch
    out: list[RiemannSurfacePoint] = []
    if f_name == "sqrt":
        out.append(lift_sqrt(path[0], branch=branch))
    else:
        out.append(lift_log(path[0], branch=branch))
    for i in range(1, len(path)):
        crossing = _crosses_negative_real_axis(path[i - 1], path[i])
        if f_name == "sqrt":
            if crossing != 0:
                branch = 1 - branch    # mod-2 flip
            out.append(lift_sqrt(path[i], branch=branch))
        else:    # log
            branch += crossing
            out.append(lift_log(path[i], branch=branch))
    return out


__all__ = [
    "RiemannSurfacePoint",
    "lift_sqrt",
    "lift_log",
    "follow_path_on_riemann_surface",
]
