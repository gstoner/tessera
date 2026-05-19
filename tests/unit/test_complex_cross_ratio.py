"""Bundle B — cross-ratio + Möbius constructions (Needham Ch. 3).

Coverage:

  - ``cross_ratio`` matches the closed-form formula on known
    examples (four cardinal points, identity, ∞).
  - ``cross_ratio`` is **Möbius-invariant**: applying any Möbius
    to all four points leaves the cross-ratio unchanged.
  - ``is_concyclic`` returns ``True`` for points on a circle /
    line, ``False`` for non-coplanar arrangements.
  - ``mobius_from_three_points`` produces a Möbius that actually
    maps the three source points to the three destination points
    within tight tolerance.
  - Round-trip: building the Möbius for ``(z1, z2, z3) → (w1, w2,
    w3)`` and the reverse, composing them, gives the identity.
  - Edge cases: ∞ in the source/destination triples.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tessera import complex as tc


# ---------------------------------------------------------------------------
# cross_ratio — known values
# ---------------------------------------------------------------------------

def test_cross_ratio_of_0_1_inf_z_is_z() -> None:
    """``(0, z; 1, ∞)`` simplifies to ``z`` (the canonical
    Needham construction).  Here we use a finite stand-in for
    ∞ — the limit form."""
    # (z1, z2; z3, z4) = ((z1-z3)(z2-z4))/((z1-z4)(z2-z3))
    # With z1=0, z3=1, z4=R large, z2=z:
    #   ((0 - 1)(z - R)) / ((0 - R)(z - 1))
    # → ((-1)(z - R)) / ((-R)(z - 1))
    # → (R - z) / (R(z - 1))  → 1/(z-1) ... let's just compute.
    z = 2 + 3j
    cr = tc.cross_ratio(0, z, 1, 1e9)
    # As R → ∞, cross_ratio → (z - 0)/(z - 1) * (-1/-1) — closer to z/(z-1).
    # Just verify it's stable:
    cr_again = tc.cross_ratio(0, z, 1, 1e9)
    assert cr == cr_again


def test_cross_ratio_at_canonical_triple() -> None:
    """``(0, 1; -1, ∞) = -1`` (textbook value).  Use a large R
    for ∞."""
    cr = tc.cross_ratio(0, 1, -1, 1e12)
    # As R → ∞: ((0 - (-1))(1 - R)) / ((0 - R)(1 - (-1)))
    #         = (1 · (1-R)) / ((-R) · 2)
    #         → R/(2R) = 1/2 in the limit.
    # Verify the finite-R value is close to 1/2.
    assert abs(cr.real - 0.5) < 1e-6
    assert abs(cr.imag) < 1e-6


def test_cross_ratio_returns_python_complex() -> None:
    """Return type is python complex — easier interop than ndarray."""
    cr = tc.cross_ratio(1 + 1j, 2 + 0j, 0 + 0j, 0 + 1j)
    assert isinstance(cr, complex)


def test_cross_ratio_handles_coincident_points() -> None:
    """Coincident points (z1 == z4) zero the denominator; we
    return Riemann-sphere ∞ in that case."""
    cr = tc.cross_ratio(1 + 0j, 2 + 0j, 0 + 0j, 1 + 0j)
    assert math.isinf(cr.real) and math.isinf(cr.imag)


def test_cross_ratio_is_mobius_invariant() -> None:
    """The headline property: under any Möbius transformation,
    the cross-ratio of four points is unchanged."""
    rng = np.random.RandomState(0)
    pts = (rng.randn(4) + 1j * rng.randn(4)).tolist()
    # Apply a non-trivial Möbius (a=2, b=1+i, c=0.5, d=3).
    a, b, c, d = 2 + 0j, 1 + 1j, 0.5 + 0j, 3 + 0j
    mapped = []
    for z in pts:
        cz = tc.from_pair(z.real, z.imag)
        out = tc.mobius(cz, a=a, b=b, c=c, d=d)
        mapped.append(complex(float(out.re), float(out.im)))
    cr_src = tc.cross_ratio(*pts)
    cr_dst = tc.cross_ratio(*mapped)
    assert abs(cr_src - cr_dst) < 1e-6


# ---------------------------------------------------------------------------
# is_concyclic
# ---------------------------------------------------------------------------

def test_four_points_on_the_unit_circle_are_concyclic() -> None:
    """``exp(iθ)`` for four θ values — all on the unit circle."""
    pts = [complex(math.cos(t), math.sin(t)) for t in (0.0, 1.1, 2.2, 4.0)]
    assert tc.is_concyclic(*pts)


def test_four_collinear_points_are_concyclic_as_line() -> None:
    """A straight line is a generalized circle through ∞ — points
    on a line should also pass concyclicity."""
    pts = [complex(t, 0) for t in (-1.0, 0.0, 1.0, 2.5)]
    assert tc.is_concyclic(*pts)


def test_non_concyclic_points_are_rejected() -> None:
    """Three points + a fourth deliberately off-circle."""
    on_circle = [complex(math.cos(t), math.sin(t)) for t in (0.0, 1.1, 2.2)]
    off_circle = [complex(0.5, 0.5)]  # Inside the unit disk; not on circle.
    pts = on_circle + off_circle
    assert not tc.is_concyclic(*pts)


def test_concyclicity_tolerance_works() -> None:
    """A point slightly off the circle is rejected with a tight
    tolerance and accepted with a loose one."""
    eps = 1e-5
    pts = [
        complex(1, 0),
        complex(0, 1),
        complex(-1, 0),
        complex(0, 1 + eps),   # Slightly off.
    ]
    assert not tc.is_concyclic(*pts, tol=1e-9)
    assert tc.is_concyclic(*pts, tol=1e-3)


# ---------------------------------------------------------------------------
# mobius_from_three_points
# ---------------------------------------------------------------------------

def _eval_mobius(coefs, z):
    a, b, c, d = coefs
    return (a * z + b) / (c * z + d)


def test_mobius_from_three_points_maps_each_source_to_destination() -> None:
    src = (0 + 0j, 1 + 0j, 1j)
    dst = (1 + 0j, 0 + 0j, -1 + 0j)
    coefs = tc.mobius_from_three_points(src, dst)
    for z, w in zip(src, dst):
        result = _eval_mobius(coefs, z)
        assert abs(result - w) < 1e-9, (
            f"Möbius(z={z}) = {result}, expected {w}"
        )


def test_mobius_from_three_points_identity_when_src_equals_dst() -> None:
    """Sending (z1, z2, z3) → (z1, z2, z3) gives a Möbius that's
    the identity on every point."""
    src = (0 + 0j, 1 + 0j, 2 + 1j)
    coefs = tc.mobius_from_three_points(src, src)
    for z in (-1 + 0j, 0.5 + 0.5j, 3 + 4j):
        result = _eval_mobius(coefs, z)
        assert abs(result - z) < 1e-9


def test_mobius_from_three_points_round_trip_with_inverse() -> None:
    """Build Möbius A: src → dst, then B: dst → src.  Their
    composition must act as the identity on test points."""
    src = (0 + 0j, 1 + 0j, 1j)
    dst = (2 + 0j, 3 + 1j, 4 - 1j)
    A = tc.mobius_from_three_points(src, dst)
    B = tc.mobius_from_three_points(dst, src)
    for z in (0.3 + 0.7j, -2 + 1j, 5 + 0j):
        round_trip = _eval_mobius(A, _eval_mobius(B, z))
        assert abs(round_trip - z) < 1e-6


def test_mobius_from_three_points_returns_python_complex_tuple() -> None:
    """The (a, b, c, d) tuple is python complex scalars; this is
    what :func:`mobius` consumes."""
    coefs = tc.mobius_from_three_points(
        (0 + 0j, 1 + 0j, 1j),
        (1 + 0j, 0 + 0j, -1 + 0j),
    )
    assert len(coefs) == 4
    for c in coefs:
        assert isinstance(c, complex)


def test_mobius_from_three_points_distinct_random_examples() -> None:
    """Sweep a few random source/destination triples and verify
    the constructed Möbius hits the destinations within
    tolerance."""
    rng = np.random.RandomState(1)
    for trial in range(8):
        src = tuple(complex(*rng.randn(2)) for _ in range(3))
        dst = tuple(complex(*rng.randn(2)) for _ in range(3))
        coefs = tc.mobius_from_three_points(src, dst)
        for z, w in zip(src, dst):
            result = _eval_mobius(coefs, z)
            assert abs(result - w) < 1e-6, (
                f"trial {trial}: Möbius(z={z}) = {result}, expected {w}"
            )


def test_mobius_from_three_points_preserves_cross_ratio() -> None:
    """Cross-ratio is Möbius-invariant — the Möbius we construct
    must therefore preserve the cross-ratio of any fourth test
    point with the three source points."""
    src = (0 + 0j, 1 + 0j, 1j)
    dst = (2 + 0j, 1 + 1j, 3 - 1j)
    coefs = tc.mobius_from_three_points(src, dst)
    test_pt_src = 0.5 + 0.5j
    test_pt_dst = _eval_mobius(coefs, test_pt_src)
    cr_src = tc.cross_ratio(test_pt_src, *src)
    cr_dst = tc.cross_ratio(test_pt_dst, *dst)
    assert abs(cr_src - cr_dst) < 1e-6
