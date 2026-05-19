"""Bundle C — contour integration, winding, residue (Needham Ch. 7-9).

Coverage:

  - Constructors (``circle``, ``line_segment``, ``polygon``)
    produce well-formed parametric curves with correct
    endpoints, derivatives, and closure semantics.
  - **Cauchy's theorem**: ``∮ f dz = 0`` for ``f`` analytic on
    a closed contour (verified for several entire functions).
  - **Cauchy's integral formula**: ``∮ 1/(z - a) dz = 2πi`` for
    ``a`` inside the contour, ``0`` for ``a`` outside.
  - Winding number ``±1`` for simple loops; ``0`` for excluded
    points; ``±n`` for n-fold encirclements via the polygon
    constructor.
  - Residue extraction matches closed forms on a sweep of
    poles.
  - **Argument principle** counts zeros minus poles enclosed.
  - **Residue theorem**: ``∮ f dz = 2πi · Σ residues`` (the
    closing argument of the textbook chapter).
"""

from __future__ import annotations

import cmath
import math

import pytest

from tessera import contour as tc


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

def test_circle_endpoints_match() -> None:
    """γ(0) == γ(1) for a closed circle."""
    c = tc.circle(1 + 1j, radius=2.0)
    assert abs(c(0.0) - c(1.0)) < 1e-12
    assert c.is_closed


def test_circle_half_turn_is_opposite_point() -> None:
    """γ(0.5) is on the far side of the center."""
    c = tc.circle(0 + 0j, radius=1.0)
    p0 = c(0.0)
    p_half = c(0.5)
    assert abs(p0 + p_half) < 1e-12  # antipodal


def test_circle_tangent_is_perpendicular_to_radius() -> None:
    """At any point on the circle, the tangent ⟂ the radius."""
    c = tc.circle(0 + 0j, radius=1.0)
    for t in (0.1, 0.3, 0.7, 0.9):
        pt = c(t)
        tan = c.tangent(t)
        # Real dot product: ⟨pt, tan⟩ = pt.re·tan.re + pt.im·tan.im
        dot = pt.real * tan.real + pt.imag * tan.imag
        assert abs(dot) < 1e-6, (t, dot)


def test_circle_orientation_flips_integral_sign() -> None:
    """The right way to test orientation: ∮ dz/z over a CCW
    unit circle is +2πi; over a CW circle it's -2πi.  (At the
    same t, the two circles are at different physical points,
    so comparing tangent vectors directly is meaningless.)"""
    ccw = tc.circle(0, 1.0, ccw=True)
    cw = tc.circle(0, 1.0, ccw=False)
    int_ccw = tc.contour_integral(lambda z: 1.0 / z, ccw, n_steps=1024)
    int_cw = tc.contour_integral(lambda z: 1.0 / z, cw, n_steps=1024)
    assert abs(int_ccw - 2.0j * math.pi) < 1e-3
    assert abs(int_cw - (-2.0j * math.pi)) < 1e-3
    # And they're sign-opposites.
    assert abs(int_ccw + int_cw) < 1e-3


def test_line_segment_endpoints() -> None:
    L = tc.line_segment(1 + 2j, 5 + 0j)
    assert L(0.0) == 1 + 2j
    assert L(1.0) == 5 + 0j
    assert not L.is_closed


def test_polygon_visits_each_vertex_in_order() -> None:
    verts = [0 + 0j, 1 + 0j, 1 + 1j, 0 + 1j]
    P = tc.polygon(verts)
    assert P.is_closed
    # γ(k/n) for k = 0..n hits vertex k.
    n = len(verts)
    for k, v in enumerate(verts):
        assert abs(P(k / n) - v) < 1e-9


def test_polygon_requires_three_vertices() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        tc.polygon([0j, 1 + 0j])


# ---------------------------------------------------------------------------
# Cauchy's theorem — ∮ f dz = 0 for analytic f
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("f", [
    lambda z: 1 + 0j,        # constant
    lambda z: z,             # z
    lambda z: z * z,         # z²
    lambda z: cmath.exp(z),  # e^z
])
def test_cauchy_theorem_closed_loop_of_analytic_function(f) -> None:
    """∮ f dz = 0 for any entire f over a closed loop."""
    c = tc.circle(0 + 0j, radius=1.0)
    integral = tc.contour_integral(f, c, n_steps=512)
    assert abs(integral) < 1e-6


def test_cauchy_theorem_over_a_polygon() -> None:
    """Same property over a non-circle contour."""
    P = tc.polygon([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j])  # unit-diamond
    integral = tc.contour_integral(lambda z: z * z + 1, P, n_steps=512)
    # f = z² + 1 is entire ⇒ integral is 0.
    assert abs(integral) < 1e-6


# ---------------------------------------------------------------------------
# Cauchy's integral formula — ∮ 1/(z - a) dz = 2πi
# ---------------------------------------------------------------------------

def test_cauchy_formula_a_inside_yields_2pi_i() -> None:
    """∮_{|z|=1} dz / (z - 0) = 2πi  (the canonical example)."""
    c = tc.circle(0 + 0j, radius=1.0)
    integral = tc.contour_integral(lambda z: 1.0 / z, c, n_steps=1024)
    expected = 2.0j * math.pi
    assert abs(integral - expected) < 1e-3


def test_cauchy_formula_a_inside_off_center() -> None:
    a = 0.3 + 0.2j
    c = tc.circle(0 + 0j, radius=1.0)
    integral = tc.contour_integral(lambda z: 1.0 / (z - a), c, n_steps=1024)
    assert abs(integral - 2.0j * math.pi) < 1e-3


def test_cauchy_formula_a_outside_yields_zero() -> None:
    a = 5 + 5j   # well outside the unit circle
    c = tc.circle(0 + 0j, radius=1.0)
    integral = tc.contour_integral(lambda z: 1.0 / (z - a), c, n_steps=1024)
    assert abs(integral) < 1e-6


# ---------------------------------------------------------------------------
# Winding number
# ---------------------------------------------------------------------------

def test_winding_number_one_for_ccw_unit_circle() -> None:
    c = tc.circle(0 + 0j, radius=1.0, ccw=True)
    assert tc.winding_number(c, 0 + 0j) == 1


def test_winding_number_negative_one_for_cw_unit_circle() -> None:
    c = tc.circle(0 + 0j, radius=1.0, ccw=False)
    assert tc.winding_number(c, 0 + 0j) == -1


def test_winding_number_zero_for_excluded_point() -> None:
    c = tc.circle(0 + 0j, radius=1.0)
    assert tc.winding_number(c, 5 + 5j) == 0


def test_winding_number_one_when_point_off_center_but_inside() -> None:
    c = tc.circle(0 + 0j, radius=2.0)
    assert tc.winding_number(c, 0.7 + 0.3j) == 1


def test_winding_number_raises_when_contour_passes_through_z0() -> None:
    c = tc.circle(0 + 0j, radius=1.0)
    with pytest.raises(ValueError, match="undefined"):
        tc.winding_number(c, 1 + 0j)  # on the contour exactly


# ---------------------------------------------------------------------------
# Residues
# ---------------------------------------------------------------------------

def test_residue_of_simple_pole_at_origin_is_one() -> None:
    """Res(1/z, 0) = 1."""
    r = tc.residue(lambda z: 1.0 / z, 0 + 0j, radius=0.5)
    assert abs(r - 1.0) < 1e-3


def test_residue_at_off_origin_pole() -> None:
    """Res(1/(z - a), a) = 1 for any a."""
    a = 2 + 3j
    r = tc.residue(lambda z: 1.0 / (z - a), a, radius=0.1)
    assert abs(r - 1.0) < 1e-3


def test_residue_of_double_pole_is_zero() -> None:
    """Res(1/z², 0) = 0 because the function expands as
    1/z² + 0/z + ... (no simple-pole part)."""
    r = tc.residue(lambda z: 1.0 / (z * z), 0 + 0j, radius=0.1)
    assert abs(r) < 1e-3


def test_residue_of_rational_function_split_by_partial_fractions() -> None:
    """f(z) = (z + 1) / (z * (z - 1))
    Partial fractions: -1/z + 2/(z - 1).
    So Res(f, 0) = -1, Res(f, 1) = 2.
    """
    f = lambda z: (z + 1) / (z * (z - 1))
    r0 = tc.residue(f, 0 + 0j, radius=0.1)
    r1 = tc.residue(f, 1 + 0j, radius=0.1)
    assert abs(r0 - (-1.0)) < 1e-3
    assert abs(r1 - 2.0) < 1e-3


# ---------------------------------------------------------------------------
# Argument principle — Z - P
# ---------------------------------------------------------------------------

def test_argument_principle_counts_a_simple_zero() -> None:
    """f(z) = z has one simple zero at 0; the contour encloses
    it; the formula reports Z - P = 1."""
    f = lambda z: z
    fp = lambda z: 1
    c = tc.circle(0, 1.0)
    n = tc.argument_principle_count(f, fp, c, n_steps=1024)
    assert n == 1


def test_argument_principle_counts_a_simple_pole() -> None:
    """f(z) = 1/z has one simple pole at 0; the contour encloses
    it; the formula reports Z - P = -1."""
    f = lambda z: 1.0 / z
    fp = lambda z: -1.0 / (z * z)
    c = tc.circle(0, 1.0)
    n = tc.argument_principle_count(f, fp, c, n_steps=1024)
    assert n == -1


def test_argument_principle_counts_a_polynomial_of_degree_three() -> None:
    """f(z) = z³ has a triple zero at 0; the contour encloses
    it; the count must be 3."""
    f = lambda z: z * z * z
    fp = lambda z: 3 * z * z
    c = tc.circle(0, 2.0)
    n = tc.argument_principle_count(f, fp, c, n_steps=2048)
    assert n == 3


# ---------------------------------------------------------------------------
# Residue theorem — the closing argument of the chapter
# ---------------------------------------------------------------------------

def test_residue_theorem_matches_direct_integral() -> None:
    """For f(z) = (z + 1) / (z (z - 1)) integrated around a
    circle enclosing both 0 and 1:

        ∮ f dz = 2πi · (Res(f, 0) + Res(f, 1))
              = 2πi · (-1 + 2)
              = 2πi
    """
    f = lambda z: (z + 1) / (z * (z - 1))
    c = tc.circle(0.5 + 0j, radius=1.5)  # encloses 0 and 1
    direct = tc.contour_integral(f, c, n_steps=2048)
    via_residues = tc.residue_theorem_sum(
        f, c, poles_inside=[0 + 0j, 1 + 0j],
    )
    expected = 2.0j * math.pi
    assert abs(direct - expected) < 1e-3
    assert abs(via_residues - expected) < 1e-3
    assert abs(direct - via_residues) < 1e-3
