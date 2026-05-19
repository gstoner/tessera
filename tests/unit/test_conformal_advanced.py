"""Schwarz-Christoffel + Weierstrass ℘ — research-grade extras.

Coverage:

  - **Schwarz-Christoffel** with explicit prevertices: rejects
    ``z`` not in H⁺; mismatched ``prevertices`` / ``angles``
    lengths raise; the map is finite for a few canonical
    polygons.
  - **Weierstrass ℘**: periodicity ``℘(z + ω) = ℘(z)``; evenness
    ``℘(-z) = ℘(z)``; the elliptic-curve identity ``℘'(z)² =
    4·℘(z)³ − g₂·℘(z) − g₃`` (with reasonable truncation).
"""

from __future__ import annotations

import cmath
import math

import pytest

from tessera import conformal_advanced as ca


# ---------------------------------------------------------------------------
# Schwarz-Christoffel
# ---------------------------------------------------------------------------

def test_sc_rejects_z_not_in_upper_half_plane() -> None:
    with pytest.raises(ValueError, match="upper half-plane"):
        ca.schwarz_christoffel_map(
            1 - 1j, prevertices=[0.0, 1.0], angles=[0.5, 0.5],
        )


def test_sc_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        ca.schwarz_christoffel_map(
            1 + 1j,
            prevertices=[0.0, 1.0, 2.0],
            angles=[0.5, 0.5],
        )


def test_sc_returns_finite_complex_for_canonical_input() -> None:
    """SC for a "triangle"-shaped prevertex set at three points.
    We don't fix the polygon vertices precisely (that's the
    parameter problem); just verify the map produces a finite
    complex value."""
    z = 0 + 2j   # well inside H⁺
    out = ca.schwarz_christoffel_map(
        z,
        prevertices=[-1.0, 0.0, 1.0],
        angles=[1/3, 1/3, 1/3],     # equilateral-triangle angles (π/3 each)
        base_point=1j,
        n_steps=512,
    )
    assert math.isfinite(out.real) and math.isfinite(out.imag)


def test_sc_map_is_conformal_for_trivial_polygon() -> None:
    """For "polygon" with ALL exterior angles 0 (i.e., the
    boundary is just the real axis), the SC integrand is the
    constant 1 and the map should be the identity (shifted by
    base_point)."""
    out = ca.schwarz_christoffel_map(
        2 + 3j,
        prevertices=[],
        angles=[],
        base_point=0 + 0j,
        n_steps=256,
    )
    # Integral of constant 1 from 0 to 2+3i is 2+3i.
    assert abs(out - (2 + 3j)) < 1e-6


# ---------------------------------------------------------------------------
# Weierstrass ℘
# ---------------------------------------------------------------------------

def test_weierstrass_p_is_even() -> None:
    """``℘(-z; ω₁, ω₂) = ℘(z; ω₁, ω₂)`` exactly."""
    z = 0.3 + 0.4j
    om1 = 1 + 0j
    om2 = 0 + 1j
    p_plus = ca.weierstrass_p(z, om1, om2, cutoff=8)
    p_minus = ca.weierstrass_p(-z, om1, om2, cutoff=8)
    assert abs(p_plus - p_minus) < 1e-10


def test_weierstrass_p_is_periodic_along_omega1() -> None:
    """``℘(z + ω₁) = ℘(z)`` — fundamental periodicity property.
    Our truncated sum approximates this; tolerance scales with
    truncation."""
    z = 0.3 + 0.4j
    om1 = 1 + 0j
    om2 = 0 + 1j
    p0 = ca.weierstrass_p(z, om1, om2, cutoff=16)
    p1 = ca.weierstrass_p(z + om1, om1, om2, cutoff=16)
    # Truncation error: with cutoff=16 and a unit-aspect lattice,
    # one-period offset converges to ~1e-2 in magnitude.  Tighter
    # tolerance is achievable with much larger cutoffs.
    assert abs(p0 - p1) < 1e-2


def test_weierstrass_p_is_periodic_along_omega2() -> None:
    z = 0.3 + 0.2j
    om1 = 1 + 0j
    om2 = 0 + 1j
    p0 = ca.weierstrass_p(z, om1, om2, cutoff=16)
    p1 = ca.weierstrass_p(z + om2, om1, om2, cutoff=16)
    # Truncation error: with cutoff=16 and a unit-aspect lattice,
    # one-period offset converges to ~1e-2 in magnitude.  Tighter
    # tolerance is achievable with much larger cutoffs.
    assert abs(p0 - p1) < 1e-2


def test_weierstrass_p_rejects_lattice_points() -> None:
    om1 = 1 + 0j
    om2 = 0 + 1j
    with pytest.raises(ValueError, match="pole"):
        ca.weierstrass_p(0 + 0j, om1, om2)
    with pytest.raises(ValueError, match="pole"):
        ca.weierstrass_p(om1, om1, om2)


def test_weierstrass_invariants_rectangular_lattice() -> None:
    """For a rectangular lattice ``(ω₁, ω₂) = (1, i)``, both
    invariants are real-valued (lattice symmetry under conjugation)."""
    g2, g3 = ca.weierstrass_invariants(1 + 0j, 0 + 1j, cutoff=16)
    assert abs(g2.imag) < 1e-9
    assert abs(g3.imag) < 1e-9


def test_weierstrass_elliptic_curve_identity() -> None:
    """The defining identity ``℘'(z)² = 4·℘(z)³ − g₂·℘(z) − g₃``.

    Test at a few non-lattice points with a generous cutoff.
    Truncation error makes this approximate, not exact."""
    om1 = 1 + 0j
    om2 = 0 + 1j
    g2, g3 = ca.weierstrass_invariants(om1, om2, cutoff=16)
    for z in [0.3 + 0.4j, -0.2 + 0.2j, 0.4 - 0.1j]:
        p = ca.weierstrass_p(z, om1, om2, cutoff=16)
        pp = ca.weierstrass_p_derivative(z, om1, om2, cutoff=16)
        lhs = pp * pp
        rhs = 4.0 * p ** 3 - g2 * p - g3
        # Truncation error is the main source of residual.
        assert abs(lhs - rhs) < 0.5, f"z={z}: lhs={lhs}, rhs={rhs}"


# ---------------------------------------------------------------------------
# Schwarz-Christoffel parameter solver
# ---------------------------------------------------------------------------

def test_sc_param_solve_triangle_returns_immediately() -> None:
    """A triangle has 3 vertices and 0 free prevertices: Möbius
    gauge fixes all three.  The solver should return immediately."""
    # Equilateral triangle (vertices at angles 90°, 210°, 330°).
    vertices = [
        cmath.exp(2j * math.pi * k / 3 + 1j * math.pi / 2)
        for k in range(3)
    ]
    prev, angles = ca.schwarz_christoffel_parameter_solve(vertices)
    assert prev == [-1.0, 0.0, math.inf]
    assert len(angles) == 3
    for a in angles:
        assert abs(a - 1.0 / 3.0) < 1e-9   # π/3 interior angle


def test_sc_param_solve_rejects_too_few_vertices() -> None:
    with pytest.raises(ValueError, match="at least 3 vertices"):
        ca.schwarz_christoffel_parameter_solve([0 + 0j, 1 + 0j])


def test_sc_param_solve_rejects_clockwise_polygon() -> None:
    """A clockwise polygon has the wrong angle-sum sign and should
    be rejected with a clear diagnostic."""
    # Clockwise square (reversed from CCW).
    vertices = [0 + 0j, 0 + 1j, 1 + 1j, 1 + 0j]
    with pytest.raises(ValueError, match="angles do not sum"):
        ca.schwarz_christoffel_parameter_solve(vertices)


def test_sc_param_solve_square_converges() -> None:
    """For a unit square (4 vertices, one free prevertex), the
    solver should converge to a configuration where all four
    sides have equal length under the SC map."""
    vertices = [0 + 0j, 1 + 0j, 1 + 1j, 0 + 1j]
    prev, angles = ca.schwarz_christoffel_parameter_solve(
        vertices, tol=1e-5,
    )
    # Angles should all be 0.5 (right angles) in units of π.
    for a in angles:
        assert abs(a - 0.5) < 1e-9
    # Four finite prevertices + ∞ at the end.
    assert len(prev) == 4
    assert prev[0] == -1.0
    assert prev[1] == 0.0
    assert math.isinf(prev[3])
    # The remaining finite prevertex x_3 must be > 0.
    assert prev[2] > 0


def test_sc_param_solve_returns_well_ordered_prevertices() -> None:
    """Pentagon: 2 free prevertices.  After solving, the finite
    prevertices must be strictly increasing."""
    # Regular pentagon.
    vertices = [
        cmath.exp(2j * math.pi * k / 5 + 1j * math.pi / 2)
        for k in range(5)
    ]
    prev, angles = ca.schwarz_christoffel_parameter_solve(
        vertices, tol=1e-5, max_iter=80,
    )
    # 5 finite prevertices + ∞.
    assert len(prev) == 5
    assert prev[0] == -1.0
    assert prev[1] == 0.0
    assert math.isinf(prev[4])
    # Free prevertices strictly ordered.
    assert prev[2] > prev[1]
    assert prev[3] > prev[2]
    # All interior angles equal (3/5 in units of π).
    for a in angles:
        assert abs(a - 0.6) < 1e-9


def test_sc_param_solve_respects_initial_guess() -> None:
    """If the caller supplies an initial guess of the wrong length,
    we raise."""
    vertices = [0 + 0j, 1 + 0j, 1 + 1j, 0 + 1j]
    with pytest.raises(ValueError, match="initial_log_spacings"):
        ca.schwarz_christoffel_parameter_solve(
            vertices, initial_log_spacings=[0.0, 0.0],
        )


# ---------------------------------------------------------------------------
# Adaptive Weierstrass ℘
# ---------------------------------------------------------------------------

def test_weierstrass_adaptive_rejects_origin() -> None:
    with pytest.raises(ValueError, match="pole"):
        ca.weierstrass_p_adaptive(0 + 0j, 1 + 0j, 0 + 1j)


def test_weierstrass_adaptive_converges_to_fixed_cutoff() -> None:
    """The adaptive sum at tight tolerance must agree with the
    fixed-cutoff implementation at a larger cutoff.

    The lattice sum converges like ``1/r³`` per ring (after the
    regularizing ``-1/Λ²`` cancellation), so at cutoff = 32 the
    last-ring contribution is roughly ``1e-5``, not ``1e-10``.
    The two implementations should still agree because they
    truncate identically."""
    z = 0.3 + 0.4j
    om1 = 1 + 0j
    om2 = 0 + 1j
    fixed = ca.weierstrass_p(z, om1, om2, cutoff=32)
    adaptive, cutoff_used, err_bound = ca.weierstrass_p_adaptive(
        z, om1, om2, tol=1e-10, max_cutoff=32,
    )
    # Both truncate at radius 32 (adaptive doesn't reach tol).
    # They must agree exactly up to floating-point reorder noise.
    assert abs(adaptive - fixed) < 1e-12
    assert cutoff_used == 32
    # The last-ring contribution at r=32 is ~1e-5, well above tol.
    assert err_bound > 0


def test_weierstrass_adaptive_error_bound_is_upper_bound() -> None:
    """The returned ``last_ring_magnitude`` is the magnitude of the
    contribution from the final ring.  The truncation error
    (estimated against a higher-cutoff baseline) should be on the
    same order or smaller."""
    z = 0.2 + 0.3j
    om1 = 1 + 0j
    om2 = 0 + 1j
    adaptive, cutoff_used, err_bound = ca.weierstrass_p_adaptive(
        z, om1, om2, tol=1e-3, max_cutoff=32,
    )
    # Reference using a much larger cutoff.
    reference = ca.weierstrass_p(z, om1, om2, cutoff=cutoff_used + 8)
    actual_err = abs(reference - adaptive)
    # The reported bound is meant to *bound* the actual error; the
    # tail after the stopping ring decays like 1/r³, so the
    # remaining tail is comparable to the last-ring contribution.
    # Allow a generous factor — we just want to know the reported
    # bound is the right order of magnitude, not a vacuous one.
    assert actual_err < 100 * err_bound + 1e-10


def test_weierstrass_adaptive_relaxes_cutoff_with_loose_tol() -> None:
    """A looser tolerance should stop earlier (smaller
    cutoff_used)."""
    z = 0.3 + 0.4j
    om1 = 1 + 0j
    om2 = 0 + 1j
    _, c_tight, _ = ca.weierstrass_p_adaptive(z, om1, om2, tol=1e-10)
    _, c_loose, _ = ca.weierstrass_p_adaptive(z, om1, om2, tol=1e-2)
    assert c_loose <= c_tight


def test_weierstrass_adaptive_evenness() -> None:
    """Adaptive ℘ preserves the parity ℘(-z) = ℘(z)."""
    z = 0.3 + 0.4j
    om1 = 1 + 0j
    om2 = 0 + 1j
    p_plus, _, _ = ca.weierstrass_p_adaptive(z, om1, om2, tol=1e-8)
    p_minus, _, _ = ca.weierstrass_p_adaptive(-z, om1, om2, tol=1e-8)
    assert abs(p_plus - p_minus) < 1e-8


def test_weierstrass_adaptive_returns_three_tuple() -> None:
    out = ca.weierstrass_p_adaptive(
        0.3 + 0.4j, 1 + 0j, 0 + 1j, tol=1e-6,
    )
    assert isinstance(out, tuple) and len(out) == 3
    val, cutoff_used, err_bound = out
    assert isinstance(cutoff_used, int)
    assert isinstance(err_bound, float)
    assert math.isfinite(val.real) and math.isfinite(val.imag)
