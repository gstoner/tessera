"""M7 Step 3 — Mobius transformation tests.

Coverage:

  - Identity-coefficient Mobius is the identity map.
  - Inversion ``z → 1/z`` round-trips.
  - The composition of two Mobius maps equals the Mobius map of
    the matrix-multiplied coefficients (the group law).
  - **Generalized-circle preservation**: a true circle in ℂ stays
    a true circle (or becomes a line) after a Mobius map.  The
    test discretizes a circle, applies the map, and checks the
    images stay equidistant from a common center within tight
    tolerance.
  - Pole policy: ``z = −d/c`` produces ``inf + inf·i`` (point at
    infinity) without raising.
  - Singular-matrix detection: ``a·d − b·c = 0`` raises.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import complex as tc


# ---------------------------------------------------------------------------
# Identity / inversion
# ---------------------------------------------------------------------------

def test_identity_mobius_is_the_identity_map() -> None:
    """a=1, b=0, c=0, d=1 ⇒ f(z) = z / 1 = z."""
    rng = np.random.RandomState(0)
    z = rng.randn(8) + 1j * rng.randn(8)
    out = tc.mobius(
        tc.from_numpy(z),
        a=1.0, b=0.0, c=0.0, d=1.0,
    ).to_numpy()
    np.testing.assert_allclose(out, z, atol=1e-12)


def test_inversion_mobius_round_trips() -> None:
    """f(z) = 1/z ⇒ f(f(z)) = z everywhere except 0."""
    rng = np.random.RandomState(1)
    z = rng.randn(8) + 1j * rng.randn(8)
    # Skip values too close to zero.
    z = z[np.abs(z) > 1e-2]
    once = tc.mobius(tc.from_numpy(z), a=0.0, b=1.0, c=1.0, d=0.0)
    twice = tc.mobius(once, a=0.0, b=1.0, c=1.0, d=0.0)
    np.testing.assert_allclose(twice.to_numpy(), z, atol=1e-9)


# ---------------------------------------------------------------------------
# Group law — composition equals matrix multiplication
# ---------------------------------------------------------------------------

def _mobius_complex(z: complex, a, b, c, d) -> complex:
    """numpy-complex reference: f(z) = (az + b) / (cz + d)."""
    return (a * z + b) / (c * z + d)


def test_mobius_composition_equals_matrix_multiplied_coefficients() -> None:
    """f ∘ g corresponds to multiplying the coefficient matrices.

    Given f₁ = (a₁z + b₁)/(c₁z + d₁) and f₂ = (a₂z + b₂)/(c₂z + d₂),
    f₁(f₂(z)) is itself a Mobius map with coefficients
    [a₁ b₁; c₁ d₁] @ [a₂ b₂; c₂ d₂].
    """
    M1 = np.array([[1.0 + 0.2j, 0.5 - 0.3j],
                   [0.1 + 0.0j, 1.0 + 0.0j]])
    M2 = np.array([[0.7 + 0.4j, 0.1 + 0.6j],
                   [0.2 + 0.1j, 0.9 + 0.0j]])
    M = M1 @ M2

    rng = np.random.RandomState(2)
    z = rng.randn(8) + 1j * rng.randn(8)
    z = z[np.abs(z) > 0.1]  # stay away from poles for stability

    a2, b2 = M2[0, 0], M2[0, 1]
    c2, d2 = M2[1, 0], M2[1, 1]
    a1, b1 = M1[0, 0], M1[0, 1]
    c1, d1 = M1[1, 0], M1[1, 1]

    expected = np.array([
        _mobius_complex(_mobius_complex(zi, a2, b2, c2, d2), a1, b1, c1, d1)
        for zi in z
    ])
    # Via the composed matrix:
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    composed = tc.mobius(
        tc.from_numpy(z),
        a=tc.from_pair(a.real, a.imag),
        b=tc.from_pair(b.real, b.imag),
        c=tc.from_pair(c.real, c.imag),
        d=tc.from_pair(d.real, d.imag),
    ).to_numpy()
    np.testing.assert_allclose(composed, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# Generalized-circle preservation
# ---------------------------------------------------------------------------

def _least_squares_circle(points: np.ndarray) -> tuple[complex, float]:
    """Fit a circle to a sequence of complex points.  Returns
    ``(center, radius)``.  Uses the algebraic form
    ``x² + y² + Dx + Ey + F = 0``."""
    x = points.real
    y = points.imag
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c0 = sol
    center = complex(cx, cy)
    radius = float(np.sqrt(cx * cx + cy * cy + c0))
    return center, radius


def test_mobius_maps_a_circle_to_a_circle() -> None:
    """A Mobius transformation maps generalized circles to
    generalized circles.  Discretize the unit circle, map it, and
    check the images are equidistant from a common center."""
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    circle = np.exp(1j * theta)
    # A non-trivial Mobius that's known to map circles to circles
    # (and doesn't have a pole on the unit circle).
    out = tc.mobius(
        tc.from_numpy(circle),
        a=1.0 + 0.0j, b=2.0 + 0.0j, c=0.0 + 0.0j, d=3.0 + 0.0j,
    ).to_numpy()
    center, radius = _least_squares_circle(out)
    residuals = np.abs(out - center) - radius
    assert float(np.abs(residuals).max()) < 1e-6, (
        f"Mobius image should be a circle within 1e-6; got max "
        f"residual {float(np.abs(residuals).max()):.3g}"
    )


def test_inversion_maps_a_circle_through_origin_to_a_line() -> None:
    """Special case: a circle that passes through the origin gets
    mapped to a straight line by inversion ``z → 1/z``.  We check
    that the image is approximately collinear."""
    # Circle centered at z=1, radius 1 ⇒ passes through origin.
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    circle = 1.0 + np.exp(1j * theta)
    # Drop the point at the origin to avoid division-by-zero edge.
    keep = np.abs(circle) > 1e-3
    circle = circle[keep]
    out = tc.mobius(
        tc.from_numpy(circle),
        a=0.0 + 0.0j, b=1.0 + 0.0j, c=1.0 + 0.0j, d=0.0 + 0.0j,
    ).to_numpy()
    # The image should lie on the line Re(z) = 0.5.  Check via
    # variance of the real part.
    np.testing.assert_allclose(out.real, 0.5, atol=1e-9)


# ---------------------------------------------------------------------------
# Pole + singular-matrix policy
# ---------------------------------------------------------------------------

def test_mobius_at_the_pole_returns_point_at_infinity() -> None:
    """f(z) = 1/z at z=0 ⇒ ∞.  We surface this as
    ``ComplexScalar(+inf, +inf)`` so a downstream
    stereographic-projection step can route it to the north pole."""
    out = tc.mobius(
        tc.from_pair(0.0, 0.0),
        a=0.0, b=1.0, c=1.0, d=0.0,
    )
    assert np.isinf(out.re).all()
    assert np.isinf(out.im).all()


def test_mobius_rejects_singular_matrix() -> None:
    """a·d − b·c = 0 ⇒ the map collapses everything to one point.
    Catch at construction."""
    with pytest.raises(ValueError, match="singular"):
        tc.mobius(
            tc.from_pair(1.0, 0.0),
            a=1.0, b=2.0, c=2.0, d=4.0,  # det = 1*4 - 2*2 = 0
        )
