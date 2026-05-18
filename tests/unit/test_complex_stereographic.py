"""M7 Step 4 — stereographic projection tests.

Coverage:

  - Forward + inverse round-trip on a sphere-distributed sample.
  - North pole maps to ∞; ∞ maps back to the north pole.
  - The south pole maps to the origin.
  - Conformality: two tangent vectors on the sphere have the
    same angle between them after projection (and vice versa).
  - Inverse-then-forward identity on ℂ.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import complex as tc


# ---------------------------------------------------------------------------
# Helpers — sample points on S²
# ---------------------------------------------------------------------------

def _sphere_sample(n: int, *, seed: int = 0) -> np.ndarray:
    """Uniformly-distributed points on the unit sphere via the
    Marsaglia / normalized-Gaussian construction."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=-1, keepdims=True)
    return pts


# ---------------------------------------------------------------------------
# Forward / inverse round-trip
# ---------------------------------------------------------------------------

def test_inverse_then_forward_is_identity_on_complex_plane() -> None:
    """For any ζ ∈ ℂ, forward(inverse(ζ)) = ζ."""
    rng = np.random.RandomState(1)
    zeta = rng.randn(32) + 1j * rng.randn(32)
    p = tc.stereographic_inverse(tc.from_numpy(zeta))
    reconstructed = tc.stereographic(p).to_numpy()
    np.testing.assert_allclose(reconstructed, zeta, atol=1e-10)


def test_forward_then_inverse_is_identity_on_the_sphere() -> None:
    """For any p ∈ S² with z ≠ 1, inverse(forward(p)) = p."""
    pts = _sphere_sample(32, seed=2)
    # Remove any sample too close to the north pole for numerics.
    pts = pts[pts[:, 2] < 0.99]
    proj = tc.stereographic(pts)
    reconstructed = tc.stereographic_inverse(proj)
    np.testing.assert_allclose(reconstructed, pts, atol=1e-10)


# ---------------------------------------------------------------------------
# Distinguished points
# ---------------------------------------------------------------------------

def test_south_pole_maps_to_origin() -> None:
    out = tc.stereographic(np.array([0.0, 0.0, -1.0]))
    assert float(out.re) == 0.0
    assert float(out.im) == 0.0


def test_north_pole_maps_to_infinity() -> None:
    out = tc.stereographic(np.array([0.0, 0.0, 1.0]))
    assert np.isinf(float(out.re))
    assert np.isinf(float(out.im))


def test_infinity_maps_back_to_north_pole() -> None:
    out = tc.stereographic_inverse(tc.from_pair(np.inf, np.inf))
    np.testing.assert_allclose(out, [0.0, 0.0, 1.0])


def test_equator_maps_to_the_unit_circle() -> None:
    """Points with z = 0 (the equator) project to |ζ| = 1."""
    theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    equator = np.stack(
        [np.cos(theta), np.sin(theta), np.zeros_like(theta)],
        axis=-1,
    )
    out = tc.stereographic(equator).to_numpy()
    np.testing.assert_allclose(np.abs(out), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Conformality on the sphere
# ---------------------------------------------------------------------------

def _tangent_pair_on_sphere(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthogonal unit tangent vectors at ``p ∈ S²``.

    Constructs them by picking an arbitrary axis ``a`` not
    parallel to ``p``, then ``t₁ = (a × p) normalized``,
    ``t₂ = p × t₁``.
    """
    a = np.array([1.0, 0.0, 0.0]) if abs(p[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(a, p)
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(p, t1)
    return t1, t2


@pytest.mark.parametrize("seed", [3, 4, 5, 6])
def test_stereographic_preserves_angles_between_tangents(seed: int) -> None:
    """Pick a point p on S², build two orthogonal tangents at p,
    project p + ε·t for each tangent, and check the angle between
    the two projected differences equals the source angle (π/2)."""
    p = _sphere_sample(1, seed=seed)[0]
    # Avoid the immediate neighborhood of the north pole (numerics).
    if p[2] > 0.9:
        p = -p
    t1, t2 = _tangent_pair_on_sphere(p)
    h = 1e-5
    # Re-project (p + h·t) back onto the sphere to stay on S².
    def _to_sphere(v):
        return v / np.linalg.norm(v)
    q1 = _to_sphere(p + h * t1)
    q2 = _to_sphere(p + h * t2)
    zeta_p = tc.stereographic(p).to_numpy()
    zeta1 = tc.stereographic(q1).to_numpy()
    zeta2 = tc.stereographic(q2).to_numpy()
    d1 = zeta1 - zeta_p
    d2 = zeta2 - zeta_p
    # Angle between the projected tangents — should be π/2.
    cos_t = float(
        (d1.real * d2.real + d1.imag * d2.imag)
        / (np.abs(d1) * np.abs(d2))
    )
    angle = np.arccos(np.clip(cos_t, -1.0, 1.0))
    assert abs(angle - np.pi / 2) < 1e-3, (
        f"projected angle {angle} should be π/2; tangents were orthogonal "
        f"on the sphere (residual: {abs(angle - np.pi / 2):.3g})"
    )


# ---------------------------------------------------------------------------
# Batched inputs preserve shape
# ---------------------------------------------------------------------------

def test_stereographic_preserves_batch_shape() -> None:
    pts = _sphere_sample(20, seed=7).reshape(4, 5, 3)
    out = tc.stereographic(pts)
    assert out.shape == (4, 5)


def test_stereographic_inverse_preserves_batch_shape() -> None:
    rng = np.random.RandomState(8)
    zeta = (rng.randn(6, 4) + 1j * rng.randn(6, 4)).astype(np.complex128)
    pts = tc.stereographic_inverse(tc.from_numpy(zeta))
    assert pts.shape == (6, 4, 3)
