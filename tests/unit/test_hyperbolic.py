"""Hyperbolic geometry primitives — Poincaré disk + upper half-plane.

Coverage:

  - Distance from 0 in the disk matches the closed form
    ``d(0, z) = 2·artanh|z|``.
  - Distance is symmetric: ``d(z, w) == d(w, z)``.
  - Triangle inequality.
  - Möbius disk-automorphisms preserve hyperbolic distance.
  - Cayley transform is an isometry between H⁺ and 𝔻.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tessera import hyperbolic as h


# ---------------------------------------------------------------------------
# Poincaré disk distance
# ---------------------------------------------------------------------------

def test_distance_from_origin_matches_2_artanh() -> None:
    for r in (0.1, 0.3, 0.5, 0.7, 0.9):
        z = complex(r, 0)
        expected = 2.0 * math.atanh(r)
        assert abs(h.poincare_distance(0, z) - expected) < 1e-9


def test_distance_is_symmetric() -> None:
    z = 0.3 + 0.4j
    w = -0.2 + 0.1j
    a = h.poincare_distance(z, w)
    b = h.poincare_distance(w, z)
    assert abs(a - b) < 1e-12


def test_distance_at_zero_is_zero() -> None:
    assert h.poincare_distance(0 + 0j, 0 + 0j) == 0.0


def test_triangle_inequality_on_disk() -> None:
    rng = np.random.RandomState(0)
    for _ in range(8):
        pts = []
        while len(pts) < 3:
            cand = complex(rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8))
            if abs(cand) < 0.9:
                pts.append(cand)
        z, w, v = pts
        d_zv = h.poincare_distance(z, v)
        d_zw = h.poincare_distance(z, w)
        d_wv = h.poincare_distance(w, v)
        # d(z, v) ≤ d(z, w) + d(w, v).  Allow tiny FP slack.
        assert d_zv <= d_zw + d_wv + 1e-9


def test_distance_diverges_near_boundary() -> None:
    """As ``|z| → 1``, the hyperbolic distance from 0 → ∞."""
    near_boundary = complex(0.9999, 0)
    d = h.poincare_distance(0, near_boundary)
    assert d > 5.0  # large; numerical scale


def test_distance_rejects_points_on_or_outside_boundary() -> None:
    with pytest.raises(ValueError, match=r"\|z\|, \|w\| < 1"):
        h.poincare_distance(1 + 0j, 0)
    with pytest.raises(ValueError, match=r"\|z\|, \|w\| < 1"):
        h.poincare_distance(0, 2 + 0j)


# ---------------------------------------------------------------------------
# Möbius isometry of the disk
# ---------------------------------------------------------------------------

def test_blaschke_automorphism_is_an_isometry() -> None:
    """A Blaschke disk-automorphism ``z ↦ (z − a)/(1 − ā·z)``
    preserves the Poincaré distance."""
    a = 0.3 + 0.1j   # center to move to 0
    # Blaschke factor: M(z) = (z - a) / (1 - conj(a) * z).
    # Coefs: a_coef=1, b_coef=-a, c_coef=-conj(a), d_coef=1.
    coefs = (1 + 0j, -a, -a.conjugate(), 1 + 0j)
    for z, w in [
        (0.2 + 0.3j, -0.1 + 0.4j),
        (0.0 + 0.0j, 0.5 + 0.0j),
        (0.4 - 0.2j, -0.3 + 0.5j),
    ]:
        d_before = h.poincare_distance(z, w)
        Mz = h.poincare_isometry_image(coefs, z)
        Mw = h.poincare_isometry_image(coefs, w)
        d_after = h.poincare_distance(Mz, Mw)
        assert abs(d_before - d_after) < 1e-6


# ---------------------------------------------------------------------------
# Upper half-plane distance
# ---------------------------------------------------------------------------

def test_upper_half_plane_distance_is_zero_at_same_point() -> None:
    z = 2 + 3j
    assert h.upper_half_plane_distance(z, z) == 0.0


def test_upper_half_plane_distance_along_vertical_geodesic() -> None:
    """For two points on the same vertical line in H⁺, the
    hyperbolic distance has the closed form ``|log(y2/y1)|``."""
    z = 0 + 1j      # y = 1
    w = 0 + 3j      # y = 3
    expected = math.log(3.0)
    assert abs(h.upper_half_plane_distance(z, w) - expected) < 1e-9


def test_upper_half_plane_distance_is_symmetric() -> None:
    z = 1 + 2j
    w = -1 + 5j
    a = h.upper_half_plane_distance(z, w)
    b = h.upper_half_plane_distance(w, z)
    assert abs(a - b) < 1e-12


def test_upper_half_plane_rejects_real_axis_points() -> None:
    with pytest.raises(ValueError, match="Im\\(z\\), Im\\(w\\) > 0"):
        h.upper_half_plane_distance(1 + 0j, 1 + 1j)


# ---------------------------------------------------------------------------
# Cayley isometry between H⁺ and 𝔻
# ---------------------------------------------------------------------------

def test_cayley_transform_round_trip() -> None:
    """``cayley_from_disk(cayley_to_disk(z)) == z``."""
    rng = np.random.RandomState(1)
    for _ in range(5):
        z = complex(rng.uniform(-2, 2), rng.uniform(0.5, 3))
        round_trip = h.cayley_from_disk(h.cayley_to_disk(z))
        assert abs(z - round_trip) < 1e-9


def test_cayley_is_an_isometry() -> None:
    """The Cayley transform preserves hyperbolic distance —
    distance in H⁺ equals distance of the images in 𝔻."""
    z = 1 + 2j
    w = -1 + 3j
    d_h = h.upper_half_plane_distance(z, w)
    z_d = h.cayley_to_disk(z)
    w_d = h.cayley_to_disk(w)
    d_d = h.poincare_distance(z_d, w_d)
    assert abs(d_h - d_d) < 1e-6
