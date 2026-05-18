"""M7 Step 7 — conformal energy example.

Demonstrates that the M7 surface composes with :mod:`tessera.energy`
rather than inventing a parallel scalar reduction.  Specifically:

  - ``conformal_energy_on_sphere(p, p) == 0`` for any ``p`` not
    at the north pole.
  - The energy minimum is at ``p = p_target``.
  - The energy is symmetric (``E(p, q) == E(q, p)``).
  - Sphere-level samples around ``p_target`` yield monotonically
    increasing energy along great-circle interpolation.
  - The same value comes out of a direct call to
    :func:`tessera.energy.norm_sq` on the 2-D difference, proving
    the M7 → M6 wiring is real.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import complex as tc
from tessera import energy as te


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_sphere(seed: int = 0, n: int = 1) -> np.ndarray:
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=-1, keepdims=True)
    return pts


# ---------------------------------------------------------------------------
# Minimum + symmetry
# ---------------------------------------------------------------------------

def test_energy_is_zero_at_target() -> None:
    p_target = np.array([0.5, 0.5, 0.5 ** 0.5 - 0.5])  # arbitrary non-pole
    p_target = p_target / np.linalg.norm(p_target)
    e = tc.conformal_energy_on_sphere(p_target, p_target)
    assert float(e) == pytest.approx(0.0, abs=1e-12)


def test_energy_is_zero_at_target_batched() -> None:
    p_target = _unit_sphere(seed=1, n=5).reshape(5, 3)
    e = tc.conformal_energy_on_sphere(p_target, p_target)
    np.testing.assert_allclose(e, 0.0, atol=1e-12)


def test_energy_is_symmetric() -> None:
    p = _unit_sphere(seed=2, n=1)[0]
    q = _unit_sphere(seed=3, n=1)[0]
    e_pq = float(tc.conformal_energy_on_sphere(p, q))
    e_qp = float(tc.conformal_energy_on_sphere(q, p))
    assert pytest.approx(e_pq, rel=1e-12) == e_qp


# ---------------------------------------------------------------------------
# Monotonicity along a great-circle path
# ---------------------------------------------------------------------------

def _great_circle_step(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
    """Linearly interpolate between two unit vectors then
    re-normalize — a simple approximation to great-circle slerp
    that's good enough for monotonicity tests."""
    mid = (1.0 - t) * p + t * q
    return mid / np.linalg.norm(mid)


def test_energy_increases_as_we_move_away_from_target() -> None:
    """Walk from the target outward along a great-circle path;
    the conformal energy must grow monotonically."""
    p_target = np.array([0.0, 0.0, -1.0])  # south pole
    p_far = np.array([1.0, 0.0, 0.0])      # equator
    ts = np.linspace(0.0, 0.9, 8)
    energies = [
        float(tc.conformal_energy_on_sphere(
            _great_circle_step(p_target, p_far, float(t)),
            p_target,
        ))
        for t in ts
    ]
    diffs = np.diff(energies)
    assert (diffs >= 0).all(), f"energy not monotone: {energies}"


# ---------------------------------------------------------------------------
# Cross-wiring: conformal energy equals direct energy.norm_sq
# of the 2-D difference of stereographic projections
# ---------------------------------------------------------------------------

def test_energy_matches_direct_norm_sq_of_stereo_difference() -> None:
    """This is the M7 → M6 wiring claim: the conformal energy is
    NOT a new primitive — it's :func:`energy.norm_sq` composed
    with the stereographic projection.  Verify the equality
    holds at a sweep of points."""
    rng = np.random.RandomState(4)
    p_target = _unit_sphere(seed=5, n=1)[0]
    for _ in range(8):
        p = _unit_sphere(seed=int(rng.randint(0, 10_000)), n=1)[0]
        zeta_p = tc.stereographic(p)
        zeta_t = tc.stereographic(p_target)
        diff = np.stack([zeta_p.re - zeta_t.re, zeta_p.im - zeta_t.im], axis=-1)
        direct = te.norm_sq(diff)
        composed = tc.conformal_energy_on_sphere(p, p_target)
        np.testing.assert_allclose(direct, composed, atol=1e-12)


# ---------------------------------------------------------------------------
# Integration with the energy whitelist — proof that the M7 surface
# is compatible with M6's energy_jit by sharing primitives
# ---------------------------------------------------------------------------

def test_energy_norm_sq_is_the_underlying_primitive() -> None:
    """Sanity that ``tessera.energy.norm_sq`` is the canonical
    name we used — if this changes, the M7 wiring should be
    re-routed through whatever replaces it."""
    diff = np.array([3.0, 4.0])
    assert float(te.norm_sq(diff)) == 25.0


# ---------------------------------------------------------------------------
# Batched behavior
# ---------------------------------------------------------------------------

def test_conformal_energy_preserves_batch_shape() -> None:
    """Batched (N, 3) inputs produce a (N,) energy vector."""
    pts = _unit_sphere(seed=6, n=10)
    p_target = pts[0]
    energies = tc.conformal_energy_on_sphere(pts, p_target)
    assert energies.shape == (10,)
    assert float(energies[0]) == pytest.approx(0.0, abs=1e-12)
