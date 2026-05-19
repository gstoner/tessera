"""Bundle C addendum — ``complex_sqrt`` with branch selection.

Coverage:

  - Principal-branch values at canonical inputs.
  - ``sqrt(z)² ≈ z`` everywhere except at the cut.
  - Branch ``branch=1`` returns the negated principal value.
  - Cut behavior: ``sqrt(-1 + 0i) = +i`` (principal cut on the
    negative real axis).
  - Matches ``numpy.sqrt`` on a complex array.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tessera import complex as tc


@pytest.mark.parametrize("z, expected", [
    (1 + 0j, 1 + 0j),
    (4 + 0j, 2 + 0j),
    (0 + 0j, 0 + 0j),
    (-1 + 0j, 0 + 1j),       # principal branch at the cut
    (-4 + 0j, 0 + 2j),
])
def test_sqrt_at_canonical_points(z, expected) -> None:
    out = tc.complex_sqrt(tc.from_numpy(z)).to_numpy()
    assert abs(out - expected) < 1e-12


def test_sqrt_squared_is_z() -> None:
    rng = np.random.RandomState(0)
    z = (rng.randn(16) + 1j * rng.randn(16)).astype(np.complex128)
    # Stay away from the cut for stability.
    z = z[z.real > -0.1]
    out = tc.complex_sqrt(tc.from_numpy(z)).to_numpy()
    round_trip = out * out
    np.testing.assert_allclose(round_trip, z, atol=1e-10)


def test_sqrt_branch_one_is_negation_of_principal() -> None:
    z = 1 + 1j
    principal = tc.complex_sqrt(tc.from_numpy(z), branch=0).to_numpy()
    other = tc.complex_sqrt(tc.from_numpy(z), branch=1).to_numpy()
    assert abs(principal + other) < 1e-12


def test_sqrt_principal_has_arg_in_right_half_plane() -> None:
    """The principal branch of sqrt sends z to the right
    half-plane: ``Re(√z) ≥ 0`` always (with ``Im(√z) ≥ 0`` on
    the positive-real-axis cut convention).  Equivalently
    ``arg(sqrt(z)) ∈ (-π/2, π/2]``."""
    rng = np.random.RandomState(1)
    z = (rng.randn(16) + 1j * rng.randn(16)).astype(np.complex128)
    z = z[np.abs(z) > 1e-3]
    out = tc.complex_sqrt(tc.from_numpy(z)).to_numpy()
    # Real part should be ≥ 0 for the principal branch.
    assert (out.real >= -1e-12).all()


def test_sqrt_matches_numpy_sqrt_on_sweep() -> None:
    rng = np.random.RandomState(2)
    z = (rng.randn(16) + 1j * rng.randn(16)).astype(np.complex128)
    z = z[z.real > -0.1]
    out = tc.complex_sqrt(tc.from_numpy(z)).to_numpy()
    np.testing.assert_allclose(out, np.sqrt(z), atol=1e-10)


def test_sqrt_invalid_branch_raises() -> None:
    with pytest.raises(ValueError, match="branch must be 0"):
        tc.complex_sqrt(1 + 0j, branch=2)


def test_sqrt_consistent_with_complex_pow_half() -> None:
    """``sqrt(z) == z^0.5`` on the principal branch (away from cut)."""
    z = 2 + 3j
    sqrt_val = tc.complex_sqrt(tc.from_numpy(z)).to_numpy()
    pow_val = tc.complex_pow(
        tc.from_numpy(z), tc.from_pair(0.5, 0.0),
    ).to_numpy()
    assert abs(sqrt_val - pow_val) < 1e-10


def test_sqrt_at_origin_is_origin() -> None:
    out = tc.complex_sqrt(tc.from_pair(0.0, 0.0)).to_numpy()
    assert out == 0 + 0j
